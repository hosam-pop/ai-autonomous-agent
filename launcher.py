"""
Autonomous launcher for the local Claw Code agent.

Boots (or verifies) the llama.cpp OpenAI-compatible server, the
Anthropic->OpenAI proxy, and the `claw` Rust CLI. Retries each step up to
MAX_ATTEMPTS total, applies corrective actions, and logs every attempt to
/opt/log.txt.

Usage:
    python /opt/claw_agent/main.py --autonomous [--prompt "..."]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("/opt/log.txt")
MODEL_PATH = Path(os.environ.get("CLAW_MODEL_PATH", "/opt/model/qwen2.5-coder-7b-instruct-q4_k_m.gguf"))
MODEL_NAME_DEFAULT = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
CLAW_BIN = Path("/opt/claw_agent/rust/target/debug/claw")
LLAMA_SERVER_BIN = Path(os.environ.get(
    "LLAMA_SERVER_BIN", "/opt/llama.cpp/llama-b8838/llama-server"
))
LLAMA_LIB_DIR = LLAMA_SERVER_BIN.parent
PROXY_MODULE = "proxy_anthropic_to_openai:app"
PROXY_DIR = Path("/opt/claw_agent")
LLAMA_LOG = Path("/tmp/llama_server.log")
PROXY_LOG = Path("/tmp/proxy.log")
MAX_ATTEMPTS = 20
DEFAULT_PROMPT = "Reply with exactly one sentence introducing yourself."
CLAW_CWD = Path("/opt/claw_agent/workspace")


@dataclass
class AttemptResult:
    ok: bool
    detail: str


def log(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.25)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except OSError:
            return False


def pick_open_port(preferred: int, fallback: int) -> int:
    return preferred if not port_in_use(preferred) else fallback


def wait_for_http(url: str, timeout: float) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 500:
                    return True
        except Exception:
            time.sleep(1.0)
    return False


def spawn(cmd: list[str], logfile: Path, cwd: Path | None = None, env: dict | None = None) -> subprocess.Popen:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    fh = logfile.open("ab")
    return subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
        env=env,
        preexec_fn=os.setsid,
    )


def ensure_llama_server(port: int, n_ctx: int = 16384) -> AttemptResult:
    if port_in_use(port):
        if wait_for_http(f"http://127.0.0.1:{port}/v1/models", 5):
            return AttemptResult(True, f"llama already healthy on :{port}")
        return AttemptResult(False, f"port {port} busy but not serving /v1/models")
    if not MODEL_PATH.exists():
        return AttemptResult(False, f"model missing at {MODEL_PATH}")
    if not LLAMA_SERVER_BIN.exists():
        return AttemptResult(False, f"llama-server missing at {LLAMA_SERVER_BIN}")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(LLAMA_LIB_DIR) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    cmd = [
        str(LLAMA_SERVER_BIN),
        "-m",
        str(MODEL_PATH),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        str(n_ctx),
        "-t",
        str(env.get("CLAW_N_THREADS", "4")),
        "--chat-template",
        "chatml",
    ]
    spawn(cmd, LLAMA_LOG, env=env)
    if wait_for_http(f"http://127.0.0.1:{port}/v1/models", 180):
        return AttemptResult(True, f"llama-server started on :{port}")
    return AttemptResult(False, "llama-server did not become healthy within 180s")


def ensure_proxy(port: int, backend_port: int) -> AttemptResult:
    if port_in_use(port):
        if wait_for_http(f"http://127.0.0.1:{port}/healthz", 5):
            return AttemptResult(True, f"proxy already healthy on :{port}")
        return AttemptResult(False, f"port {port} busy but not serving /healthz")
    env = os.environ.copy()
    env["OPENAI_BACKEND"] = f"http://127.0.0.1:{backend_port}/v1"
    env["BACKEND_MODEL"] = env.get("BACKEND_MODEL", "qwen-coder")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        PROXY_MODULE,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    spawn(cmd, PROXY_LOG, cwd=PROXY_DIR, env=env)
    if wait_for_http(f"http://127.0.0.1:{port}/healthz", 60):
        return AttemptResult(True, f"proxy started on :{port}")
    return AttemptResult(False, "proxy did not become healthy within 60s")


def run_claw(prompt: str, proxy_port: int, timeout: int = 1800) -> AttemptResult:
    if not CLAW_BIN.exists():
        return AttemptResult(False, f"claw binary missing at {CLAW_BIN}")
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = "local-dummy"
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{proxy_port}"
    env["ANTHROPIC_MODEL"] = env.get("ANTHROPIC_MODEL", "claude-3-5-sonnet")
    # Disable any external provider fallbacks.
    for var in (
        "XAI_BASE_URL",
        "DASHSCOPE_BASE_URL",
        "OPENAI_BASE_URL",
    ):
        env.pop(var, None)
    cmd = [str(CLAW_BIN), "--output-format", "json", "prompt", prompt]
    CLAW_CWD.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(CLAW_CWD),
        )
    except subprocess.TimeoutExpired:
        return AttemptResult(False, f"claw timed out after {timeout}s")
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-400:]
        return AttemptResult(False, f"claw exited {result.returncode}: {tail!r}")
    out = (result.stdout or "").strip()
    try:
        parsed = json.loads(out.splitlines()[-1])
        msg = parsed.get("message") or parsed.get("output") or ""
    except Exception:
        msg = out[:400]
    return AttemptResult(True, f"claw ok; message={msg[:200]!r}")


def run_autonomous(prompt: str) -> int:
    proxy_port = int(os.environ.get("CLAW_PROXY_PORT", "8080"))
    llama_port = int(os.environ.get("CLAW_LLAMA_PORT", "8081"))
    attempt = 0
    last_error = "no attempts recorded"
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        try:
            log(f"attempt={attempt} stage=llama target_port={llama_port}")
            r = ensure_llama_server(llama_port)
            log(f"attempt={attempt} stage=llama ok={r.ok} detail={r.detail!r}")
            if not r.ok:
                last_error = r.detail
                if "port" in r.detail and "busy" in r.detail:
                    llama_port = 8082 if llama_port == 8081 else llama_port + 1
                    log(f"attempt={attempt} action=switch_llama_port->{llama_port}")
                elif "model missing" in r.detail:
                    log(f"attempt={attempt} action=redownload_model")
                    subprocess.run([
                        "hf", "download",
                        MODEL_NAME_DEFAULT,
                        MODEL_PATH.name,
                        "--local-dir", str(MODEL_PATH.parent),
                    ], check=False)
                continue

            log(f"attempt={attempt} stage=proxy target_port={proxy_port}")
            r = ensure_proxy(proxy_port, llama_port)
            log(f"attempt={attempt} stage=proxy ok={r.ok} detail={r.detail!r}")
            if not r.ok:
                last_error = r.detail
                if "port" in r.detail and "busy" in r.detail:
                    proxy_port = 8090 if proxy_port == 8080 else proxy_port + 1
                    log(f"attempt={attempt} action=switch_proxy_port->{proxy_port}")
                continue

            log(f"attempt={attempt} stage=claw prompt={prompt[:80]!r}")
            r = run_claw(prompt, proxy_port)
            log(f"attempt={attempt} stage=claw ok={r.ok} detail={r.detail!r}")
            if not r.ok and "maximum context length" in r.detail:
                log(f"attempt={attempt} action=restart_llama_with_larger_context")
                subprocess.run(["pkill", "-f", "llama_cpp.server"], check=False)
                time.sleep(4)
            if r.ok:
                log(
                    f"SUCCESS attempts={attempt} proxy_port={proxy_port} "
                    f"llama_port={llama_port} model={MODEL_PATH.name}"
                )
                print(
                    f"\n[READY] Autonomous agent is running. Proxy http://127.0.0.1:{proxy_port} "
                    f"-> llama http://127.0.0.1:{llama_port} (model={MODEL_PATH.name}). "
                    f"Attempts={attempt}."
                )
                return 0
            last_error = r.detail
        except Exception as exc:  # noqa: BLE001
            last_error = f"unhandled {type(exc).__name__}: {exc}"
            log(f"attempt={attempt} unhandled={last_error!r}")
        time.sleep(2.0)
    log(f"FAILURE attempts={attempt} last_error={last_error!r}")
    print(f"\n[FAILED] last_error={last_error}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Claw Code autonomous launcher")
    parser.add_argument("--autonomous", action="store_true")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    if not args.autonomous:
        parser.error("must be run with --autonomous")
    LOG_PATH.touch(exist_ok=True)
    log("LAUNCH mode=autonomous")
    return run_autonomous(args.prompt)


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    raise SystemExit(main())
