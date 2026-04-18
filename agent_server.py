"""Devin-style local autonomous agent.

Architecture:
  - Planner   → decomposes user request into a task list (JSON)
  - Executor  → for each task runs a ReAct loop using typed tools
  - Critic    → reviews outputs, decides done/retry, records notes
  - Memory    → appends every task + critique to workspace/.memory/history.json

Endpoints:
  GET  /           — single-page UI
  POST /run        — full pipeline (plan → execute → critique), SSE stream
  POST /agent      — single-step ReAct loop, SSE (fast mode)
  POST /claw       — delegate to Claw Code Rust CLI, SSE (slow mode)
  GET  /files      — list workspace files
  GET  /file       — read a workspace file
  POST /reset      — wipe workspace
  GET  /memory     — last N memory entries
  GET  /healthz    — backend/claw status
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# ---------------------------- config ----------------------------
BACKEND = os.environ.get("OPENAI_BACKEND", "http://127.0.0.1:8081/v1")
MODEL_LABEL = os.environ.get("MODEL_LABEL", "Qwen2.5-Coder-7B (local)")
WORKSPACE = Path(os.environ.get("AGENT_WORKSPACE", "/home/ubuntu/agent_workspace")).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)
MEMORY_DIR = WORKSPACE / ".memory"
MEMORY_DIR.mkdir(exist_ok=True)
MEMORY_FILE = MEMORY_DIR / "history.json"

CLAW_BIN = Path(os.environ.get("CLAW_BIN", "/opt/claw_agent/rust/target/debug/claw"))
CLAW_CWD = Path(os.environ.get("CLAW_CWD", "/opt/claw_agent/workspace"))
CLAW_CWD.mkdir(parents=True, exist_ok=True)
CLAW_PROXY = os.environ.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8080")

MAX_STEPS_PER_TASK = int(os.environ.get("AGENT_STEPS", "6"))
MAX_TASKS = int(os.environ.get("AGENT_TASKS", "3"))
SHELL_TIMEOUT = int(os.environ.get("AGENT_SHELL_TIMEOUT", "45"))
PY_TIMEOUT = int(os.environ.get("AGENT_PY_TIMEOUT", "45"))
CLAW_TIMEOUT = int(os.environ.get("CLAW_TIMEOUT", "1800"))
MAX_OUT_BYTES = 12_000
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "600"))

app = FastAPI()

# ---------------------------- prompts ----------------------------

PLANNER_PROMPT = f"""You are the PLANNER of a local autonomous coding agent.
Given the user's request, output a JSON plan:

  {{"plan": [{{"task": "<concrete sub-goal>"}}, ...], "summary": "<one-line goal>"}}

Rules:
- 1 to {MAX_TASKS} tasks, each actionable in under {MAX_STEPS_PER_TASK} tool calls.
- Output ONLY the JSON. No prose, no code fences.
- If the request is just a question with no file/code changes, return a plan of one "answer" task.
- Order matters: earlier tasks produce artefacts later tasks use.
"""

EXECUTOR_PROMPT = f"""You are the EXECUTOR of a local autonomous coding agent.
Workspace: `{WORKSPACE}` (all file paths are relative to it).
You MUST output EXACTLY ONE JSON object per turn, no prose, no fences.

Available tools:
  {{"tool": "shell",   "command": "<bash>"}}
  {{"tool": "write",   "path": "file.ext", "content": "<full content>"}}
  {{"tool": "read",    "path": "file.ext"}}
  {{"tool": "list",    "path": "."}}
  {{"tool": "search",  "pattern": "<regex>", "path": "."}}
  {{"tool": "python",  "code": "<python source to run>"}}
  {{"tool": "install", "package": "<pip name>"}}
  {{"tool": "finish",  "message": "<short result for user>"}}

After each tool call, you receive OBSERVATION JSON. Keep going until you emit `finish`
with a concise result for THIS task. Steps per task are capped at {MAX_STEPS_PER_TASK}.
Prefer small, verifiable steps. Never fabricate files — read or list first."""

CRITIC_PROMPT = """You are the CRITIC of a local autonomous coding agent.
Given the user's original request, the plan, and the per-task results, output ONLY JSON:

  {"ok": true|false, "score": 0-100, "notes": "<1-3 sentences>", "final_answer": "<what to tell user>"}

- `ok` = was the user's goal achieved well?
- `score` = quality 0-100.
- `final_answer` = polished message for the user (may include markdown). Must be self-contained.
"""

# ---------------------------- low-level helpers ----------------------------

def _safe(rel: str) -> Path:
    p = (WORKSPACE / rel).resolve()
    if not str(p).startswith(str(WORKSPACE)):
        raise ValueError("path escapes workspace")
    return p


def sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    m = JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:  # noqa: BLE001
            pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:  # noqa: BLE001
                    return None
    return None


async def llm(messages: list[dict[str, Any]], max_tokens: int = LLM_MAX_TOKENS, temperature: float = 0.2) -> str:
    payload = {
        "model": "local",
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{BACKEND}/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")


# ---------------------------- tools ----------------------------

def tool_shell(command: str) -> dict[str, Any]:
    try:
        r = subprocess.run(["bash", "-lc", command], cwd=str(WORKSPACE),
                           capture_output=True, text=True, timeout=SHELL_TIMEOUT)
        return {"stdout": r.stdout[-MAX_OUT_BYTES:], "stderr": r.stderr[-MAX_OUT_BYTES:], "exit": r.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"timeout after {SHELL_TIMEOUT}s", "exit": 124}
    except Exception as exc:  # noqa: BLE001
        return {"stdout": "", "stderr": str(exc), "exit": 1}


def tool_write(path: str, content: str) -> dict[str, Any]:
    try:
        p = _safe(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"ok": True, "path": str(p.relative_to(WORKSPACE)), "bytes": len(content)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def tool_read(path: str) -> dict[str, Any]:
    try:
        p = _safe(path)
        if not p.exists():
            return {"ok": False, "error": "not found"}
        data = p.read_text(encoding="utf-8", errors="replace")
        return {"ok": True, "content": data[:MAX_OUT_BYTES], "bytes": len(data),
                "truncated": len(data) > MAX_OUT_BYTES}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def tool_list(path: str = ".") -> dict[str, Any]:
    try:
        base = _safe(path)
        if not base.exists():
            return {"ok": False, "error": "not found"}
        items = []
        for p in sorted(base.iterdir()):
            items.append({"name": p.name, "type": "dir" if p.is_dir() else "file",
                          "size": p.stat().st_size if p.is_file() else 0})
        return {"ok": True, "path": str(base.relative_to(WORKSPACE)) or ".", "items": items}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def tool_search(pattern: str, path: str = ".") -> dict[str, Any]:
    try:
        base = _safe(path)
        r = subprocess.run(["rg", "-n", "--max-count", "20", "--no-heading", pattern, str(base)],
                           capture_output=True, text=True, timeout=20)
        matches = [ln for ln in r.stdout.splitlines()[:50] if ln]
        return {"ok": True, "matches": matches, "pattern": pattern}
    except FileNotFoundError:
        # fallback to grep
        try:
            r = subprocess.run(["grep", "-rn", "--max-count=20", pattern, str(base)],
                               capture_output=True, text=True, timeout=20)
            return {"ok": True, "matches": r.stdout.splitlines()[:50], "pattern": pattern}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def tool_python(code: str) -> dict[str, Any]:
    try:
        tmp = WORKSPACE / ".__run.py"
        tmp.write_text(code, encoding="utf-8")
        r = subprocess.run(["python3", str(tmp)], cwd=str(WORKSPACE),
                           capture_output=True, text=True, timeout=PY_TIMEOUT)
        return {"stdout": r.stdout[-MAX_OUT_BYTES:], "stderr": r.stderr[-MAX_OUT_BYTES:], "exit": r.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"timeout after {PY_TIMEOUT}s", "exit": 124}
    except Exception as exc:  # noqa: BLE001
        return {"stdout": "", "stderr": str(exc), "exit": 1}
    finally:
        try:
            (WORKSPACE / ".__run.py").unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


def tool_install(package: str) -> dict[str, Any]:
    if not re.match(r"^[A-Za-z0-9_.\-]+(\[[^\]]+\])?(==[\w\.\-]+)?$", package):
        return {"ok": False, "error": "invalid package name"}
    try:
        r = subprocess.run(["pip", "install", "--user", "--quiet", package],
                           capture_output=True, text=True, timeout=120)
        return {"ok": r.returncode == 0, "stdout": r.stdout[-2000:], "stderr": r.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "pip install timeout"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def dispatch_tool(action: dict[str, Any]) -> dict[str, Any]:
    t = action.get("tool") or action.get("action")
    if t == "shell":
        return tool_shell(str(action.get("command", "")))
    if t == "write":
        return tool_write(str(action.get("path", "")), str(action.get("content", "")))
    if t == "read":
        return tool_read(str(action.get("path", "")))
    if t == "list":
        return tool_list(str(action.get("path", ".")))
    if t == "search":
        return tool_search(str(action.get("pattern", "")), str(action.get("path", ".")))
    if t == "python":
        return tool_python(str(action.get("code", "")))
    if t == "install":
        return tool_install(str(action.get("package", "")))
    return {"ok": False, "error": f"unknown tool {t!r}"}


# ---------------------------- workspace helpers ----------------------------

def list_workspace() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(WORKSPACE.rglob("*")):
        if any(part.startswith(".") for part in p.relative_to(WORKSPACE).parts):
            continue
        if p.is_file():
            try:
                st = p.stat()
                out.append({"path": str(p.relative_to(WORKSPACE)), "size": st.st_size, "mtime": int(st.st_mtime)})
            except Exception:  # noqa: BLE001
                pass
    return out


def load_memory(limit: int = 20) -> list[dict[str, Any]]:
    if not MEMORY_FILE.exists():
        return []
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data[-limit:]
    except Exception:  # noqa: BLE001
        pass
    return []


def save_memory(entry: dict[str, Any]) -> None:
    data = []
    if MEMORY_FILE.exists():
        try:
            data = json.loads(MEMORY_FILE.read_text(encoding="utf-8")) or []
        except Exception:  # noqa: BLE001
            data = []
    data.append(entry)
    MEMORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------- endpoints ----------------------------

@app.get("/healthz")
async def healthz() -> dict:
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            r = await client.get(f"{BACKEND}/models")
            return {"ok": r.status_code < 500, "backend": BACKEND, "workspace": str(WORKSPACE),
                    "claw_bin": str(CLAW_BIN), "claw_available": CLAW_BIN.exists(),
                    "memory_entries": len(load_memory(10_000))}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "backend": BACKEND, "error": str(exc)}


@app.get("/files")
def files() -> JSONResponse:
    return JSONResponse(content={"files": list_workspace(), "root": str(WORKSPACE)})


@app.get("/file")
def file_content(path: str) -> JSONResponse:
    try:
        p = _safe(path)
        if not p.exists() or not p.is_file():
            return JSONResponse(status_code=404, content={"error": "not found"})
        data = p.read_text(encoding="utf-8", errors="replace")
        return JSONResponse(content={"path": path, "content": data[:200_000]})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=400, content={"error": str(exc)})


@app.post("/reset")
def reset() -> JSONResponse:
    subprocess.run(["rm", "-rf", str(WORKSPACE)], check=False)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(exist_ok=True)
    return JSONResponse(content={"ok": True, "workspace": str(WORKSPACE)})


@app.get("/memory")
def memory(limit: int = 20) -> JSONResponse:
    return JSONResponse(content={"entries": load_memory(limit)})


# ---------------------------- /agent (fast ReAct) ----------------------------

@app.post("/agent")
async def agent_endpoint(request: Request) -> StreamingResponse:
    body = await request.json()
    user_msg = (body.get("message") or "").strip()
    history = body.get("history") or []
    if not user_msg:
        return StreamingResponse(iter([sse("error", {"error": "empty message"})]),
                                 media_type="text/event-stream")

    async def gen() -> AsyncIterator[bytes]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": EXECUTOR_PROMPT}]
        for t in history[-6:]:
            messages.append({"role": t["role"], "content": t["content"]})
        messages.append({"role": "user", "content": user_msg})
        yield sse("start", {"mode": "agent", "max_steps": MAX_STEPS_PER_TASK})
        for step in range(1, MAX_STEPS_PER_TASK + 1):
            yield sse("thinking", {"step": step})
            try:
                raw = await llm(messages)
            except Exception as exc:  # noqa: BLE001
                yield sse("error", {"error": f"llm: {exc}"}); return
            yield sse("raw", {"step": step, "text": raw})
            action = extract_json(raw) or {"tool": "finish", "message": raw[:400]}
            yield sse("action", {"step": step, "action": action})
            if (action.get("tool") or action.get("action")) == "finish":
                yield sse("final", {"message": str(action.get("message", ""))}); break
            obs = dispatch_tool(action)
            yield sse("observation", {"step": step, "observation": obs})
            messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
            messages.append({"role": "user", "content": f"OBSERVATION:\n{json.dumps(obs, ensure_ascii=False)[:2000]}"})
        else:
            yield sse("final", {"message": "(max steps reached without finish)"})
        yield sse("files", {"files": list_workspace()})
        yield sse("end", {})

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------- /run (Plan → Execute → Critique) ----------------------------

@app.post("/run")
async def run_endpoint(request: Request) -> StreamingResponse:
    body = await request.json()
    user_msg = (body.get("message") or "").strip()
    if not user_msg:
        return StreamingResponse(iter([sse("error", {"error": "empty message"})]),
                                 media_type="text/event-stream")

    async def gen() -> AsyncIterator[bytes]:
        started = time.monotonic()
        yield sse("start", {"mode": "run", "max_tasks": MAX_TASKS, "max_steps_per_task": MAX_STEPS_PER_TASK})

        # -------- PLAN --------
        yield sse("phase", {"phase": "plan"})
        plan_messages = [{"role": "system", "content": PLANNER_PROMPT},
                         {"role": "user", "content": user_msg}]
        try:
            plan_raw = await llm(plan_messages, max_tokens=500, temperature=0.1)
        except Exception as exc:  # noqa: BLE001
            yield sse("error", {"error": f"planner: {exc}"}); return
        yield sse("raw", {"phase": "plan", "text": plan_raw})
        plan_obj = extract_json(plan_raw) or {"plan": [{"task": user_msg}], "summary": user_msg[:80]}
        tasks = (plan_obj.get("plan") or [])[:MAX_TASKS]
        if not tasks:
            tasks = [{"task": user_msg}]
        yield sse("plan", {"summary": plan_obj.get("summary", ""), "tasks": tasks})

        # -------- EXECUTE each task --------
        task_outcomes: list[dict[str, Any]] = []
        for idx, t in enumerate(tasks):
            task_text = str(t.get("task", "")).strip()
            yield sse("task_start", {"idx": idx, "task": task_text})
            exec_messages: list[dict[str, Any]] = [
                {"role": "system", "content": EXECUTOR_PROMPT},
                {"role": "user", "content": f"GOAL: {user_msg}\n\nCURRENT TASK ({idx+1}/{len(tasks)}): {task_text}"},
            ]
            task_log: list[dict[str, Any]] = []
            final_msg = ""
            for step in range(1, MAX_STEPS_PER_TASK + 1):
                yield sse("thinking", {"task": idx, "step": step})
                try:
                    raw = await llm(exec_messages, max_tokens=700)
                except Exception as exc:  # noqa: BLE001
                    yield sse("error", {"error": f"executor: {exc}"}); return
                yield sse("raw", {"task": idx, "step": step, "text": raw})
                action = extract_json(raw) or {"tool": "finish", "message": raw[:400]}
                yield sse("action", {"task": idx, "step": step, "action": action})
                tname = action.get("tool") or action.get("action")
                if tname == "finish":
                    final_msg = str(action.get("message", ""))
                    task_log.append({"step": step, "action": action, "observation": None})
                    break
                obs = dispatch_tool(action)
                yield sse("observation", {"task": idx, "step": step, "observation": obs})
                task_log.append({"step": step, "action": action, "observation": obs})
                exec_messages.append({"role": "assistant", "content": json.dumps(action, ensure_ascii=False)})
                exec_messages.append({"role": "user", "content": f"OBSERVATION:\n{json.dumps(obs, ensure_ascii=False)[:2000]}"})
            else:
                final_msg = "(max steps reached without finish)"
            yield sse("task_done", {"idx": idx, "result": final_msg, "steps": len(task_log)})
            task_outcomes.append({"task": task_text, "result": final_msg, "log": task_log})
            yield sse("files", {"files": list_workspace()})

        # -------- CRITIQUE --------
        yield sse("phase", {"phase": "critique"})
        critique_user = json.dumps({
            "user_request": user_msg,
            "plan": tasks,
            "results": [{"task": o["task"], "result": o["result"]} for o in task_outcomes],
            "workspace_files": [f["path"] for f in list_workspace()],
        }, ensure_ascii=False)[:6000]
        try:
            critique_raw = await llm(
                [{"role": "system", "content": CRITIC_PROMPT},
                 {"role": "user", "content": critique_user}],
                max_tokens=500, temperature=0.1,
            )
        except Exception as exc:  # noqa: BLE001
            yield sse("error", {"error": f"critic: {exc}"}); return
        yield sse("raw", {"phase": "critique", "text": critique_raw})
        critique = extract_json(critique_raw) or {
            "ok": True, "score": 70,
            "notes": "(critic output not parsable; returning best-effort)",
            "final_answer": task_outcomes[-1]["result"] if task_outcomes else critique_raw[:600],
        }
        yield sse("critique", critique)

        # -------- MEMORY --------
        entry = {
            "at": int(time.time()),
            "user_request": user_msg,
            "plan": tasks,
            "tasks": task_outcomes,
            "critique": critique,
            "duration_s": round(time.monotonic() - started, 1),
        }
        save_memory(entry)
        yield sse("memory", {"entry_count": len(load_memory(10_000))})

        yield sse("final", {"message": critique.get("final_answer") or "(no answer)"})
        yield sse("end", {"duration_s": round(time.monotonic() - started, 1)})

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------- /claw ----------------------------

@app.post("/claw")
async def claw_endpoint(request: Request) -> StreamingResponse:
    body = await request.json()
    user_msg = (body.get("message") or "").strip()
    if not user_msg:
        return StreamingResponse(iter([sse("error", {"error": "empty message"})]),
                                 media_type="text/event-stream")

    async def gen() -> AsyncIterator[bytes]:
        env = os.environ.copy()
        env.setdefault("ANTHROPIC_API_KEY", "local-dummy")
        env.setdefault("ANTHROPIC_BASE_URL", CLAW_PROXY)
        env.setdefault("ANTHROPIC_MODEL", "claude-3-5-sonnet")
        yield sse("start", {"mode": "claw", "cwd": str(CLAW_CWD), "timeout": CLAW_TIMEOUT})
        yield sse("thinking", {"step": 1, "note": "launching claw CLI — first run may take several minutes"})
        cmd = [str(CLAW_BIN), "--output-format", "json", "prompt", user_msg]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(CLAW_CWD), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:  # noqa: BLE001
            yield sse("error", {"error": f"spawn: {exc}"}); return

        started = time.monotonic()
        comm = asyncio.ensure_future(proc.communicate())
        llama_log = Path("/tmp/llama_server.log")
        last_size = 0
        while not comm.done():
            try:
                await asyncio.wait_for(asyncio.shield(comm), timeout=2.0)
                break
            except asyncio.TimeoutError:
                pass
            elapsed = time.monotonic() - started
            stage = "prompt eval"
            try:
                if llama_log.exists():
                    sz = llama_log.stat().st_size
                    if sz != last_size:
                        last_size = sz
                        with llama_log.open("rb") as f:
                            f.seek(max(0, sz - 4096))
                            tail = f.read().decode("utf-8", errors="replace")
                        for ln in reversed(tail.splitlines()):
                            if any(k in ln for k in ("eval time", "prompt processing", "tokens per second")):
                                stage = ln.strip()[-140:]; break
            except Exception:  # noqa: BLE001
                pass
            if elapsed > CLAW_TIMEOUT:
                proc.kill()
                yield sse("error", {"error": f"claw timed out after {CLAW_TIMEOUT}s"}); return
            yield sse("heartbeat", {"elapsed_s": round(elapsed, 1), "stage": stage})

        stdout, stderr = comm.result()
        so, se = stdout.decode(errors="replace"), stderr.decode(errors="replace")
        yield sse("raw", {"step": 1, "text": so[-4000:]})
        if se.strip():
            yield sse("stderr", {"text": se[-2000:]})
        if proc.returncode != 0:
            yield sse("error", {"error": f"claw exit {proc.returncode}", "tail": se[-500:]}); return
        msg = ""
        for ln in reversed(so.splitlines()):
            ln = ln.strip()
            if ln.startswith("{"):
                try:
                    j = json.loads(ln); msg = str(j.get("message") or j); yield sse("claw_payload", {"payload": j}); break
                except Exception:  # noqa: BLE001
                    continue
        yield sse("final", {"message": msg or so.strip().splitlines()[-1] if so.strip() else "(no output)"})
        yield sse("end", {})

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------- UI ----------------------------

INDEX_HTML_PATH = Path(__file__).parent / "agent_ui.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html = INDEX_HTML_PATH.read_text(encoding="utf-8")
    return html.replace("__MODEL_LABEL__", MODEL_LABEL)
