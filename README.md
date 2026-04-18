# ai-autonomous-agent

A **fully local, autonomous coding agent** with a Devin-style web UI, powered by
[llama.cpp](https://github.com/ggerganov/llama.cpp) running
**Qwen2.5-Coder-7B-Instruct** (or any GGUF model you pick). Zero external APIs,
zero telemetry, runs offline on a single Linux box.

It offers **four execution modes** side-by-side:

| Mode | Endpoint | What it does |
|---|---|---|
| 🧠 **Plan-Execute-Critique** | `POST /run` | Planner LLM → decomposes into tasks → Executor ReAct loop per task → Critic summarizes |
| ⚡ **Fast Agent** | `POST /agent` | Single ReAct loop with the typed tool set — fastest path for simple jobs |
| 💎 **Aider** | `POST /aider` | Delegates to [Aider](https://github.com/Aider-AI/aider) OSS coding agent (whole-file edit format) |
| 🦀 **Claw Code** | `POST /claw` | Delegates to the [Claw Code](https://github.com/ultraworkers/claw-code) Rust CLI via an Anthropic↔OpenAI shim |

## Speed notes

On a 4-core CPU box with no GPU, model size is the #1 latency factor. Defaults ship with
**Qwen2.5-Coder-1.5B-Instruct** — roughly **10–20× faster** than the 7B variant on the same hardware.
A typical 3-step ReAct run (write → read → finish) completes in ~20 seconds; Aider writes a small
file in ~13 seconds. Claw mode remains available but slower because it injects a large system prompt.

If you have access to a larger machine or an external OpenAI-compatible endpoint (e.g. an RTX
box, Modal, RunPod, a Colab-hosted llama-server), just point `OPENAI_BACKEND` at it and the whole
stack will use the stronger model with no code changes.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Browser UI (single file)                     │
│   Chat · Plan · Activity · Files · Memory · Logs  (SSE stream)   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                    agent_server.py  (FastAPI, :7860)
      ┌──────────────────────┼──────────────────────┐
      │                      │                      │
      ▼                      ▼                      ▼
  /run                   /agent                  /claw
 Planner + Exec          ReAct loop            Rust CLI process
  + Critic               (tools)                    │
   │                         │                      ▼
   └─────────┬───────────────┘           proxy_anthropic_to_openai.py  (:8080)
             ▼                                       │
                             llama-server (:8081, OpenAI API)
                             model: Qwen2.5-Coder-7B-Instruct (GGUF)
```

All file I/O and shell happens inside a sandboxed workspace
(`/home/ubuntu/agent_workspace` by default).

## Tools the agent can call

```jsonc
{"tool": "shell",   "command": "<bash>"}            // 45s timeout
{"tool": "write",   "path": "file", "content": "…"} // sandboxed
{"tool": "read",    "path": "file"}
{"tool": "list",    "path": "."}
{"tool": "search",  "pattern": "<regex>", "path": "."}
{"tool": "python",  "code": "<python>"}             // 45s timeout
{"tool": "install", "package": "<pip name>"}        // pip install --user
{"tool": "finish",  "message": "<result>"}
```

## Memory & self-reflection

After every `/run`, the Critic emits `{ok, score, notes, final_answer}` and the
full task log is appended to `workspace/.memory/history.json`. The UI renders
this history in the **Memory** tab.

## Quick start

```bash
# 1) download a GGUF model (Qwen2.5-Coder-1.5B-Instruct, ~1 GB) — recommended default
mkdir -p /opt/model
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --local-dir /opt/model
#    …or the 7B variant if you have ≥8GB RAM / a GPU:
# hf download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF qwen2.5-coder-7b-instruct-q4_k_m.gguf --local-dir /opt/model

# 2) run llama-server (native, fastest)
/opt/llama.cpp/llama-server \
    -m /opt/model/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
    --host 127.0.0.1 --port 8081 -c 8192 -t 4 --chat-template chatml &

# 3) run the Anthropic↔OpenAI proxy (only needed for Claw mode)
python3 -m uvicorn proxy_anthropic_to_openai:app --host 127.0.0.1 --port 8080 &

# 4) (optional) install Aider for 💎 mode
python3 -m venv /opt/aider_venv && /opt/aider_venv/bin/pip install aider-chat

# 5) run the agent server
python3 -m uvicorn agent_server:app --host 0.0.0.0 --port 7860

# open http://localhost:7860
```

Or use the all-in-one launcher with built-in retry and health checks:

```bash
python3 launcher.py --autonomous
```

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `OPENAI_BACKEND` | `http://127.0.0.1:8081/v1` | llama.cpp OpenAI endpoint |
| `AGENT_WORKSPACE` | `/home/ubuntu/agent_workspace` | Sandbox root |
| `AGENT_TASKS` | `3` | Max plan tasks |
| `AGENT_STEPS` | `6` | Max ReAct steps per task |
| `AGENT_SHELL_TIMEOUT` | `45` | Shell tool timeout (s) |
| `AGENT_PY_TIMEOUT` | `45` | Python tool timeout (s) |
| `CLAW_TIMEOUT` | `1800` | Claw CLI total timeout (s) |
| `ANTHROPIC_BASE_URL` | `http://127.0.0.1:8080` | Proxy URL for Claw |
| `AIDER_BIN` | `/opt/aider_venv/bin/aider` | Path to Aider binary |
| `AIDER_CWD` | `/home/ubuntu/aider_workspace` | Aider workspace |
| `AIDER_MODEL` | `openai/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` | Aider --model flag |
| `MODEL_LABEL` | `Qwen2.5-Coder-1.5B (local)` | Shown in the UI header |

## Files

```
agent_server.py                  — FastAPI app with /run, /agent, /claw, /files, /memory
agent_ui.html                    — Single-file Devin-style UI (dark, mobile-ready)
proxy_anthropic_to_openai.py     — Anthropic Messages API → OpenAI ChatCompletions bridge
launcher.py                      — Autonomous launcher (starts llama-server + proxy, retries up to 20)
```

## Safety

- Shell + Python tools run **without sudo** and only inside the workspace.
- `_safe()` helper rejects any path that resolves outside the workspace.
- `install` allow-lists package names to `^[A-Za-z0-9_.\-]+(\[…\])?(==\d…)?$`.
- Hard timeouts on shell (45 s), Python (45 s), and Claw (30 min).

## License

MIT — do whatever.
