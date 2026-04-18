# ai-autonomous-agent

A **fully local, autonomous coding agent** with a Devin-style web UI, powered by
[llama.cpp](https://github.com/ggerganov/llama.cpp) running
**Qwen2.5-Coder-7B-Instruct** (or any GGUF model you pick). Zero external APIs,
zero telemetry, runs offline on a single Linux box.

It offers **three execution modes** side-by-side:

| Mode | Endpoint | What it does |
|---|---|---|
| 🧠 **Plan-Execute-Critique** | `POST /run` | Planner LLM → decomposes into tasks → Executor ReAct loop per task → Critic summarizes |
| ⚡ **Fast Agent** | `POST /agent` | Single ReAct loop with the typed tool set — fastest path for simple jobs |
| 🦀 **Claw Code** | `POST /claw` | Delegates to the [Claw Code](https://github.com/ultraworkers/claw-code) Rust CLI via an Anthropic↔OpenAI shim |

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
# 1) download a GGUF model (Qwen2.5-Coder-7B-Instruct, ~4.4 GB)
mkdir -p /opt/model
hf download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF qwen2.5-coder-7b-instruct-q4_k_m.gguf --local-dir /opt/model

# 2) run llama-server (native, fastest)
/opt/llama.cpp/llama-server \
    -m /opt/model/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    --host 127.0.0.1 --port 8081 -c 16384 -t 4 --chat-template chatml &

# 3) run the Anthropic↔OpenAI proxy (only needed for Claw mode)
python3 -m uvicorn proxy_anthropic_to_openai:app --host 127.0.0.1 --port 8080 &

# 4) run the agent server
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
| `MODEL_LABEL` | `Qwen2.5-Coder-7B (local)` | Shown in the UI header |

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
