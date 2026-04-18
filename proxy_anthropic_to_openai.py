"""
Minimal Anthropic -> OpenAI compatibility proxy.

Claw Code speaks the Anthropic /v1/messages wire format. This proxy accepts
that format on an Anthropic-like port (default 8080) and forwards to an
OpenAI-compatible backend (default http://localhost:8081/v1, served by
llama-cpp-python) so the local GGUF model acts as the LLM engine.

Supports both regular JSON responses and SSE streaming.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

BACKEND = os.environ.get("OPENAI_BACKEND", "http://127.0.0.1:8081/v1")
BACKEND_MODEL = os.environ.get("BACKEND_MODEL", "qwen-coder")
REQUEST_TIMEOUT = float(os.environ.get("PROXY_TIMEOUT", "600"))

app = FastAPI(title="Anthropic->OpenAI proxy")


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                parts.append(
                    f"<tool_use name={block.get('name')!r} input={json.dumps(block.get('input', {}))}>"
                )
            elif btype == "tool_result":
                tc = block.get("content", "")
                if isinstance(tc, list):
                    tc = "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b) for b in tc
                    )
                parts.append(f"<tool_result>{tc}</tool_result>")
        return "\n".join(parts)
    return str(content)


def anthropic_to_openai(payload: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system = payload.get("system")
    if isinstance(system, list):
        system = "\n\n".join(_extract_text(s) for s in system)
    if system:
        messages.append({"role": "system", "content": system})
    for msg in payload.get("messages", []):
        role = msg.get("role", "user")
        if role not in ("user", "assistant", "system"):
            role = "user"
        messages.append({"role": role, "content": _extract_text(msg.get("content", ""))})
    openai_payload: dict[str, Any] = {
        "model": BACKEND_MODEL,
        "messages": messages,
        "stream": bool(payload.get("stream")),
    }
    if payload.get("max_tokens") is not None:
        openai_payload["max_tokens"] = payload["max_tokens"]
    if payload.get("temperature") is not None:
        openai_payload["temperature"] = payload["temperature"]
    if payload.get("top_p") is not None:
        openai_payload["top_p"] = payload["top_p"]
    if payload.get("stop_sequences"):
        openai_payload["stop"] = payload["stop_sequences"]
    return openai_payload


def openai_to_anthropic(resp: dict[str, Any], model: str) -> dict[str, Any]:
    choice = (resp.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    finish = choice.get("finish_reason") or "stop"
    stop_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
    usage = resp.get("usage") or {}
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_map.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def stream_anthropic(
    backend_stream: AsyncIterator[bytes], model: str
) -> AsyncIterator[bytes]:
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    yield _sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    finish: str | None = None
    in_tokens = 0
    out_tokens = 0
    buf = b""
    async for chunk in backend_stream:
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line or not line.startswith(b"data:"):
                continue
            payload_raw = line[5:].strip()
            if payload_raw == b"[DONE]":
                continue
            try:
                piece = json.loads(payload_raw)
            except json.JSONDecodeError:
                continue
            choice = (piece.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            text = delta.get("content") or ""
            if text:
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            usage = piece.get("usage") or {}
            if usage:
                in_tokens = usage.get("prompt_tokens", in_tokens)
                out_tokens = usage.get("completion_tokens", out_tokens)
    yield _sse(
        "content_block_stop", {"type": "content_block_stop", "index": 0}
    )
    stop_map = {"stop": "end_turn", "length": "max_tokens"}
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_map.get(finish or "stop", "end_turn"),
                "stop_sequence": None,
            },
            "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens},
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=5.0) as c:
        try:
            r = await c.get(f"{BACKEND}/models")
            backend_ok = r.status_code == 200
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "backend": BACKEND, "error": str(exc)}
    return {"ok": True, "backend": BACKEND, "backend_ok": backend_ok}


async def _single_completion(openai_payload: dict[str, Any]) -> tuple[int, dict[str, Any] | str]:
    payload = dict(openai_payload)
    payload["stream"] = False
    url = f"{BACKEND}/chat/completions"
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(url, json=payload)
    if r.status_code >= 400:
        return r.status_code, r.text[:500]
    try:
        return 200, r.json()
    except Exception as exc:  # noqa: BLE001
        return 500, f"invalid backend JSON: {exc}"


def _synth_stream(
    resp: dict[str, Any], model: str, chunk: int = 48
) -> AsyncIterator[bytes]:
    async def gen() -> AsyncIterator[bytes]:
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        choice = (resp.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        finish = choice.get("finish_reason") or "stop"
        usage = resp.get("usage") or {}
        yield _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": 0,
                    },
                },
            },
        )
        yield _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        for i in range(0, len(text), chunk):
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text[i : i + chunk]},
                },
            )
        yield _sse(
            "content_block_stop", {"type": "content_block_stop", "index": 0}
        )
        stop_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_map.get(finish, "end_turn"),
                    "stop_sequence": None,
                },
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                },
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})

    return gen()


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    payload = await request.json()
    model = payload.get("model", "claude-like")
    openai_payload = anthropic_to_openai(payload)
    want_stream = bool(openai_payload.get("stream"))
    status, body = await _single_completion(openai_payload)
    if status >= 400:
        return JSONResponse(
            status_code=status,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": str(body)[:500]},
            },
        )
    if want_stream:
        return StreamingResponse(
            _synth_stream(body, model), media_type="text/event-stream"
        )
    return JSONResponse(content=openai_to_anthropic(body, model))


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request) -> Response:
    payload = await request.json()
    # Very rough heuristic; good enough for a local standalone agent.
    text = json.dumps(payload)
    return JSONResponse(content={"input_tokens": max(1, len(text) // 4)})


@app.get("/")
async def root() -> dict[str, Any]:
    return {"service": "anthropic-openai-proxy", "backend": BACKEND, "time": time.time()}
