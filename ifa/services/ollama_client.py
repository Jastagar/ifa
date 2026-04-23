"""Thin HTTP client for Ollama's /api/chat and /api/tags endpoints.

Stage 1 uses Ollama's native tool-calling (tools=[...]) via /api/chat rather
than the subprocess CLI. Tool-result messages use the `tool_name` field per
Ollama's spec (not `name`). The helper `build_tool_result_message` isolates
that spec quirk so callers don't drift from it.
"""
import httpx

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 60.0


def build_tool_result_message(tool_name: str, content: str) -> dict:
    """Build a tool-role message for Ollama's /api/chat.

    Ollama's spec uses `tool_name` as the field identifying which tool
    produced the result. Not `name` (OpenAI-style) — keeping them distinct.
    """
    return {"role": "tool", "tool_name": tool_name, "content": content}


def chat(model: str, messages: list[dict], tools: list[dict] | None = None,
         timeout: float = DEFAULT_TIMEOUT) -> dict:
    """POST to /api/chat and return the parsed response dict.

    Raises httpx.HTTPError on network/timeout/status failures. Raises
    KeyError if the response shape is unexpected — caller handles both.
    """
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if tools is not None:
        payload["tools"] = tools

    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def check_health(required_model: str) -> None:
    """Verify Ollama is running, the model is pulled, and tool-calling works.

    Three checks in sequence:
      1. /api/tags responds (Ollama is running)
      2. a model whose name starts with `required_model` is present
      3. a minimal POST to /api/chat with a tools payload succeeds
         (proves this Ollama version actually accepts tool-calling)

    Raises RuntimeError with an actionable message on any failure.
    """
    # Check 1+2: list tags, verify model present
    try:
        tags_response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        tags_response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        raise RuntimeError(
            f"Ollama is not running at {OLLAMA_BASE_URL}. "
            "Start it with `ollama serve` (or install it from https://ollama.com)."
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Ollama responded with an error at {OLLAMA_BASE_URL}: {exc}"
        ) from exc

    models = tags_response.json().get("models", [])
    model_names = [m.get("name", "") for m in models]
    if not any(name.startswith(required_model) for name in model_names):
        raise RuntimeError(
            f"Model `{required_model}` is not pulled in Ollama. "
            f"Run `ollama pull {required_model}` and try again. "
            f"(Ollama reported these models: {model_names or 'none'})"
        )

    # Check 3: conformance probe — POST /api/chat with tools to verify
    # Ollama actually accepts the tool-calling payload shape.
    probe_tool = {
        "type": "function",
        "function": {
            "name": "_ifa_probe",
            "description": "Probe tool used at startup to verify tool-calling is supported. Do not call.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    try:
        chat(
            model=required_model,
            messages=[{"role": "user", "content": "ok"}],
            tools=[probe_tool],
            timeout=30.0,
        )
    except httpx.HTTPStatusError as exc:
        # 4xx/5xx suggests the tools payload shape is rejected.
        raise RuntimeError(
            f"Ollama accepted a connection but rejected a tool-calling request "
            f"(status {exc.response.status_code}). Your Ollama version may be "
            f"too old — upgrade to the latest release and try again."
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Ollama tool-calling conformance probe failed: {exc}. "
            "Check that the model is fully loaded and Ollama is responsive."
        ) from exc
