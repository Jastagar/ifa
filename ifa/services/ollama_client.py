import httpx
import json
import base64
import io
from PIL import Image
from ifa.config.settings import OLLAMA_KEEP_ALIVE, OLLAMA_NUM_CTX, OLLAMA_THINK, MAIN_OLLAMA_URL

DEFAULT_TIMEOUT = 60.0

def build_tool_result_message(tool_name: str, content: str) -> dict:
    return {"role": "tool", "tool_name": tool_name, "content": content}

class OllamaError(RuntimeError):
    pass


def stream_chat(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    think: bool = OLLAMA_THINK,
    timeout: float = DEFAULT_TIMEOUT,
):
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": think,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": OLLAMA_NUM_CTX},
    }

    if tools is not None:
        payload["tools"] = tools

    with httpx.stream(
        "POST",
        f"{MAIN_OLLAMA_URL}/api/chat",
        json=payload,
        timeout=timeout,
    ) as response:
        status = getattr(response, "status_code", 200)
        if isinstance(status, int) and status >= 400:
            body = response.read().decode("utf-8", "replace")
            try:
                detail = json.loads(body).get("error") or body
            except ValueError:
                detail = body
            raise OllamaError(f"Ollama answered HTTP {response.status_code}: {detail[:500]}")

        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise OllamaError(f"Ollama error: {str(chunk['error'])[:500]}")
            yield chunk

def chat(
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    think: bool = OLLAMA_THINK,
    images: list[Image.Image] | None = None,
    format: dict | None = None,
) -> dict:
    messages = [message.copy() for message in messages]

    if images:
        encoded_images: list[str] = []

        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")

            encoded_images.append(
                base64.b64encode(buffer.getvalue()).decode("utf-8")
            )

        for message in reversed(messages):
            if message["role"] == "user":
                message["images"] = encoded_images
                break
        else:
            raise ValueError("No user message found to attach images to.")

    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": think,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": OLLAMA_NUM_CTX},
    }

    if tools is not None:
        payload["tools"] = tools
    if format is not None:
        payload["format"] = format

    response = httpx.post(
        f"{MAIN_OLLAMA_URL}/api/chat",
        json=payload,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()

def check_health(required_model: str) -> None:
    try:
        tags_response = httpx.get(f"{MAIN_OLLAMA_URL}/api/tags", timeout=5.0)
        tags_response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        raise RuntimeError(
            f"Ollama is not running at {MAIN_OLLAMA_URL}. "
            "Start it with `ollama serve` (or install it from https://ollama.com)."
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Ollama responded with an error at {MAIN_OLLAMA_URL}: {exc}"
        ) from exc

    models = tags_response.json().get("models", [])
    model_names = [m.get("name", "") for m in models]
    if not any(name.startswith(required_model) for name in model_names):
        raise RuntimeError(
            f"Model `{required_model}` is not pulled in Ollama. "
            f"Run `ollama pull {required_model}` and try again. "
            f"(Ollama reported these models: {model_names or 'none'})"
        )

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
