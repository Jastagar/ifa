"""Local HTTP activation endpoint for an already-running Ifa instance."""
from __future__ import annotations

import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from ifa.core.agent_stream import agent_turn_stream
from ifa.core.context import AgentContext
from ifa.core.memory import Memory

MAX_TEXT_CHARS = 4_000


class ActivationService:
    """Runs API activations through the same serialized agent session as voice."""

    def __init__(
        self,
        ctx: AgentContext,
        memory: Memory,
        on_sentence: Callable[[str], None],
    ) -> None:
        self._ctx = ctx
        self._memory = memory
        self._on_sentence = on_sentence
        self._turn_lock = threading.Lock()

    def activate(self, text: str, context: object | None = None, speak: bool = True) -> str:
        text = text.strip()
        if not text:
            raise ValueError("`text` must be a non-empty string")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"`text` must be at most {MAX_TEXT_CHARS} characters")

        if context is not None:
            context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            if len(context_json) > MAX_TEXT_CHARS:
                raise ValueError(f"`context` must be at most {MAX_TEXT_CHARS} characters when encoded")
            text = f"{text}\n\nActivation context (reference data): {context_json}"

        # Ollama, tools, memory, and speech are session state. One active turn
        # prevents an HTTP request from interleaving with a voice request.
        with self._turn_lock:
            return agent_turn_stream(
                user_text=text,
                ctx=self._ctx,
                memory=self._memory,
                on_sentence=self._on_sentence if speak else None,
            )


def start_activation_server(
    service: ActivationService,
    input_mode,
    host: str | None = None,
    port: int | None = None,
    token: str | None = None,
) -> ThreadingHTTPServer:
    """Start the local activation listener in a daemon thread."""
    host = host or os.environ.get("IFA_API_HOST", "127.0.0.1")
    port = port or int(os.environ.get("IFA_API_PORT", "8787"))
    token = token if token is not None else os.environ.get("IFA_API_TOKEN", "")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            print(f"[api] {self.address_string()} - {format % args}")

        def _send_json(self, status: HTTPStatus, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:
            print("===== REQUEST =====")
            print(self.command, self.path)
            print(self.headers)

            content_length = int(self.headers.get("Content-Length", "0"))
            print("Content-Length =", content_length)
            
            if self.path != "/activate":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return

            if token and self.headers.get("X-IFA-Token") != token:
                self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "invalid API token"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))

                if content_length <= 0:
                    raise ValueError("request body is missing")

                if content_length > 100_000:
                    raise ValueError("request body is too large")

                payload = json.loads(self.rfile.read(content_length))

                context = payload.get("context")

                input_mode._listener.start_listening_from_api(
                    api_context=json.dumps(context, ensure_ascii=False)
                )

            except (ValueError, json.JSONDecodeError) as exc:
                print(exc)
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            except Exception as exc:
                import traceback

                traceback.print_exc()

                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": str(exc)}
                )
                return

            self._send_json(HTTPStatus.ACCEPTED, {})

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, name="ifa-api", daemon=True).start()
    print(f"[api] listening on http://{host}:{port} (POST /activate)")
    return server
