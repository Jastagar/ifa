import re
import uuid

import httpx

from ifa.config.settings import OLLAMA_MODEL
from ifa.core.context import AgentContext
from ifa.core.memory import Memory
from ifa.core.personality import persona, tool_framing, remember_nudge
from ifa.services.ollama_client import (
    build_tool_result_message,
    stream_chat,
)
from pathlib import Path
from ifa.tools import registry

MEMORY_FILE = Path("./memory.md")
def load_memories() -> str:
    if not MEMORY_FILE.exists():
        return ""

    return MEMORY_FILE.read_text(encoding="utf-8").strip()
MODEL = OLLAMA_MODEL

MAX_ITERATIONS = 1
# Submit each completed model sentence immediately. The TTS producer runs
# independently from playback and can pre-generate every available sentence.
SPEECH_BATCH_SENTENCES = 1
SPEECH_BATCH_CHARS = 220


def _build_system_prompt(nonce: str, facts: list[str] | None = None) -> str:
    parts = [persona, tool_framing(nonce), remember_nudge]
    memory = load_memories()
    if memory:
        parts.append(
            "Long-term memory:\n"
            "The following information has been intentionally remembered from previous conversations.\n"
            "Treat it as persistent context.\n\n"
            f"{memory}"
        )
    return "\n\n".join(parts)

def _new_nonce() -> str:
    return f"TOOL_RESULT_{uuid.uuid4().hex[:12]}"


def _extract_chunks(buffer: str):

    matches = list(
        re.finditer(r'[^.!?]+[.!?]', buffer)
    )

    completed = []

    last_end = 0

    for m in matches:
        completed.append(
            m.group(0).strip()
        )
        last_end = m.end()

    remaining = buffer[last_end:]

    # fallback flush for long running text
    if len(remaining.split()) >= 12:

        split_idx = remaining.rfind(" ")

        if split_idx > 0:

            completed.append(
                remaining[:split_idx].strip()
            )

            remaining = remaining[split_idx:].strip()

    return completed, remaining


def _flush_speech_batch(parts: list[str], on_sentence) -> None:
    """Send a short multi-sentence phrase to TTS as one natural utterance."""
    if parts and on_sentence:
        on_sentence(" ".join(parts))
    parts.clear()


def _build_messages(user_text: str, ctx: AgentContext, memory: Memory, nonce: str) -> list[dict]:
    return [
        {"role": "system", "content": _build_system_prompt(nonce)},
        {"role": "user", "content": user_text},
    ]


def _stream_assistant_message(
    messages: list[dict],
    allow_tools: bool,
    on_sentence,
) -> tuple[dict, list[str]]:
    """Stream one assistant response and emit full phrase batches early."""
    full_text = ""
    sentence_buffer = ""
    speech_parts: list[str] = []
    tool_calls: list[dict] = []
    tools = registry.as_ollama_schema() if allow_tools else None

    for chunk in stream_chat(model=MODEL, messages=messages, tools=tools):
        message = chunk.get("message") or {}
        tool_calls.extend(message.get("tool_calls") or [])
        content = message.get("content") or ""
        if not content:
            continue

        full_text += content
        sentence_buffer += content
        completed, sentence_buffer = _extract_chunks(sentence_buffer)
        for sentence in completed:
            speech_parts.append(sentence)
            if (
                len(speech_parts) >= SPEECH_BATCH_SENTENCES
                or sum(len(part) for part in speech_parts) >= SPEECH_BATCH_CHARS
            ):
                _flush_speech_batch(speech_parts, on_sentence)

    if sentence_buffer.strip():
        speech_parts.append(sentence_buffer.strip())

    return (
        {"role": "assistant", "content": full_text, "tool_calls": tool_calls},
        speech_parts,
    )


def _tool_call_errors(tool_calls: list[dict]) -> list[str]:
    errors = []
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        if not isinstance(function.get("name"), str):
            errors.append("missing function.name")
        if not isinstance(function.get("arguments"), dict):
            errors.append(f"arguments for `{function.get('name')}` must be an object")
    return errors


def _append_tool_results(
    messages: list[dict], tool_calls: list[dict], ctx: AgentContext, nonce: str
) -> None:
    """Dispatch valid tool calls and append their safely-delimited results."""
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        name = function.get("name")
        args = function.get("arguments", {})
        if not name:
            messages.append(build_tool_result_message("unknown", "ERROR: missing tool name"))
            return

        result = registry.dispatch(name, args, ctx)
        messages.append(build_tool_result_message(
            tool_name=name,
            content=registry.delimit_as_data(nonce, name, result),
        ))
        if isinstance(result, str) and result.startswith("ERROR"):
            return


def _remember_turn(memory: Memory, user_text: str, assistant_text: str) -> None:
    memory.add(role="user", content=user_text)
    memory.add(role="assistant", content=assistant_text)


def agent_turn_stream(
    user_text: str,
    ctx: AgentContext,
    memory: Memory,
    on_sentence=None,
) -> str:

    nonce = _new_nonce()
    messages = _build_messages(user_text, ctx, memory, nonce)
    tool_hops = 0
    retries_remaining = 1

    while True:
        try:
            assistant_msg, pending_speech = _stream_assistant_message(
                messages, allow_tools=tool_hops == 0, on_sentence=on_sentence
            )
        except (KeyError, httpx.HTTPError):
            return "I hit a problem talking to the language model."

        tool_calls = assistant_msg["tool_calls"]
        if tool_calls:
            if tool_hops >= MAX_ITERATIONS:
                return "I couldn't finish that in one step."

            malformed_reasons = _tool_call_errors(tool_calls)
            if malformed_reasons and retries_remaining:
                retries_remaining -= 1
                messages.append(assistant_msg)
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous tool call "
                        "was invalid: "
                        f"{'; '.join(malformed_reasons)}."
                    ),
                })
                continue
            if malformed_reasons:
                return "I couldn't figure out how to do that."
            tool_hops += 1
            messages.append(assistant_msg)
            _append_tool_results(messages, tool_calls, ctx, nonce)
            continue

        _flush_speech_batch(pending_speech, on_sentence)
        assistant_text = assistant_msg["content"]
        _remember_turn(memory, user_text, assistant_text)
        return assistant_text
