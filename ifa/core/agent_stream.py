import re
import uuid

import httpx

from ifa.config.settings import OLLAMA_MODEL
from ifa.core.context import AgentContext
from ifa.core.memory import Memory, tool_step
from ifa.core.personality import tool_framing
from ifa.services import prompts
from ifa.services.ollama_client import (
    build_tool_result_message,
    stream_chat,
)
from pathlib import Path
from ifa.skills.vibe.vibe import VibeManager
from ifa.tools import registry
from ifa.utils import latency

MEMORY_FILE = Path("./memory.md")
def load_memories() -> str:
    if not MEMORY_FILE.exists():
        return ""

    return MEMORY_FILE.read_text(encoding="utf-8").strip()
MODEL = OLLAMA_MODEL

MAX_TOOL_HOPS = 3
SPEECH_BATCH_SENTENCES = 1
SPEECH_BATCH_CHARS = 220


def _build_system_prompt(nonce: str) -> str:
    parts = [prompts.get("persona"), tool_framing(nonce), VibeManager.get_system_based_on_vibe()]
    memory = load_memories()
    if memory:
        parts.append(
            "Long-term memory:\n"
            "The following information has been intentionally remembered from previous conversations.\n"
            "Treat it as persistent context.\n\n"
            f"{memory}"
        )
    return "\n\n".join(parts)

TOOL_RESULT_MARKER = "TOOL_RESULT_"
FABRICATION_CORRECTION = (
    "[Ifa system note, not from the user: you wrote a tool result yourself. Tool "
    "results only come from real tool calls. Call the tool instead of imitating it.]"
)


ACTION_CLAIM = re.compile(
    r"^(?:(?:ok(?:ay)?|sure|alright|done|got it|sir|boss)[,!.\s]+)*(?:now\s+)?"
    r"(?:playing|paused|pausing|skipping|skipped|resuming|resumed|opening|opened|launching|launched|"
    r"starting|started|running|setting|sending|sent|switching|switched)\b"
    r"|\b(?:is|'s|are)\s+(?:now\s+)?(?:playing|open|opened|paused|running)\b",
    re.IGNORECASE,
)
ACTION_CLAIM_CORRECTION = (
    "[Ifa system note, not from the user: you said you did something, but no tool was called "
    "in this turn, so nothing happened. If Sir asked for an action, call the right tool now. "
    "If he was only chatting, answer again without saying you did anything.]"
)


def _new_nonce() -> str:
    return f"{TOOL_RESULT_MARKER}{uuid.uuid4().hex[:12]}"


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

    if len(remaining.split()) >= 12:
        split_idx = remaining.rfind(" ")

        if split_idx > 0:
            completed.append(
                remaining[:split_idx].strip()
            )

            remaining = remaining[split_idx:].strip()

    return completed, remaining


def _flush_speech_batch(parts: list[str], on_sentence) -> None:
    if parts and on_sentence:
        on_sentence(" ".join(parts))
    parts.clear()


def _notification_hint(ctx: AgentContext) -> str:
    unread = ctx.notifications.inbox.unread_count() if ctx.notifications else 0
    if not unread:
        return ""
    return (
        f"[Ifa system note, not from the user: {unread} unread phone notification(s) "
        "arrived recently. If the user asks about them (e.g. 'what is it?'), call "
        "read_notifications to get them. Never guess what they say.]\n"
    )


def _build_messages(user_text: str, ctx: AgentContext, memory: Memory, nonce: str) -> list[dict]:
    return [
        {"role": "system", "content": _build_system_prompt(nonce)},
        *memory.recent(),
        {"role": "user", "content": _notification_hint(ctx) + user_text},
    ]


def _stream_assistant_message(
    messages: list[dict],
    allow_tools: bool,
    on_sentence,
    guard_claims: bool = False,
) -> tuple[dict, list[str]]:
    full_text = ""
    fabricated = False
    claimed = False
    first_checked = not guard_claims
    sentence_buffer = ""
    speech_parts: list[str] = []
    tool_calls: list[dict] = []
    tools = registry.as_ollama_schema() if allow_tools else None
    timer = latency.current()
    started_ms = timer.now_ms()
    first_chunk_ms = None
    final_chunk: dict = {}

    for chunk in stream_chat(model=MODEL, messages=messages, tools=tools):
        if first_chunk_ms is None:
            first_chunk_ms = timer.now_ms()
        if chunk.get("done"):
            final_chunk = chunk
        message = chunk.get("message") or {}
        tool_calls.extend(message.get("tool_calls") or [])
        content = message.get("content") or ""
        if not content:
            continue

        full_text += content
        if fabricated or TOOL_RESULT_MARKER in full_text:
            fabricated = True
            continue
        if claimed:
            continue
        sentence_buffer += content
        completed, sentence_buffer = _extract_chunks(sentence_buffer)
        for sentence in completed:
            if not first_checked:
                first_checked = True
                if ACTION_CLAIM.search(sentence):
                    claimed = True
                    break
            speech_parts.append(sentence)
            if (
                len(speech_parts) >= SPEECH_BATCH_SENTENCES
                or sum(len(part) for part in speech_parts) >= SPEECH_BATCH_CHARS
            ):
                _flush_speech_batch(speech_parts, on_sentence)

    tail = sentence_buffer.strip()
    if tail and not first_checked and ACTION_CLAIM.search(tail):
        claimed = True
    if tail and not fabricated and not claimed:
        speech_parts.append(tail)

    timer.record_llm("llm", started_ms, first_chunk_ms, final_chunk, len(tool_calls))
    return (
        {"role": "assistant", "content": full_text, "tool_calls": tool_calls,
         "fabricated": fabricated, "claimed": claimed and not tool_calls},
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
) -> list[tuple[str, str]]:
    results = []
    for tool_call in tool_calls:
        function = tool_call["function"]
        name = function["name"]
        result = registry.dispatch(name, function["arguments"], ctx)
        results.append((name, result))
        messages.append(build_tool_result_message(
            tool_name=name,
            content=registry.delimit_as_data(nonce, name, result),
        ))
    return results


def _direct_answer(results: list[tuple[str, str]]) -> str | None:
    tools = [registry.get(name) for name, _ in results]
    if not all(tool is not None and tool.return_direct for tool in tools):
        return None
    if any(result.startswith("ERROR") for _, result in results):
        return None
    return "\n".join(result for _, result in results)


def agent_turn_stream(
    user_text: str,
    ctx: AgentContext,
    memory: Memory,
    on_sentence=None,
    on_tool_calls=None,
) -> str:
    ctx.turn += 1
    ctx.user_text = user_text
    nonce = _new_nonce()
    messages = _build_messages(user_text, ctx, memory, nonce)
    tool_hops = 0
    retries_remaining = 1
    steps: list[dict] = []

    while True:
        allow_tools = tool_hops < MAX_TOOL_HOPS
        try:
            assistant_msg, pending_speech = _stream_assistant_message(
                messages, allow_tools=allow_tools, on_sentence=on_sentence,
                guard_claims=allow_tools and tool_hops == 0,
            )
        except (KeyError, httpx.HTTPError):
            return "I hit a problem talking to the language model."

        claimed = assistant_msg.pop("claimed")
        if assistant_msg.pop("fabricated"):
            if not retries_remaining:
                return "I couldn't figure out how to do that."
            retries_remaining -= 1
            messages.append({"role": "user", "content": FABRICATION_CORRECTION})
            continue

        if claimed:
            print(f"[agent] reply claimed an action without a tool call: {assistant_msg['content'][:80]!r}")
            if retries_remaining:
                retries_remaining -= 1
                messages.append({"role": "assistant", "content": assistant_msg["content"]})
                messages.append({"role": "user", "content": ACTION_CLAIM_CORRECTION})
                continue
            pending_speech = [assistant_msg["content"]]

        tool_calls = assistant_msg["tool_calls"] if allow_tools else []
        if tool_calls:
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
            if on_tool_calls:
                on_tool_calls([tc["function"]["name"] for tc in tool_calls])
            messages.append(assistant_msg)
            results = _append_tool_results(messages, tool_calls, ctx, nonce)
            steps.extend(tool_step(assistant_msg, results))
            direct = _direct_answer(results)
            if direct is not None:
                if on_sentence:
                    on_sentence(direct)
                memory.add_turn(user_text, steps, direct)
                return direct
            continue

        _flush_speech_batch(pending_speech, on_sentence)
        assistant_text = assistant_msg["content"]
        memory.add_turn(user_text, steps, assistant_text)
        return assistant_text
