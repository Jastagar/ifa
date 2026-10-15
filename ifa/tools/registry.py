"""Tool registry: declare tools once, dispatch by name, export Ollama schema.

Each tool declares a name, description, JSON Schema for parameters, and a
handler callable. `dispatch()` validates the LLM's supplied args against
the schema before calling the handler — this is the R18 confused-deputy
defense, so jsonschema is a hard dep (not optional).

`delimit_as_data(nonce, tool_name, result)` wraps a tool result string in
a nonce-keyed envelope. The nonce is generated per agent turn and embedded
in the system prompt so the LLM knows where data blocks start/end; a
crafted response cannot close the envelope without guessing the nonce.
"""
from dataclasses import dataclass
from typing import Callable

from jsonschema import ValidationError, validate

from ifa.core.context import AgentContext


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable[[dict, AgentContext], str]


_TOOLS: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    """Register a tool. Last registration wins if the name collides."""
    _TOOLS[tool.name] = tool


def get(name: str) -> Tool | None:
    return _TOOLS.get(name)


def all_tools() -> list[Tool]:
    return list(_TOOLS.values())


def clear() -> None:
    """Reset the registry. Test-only — production code registers once at import time."""
    _TOOLS.clear()


def as_ollama_schema() -> list[dict]:
    """Return the tool list in Ollama's /api/chat `tools` field format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        for tool in _TOOLS.values()
    ]


def dispatch(tool_name: str, args: dict, ctx: AgentContext) -> str:
    """Validate args against the tool's schema, then invoke the handler.

    Returns a string result (including error strings — the LLM reads them).
    Never raises; all errors become structured result strings so the agent
    loop can feed them back to the LLM on retry.
    """
    tool = _TOOLS.get(tool_name)
    if tool is None:
        return f"ERROR: unknown tool `{tool_name}`. Available tools: {list(_TOOLS.keys())}"

    try:
        validate(instance=args, schema=tool.parameters)
    except ValidationError as exc:
        return f"ERROR: invalid arguments for `{tool_name}`: {exc.message}"

    try:
        return tool.handler(args, ctx)
    except Exception as exc:  # handler bugs shouldn't crash the agent loop
        return f"ERROR: tool `{tool_name}` failed: {exc}"


def delimit_as_data(nonce: str, tool_name: str, result: str) -> str:
    """Wrap a tool result in a nonce-keyed delimiter block.

    The nonce is generated per agent_turn and embedded in the system prompt
    so the LLM treats the wrapper as authoritative. Crafted tool content
    cannot close the block without guessing the nonce.
    """
    return (
        f"<{nonce}_START tool={tool_name}>\n"
        f"{result}\n"
        f"<{nonce}_END>"
    )
