from dataclasses import dataclass
from typing import Callable

from jsonschema import ValidationError, validate

from ifa.core.context import AgentContext


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict, AgentContext], str]
    return_direct: bool = False


_TOOLS: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    _TOOLS[tool.name] = tool


def get(name: str) -> Tool | None:
    return _TOOLS.get(name)


def all_tools() -> list[Tool]:
    return list(_TOOLS.values())


def clear() -> None:
    _TOOLS.clear()


def as_ollama_schema() -> list[dict]:
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
    tool = _TOOLS.get(tool_name)
    if tool is None:
        return f"ERROR: unknown tool `{tool_name}`. Available tools: {list(_TOOLS.keys())}"

    try:
        validate(instance=args, schema=tool.parameters)
    except ValidationError as exc:
        return f"ERROR: invalid arguments for `{tool_name}`: {exc.message}"

    try:
        return tool.handler(args, ctx)
    except Exception as exc:
        return f"ERROR: tool `{tool_name}` failed: {exc}"


def delimit_as_data(nonce: str, tool_name: str, result: str) -> str:
    return (
        f"<{nonce}_START tool={tool_name}>\n"
        f"{result}\n"
        f"<{nonce}_END>"
    )
