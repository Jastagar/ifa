"""`get_time` tool — adapter wrapping ifa.skills.system.TimeSkill."""
from ifa.core.context import AgentContext
from ifa.skills.system import TimeSkill
from ifa.tools.registry import Tool, register

_skill = TimeSkill()


def _handler(args: dict, ctx: AgentContext) -> str:
    return _skill.handle("")


TOOL = Tool(
    name="get_time",
    description="Return the current local time. Call this when the user asks for the time, clock, or similar.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
    handler=_handler,
)

register(TOOL)
