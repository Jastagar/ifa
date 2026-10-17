from ifa.core.context import AgentContext
from ifa.supervisor import is_supervised
from ifa.tools.registry import Tool

RESTART_REPLY = "Restarting, Sir."


def _end(args: dict, ctx: AgentContext) -> str:
    ctx.end_conversation = True
    return ""


def _restart(args: dict, ctx: AgentContext) -> str:
    if not is_supervised():
        return "ERROR: Ifa wasn't started with python -m ifa.main, so she can't restart herself. Restart her by hand."
    ctx.restart_requested = True
    return RESTART_REPLY


END_TOOL = Tool(
    name="end_conversation",
    description=(
        "Sir is done talking for now: he says goodbye, thanks you without asking for anything more, "
        "or tells you to stop listening. Ends the conversation silently, with no reply. "
        "Don't call it if the same message also asks for something."
    ),
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
    handler=_end,
    return_direct=True,
)

RESTART_TOOL = Tool(
    name="restart_ifa",
    description=(
        "Restart Ifa herself (this assistant) to reload her code and settings, when Sir asks you to "
        "restart or reboot yourself. Never use it for the computer or other apps."
    ),
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
    handler=_restart,
    return_direct=True,
)
