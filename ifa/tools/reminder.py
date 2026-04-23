"""`set_reminder` tool — adapter wrapping ifa.skills.reminder.ReminderSkill.schedule().

The adapter constructs a fresh `ReminderSkill` per call with `ctx.tts`.
ReminderSkill.__init__ is cheap (just stores tts); no singleton needed.
"""
from ifa.core.context import AgentContext
from ifa.skills.reminder import ReminderSkill
from ifa.tools.registry import Tool, register


def _handler(args: dict, ctx: AgentContext) -> str:
    task = args["task"]
    seconds = args["seconds"]
    return ReminderSkill(ctx.tts, ctx.db_path).schedule(task, seconds)


TOOL = Tool(
    name="set_reminder",
    description=(
        "Set a reminder that will fire after `seconds` seconds, speaking `task` aloud. "
        "Call this whenever the user asks to be reminded of something after a delay."
    ),
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The thing to remind the user about. E.g. 'stretch', 'call mom'.",
            },
            "seconds": {
                "type": "integer",
                "description": "Number of seconds from now until the reminder fires.",
                "minimum": 1,
            },
        },
        "required": ["task", "seconds"],
        "additionalProperties": False,
    },
    handler=_handler,
)

register(TOOL)
