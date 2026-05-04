from ifa.core.context import AgentContext
from ifa.skills.home import Home
from ifa.skills.enums import Vibes
from ifa.tools.registry import Tool, register

def _handler(args: dict, ctx: AgentContext) -> str:
    to_vibe = args["to_vibe"]
    return Home(ctx.tts).change_vibe(to_vibe)

TOOL = Tool(
    name="select_vibe",
    description=(
        "call this when user says about turning on the vibe or gaming vibe or gaming mode or its time to play (usually gets misread as wipes/bikes) or turning on/off the vibe."
        "The vibe that we currently have is 'NORMAL' and 'GAMING', so when user says turn on the vibe that would usually mean 'gaming'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "to_vibe": {
                "type": "string",
                "enum": [m.value for m in Vibes],
                "description": "Target vibe",
            },
        },
        "required": ["to_vibe"],
        "additionalProperties": False,
    },
    handler=_handler,
)

register(TOOL)