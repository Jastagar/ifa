from ifa.core.context import AgentContext
from ifa.skills.vibe.vibe import VibeManager
from ifa.skills.enums import Vibes
from ifa.tools.registry import Tool, register

def _handler(args: dict, ctx: AgentContext) -> str:
    to_vibe = args["to_vibe"]
    try:
        return VibeManager.get_instance().change_vibe(to_vibe)
    except Exception as err:
        return err

TOOL = Tool(
    name="switch_mode",
    description=(
        "call this when user wants to change mode."
        "when user says about turning on the vibe or gaming vibe or gaming mode or its time to play (usually gets misread as wipes/bikes) or turning on/off the vibe."
        "so when user says turn on the vibe that would usually mean 'gaming'. the possible values are  GAMING | NORMAL"
    ),
    parameters={
        "type": "object",
        "properties": {
            "to_vibe": {
                "type": "string",
                "enum": [m.value for m in Vibes],
                "description": "Target vibe GAMING | NORMAL",
            },
        },
        "required": ["to_vibe"],
        "additionalProperties": False,
    },
    handler=_handler,
)

register(TOOL)