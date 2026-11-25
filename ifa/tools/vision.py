from ifa.core.context import AgentContext
from ifa.skills.vision.vision import VisionSkill
from ifa.tools.registry import Tool, register


def _handler(args: dict, ctx: AgentContext) -> str:
    prompt = args["prompt"]

    vision = VisionSkill()

    return vision.analyze(prompt)


TOOL = Tool(
    name="analyze_screen",
    description=(
        "Look at Sir's PC screen (a screenshot) and answer about what is SHOWN: text, errors, windows, pages, "
        "images, buttons. Use it when he talks about seeing or reading: 'what's on my screen', 'read this', "
        "'this error', 'what am I looking at', 'which button'. For something he HEARD (people talking, TV, a video, "
        "a call, 'what did he say'), use recall_recent_audio instead. If it's unclear whether he means something "
        "seen or heard ('is it true?', 'explain that'), try recall_recent_audio first and use this only if that "
        "found no speech."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "The visual question to answer about the user's current screen."
                ),
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
    handler=_handler,
)

register(TOOL)