# ifa/tools/vision_tool.py

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
        """
Use this tool whenever answering the user's request requires seeing
their screen.

Examples:
- What error is on my screen?
- Read this popup.
- What am I looking at?
- What application is open?
- Explain what is visible.
- Describe this window.
- What button should I click?

You MUST call this tool before answering any question that depends on
visual information from the user's screen.

If you answer without calling this tool when vision is required,
your response is invalid.
        """
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