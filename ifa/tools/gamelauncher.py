from ifa.core.context import AgentContext
from ifa.skills.games.launcher import GameLauncher
from ifa.tools.registry import Tool, register


def _handler(args: dict, ctx: AgentContext) -> str:
    game_name = args["game"]

    launcher = GameLauncher()

    return launcher.launch(game_name)


TOOL = Tool(
    name="launch_game",
    description=("""
Use this tool whenever the user wants to open, launch, start, or run a PC game.

This tool launches the actual game executable/shortcut on the user's computer.

Examples:
- "Launch Factorio"
- "Open Minecraft"
- "Start Red Dead Redemption 2"
- "Run Stardew Valley"

Do NOT create a workflow for game launching.
Do NOT explain gameplay.
Do NOT provide game instructions.

If the user wants a game opened, call this tool.
"""
    ),
    parameters={
        "type": "object",
        "properties": {
            "game": {
                "type": "string",
                "description": (
                    "Name of the game to launch. "
                    "Examples: Factorio, Minecraft, Valorant, "
                    "Red Dead Redemption 2."
                )
            }
        },
        "required": ["game"],
        "additionalProperties": False,
    },
    handler=_handler,
)

register(TOOL)