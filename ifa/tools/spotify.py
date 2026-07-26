# ifa/tools/spotify_tool.py

from ifa.core.context import AgentContext
from ifa.skills.spotify import SpotifySkill
from ifa.tools.registry import Tool, register


def _handler(args: dict, ctx: AgentContext) -> str:
    action = args["action"]
    query = args.get("query")

    print("CALLING SPOTIFY")
    spotify = SpotifySkill(ctx.tts)

    if action == "play":
        if not query:
            return "Missing query for play"
        return spotify.play(query)

    elif action == "pause":
        return spotify.pause()

    elif action == "next":
        return spotify.next()

    return "Invalid action"


TOOL = Tool(
    name="spotify_control",
    description=(
        """
If the user asks to perform an action (play music, stop music, play song etc):

- You MUST call a spotify_control tool
- You MUST NOT pretend the action succeeded
- You MUST NOT return a normal message instead of calling a tool

If you fail to call a tool, your response is invalid.
In case you play wrong song, Call the tool again
        """
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["play", "pause", "next"]
            },
            "query": {
                "type": "string"
            }
        },
        "required": ["action"],
        "additionalProperties": False,
    },
    handler=_handler,
)

register(TOOL)