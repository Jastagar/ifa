from ifa.core.context import AgentContext
from ifa.skills.spotify import SpotifySkill
from ifa.tools.registry import Tool, register

_skill: SpotifySkill | None = None


def _spotify(ctx: AgentContext) -> SpotifySkill:
    global _skill
    if _skill is None:
        _skill = SpotifySkill(ctx.tts)
    return _skill


def _handler(args: dict, ctx: AgentContext) -> str:
    action = args["action"]
    spotify = _spotify(ctx)
    if action == "play":
        return spotify.play(args.get("query"))
    if action == "pause":
        return spotify.pause()
    if action == "next":
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
Use action "play" without a query to resume whatever was playing.
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
