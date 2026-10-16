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
    if action == "search":
        if not args.get("query"):
            return "ERROR: search needs a query."
        return spotify.search(args["query"], args.get("artist"))
    if action == "play":
        return spotify.play(args.get("query"), args.get("choice"), args.get("artist"))
    if action == "pause":
        return spotify.pause()
    if action == "next":
        return spotify.next()
    return "Invalid action"


TOOL = Tool(
    name="spotify_control",
    description=(
        "Control Spotify through its API. It never opens or shows the app: if Sir says to open Spotify, "
        "call open_app with app spotify first. "
        "To play a song, use two steps. First call action search with query set to the song name as Sir said it, "
        "and artist if he named one. Then read the numbered results and call action play with choice set to the "
        "number of the closest match, preferring the original over remix, slowed, lofi or cover versions unless "
        "he asked for one. Never say a song is playing before the play call succeeds. "
        "Action play with no choice resumes playback; pause and next control what is playing."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["search", "play", "pause", "next"]},
            "query": {"type": "string", "description": "Song name for search, as Sir said it."},
            "artist": {"type": "string", "description": "Artist name, only if Sir named one."},
            "choice": {"type": "integer", "minimum": 1, "description": "Result number from the last search."},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
    handler=_handler,
)

register(TOOL)
