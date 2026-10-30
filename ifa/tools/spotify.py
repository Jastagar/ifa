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
        return _quietly(spotify.play(args.get("query"), args.get("choice"), args.get("artist")), ctx, music_on=True)
    if action == "pause":
        return _quietly(spotify.pause(), ctx, music_on=False)
    if action == "now_playing":
        return spotify.now_playing()
    if action == "next":
        return _quietly(spotify.next(), ctx, music_on=True)
    return "Invalid action"


SUCCESS = ("Playing ", "Resumed ", "Paused", "Skipped ")


def _quietly(result: str, ctx: AgentContext, music_on: bool) -> str:
    # The music starting or stopping is the answer, so Ifa says nothing after it. With music on she also
    # stops listening for a follow-up, which would only hear the song. Failures still go back to the model.
    if result.startswith(SUCCESS):
        ctx.finish_silently = True
        ctx.end_conversation = ctx.end_conversation or music_on
    return result


TOOL = Tool(
    name="spotify_control",
    description=(
        "Control Spotify through its API. It never opens or shows the app: if Sir says to open Spotify, "
        "call open_app with app spotify first. "
        "To play a song, use two steps. First call action search with query set to the song name as Sir said it, "
        "and artist if he named one. Then read the numbered results and call action play with choice set to the "
        "number of the closest match, preferring the original over remix, slowed, lofi or cover versions unless "
        "he asked for one. Never say a song is playing before the play call succeeds. "
        "Action play with no choice resumes playback; pause and next control what is playing. "
        "Pause and now_playing return the song's name, so use it when Sir says 'this song' or 'it', "
        "for example to play the same song on YouTube with open_website."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["search", "play", "pause", "next", "now_playing"]},
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
