from ifa.core.context import AgentContext
from ifa.skills import browser
from ifa.tools.registry import Tool


def _handler(args: dict, ctx: AgentContext) -> str:
    site = args["site"]
    query = (args.get("query") or "").strip()
    private = bool(args.get("private"))

    if site == "youtube":
        if not query:
            url, what = browser.YOUTUBE, "YouTube"
        elif args.get("play", True):
            url = browser.first_youtube_video(query)
            what = f"the top YouTube video for {query}, playing"
            if url is None:
                url, what = browser.youtube_search_url(query), f"YouTube results for {query} (couldn't pick a video)"
        else:
            url, what = browser.youtube_search_url(query), f"YouTube results for {query}"
    elif site == "google":
        if not query:
            return "ERROR: google needs a query."
        url, what = browser.google_search_url(query), f"Google results for {query}"
    else:
        if not args.get("url"):
            return "ERROR: site url needs url."
        url = browser.normalize_url(args["url"])
        what = url

    print(f"[browser] opening {url}")
    where = browser.open_url(url, private=private)
    return f"Opened {what} in {where}."


TOOL = Tool(
    name="open_website",
    description=(
        "Open a website in the browser (Brave). Use it for YouTube, Google searches and any website. "
        "For 'play X on YouTube' use site youtube with query X and play true: it opens the top video and plays it. "
        "For 'search YouTube for X' use play false to show the results instead. "
        "When Sir says 'this song' or 'it', get the name first (spotify_control pause or now_playing returns it) "
        "and put the song and artist in query. "
        "Use private true for adult or NSFW content, or when he asks for incognito or private."
    ),
    parameters={
        "type": "object",
        "properties": {
            "site": {"type": "string", "enum": ["youtube", "google", "url"]},
            "query": {"type": "string", "description": "What to search or play, e.g. 'Tum Hi Ho Arijit Singh'."},
            "play": {"type": "boolean", "description": "YouTube only: true opens and plays the top video (default), false shows results."},
            "url": {"type": "string", "description": "Only for site url: the address, e.g. github.com or https://news.ycombinator.com."},
            "private": {"type": "boolean", "description": "Open in a private window."},
        },
        "required": ["site"],
        "additionalProperties": False,
    },
    handler=_handler,
)
