"""Web search skill for DuckDuckGo API."""
from ifa.core.context import AgentContext
from ifa.tools.registry import Tool, register


class WebSearchSkill:
    """Skill for performing web searches using DuckDuckGo API."""

    _BASE_URL = "https://api.duckduckgo.com/search"
    _PARAMS = {"q": "{query}", "format": "json", "pretty": 1}

    def __init__(self, tts=None):
        self.tts = tts

    @property
    def _headers(self) -> dict:
        """Return headers for requests."""
        return {
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
        }

    def _fetch(self) -> dict:
        """Fetch search results from DuckDuckGo."""
        import requests

        resp = requests.get(self._BASE_URL, params=self._PARAMS, headers=self._headers)
        resp.raise_for_status()
        return resp.json()

    def handle(self, text: str) -> str:
        """Handle the web search request."""
        import re

        query = re.search(r"web\s+search\s+(.+)", text.lower())
        if not query or not query.group(1):
            return "I can help you search the web. Try saying 'web search <your question>'."

        query_text = query.group(1)
        results = self._fetch()

        if "Results" in results and len(results["Results"]) > 0:
            lines = [f"Here are the top results for '{query_text}':"]
            for i, r in enumerate(results["Results"][:5], 1):
                title = r.get("Title", "Unknown")
                url = r.get("Url", "")
                abstract = r.get("Abstract", "")
                lines.append(f"   {i}. [{title}]({url})\n      {abstract}")
            lines.append("")

            return "\n".join(lines)

        # DuckDuckGo sometimes returns an IImage result instead of text
        if "IResult" in results:
            ir = results["IResult"]
            title = ir.get("FirstURLTitle", "") or ir.get("TextTitle", "")
            url = ir.get("FirstURL", "") or ir.get("TextUrl", "")
            return f"I found: {title} -> {url}"

        return "Sorry, I couldn't find any relevant web results."


def _handler(args: dict, ctx: AgentContext) -> str:
    """Handler function for the web search tool."""
    skill = WebSearchSkill(ctx.tts)
    query = args.get("query", "")
    if not query.strip():
        return "Please specify a search query."

    results = skill._fetch()

    if "Results" in results and len(results["Results"]) > 0:
        lines = [f"Here are the top results for '{query}':"]
        for i, r in enumerate(results["Results"][:5], 1):
            title = r.get("Title", "Unknown")
            url = r.get("Url", "")
            abstract = r.get("Abstract", "")
            lines.append(f"   {i}. [{title}]({url})\n      {abstract}")
        lines.append("")

        return "\n".join(lines)

    # DuckDuckGo sometimes returns an IImage result instead of text
    if "IResult" in results:
        ir = results["IResult"]
        title = ir.get("FirstURLTitle", "") or ir.get("TextTitle", "")
        url = ir.get("FirstURL", "") or ir.get("TextUrl", "")
        return f"I found: {title} -> {url}"

    return "Sorry, I couldn't find any relevant web results."


TOOL = Tool(
    name="search_web",
    description="Search the web using DuckDuckGo for current information. Call this when you need real-time data about topics like news, facts, or recent events.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query string."},
        },
        "additionalProperties": False,
    },
    handler=_handler,
)

register(TOOL)