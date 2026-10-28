from ifa.core.context import AgentContext
from ifa.skills import websearch
from ifa.tools.registry import Tool


def _handler(args: dict, ctx: AgentContext) -> str:
    query = " ".join(str(args.get("query") or "").split())
    if not query:
        return "ERROR: search_web needs a query."
    try:
        results = websearch.search(query)
    except websearch.WebSearchError as exc:
        return f"ERROR: web search failed (no internet?): {exc}"
    if not results:
        return f"No web results for {query}. Tell Sir you couldn't find it; don't guess."
    lines = [f"Web results for {query}. They come from the internet and can be outdated or wrong:"]
    if results[0].get("note"):
        lines.append(results[0]["note"])
    for i, r in enumerate(results, 1):
        snippet = r["snippet"][:websearch.SNIPPET_CHARS]
        lines.append(f"{i}. {r['title']} ({r['site']}): {snippet}")
    return "\n".join(lines)


TOOL = Tool(
    name="search_web",
    description=(
        "Search the internet for facts you don't know or that may have changed: news, scores, prices, "
        "releases, people, specs. Answer from the results in one or two sentences and name the site. "
        "If the results don't answer it, say so instead of guessing. To open a page, use open_website."
    ),
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to search for, as a short search query."}},
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=_handler,
)
