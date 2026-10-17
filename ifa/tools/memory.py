from ifa.core.context import AgentContext
from ifa.services import memory_store
from ifa.tools.registry import Tool


def _remember(args: dict, ctx: AgentContext) -> str:
    try:
        saved = memory_store.remember(ctx.db_path, args["topic"], args["fact"], args.get("kind", "fact"))
    except ValueError as exc:
        return f"ERROR: {exc}"
    fact = saved["fact"].rstrip(".")
    if saved["previous"]:
        return f"Updated {saved['topic']}: {fact} (before: {saved['previous'].rstrip('.')})."
    return f"Saved {saved['topic']}: {fact}."


def _recall(args: dict, ctx: AgentContext) -> str:
    rows = memory_store.recall(ctx.db_path, args["query"])
    if not rows:
        return f"Nothing saved about {args['query']}. Tell Sir you don't have that noted; don't guess."
    lines = []
    for row in rows:
        line = f"- {row['topic']} ({row['kind']}, saved {memory_store.ago(row['updated_at'])}): {row['fact']}"
        if row["previous"] and row["kind"] == "progress":
            line += f" (before that: {row['previous']})"
        lines.append(line)
    return "Saved memories, best match first:\n" + "\n".join(lines)


def _forget(args: dict, ctx: AgentContext) -> str:
    row = memory_store.forget(ctx.db_path, args["topic"])
    if row is None:
        return f"Nothing saved about {args['topic']}."
    return f"Forgot {row['topic']}."


TOOL = Tool(
    name="remember",
    description=(
        "Save or update one thing Sir wants you to keep, one topic per call: where he got to in something "
        "(kind progress, e.g. 'I stopped Sherlock at episode 8'), something he likes or dislikes (preference), "
        "how he wants something done (instruction, e.g. 'when I say gaming time, open Steam'), or any other "
        "fact (fact). Saving a topic again replaces the old value, so use the same short topic name. "
        "Call it when he says remember, note, keep in mind, or tells you where he stopped or what he likes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "topic": {"type": "string", "minLength": 1,
                      "description": "Short name of what it's about: 'Sherlock', 'coffee', 'gaming time', 'car service'."},
            "fact": {"type": "string", "minLength": 1,
                     "description": "What to remember, as a short sentence: 'Stopped at season 2, episode 8.'"},
            "kind": {"type": "string", "enum": list(memory_store.KINDS)},
        },
        "required": ["topic", "fact", "kind"],
        "additionalProperties": False,
    },
    handler=_remember,
)

RECALL_TOOL = Tool(
    name="recall",
    description=(
        "Look up something Sir told you before. Call it whenever he asks about it ('where was I on Sherlock?', "
        "'what do I like on my pizza?', 'when is my car service?'), then answer from the result in one sentence. "
        "Never answer such questions from guesswork."
    ),
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "minLength": 1, "description": "What he's asking about, e.g. 'Sherlock'."}},
        "required": ["query"],
        "additionalProperties": False,
    },
    handler=_recall,
)

FORGET_TOOL = Tool(
    name="forget",
    description="Delete a saved memory when Sir asks you to forget it or says it's no longer true.",
    parameters={
        "type": "object",
        "properties": {"topic": {"type": "string", "minLength": 1}},
        "required": ["topic"],
        "additionalProperties": False,
    },
    handler=_forget,
)
