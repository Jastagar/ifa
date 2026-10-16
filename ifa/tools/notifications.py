"""`read_notifications` tool — the only way notification text reaches the model.

Going through a tool (instead of pasting notifications into the prompt) keeps
third-party text out of the main conversation and tells us exactly when a
notification was read.

It is a terminal tool (return_direct): a focused summarizer writes one
sentence per notification and that is spoken as-is. Letting the main model
rephrase it brought back offers to reply, extra sentences, and "if I'm free"
instead of "if you're free".
"""
from ifa.core.context import AgentContext
from ifa.tools.registry import Tool, register


def _handler(args: dict, ctx: AgentContext) -> str:
    center = ctx.notifications
    if center is None:
        return "Notifications are not connected."
    if args.get("scope", "new") == "older":
        bursts = center.inbox.older()
        return center.summarize(bursts) if bursts else "You have no older notifications."
    bursts = center.inbox.take_unread()
    return center.summarize(bursts) if bursts else "You have no new notifications."


TOOL = Tool(
    name="read_notifications",
    description=(
        "Read the user's phone notifications. Use scope 'new' when the user asks what a "
        "notification was ('what is it?', 'what did they say?', 'any new messages?'). Use "
        "scope 'older' ONLY when the user explicitly asks for older, earlier or missed "
        "notifications. The result is read out to the user as-is."
    ),
    parameters={
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["new", "older"],
                "description": "'new' for recent unread notifications, 'older' for earlier ones.",
            },
        },
        "additionalProperties": False,
    },
    handler=_handler,
    return_direct=True,
)

register(TOOL)
