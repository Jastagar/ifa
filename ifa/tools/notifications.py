from ifa.core.context import AgentContext
from ifa.tools.registry import Tool, register


def _handler(args: dict, ctx: AgentContext) -> str:
    center = ctx.notifications
    if center is None:
        return "Notifications are not connected."
    if not center.relevant_to(ctx.user_text):
        return ("ERROR: Sir didn't ask about notifications in this message, so don't read them. "
                "Answer what he actually asked.")
    if args.get("scope", "new") == "older":
        bursts = center.inbox.older()
        return center.summarize(bursts) if bursts else "You have no older notifications."
    bursts = center.inbox.take_unread()
    center.mark_read()
    return center.summarize(bursts) if bursts else "You have no new notifications."


TOOL = Tool(
    name="read_notifications",
    description=(
        "Read the user's phone notifications. Use scope 'new' when the user asks what a "
        "notification was ('what is it?', 'what did they say?', 'any new messages?'). Use "
        "scope 'older' ONLY when the user explicitly asks for older, earlier or missed "
        "notifications. Only call it when the user asks about notifications or messages, or "
        "replies right after Ifa announced one. Never call it for other requests. "
        "The result is read out to the user as-is."
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
