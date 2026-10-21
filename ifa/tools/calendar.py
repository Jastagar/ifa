from ifa.core.context import AgentContext
from ifa.skills.google_calendar import GoogleCalendar
from ifa.tools.registry import Tool

_calendar: GoogleCalendar | None = None


def _handler(args: dict, ctx: AgentContext) -> str:
    global _calendar
    _calendar = _calendar or GoogleCalendar()
    action = args["action"]
    if action == "list":
        return _calendar.list(args.get("day"), args.get("days"), args.get("query"))
    if action == "add":
        return _calendar.add(args.get("text", ""))
    if action == "remove":
        return _calendar.remove(args.get("query", ""), args.get("day"), args.get("days"))
    return "ERROR: unknown action"


TOOL = Tool(
    name="calendar",
    description=(
        "Sir's Google Calendar. "
        "list: events for a day or range. day is today, tomorrow, a weekday (monday), this week, next week "
        "or a date (2026-10-25); days is how many days (optional); query filters by words (optional). "
        "add: create an event from one English sentence in text that says what and when, e.g. "
        "'Dentist with Dr Rao tomorrow at 5pm for 30 minutes'. Then tell Sir the date and time Google understood. "
        "remove: delete the one event matching query (add day to narrow it). If several match, ask him which. "
        "For reminders that only need a spoken alert, use set_reminder instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "add", "remove"]},
            "day": {"type": "string", "description": "today, tomorrow, monday, this week, next week, or YYYY-MM-DD"},
            "days": {"type": "integer", "minimum": 1, "maximum": 31},
            "query": {"type": "string", "description": "Words in the event title, for list or remove."},
            "text": {"type": "string", "description": "For add: what and when, in one English sentence."},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
    handler=_handler,
)
