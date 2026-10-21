from __future__ import annotations

import os
import re
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKEN_FILE = REPO_ROOT / ".google_token.json"
DEFAULT_CREDENTIALS = REPO_ROOT / "google_credentials.json"
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
EVENTS_URL = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
MAX_EVENTS = 20
LOGIN_HINT = "Google Calendar isn't connected yet. Run 'Log in to Google Calendar' in the launcher's Setup list."
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DATE_FORMATS = ("%Y-%m-%d", "%d %b", "%d %B", "%b %d", "%B %d", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y")


class CalendarError(Exception):
    pass


def credentials_path() -> Path:
    return Path(os.environ.get("IFA_GOOGLE_CREDENTIALS", "").strip() or DEFAULT_CREDENTIALS)


def resolve_range(day: str | None, days: int | None, today: date) -> tuple[date, date, str]:
    text = " ".join((day or "today").lower().replace(",", " ").split())
    if text in ("today", "aaj", "now"):
        start, label = today, "today"
    elif text in ("tomorrow", "kal"):
        start, label = today + timedelta(days=1), "tomorrow"
    elif text == "yesterday":
        start, label = today - timedelta(days=1), "yesterday"
    elif text in ("day after tomorrow", "parso"):
        start, label = today + timedelta(days=2), "the day after tomorrow"
    elif text in ("this week", "week"):
        return today, today + timedelta(days=7 - today.weekday()), "this week"
    elif text == "next week":
        start = today + timedelta(days=7 - today.weekday())
        return start, start + timedelta(days=7), "next week"
    elif text.removeprefix("next ").removeprefix("on ") in WEEKDAYS:
        target = WEEKDAYS.index(text.removeprefix("next ").removeprefix("on "))
        start = today + timedelta(days=(target - today.weekday()) % 7)
        if text.startswith("next ") and start == today:
            start += timedelta(days=7)
        label = start.strftime("%A")
    else:
        start = _parse_date(text, today)
        label = start.strftime("%A %d %B")
    span = max(1, min(int(days or 1), 31))
    return start, start + timedelta(days=span), label if span == 1 else f"{span} days from {label}"


DAY_WORDS = re.compile(
    r"\b(?:on\s+|next\s+|this\s+)?(today|tomorrow|yesterday|day after tomorrow|this week|next week|"
    + "|".join(WEEKDAYS) + r")\b", re.IGNORECASE)


def split_day(query: str | None, day: str | None) -> tuple[str | None, str | None]:
    if not query or day:
        return query, day
    match = DAY_WORDS.search(query)
    if not match:
        return query, day
    rest = " ".join((query[:match.start()] + " " + query[match.end():]).split())
    return rest or None, match.group(0).lower().removeprefix("on ").strip()


def _parse_date(text: str, today: date) -> date:
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text)
    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
        if "%Y" not in fmt:
            parsed = parsed.replace(year=today.year)
            if parsed < today - timedelta(days=30):
                parsed = parsed.replace(year=today.year + 1)
        return parsed
    raise CalendarError(f"I couldn't read the date '{text}'. Use today, tomorrow, a weekday, this week, or a date like 2026-10-25.")


def _clock(dt: datetime) -> str:
    hour = dt.hour % 12 or 12
    return f"{hour}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def describe(event: dict) -> str:
    title = event.get("summary") or "(no title)"
    start, end = event.get("start", {}), event.get("end", {})
    if "dateTime" in start:
        begin = datetime.fromisoformat(start["dateTime"].replace("Z", "+00:00")).astimezone()
        finish = datetime.fromisoformat(end["dateTime"].replace("Z", "+00:00")).astimezone() if "dateTime" in end else None
        when = f"{begin.strftime('%a %d %b')}, {_clock(begin)}" + (f" to {_clock(finish)}" if finish else "")
    else:
        when = f"{date.fromisoformat(start.get('date', '1970-01-01')).strftime('%a %d %b')}, all day"
    where = f" at {event['location']}" if event.get("location") else ""
    return f"{title} ({when}){where}"


def explain(exc: Exception) -> str:
    if isinstance(exc, CalendarError):
        return str(exc)
    text = str(exc)
    if "invalid_grant" in text or "Token has been expired or revoked" in text:
        return ("The Google login has expired. " + LOGIN_HINT + " If this happens every week, set your OAuth app "
                "to 'In production' in Google Cloud: apps in testing lose their login after 7 days.")
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 401:
        return "The Google login isn't valid any more. " + LOGIN_HINT
    if status == 403:
        return "Google refused access. Check that the Google Calendar API is enabled in your Google Cloud project."
    if status == 404:
        return "That event no longer exists."
    import requests
    if isinstance(exc, requests.RequestException):
        return "Couldn't reach Google Calendar. Check the internet connection."
    return f"Google Calendar failed: {text}"


class GoogleCalendar:
    def __init__(self, session=None, now=datetime.now) -> None:
        self._session = session
        self._now = now

    def _http(self):
        if self._session is None:
            if not TOKEN_FILE.exists():
                raise CalendarError(LOGIN_HINT)
            from google.auth.transport.requests import AuthorizedSession
            from google.oauth2.credentials import Credentials

            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
            self._session = AuthorizedSession(creds)
        return self._session

    def _call(self, method: str, url: str, **kwargs) -> dict:
        response = self._http().request(method, url, timeout=15, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else {}

    def _local(self, day: date) -> str:
        return datetime.combine(day, time.min).astimezone().isoformat()

    def _events(self, start: date, end: date, query: str | None) -> list[dict]:
        params = {"timeMin": self._local(start), "timeMax": self._local(end), "singleEvents": "true",
                  "orderBy": "startTime", "maxResults": MAX_EVENTS}
        if query:
            params["q"] = query
        return self._call("GET", EVENTS_URL, params=params).get("items", [])

    def _run(self, action, *args) -> str:
        try:
            return action(*args)
        except Exception as exc:
            raise CalendarError(explain(exc)) from exc

    def list(self, day: str | None = None, days: int | None = None, query: str | None = None) -> str:
        return self._run(self._list, day, days, query)

    def add(self, text: str) -> str:
        return self._run(self._add, text)

    def remove(self, query: str, day: str | None = None, days: int | None = None) -> str:
        return self._run(self._remove, query, day, days)

    def _list(self, day, days, query) -> str:
        query, day = split_day(query, day)
        start, end, label = resolve_range(day, days, self._now().date())
        events = self._events(start, end, query)
        if not events:
            return f"Nothing on the calendar {label}" + (f" matching '{query}'." if query else ".")
        lines = "\n".join(f"- {describe(e)}" for e in events)
        more = "\n(more events not shown)" if len(events) >= MAX_EVENTS else ""
        return f"Calendar {label}:\n{lines}{more}"

    def _add(self, text: str) -> str:
        text = " ".join((text or "").split())
        if not text:
            raise CalendarError("Say what to add and when, e.g. 'Dentist tomorrow at 5pm'.")
        event = self._call("POST", f"{EVENTS_URL}/quickAdd", params={"text": text})
        return f"Added: {describe(event)}. That's how Google read it; correct it if the date or time is wrong."

    def _remove(self, query, day, days) -> str:
        query, day = split_day((query or "").strip(), day)
        query = (query or "").strip()
        if not query:
            raise CalendarError("Say which event to remove.")
        today = self._now().date()
        if day:
            start, end, label = resolve_range(day, days, today)
        else:
            start, end, label = today, today + timedelta(days=days or 14), "in the next two weeks"
        events = self._events(start, end, query)
        if not events:
            return f"No event matching '{query}' {label}."
        if len(events) > 1:
            options = "\n".join(f"{i}. {describe(e)}" for i, e in enumerate(events[:5], 1))
            return f"Several events match '{query}'. Ask Sir which one, then remove it with a more specific query or day:\n{options}"
        self._call("DELETE", f"{EVENTS_URL}/{events[0]['id']}")
        return f"Removed: {describe(events[0])}. It can be restored from Google Calendar's trash."


def login() -> int:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    path = credentials_path()
    if not path.exists():
        print(f"Missing {path.name}. In Google Cloud Console: enable the Google Calendar API, create an OAuth "
              f"client of type 'Desktop app', download its JSON and save it as {path}.", file=sys.stderr)
        return 1
    from google_auth_oauthlib.flow import InstalledAppFlow

    print("Opening Google in your browser. Approve calendar access for Ifa.", flush=True)
    flow = InstalledAppFlow.from_client_secrets_file(str(path), SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True, prompt="consent", access_type="offline")
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    os.chmod(TOKEN_FILE, 0o600)
    try:
        print("Logged in to Google Calendar. " + GoogleCalendar().list("today").splitlines()[0])
    except CalendarError as exc:
        print(f"Logged in, but the test read failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(login())
