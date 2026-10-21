import unittest
from datetime import date, datetime
from unittest.mock import patch

from ifa.skills import google_calendar as gc
from ifa.skills.google_calendar import CalendarError, GoogleCalendar, describe, resolve_range

TODAY = date(2026, 10, 22)


def event(title, start, end=None, location=None, id="e1"):
    ev = {"id": id, "summary": title, "start": {"dateTime": start}, "end": {"dateTime": end or start}}
    if location:
        ev["location"] = location
    return ev


class FakeResponse:
    def __init__(self, data=None, status=200):
        self._data, self.status_code = data, status
        self.content = b"x" if data is not None else b""

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}", response=self)


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, timeout=None, **kwargs):
        self.calls.append((method, url, kwargs.get("params", {})))
        return self.responses.pop(0)


class RangeTests(unittest.TestCase):
    def test_relative_days(self):
        self.assertEqual(resolve_range(None, None, TODAY)[:2], (TODAY, date(2026, 10, 23)))
        self.assertEqual(resolve_range("tomorrow", None, TODAY)[0], date(2026, 10, 23))
        self.assertEqual(resolve_range("kal", None, TODAY)[0], date(2026, 10, 23))
        self.assertEqual(resolve_range("today", 3, TODAY)[1], date(2026, 10, 25))

    def test_weekdays_and_weeks(self):
        self.assertEqual(resolve_range("friday", None, TODAY)[0], date(2026, 10, 23))
        self.assertEqual(resolve_range("thursday", None, TODAY)[0], TODAY)
        self.assertEqual(resolve_range("next thursday", None, TODAY)[0], date(2026, 10, 29))
        self.assertEqual(resolve_range("this week", None, TODAY)[:2], (TODAY, date(2026, 10, 26)))
        self.assertEqual(resolve_range("next week", None, TODAY)[:2], (date(2026, 10, 26), date(2026, 11, 2)))

    def test_dates(self):
        self.assertEqual(resolve_range("2026-11-05", None, TODAY)[0], date(2026, 11, 5))
        self.assertEqual(resolve_range("5th november", None, TODAY)[0], date(2026, 11, 5))
        self.assertEqual(resolve_range("jan 3", None, TODAY)[0], date(2027, 1, 3))
        with self.assertRaises(CalendarError):
            resolve_range("someday", None, TODAY)


class SplitDayTests(unittest.TestCase):
    def test_day_words_move_out_of_the_search(self):
        self.assertEqual(gc.split_day("gym session Friday", None), ("gym session", "friday"))
        self.assertEqual(gc.split_day("dentist on tomorrow", None), ("dentist", "tomorrow"))
        self.assertEqual(gc.split_day("gym", "friday"), ("gym", "friday"))
        self.assertEqual(gc.split_day("standup", None), ("standup", None))

    def test_remove_uses_the_moved_day(self):
        session = FakeSession(FakeResponse({"items": [event("Gym", "2026-10-23T07:00:00Z", id="g")]}), FakeResponse(None))
        cal = GoogleCalendar(session=session, now=lambda: datetime(2026, 10, 22, 9, 0))
        cal.remove("gym session Friday")
        params = session.calls[0][2]
        self.assertEqual(params["q"], "gym session")
        self.assertIn("2026-10-23T00:00:00", params["timeMin"])


class DescribeTests(unittest.TestCase):
    def test_timed_and_all_day(self):
        text = describe(event("Standup", "2026-10-22T10:00:00+05:30", "2026-10-22T10:30:00+05:30", "Zoom"))
        self.assertIn("Standup (Thu 22 Oct, ", text)
        self.assertIn(" at Zoom", text)
        all_day = describe({"summary": "Holiday", "start": {"date": "2026-10-24"}, "end": {"date": "2026-10-25"}})
        self.assertEqual(all_day, "Holiday (Sat 24 Oct, all day)")


class CalendarTests(unittest.TestCase):
    def cal(self, *responses):
        self.session = FakeSession(*responses)
        return GoogleCalendar(session=self.session, now=lambda: datetime(2026, 10, 22, 9, 0))

    def test_list(self):
        cal = self.cal(FakeResponse({"items": [event("Standup", "2026-10-22T10:00:00Z")]}))
        out = cal.list("today")
        self.assertTrue(out.startswith("Calendar today:\n- Standup"))
        method, url, params = self.session.calls[0]
        self.assertEqual((method, params["singleEvents"], params["orderBy"]), ("GET", "true", "startTime"))
        self.assertIn("2026-10-22T00:00:00", params["timeMin"])

    def test_empty_list(self):
        self.assertEqual(self.cal(FakeResponse({"items": []})).list("tomorrow", query="gym"),
                         "Nothing on the calendar tomorrow matching 'gym'.")

    def test_add_uses_quick_add_and_says_what_google_understood(self):
        cal = self.cal(FakeResponse(event("Dentist", "2026-10-23T17:00:00+05:30", "2026-10-23T17:30:00+05:30")))
        out = cal.add("Dentist tomorrow at 5pm for 30 minutes")
        self.assertTrue(out.startswith("Added: Dentist (Fri 23 Oct, "))
        self.assertTrue(self.session.calls[0][1].endswith("/events/quickAdd"))
        self.assertEqual(self.session.calls[0][2]["text"], "Dentist tomorrow at 5pm for 30 minutes")

    def test_remove_one_match(self):
        cal = self.cal(FakeResponse({"items": [event("Dentist", "2026-10-23T17:00:00Z", id="abc")]}), FakeResponse(None))
        out = cal.remove("dentist")
        self.assertTrue(out.startswith("Removed: Dentist"))
        self.assertEqual(self.session.calls[1][:2], ("DELETE", gc.EVENTS_URL + "/abc"))

    def test_remove_several_matches_asks_instead_of_deleting(self):
        cal = self.cal(FakeResponse({"items": [event("Gym", "2026-10-23T07:00:00Z", id="1"),
                                               event("Gym", "2026-10-25T07:00:00Z", id="2")]}))
        out = cal.remove("gym")
        self.assertIn("Several events match 'gym'", out)
        self.assertEqual([c[0] for c in self.session.calls], ["GET"])

    def test_remove_nothing(self):
        self.assertIn("No event matching 'yoga'", self.cal(FakeResponse({"items": []})).remove("yoga", day="friday"))

    def test_errors_become_sentences(self):
        with self.assertRaises(CalendarError) as cm:
            self.cal(FakeResponse({}, status=403)).list("today")
        self.assertIn("Calendar API is enabled", str(cm.exception))

    def test_not_logged_in(self):
        with patch.object(gc, "TOKEN_FILE", gc.REPO_ROOT / "does-not-exist.json"):
            with self.assertRaises(CalendarError) as cm:
                GoogleCalendar().list("today")
        self.assertEqual(str(cm.exception), gc.LOGIN_HINT)

    def test_expired_testing_login_explains_the_seven_day_limit(self):
        self.assertIn("7 days", gc.explain(Exception("invalid_grant: Token has been expired or revoked.")))


class ToolTests(unittest.TestCase):
    def test_registered(self):
        from ifa.tools import calendar, registry, register_all

        register_all()
        tool = registry.get("calendar")
        self.assertEqual(tool.parameters["properties"]["action"]["enum"], ["list", "add", "remove"])
        with patch.object(calendar, "_calendar", None), patch.object(gc.GoogleCalendar, "list", return_value="ok") as listed:
            self.assertEqual(registry.dispatch("calendar", {"action": "list", "day": "today"}, None), "ok")
        listed.assert_called_once_with("today", None, None)


if __name__ == "__main__":
    unittest.main()
