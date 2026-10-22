"""Parsing /api/leaves/dashboard, from a trimmed copy of a live response.

Runs without Ollama, Postgres or a network.
"""

import os
from datetime import date
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from rag import tools  # noqa: E402


DASHBOARD = {"error": None, "data": {
    "employee_id": "emp",
    "leaves": {
        "leave_data": [
            {
                "leave_type": {"leave_type_id": "t-el", "name": "el-1",
                               "is_sick_leave": False, "is_paid": True},
                "total_allocated": 21.5, "balance": 19.5, "used": 2.0,
                "quota_unit": "days",
                "configuration": {
                    "applying_leave": {
                        "allow_half_day": False,
                        "prior_notice_rules": [{"id": 15, "calendar_days_notice": 7,
                                                "working_days_needed": 7,
                                                "min_leave_days_required": 5}],
                    },
                    "leave_approval": {"requires_approval": True, "levels": [
                        {"approval_assignees": [{"employee": {
                            "employee_id": "other", "full_name": "Someone Else",
                            "email": "someone@example.com",
                            "date_of_birth": "2008-08-04"}}]}]},
                },
            },
            {
                # A comp-off type: no applying_leave block at all.
                "leave_type": {"leave_type_id": "t-co", "name": "CompOff-v"},
                "total_allocated": 0.0, "balance": 0.0, "used": 0.0,
                "configuration": {"applying_leave": None, "leave_approval": None},
            },
        ],
    },
    "upcoming_holidays": {"name": "Gandhi", "date": "2026-10-14",
                          "description": None, "is_optional": False},
}}

# /api/self/leave-applications/?type=upcoming, trimmed. A leave cancelled and
# applied for again leaves both on the same date.
UPCOMING = {"error": None, "data": {"count": 3, "next": None, "results": [
    {"leave_log_id": "log-1", "leave_type": {"leave_type_id": "t-el", "name": "el-1"},
     "start_date": "2026-10-12", "end_date": "2026-10-12", "duration_days": 1.0,
     "status": "Pending"},
    {"leave_log_id": "log-0", "leave_type": {"leave_type_id": "t-el", "name": "el-1"},
     "start_date": "2026-10-12", "end_date": "2026-10-12", "duration_days": 1.0,
     "status": "Cancelled"},
    {"leave_log_id": "log-2", "leave_type": {"leave_type_id": "t-el", "name": "el-1"},
     "start_date": "2026-10-27", "end_date": "2026-10-28", "duration_days": 2.0,
     "status": "Approved"},
]}}


def test_balances_come_from_the_entry_not_the_nested_type():
    """The old walker matched the nested leave_type and lost every balance."""
    el = tools.leave_types(DASHBOARD)[0]
    assert (el["id"], el["name"]) == ("t-el", "el-1")
    assert (el["balance"], el["used"], el["total_allocated"]) == (19.5, 2.0, 21.5)
    assert el["half_day_allowed"] is False
    assert el["prior_notice"] == ["7 calendar days' notice for 5 or more days of leave"]


def test_a_type_without_rules_still_parses():
    co = tools.leave_types(DASHBOARD)[1]
    assert co["name"] == "CompOff-v"
    assert co["half_day_allowed"] is False and co["prior_notice"] == []


def test_approvers_never_reach_the_model():
    assert "someone@example.com" not in str(tools.leave_types(DASHBOARD))


def test_upcoming_leave_and_holiday():
    assert tools.upcoming_leaves(UPCOMING) == [
        {"id": "log-1", "leave_type": "el-1", "start_date": "2026-10-12",
         "end_date": "2026-10-12", "days": 1.0, "approved": False},
        {"id": "log-2", "leave_type": "el-1", "start_date": "2026-10-27",
         "end_date": "2026-10-28", "days": 2.0, "approved": True},
    ]  # the cancelled one is not booked
    assert tools.next_holiday(DASHBOARD) == {
        "name": "Gandhi", "date": "2026-10-14", "optional": False}


def test_missing_parts_are_empty():
    assert tools.leave_types({"data": {}}) == []
    assert tools.upcoming_leaves({}) == []
    assert tools.next_holiday({"data": {"upcoming_holidays": None}}) is None


def test_half_day_refused_before_submitting(monkeypatch):
    sent = []
    monkeypatch.setattr(tools, "fetch_dashboard", lambda token: DASHBOARD)
    monkeypatch.setattr(tools, "hrms_request", lambda *a, **k: sent.append(a) or {})
    result = tools.apply_leave.func("tok", "el-1", "2026-10-20", "2026-10-20",
                                    is_half_day=True)
    assert "half day" in result["error"] and sent == []


CANCELLED = {"error": None, "data": {
    "leave_log_id": "log-1",
    "employee_details": {"full_name": "Me", "date_of_birth": "2008-02-06"},
    "leave_type": {"leave_type_id": "t-el", "name": "el-1"},
    "start_date": "2026-10-12", "end_date": "2026-10-12", "duration_days": 1.0,
    "status": "Cancelled",
}}


def test_cancel_finds_the_leave_by_date(monkeypatch):
    sent = []
    monkeypatch.setattr(tools, "fetch_upcoming", lambda token: UPCOMING)

    def request(url, token, body=None, method=None):
        sent.append((url, body, method))
        return CANCELLED
    monkeypatch.setattr(tools, "hrms_request", request)

    result = tools.cancel_leave.func("tok", "2026-10-12")
    assert sent == [(f"{tools.ATTENDANCE_BASE_URL}/api/leave-applications/log-1/cancel",
                     {}, "PUT")]
    assert result == {"status": "Cancelled", "leave_type": "el-1",
                      "start_date": "2026-10-12", "end_date": "2026-10-12", "days": 1.0}


def test_cancel_with_no_matching_leave_sends_nothing(monkeypatch):
    sent = []
    monkeypatch.setattr(tools, "fetch_upcoming", lambda token: UPCOMING)
    monkeypatch.setattr(tools, "hrms_request", lambda *a, **k: sent.append(a) or {})
    result = tools.cancel_leave.func("tok", "2026-11-01")
    assert "No upcoming leave" in result["error"] and sent == []
    assert "id" not in result["upcoming_leaves"][0]


class Context:
    access_token = None
    profile = ""
    credentials = ("me@example.com", "secret", "acme")


class Runtime:
    def __init__(self):
        self.context = Context()


def test_login_posts_the_credentials_and_keeps_the_token(monkeypatch):
    sent = []

    def request(url, token, body=None, method=None):
        sent.append((url, token, body))
        if url.endswith("/login"):
            return {"status": "success", "data": {"token": "tok-1"}}
        return {"data": {"full_name": "Me"}}
    monkeypatch.setattr(tools, "hrms_request", request)

    runtime = Runtime()
    result = tools.login.func(runtime)
    assert sent[0] == (f"{tools.AUTH_BASE_URL}/api/users/login", None, {
        "email": "me@example.com", "password": "secret",
        "remember_me": True, "subdomain": "acme"})
    assert sent[1][1] == "tok-1"
    assert runtime.context.access_token == "tok-1"
    assert runtime.context.credentials is None  # dropped once used
    assert result.startswith("Logged in") and "secret" not in result


def test_failed_login_keeps_no_session(monkeypatch):
    monkeypatch.setattr(tools, "hrms_request", lambda *a, **k: {
        "error": "HRMS returned 401", "detail": "Invalid credentials"})
    runtime = Runtime()
    result = tools.login.func(runtime)
    assert result.startswith("Login failed: HRMS returned 401")
    assert runtime.context.access_token is None and runtime.context.credentials is None


def test_login_without_credentials_sends_nothing(monkeypatch):
    sent = []
    monkeypatch.setattr(tools, "hrms_request", lambda *a, **k: sent.append(a) or {})
    runtime = Runtime()
    runtime.context.credentials = None
    assert tools.login.func(runtime).startswith("Login cancelled") and sent == []


def test_weekend_only_dates_are_turned_back():
    assert tools.weekend_only("2026-10-17", "2026-10-17")["error"].startswith(
        "2026-10-17 (Saturday) is a weekly off")
    assert tools.weekend_only("2026-10-17", "2026-10-18") is not None  # Sat-Sun


def test_a_weekday_or_a_range_across_a_weekend_goes_through():
    assert tools.weekend_only("2026-10-16", "2026-10-16") is None  # Friday
    assert tools.weekend_only("2026-10-16", "2026-10-19") is None  # Fri-Mon
    assert tools.weekend_only("not a date", "2026-10-18") is None  # the HRMS's call


def test_apply_wfh_refuses_a_weekend_without_calling_the_hrms(monkeypatch):
    monkeypatch.setattr(tools, "hrms_request",
                        lambda *a, **k: pytest.fail("the HRMS was called"))
    result = tools.apply_wfh.func("token", "2026-10-18", "2026-10-18")
    assert "weekly off" in result["error"]


HOLIDAYS = {"error": None, "data": [
    {"name": "Independence Day", "date": "2026-08-15", "is_optional": False, "is_active": True},
    {"name": "Republic Day", "date": "2027-01-26", "is_optional": False, "is_active": True},
    {"name": "Testijng", "date": "2026-10-23", "is_optional": True, "is_active": True},
    {"name": "shubhamtest", "date": "2026-11-20", "is_optional": False, "is_active": True},
    {"name": "Diwali", "date": "2026-11-08", "is_optional": False, "is_active": True},
    {"name": "Old", "date": "2026-12-01", "is_optional": False, "is_active": False},
]}


def test_holidays_are_upcoming_real_and_in_order():
    assert tools.holiday_list(HOLIDAYS, date(2026, 10, 19)) == [
        {"name": "Diwali", "date": "2026-11-08", "weekday": "Sunday", "optional": False},
        {"name": "Republic Day", "date": "2027-01-26", "weekday": "Tuesday", "optional": False},
    ]


def test_past_holidays_only_when_asked():
    names = [h["name"] for h in tools.holiday_list(HOLIDAYS, date(2026, 10, 19), include_past=True)]
    assert names == ["Independence Day", "Diwali", "Republic Day"]


def test_backdating_only_within_the_current_month():
    today = date(2026, 10, 21)
    assert tools.past_month("2026-10-05", today) is None  # same month
    assert tools.past_month("2026-10-01", today) is None  # first of the month
    assert tools.past_month("2026-11-03", today) is None  # future
    error = tools.past_month("2026-09-30", today)["error"]
    assert "past month" in error and "2026-10-01" in error
    assert tools.past_month("2025-10-21", today) is not None  # same month, last year
    assert tools.past_month("not a date", today) is None  # the HRMS's call


def test_leave_history_carries_no_personal_record(monkeypatch):
    raw = {"error": None, "data": {"count": 1, "results": [
        dict(UPCOMING["data"]["results"][0], is_half_day=False,
             employee_details={"full_name": "Me", "date_of_birth": "2008-02-06"}),
    ]}}
    monkeypatch.setattr(tools, "hrms_request", lambda *a, **k: raw)
    result = tools.get_leave_history.func("tok", type="upcoming")
    assert result == {"total": 1, "leaves": [{
        "leave_type": "el-1", "start_date": "2026-10-12", "end_date": "2026-10-12",
        "days": 1.0, "half_day": False, "status": "Pending"}]}
