import json
import os
import re
import urllib.error
import urllib.request
import uuid
from datetime import date, datetime, timedelta, timezone
from functools import cache
from typing import Annotated
from urllib.parse import quote, urlencode

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, tool

from .chain import doc_label
from .config import (
    HRMS_TIMEOUT,
    MMR_FETCH_K,
    MMR_LAMBDA,
    RELEVANCE_THRESHOLD,
    TOP_K,
)
from .db import store
from .profile import profile_from


AUTH_BASE_URL = os.getenv("HRMS_AUTH_URL", "https://auth.dev2.autoscal.com")
# The tenant the login form posts to: the subdomain of the HRMS web app. The
# login prompt offers it as the default.
SUBDOMAIN = os.getenv("HRMS_SUBDOMAIN", "dev1")
ATTENDANCE_BASE_URL = os.getenv("HRMS_ATTENDANCE_URL", "https://attendence.dev2.autoscal.com")
JOB_BASE_URL = os.getenv("HRMS_JOB_URL", "https://job.dev2.autoscal.com")
# The HRMS rejects a request without its client id, alongside the bearer token.
CLIENT_ID = os.getenv("HRMS_CLIENT_ID", "02711038-26af-4022-8cad-d8e058bc8d89")
# Statuses that count as a current employee; the HRMS UI sends the same list.
ACTIVE_MEMBER_STATUSES = ["onboarded", "PreClearance", "Clearance", "InExit"]
# The regular "Work From Home" type, the only one the HRMS offers so far.
WFH_TYPE_ID = os.getenv("HRMS_WFH_TYPE_ID", "a44233a3-e99c-4410-b4a1-a19e63ddff48")
# Where save_feedback_form puts downloaded feedback forms.
FEEDBACK_DIR = os.getenv("FEEDBACK_DIR", "feedback")


def hrms_send(url: str, access_token: str | None, data: bytes | None = None,
              content_type: str | None = None, accept: str = "application/json",
              method: str | None = None) -> bytes | dict:
    """The raw response body of an HRMS call as the employee, or an error dict.
    GET, or POST when there is data, unless `method` says otherwise. No token
    sends no Authorization header."""
    headers = {"Accept": accept, "x-client-id": CLIENT_ID}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=HRMS_TIMEOUT) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", "replace").strip()
        return {"error": f"HRMS returned {exc.code}", "detail": detail}
    except (urllib.error.URLError, OSError) as exc:
        return {"error": f"Cannot reach the HRMS: {exc}"}


def hrms_request(url: str, access_token: str | None, body: dict | None = None,
                 method: str | None = None) -> dict:
    """Call a JSON endpoint as the employee: GET, or POST when there is a body,
    unless `method` says otherwise. Errors come back as a dict rather than
    raising, so the model can explain them instead of the loop crashing."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    raw = hrms_send(url, access_token, data,
                    content_type=None if body is None else "application/json",
                    method=method)
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"error": "HRMS did not return JSON"}


@tool
def get_user_details(access_token: Annotated[str, InjectedToolArg]) -> dict:
    """Get the logged-in employee's own HRMS record: name, email, role, band,
    manager, joining date and similar personal details."""
    result = hrms_request(f"{AUTH_BASE_URL}/api/users/me", access_token)
    if result.get("error"):
        return result
    # Pruned like the login profile: the raw record carries credentials, pay
    # and personal data that must never reach the model.
    return {"profile": profile_from(result)}


def is_test_holiday(name: str) -> bool:
    """Entries HR made to try the HRMS out: "Test Floater 1", "shubhamtest"."""
    return "test" in name.lower()


def holiday_list(data, today: date, include_past: bool = False) -> list[dict]:
    """Active, real holidays from an /api/holidays response, by date: only
    those from today on unless `include_past`."""
    holidays = []
    for holiday in data.get("data") or []:
        name = (holiday.get("name") or "").strip()
        try:
            when = date.fromisoformat(holiday.get("date") or "")
        except ValueError:
            continue
        if (not holiday.get("is_active", True) or is_test_holiday(name)
                or (when < today and not include_past)):
            continue
        holidays.append({"name": name, "date": when.isoformat(),
                         "weekday": f"{when:%A}", "optional": holiday.get("is_optional")})
    return sorted(holidays, key=lambda h: h["date"])


@tool(parse_docstring=True)
def get_holidays(
    access_token: Annotated[str, InjectedToolArg],
    include_past: bool = False,
) -> dict:
    """Get the company holidays from today on, by date, with whether each is
    optional (restricted). List them as they come back; past ones are left out.

    Args:
        include_past: Also return holidays already gone, only when the employee
            asks about past holidays or a whole year.
    """
    data = hrms_request(f"{ATTENDANCE_BASE_URL}/api/holidays", access_token)
    if data.get("error"):
        return data
    today = date.today()
    return {"today": today.isoformat(),
            "holidays": holiday_list(data, today, include_past)}


@tool(parse_docstring=True)
def get_leave_history(
    access_token: Annotated[str, InjectedToolArg],
    type: str = "past",
    status: str = "",
    page: int = 1,
    page_size: int = 5,
) -> dict:
    """Get the employee's own leave applications.

    Args:
        type: "past" for leave already taken, "upcoming" for leave still ahead.
        status: Only applications with this status. Empty for every status.
        page: Page number, starting at 1.
        page_size: Applications per page.
    """
    query = urlencode({"name": "", "status": status, "type": type,
                       "page": page, "page_size": page_size})
    return hrms_request(f"{ATTENDANCE_BASE_URL}/api/self/leave-applications/?{query}",
                        access_token)


@tool(parse_docstring=True)
def get_team_members(
    access_token: Annotated[str, InjectedToolArg],
    name: str = "",
) -> dict:
    """List colleagues with their id, name and official email. Use it to find
    the ids of the people to notify about a leave application.

    Args:
        name: Only colleagues whose name contains this text. Empty for everyone.
    """
    body = {
        "page": 1,
        "limit": 9007199254740991,  # what the HRMS UI sends for "all"
        # Only what identifies a person: the whole team's full records would
        # not fit in the model's context.
        "select": {"id": True, "full_name": True, "official_email": True},
        "status": ACTIVE_MEMBER_STATUSES,
    }
    result = hrms_request(
        f"{JOB_BASE_URL}/api/team-members/all?sortBy=full_name&sortOrder=asc",
        access_token, body,
    )
    if name and isinstance(result.get("data"), list):
        wanted = name.lower()
        result["data"] = [m for m in result["data"]
                          if wanted in (m.get("full_name") or "").lower()]
    return result


def dashboard_part(data, *keys):
    """`data["data"][keys...]` of a leave-dashboard response, or None."""
    node = data.get("data") if isinstance(data, dict) else None
    for key in keys:
        node = node.get(key) if isinstance(node, dict) else None
    return node


def prior_notice(rules) -> list[str]:
    """A type's prior-notice rules as sentences the model can quote."""
    return [f"{r['calendar_days_notice']} calendar days' notice for "
            f"{r['min_leave_days_required']} or more days of leave"
            for r in rules or []
            if r.get("calendar_days_notice") is not None]


def leave_types(data) -> list[dict]:
    """The leave types in a leave-dashboard response, each as {id, name, ...}
    with its balance and the rules that decide whether an application goes
    through. The rest of each entry is scheme configuration, and its approval
    chain holds other employees' records: neither reaches the model."""
    types = []
    for entry in dashboard_part(data, "leaves", "leave_data") or []:
        kind = entry.get("leave_type") or {}
        if not (kind.get("leave_type_id") and kind.get("name")):
            continue
        config = entry.get("configuration") or {}
        applying = config.get("applying_leave") or {}
        approval = config.get("leave_approval") or {}
        types.append({
            "id": str(kind["leave_type_id"]),
            "name": kind["name"],
            "balance": entry.get("balance"),
            "used": entry.get("used"),
            "total_allocated": entry.get("total_allocated"),
            "unit": entry.get("quota_unit", "days"),
            "is_paid": kind.get("is_paid"),
            "is_sick_leave": kind.get("is_sick_leave"),
            "half_day_allowed": bool(applying.get("allow_half_day")),
            "needs_approval": approval.get("requires_approval", True),
            "prior_notice": prior_notice(applying.get("prior_notice_rules")),
        })
    return types


def upcoming_leaves(data) -> list[dict]:
    """The employee's booked leave still ahead, from a leave-dashboard response."""
    return [{
        "id": leave.get("leave_log_id"),
        "leave_type": leave.get("leave_type_name"),
        "start_date": leave.get("start_date"),
        "end_date": leave.get("end_date"),
        "days": leave.get("duration_days"),
        "approved": leave.get("is_approved"),
    } for leave in dashboard_part(data, "leaves", "upcoming_leaves") or []]


def next_holiday(data) -> dict | None:
    """The next company holiday, from a leave-dashboard response."""
    holiday = dashboard_part(data, "upcoming_holidays")
    if not isinstance(holiday, dict) or not holiday.get("date"):
        return None
    return {"name": holiday.get("name"), "date": holiday.get("date"),
            "optional": holiday.get("is_optional")}


def without_id(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k != "id"}


def weekend_only(start_date: str, end_date: str) -> dict | None:
    """An error dict when every day from start to end is a Saturday or Sunday:
    those are weekly offs, so there is nothing to apply for. None otherwise,
    including for dates the HRMS should reject itself."""
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except ValueError:
        return None
    days = [start + timedelta(days=n) for n in range((end - start).days + 1)]
    if days and all(d.weekday() >= 5 for d in days):
        when = ", ".join(f"{d.isoformat()} ({d:%A})" for d in days)
        return {"error": f"{when} is a weekly off, not a working day, so no leave "
                         "or WFH is needed. Tell the employee; do not retry."}
    return None


def past_month(start_date: str, today: date | None = None) -> dict | None:
    """An error dict when the leave or WFH starts before the current month:
    backdating is only allowed within this month. None otherwise, including
    for dates the HRMS should reject itself."""
    try:
        start = date.fromisoformat(start_date)
    except ValueError:
        return None
    first = (today or date.today()).replace(day=1)
    if start < first:
        return {"error": f"{start.isoformat()} is in a past month. Leave and WFH can "
                         f"only be backdated within the current month (from "
                         f"{first.isoformat()}). Tell the employee; do not retry."}
    return None


def date_problem(start_date: str, end_date: str) -> dict | None:
    """The first rule these dates break, as an error dict, or None."""
    return weekend_only(start_date, end_date) or past_month(start_date)


def match_leave_type(types: list[dict], wanted: str) -> dict | None:
    """The one type whose name is `wanted`, or the only one containing it:
    "casual" finds "Casual Leave". None when nothing or several match."""
    wanted = wanted.strip().lower()
    exact = [t for t in types if t["name"].lower() == wanted]
    if exact:
        return exact[0]
    partial = [t for t in types if wanted in t["name"].lower()]
    return partial[0] if len(partial) == 1 else None


def fetch_dashboard(access_token: str) -> dict:
    """The raw leave dashboard: leave types, upcoming leave, next holiday."""
    return hrms_request(f"{ATTENDANCE_BASE_URL}/api/leaves/dashboard", access_token)


@tool
def get_leave_balance(access_token: Annotated[str, InjectedToolArg]) -> dict:
    """Get the leave types the employee can apply for with the balance of each,
    their leave already booked ahead, and the next company holiday. Use the
    type names it returns as apply_leave's leave_type."""
    data = fetch_dashboard(access_token)
    if data.get("error"):
        return data
    # The ids stay here: apply_leave and cancel_leave look them up themselves.
    return {
        "leave_types": [without_id(t) for t in leave_types(data)],
        "upcoming_leaves": [without_id(l) for l in upcoming_leaves(data)],
        "next_holiday": next_holiday(data),
    }


@tool(parse_docstring=True)
def apply_leave(
    access_token: Annotated[str, InjectedToolArg],
    leave_type: str,
    start_date: str,
    end_date: str,
    is_half_day: bool = False,
    people_to_notify: list[str] | None = None,
) -> dict:
    """Submit a leave application for the employee once they have asked to
    apply. The employee is shown the details and confirms before it is sent.
    Leave is for working days only, never a Saturday or Sunday, and can only be
    backdated within the current month.

    Args:
        leave_type: Name of the leave type, such as "Casual Leave", as
            get_leave_balance lists it.
        start_date: First day of leave, YYYY-MM-DD.
        end_date: Last day of leave, YYYY-MM-DD. Same as start_date for one day.
        is_half_day: True for a half-day leave on a single date.
        people_to_notify: Ids of colleagues to notify, from get_team_members.
            Omit to notify nobody.
    """
    problem = date_problem(start_date, end_date)
    if problem:
        return problem
    data = fetch_dashboard(access_token)
    if data.get("error"):
        return data
    types = leave_types(data)
    match = match_leave_type(types, leave_type)
    if match is None:
        return {"error": f"No single leave type matches {leave_type!r}. "
                         "Ask the employee which one they mean.",
                "leave_types": [t["name"] for t in types]}
    # Caught here rather than by the HRMS, whose rejection says less.
    if is_half_day and not match["half_day_allowed"]:
        return {"error": f"{match['name']} cannot be taken as a half day. "
                         "Ask the employee whether to apply for the full day."}
    body = {
        "leave_type_id": match["id"],
        "start_date": start_date,
        "end_date": end_date,
        "is_half_day": is_half_day,
        "people_to_notify": people_to_notify or [],
        "attachments": [],
    }
    return hrms_request(f"{ATTENDANCE_BASE_URL}/api/leave-applications", access_token, body)


@tool(parse_docstring=True)
def cancel_leave(
    access_token: Annotated[str, InjectedToolArg],
    start_date: str,
    leave_type: str = "",
) -> dict:
    """Cancel one of the employee's upcoming leave applications. Only call this
    once they have clearly asked to cancel it.

    Args:
        start_date: First day of the leave to cancel, YYYY-MM-DD.
        leave_type: Name of its leave type. Only needed when two leaves start
            on the same day.
    """
    data = fetch_dashboard(access_token)
    if data.get("error"):
        return data
    upcoming = upcoming_leaves(data)
    found = [l for l in upcoming if l["start_date"] == start_date
             and (not leave_type
                  or leave_type.strip().lower() in (l["leave_type"] or "").lower())]
    if len(found) != 1:
        problem = "No upcoming leave" if not found else "More than one upcoming leave"
        return {"error": f"{problem} starts on {start_date}"
                         + (f" with type {leave_type!r}" if leave_type else "")
                         + ". Ask the employee which one they mean.",
                "upcoming_leaves": [without_id(l) for l in upcoming]}
    result = hrms_request(
        f"{ATTENDANCE_BASE_URL}/api/leave-applications/{found[0]['id']}/cancel",
        access_token, {}, method="PUT",
    )
    if result.get("error"):
        return result
    # Only the leave itself: the response also carries the employee's record.
    leave = result.get("data") or {}
    return {
        "status": leave.get("status"),
        "leave_type": (leave.get("leave_type") or {}).get("name"),
        "start_date": leave.get("start_date"),
        "end_date": leave.get("end_date"),
        "days": leave.get("duration_days"),
    }


@tool
def get_wfh_balance(access_token: Annotated[str, InjectedToolArg]) -> dict:
    """Get how many work-from-home days the employee has left."""
    return hrms_request(f"{ATTENDANCE_BASE_URL}/api/wfh/balance", access_token)


@tool(parse_docstring=True)
def get_wfh_history(
    access_token: Annotated[str, InjectedToolArg],
    type: str = "upcoming",
    status: list[str] | None = None,
) -> dict:
    """Get the employee's own work-from-home requests.

    Args:
        type: "upcoming" for WFH days still ahead, "past" for ones already done.
        status: Only requests with these statuses, such as APPROVED or PENDING.
            Omit for every status.
    """
    query = urlencode({"status": status or [], "type": type}, doseq=True)
    return hrms_request(f"{ATTENDANCE_BASE_URL}/api/wfh/request_list/?{query}", access_token)


@tool(parse_docstring=True)
def apply_wfh(
    access_token: Annotated[str, InjectedToolArg],
    start_date: str,
    end_date: str,
    is_half_day: bool = False,
) -> dict:
    """Submit a work-from-home request for the employee once they have asked
    to apply. The employee is shown the details and confirms before it is sent.
    WFH is for working days only, never a Saturday or Sunday, and can only be
    backdated within the current month.

    Args:
        start_date: First WFH day, YYYY-MM-DD.
        end_date: Last WFH day, YYYY-MM-DD. Same as start_date for one day.
        is_half_day: True for a half-day WFH on a single date.
    """
    problem = date_problem(start_date, end_date)
    if problem:
        return problem
    body = {
        "wfh_type_id": WFH_TYPE_ID,
        "start_date": start_date,
        "end_date": end_date,
        "is_half_day": is_half_day,
        "attachments": [],
    }
    return hrms_request(f"{ATTENDANCE_BASE_URL}/api/wfh/requests/", access_token, body)


def local_time(stamp: str | None) -> str | None:
    """An HRMS UTC timestamp in local time: an expiry of 18:30Z is midnight
    in India, the next day there."""
    if not stamp:
        return None
    try:
        moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return stamp
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


def feedback_request_summary(entry: dict, team_member_id: str) -> dict:
    """One feedback request as the model needs it. The rest of the entry is
    the form twice over and every participant's record."""
    expiry = entry.get("expiry_date")
    expired = False
    if expiry:
        try:
            expired = datetime.fromisoformat(expiry.replace("Z", "+00:00")) \
                < datetime.now(timezone.utc)
        except ValueError:
            pass
    return {
        "id": entry.get("id"),
        "project": (entry.get("project") or {}).get("name"),
        "form": (entry.get("feedback_template") or {}).get("name"),
        "category": entry.get("category"),
        "status": entry.get("status"),
        "feedback_for": (entry.get("feedback_for_date") or "")[:10] or None,
        "due": local_time(expiry),
        "expired": expired,
        # The employee is among the receivers too, but not on their own form.
        "colleagues": sum(1 for r in entry.get("receivers") or []
                          if r.get("team_member_id") != team_member_id),
        "submissions": f"{entry.get('total_submission_count', 0)} of "
                       f"{entry.get('total_request', 0)}",
    }


def fetch_feedback_requests(access_token: str) -> list[dict] | dict:
    """The feedback requests the employee has been asked to fill, summarised,
    or the HRMS error dict."""
    me = hrms_request(f"{AUTH_BASE_URL}/api/users/me", access_token)
    if me.get("error"):
        return me
    team_member_id = (me.get("data") or {}).get("team_member_id")
    if not team_member_id:
        return {"error": "The employee's HRMS record has no team member id."}
    query = urlencode({"action": "receivedRequests", "team_member_id": team_member_id})
    result = hrms_request(f"{JOB_BASE_URL}/api/team-feedback/requests/?{query}",
                          access_token)
    if result.get("error"):
        return result
    return [feedback_request_summary(r, team_member_id) for r in result.get("data") or []]


def find_feedback_request(access_token: str, wanted: str) -> dict:
    """The one request `wanted` names - by project, form name or id - or the
    only one there is when `wanted` is empty. Otherwise an error dict that
    lists them, so the model can ask which one.

    The model names a request rather than passing its id: chat memory keeps
    only the answers, so an id from an earlier turn is gone, and a 4B model
    asked for "this request" made one up."""
    requests = fetch_feedback_requests(access_token)
    if isinstance(requests, dict):
        return requests
    if not requests:
        return {"error": "The employee has no feedback requests to fill."}
    wanted = wanted.strip().lower()
    found = [r for r in requests if not wanted or wanted == r["id"]
             or wanted in (r["project"] or "").lower()
             or wanted in (r["form"] or "").lower()]
    if len(found) == 1:
        return found[0]
    problem = "No feedback request matches" if not found else "More than one feedback request matches"
    return {"error": f"{problem} {wanted!r}. Ask the employee which one they mean."
                     if wanted else "The employee has several feedback requests. "
                                    "Ask which one they mean.",
            "requests": [without_id(r) for r in requests]}


@tool
def get_feedback_requests(access_token: Annotated[str, InjectedToolArg]) -> dict:
    """List the team-feedback requests the employee has been asked to fill,
    with their project, form name and due date."""
    requests = fetch_feedback_requests(access_token)
    if isinstance(requests, dict):
        return requests
    # The ids stay here: the feedback tools look them up by name themselves.
    return {"requests": [without_id(r) for r in requests]}


def feedback_url(request_id: str, path: str) -> str:
    # Quoted: the id comes from the model and ends up in the path.
    return f"{JOB_BASE_URL}/api/team-feedback/requests/{quote(request_id.strip(), safe='')}/{path}"


def download_feedback_form(access_token: str, request_id: str) -> bytes | dict:
    """The blank CSV form of a feedback request: one column per colleague,
    one row per score or remark."""
    return hrms_send(feedback_url(request_id, "export/?action=form&format=csv"),
                     access_token, accept="text/csv")


def parse_feedback_form(access_token: str, request_id: str, csv_bytes: bytes) -> dict:
    """The HRMS's reading of a feedback CSV: the form template, and per
    colleague the form with each filled cell as a field's `value`. The HRMS
    does not check the scores' range here."""
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="feedback-{request_id}.csv"\r\n'
            "Content-Type: text/csv\r\n\r\n").encode() \
        + csv_bytes + f"\r\n--{boundary}--\r\n".encode()
    raw = hrms_send(feedback_url(request_id, "parse-excel/"), access_token, body,
                    content_type=f"multipart/form-data; boundary={boundary}")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {"error": "HRMS did not return JSON"}


def by_order(node: dict) -> list[tuple[str, dict]]:
    """A form node's children, in the order the form shows them."""
    return sorted(((k, v) for k, v in node.items() if isinstance(v, dict)),
                  key=lambda kv: kv[1].get("order", 0))


def feedback_outline(form_fields: dict) -> list[dict]:
    """The form's sections in order, each with the aspects scored in it. A
    top-level field such as Kudos is a section of its own with no aspects."""
    sections = []
    for key, node in by_order(form_fields):
        if node.get("isField"):
            sections.append({"name": node.get("label", key), "free_text": True})
            continue
        aspects, section_remarks = [], False
        for child_key, child in by_order(node.get("child") or {}):
            if child.get("isField"):  # the one remark for the whole section
                section_remarks = True
                continue
            score = (child.get("child") or {}).get("score") or {}
            aspects.append({
                "name": child.get("label", child_key),
                "description": child.get("description"),
                "score_range": [score.get("minValue"), score.get("maxValue")],
                # The scores the form asks a remark for.
                "remarks_for_scores": score.get("remarksCheckpoint") or [],
            })
        sections.append({"name": node.get("label", key), "aspects": aspects,
                         "section_remarks": section_remarks})
    return sections


def has_value(node) -> bool:
    """Whether any field under a form node has an answer filled in."""
    if not isinstance(node, dict):
        return False
    if node.get("value") not in (None, ""):
        return True
    return any(has_value(v) for v in node.values())


def feedback_colleagues(receivers) -> list[dict]:
    """Who the form is about, by name and email. The rest of each entry is the
    whole form again and profile-picture links."""
    colleagues = []
    for entry in receivers or []:
        person = entry.get("receiver") or {}
        colleague = {
            "name": " ".join(filter(None, [person.get("firstName"), person.get("lastName")])),
            "email": person.get("email"),
        }
        if entry.get("isEditable") is False:
            colleague["editable"] = False
        if has_value(entry.get("formFields")):
            colleague["has_answers"] = True
        colleagues.append(colleague)
    return colleagues


@tool(parse_docstring=True)
def get_feedback_form(
    access_token: Annotated[str, InjectedToolArg],
    request: str = "",
) -> dict:
    """Get a team-feedback form the employee has to fill: the colleagues they
    give feedback to, and the sections and aspects each colleague is scored on.

    Args:
        request: Project or form name of the feedback request, as
            get_feedback_requests lists it. Empty when the employee has only one.
    """
    found = find_feedback_request(access_token, request)
    if found.get("error"):
        return found
    form = download_feedback_form(access_token, found["id"])
    if isinstance(form, dict):
        return form
    parsed = parse_feedback_form(access_token, found["id"], form)
    if parsed.get("error"):
        return parsed
    data = parsed.get("data") or {}
    template = (data.get("feedbackTemplate") or {}).get("form_fields") or {}
    result: dict = {
        "request": without_id(found),
        "colleagues": feedback_colleagues(data.get("receivers")),
        "sections": feedback_outline(template),
    }
    # unmatchedEmails / missingFields: only worth showing when there are some.
    problems = {k: v for k, v in (data.get("errors") or {}).items() if v}
    if problems:
        result["problems"] = problems
    return result


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "feedback"


@tool(parse_docstring=True)
def save_feedback_form(
    access_token: Annotated[str, InjectedToolArg],
    request: str = "",
) -> dict:
    """Download a team-feedback form as a CSV file the employee can fill in,
    and say where it was saved.

    Args:
        request: Project or form name of the feedback request, as
            get_feedback_requests lists it. Empty when the employee has only one.
    """
    found = find_feedback_request(access_token, request)
    if found.get("error"):
        return found
    form = download_feedback_form(access_token, found["id"])
    if isinstance(form, dict):
        return form
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    # Stamped, so a second download never overwrites a form being filled in.
    name = f"{slug(found['form'] or '')}_{slug(found['project'] or '')}_" \
           f"{datetime.now():%Y%m%d_%H%M%S}.csv"
    path = os.path.join(FEEDBACK_DIR, name)
    with open(path, "wb") as fh:
        fh.write(form)
    return {"saved_to": os.path.abspath(path), "request": without_id(found),
            "fill_with": f"python -m app.rag feedback {path}"}


def feedback_files() -> list[str]:
    """The CSV files in the feedback folder, newest first."""
    if not os.path.isdir(FEEDBACK_DIR):
        return []
    names = [n for n in os.listdir(FEEDBACK_DIR) if n.lower().endswith(".csv")]
    return sorted(names, key=lambda n: os.path.getmtime(os.path.join(FEEDBACK_DIR, n)),
                  reverse=True)


def find_feedback_file(wanted: str) -> str | dict:
    """The path of the one feedback CSV named `wanted`, or the only one whose
    name contains it. Only the base name counts, so the model cannot point the
    upload at a file outside the feedback folder."""
    files = feedback_files()
    if not files:
        return {"error": f"No feedback CSV files in {os.path.abspath(FEEDBACK_DIR)}."}
    name = os.path.basename(wanted.strip())
    found = [n for n in files if n == name] \
        or [n for n in files if name and name.lower() in n.lower()]
    if len(found) == 1:
        return os.path.join(FEEDBACK_DIR, found[0])
    problem = "No feedback file matches" if not found else "More than one feedback file matches"
    return {"error": f"{problem} {wanted!r}. Ask the employee to pick one of these.",
            "files": files}


@tool
def list_feedback_files() -> dict:
    """List the feedback CSV files in the feedback folder, newest first, so the
    employee can pick the one to upload. Filled-in forms have "_filled_" in
    their name; the others are blank downloads."""
    files = feedback_files()
    if not files:
        return {"error": f"No feedback CSV files in {os.path.abspath(FEEDBACK_DIR)}."}
    return {"folder": os.path.abspath(FEEDBACK_DIR),
            "files": [{"file": n, "filled": "_filled_" in n} for n in files]}


@tool(parse_docstring=True)
def submit_feedback_form(
    access_token: Annotated[str, InjectedToolArg],
    file: str,
    request: str = "",
) -> dict:
    """Upload a filled-in feedback CSV from the feedback folder to the HRMS.

    Args:
        file: The CSV's file name exactly as list_feedback_files returns it.
        request: Project or form name of the feedback request, as
            get_feedback_requests lists it. Empty when the employee has only one.
    """
    path = find_feedback_file(file)
    if isinstance(path, dict):
        return path
    found = find_feedback_request(access_token, request)
    if found.get("error"):
        return found
    with open(path, "rb") as fh:
        csv_bytes = fh.read()
    parsed = parse_feedback_form(access_token, found["id"], csv_bytes)
    if parsed.get("error"):
        return parsed
    data = parsed.get("data") or {}
    colleagues = feedback_colleagues(data.get("receivers"))
    result: dict = {
        "uploaded": os.path.basename(path),
        "request": without_id(found),
        "colleagues_with_answers": [c["name"] for c in colleagues if c.get("has_answers")],
    }
    problems = {k: v for k, v in (data.get("errors") or {}).items() if v}
    if problems:
        result["problems"] = problems
    return result


@cache
def policy_store():
    """One PGVector connection for the session, opened on first use."""
    return store()


def find_policy(query: str, k: int = TOP_K) -> list[dict]:
    """The policy chunks for a query, each with its source; empty when nothing
    in the index is about it."""
    vectors = policy_store()
    # MMR always returns k chunks whether they match or not, so check the best
    # match first: below the threshold, nothing in the index is about this.
    top = vectors.similarity_search_with_relevance_scores(query, k=1)
    if not top or top[0][1] < RELEVANCE_THRESHOLD:
        return []
    docs = vectors.max_marginal_relevance_search(
        query, k=k, fetch_k=max(MMR_FETCH_K, k * 5), lambda_mult=MMR_LAMBDA,
    )
    return [{"source": doc_label(d), "text": d.page_content} for d in docs]


@tool(parse_docstring=True)
def search_policy(query: str) -> dict:
    """Search the company policy documents: leave rules, WFH policy,
    reimbursement, appraisal, conduct and other HR policy. Use it for any
    question about what the policy says, and quote what comes back.

    Args:
        query: What to look up, as a standalone question or phrase.
    """
    results = find_policy(query)
    if not results:
        return {"results": [], "note": "No policy document covers this."}
    return {"results": results}


def password_login(email: str, password: str, subdomain: str = SUBDOMAIN) -> str | dict:
    """The access token for these credentials, or the HRMS error dict."""
    result = hrms_request(f"{AUTH_BASE_URL}/api/users/login", None, {
        "email": email, "password": password,
        "remember_me": True, "subdomain": subdomain,
    })
    token = (result.get("data") or {}).get("token") if isinstance(result, dict) else None
    if token:
        return token
    return result if result.get("error") else {"error": "The HRMS returned no token",
                                               "detail": result.get("message", "")}


@tool
def login(runtime: ToolRuntime) -> str:
    """Log the employee in to the HRMS with their email, password and company
    subdomain, which they type in themselves when asked to confirm. Call this when they ask to
    log in, or before any HRMS tool when they are not logged in yet."""
    # Typed in at the confirmation (agent.decide), kept in the run context
    # rather than a message, and dropped as soon as they are used.
    credentials, runtime.context.credentials = runtime.context.credentials, None
    if not credentials:
        return "Login cancelled: no email or password was entered."
    token = password_login(*credentials)
    if isinstance(token, dict):
        return f"Login failed: {token['error']}. {token.get('detail', '')}".strip()
    # The session lives in the run context, never in a message: the model does
    # not see the token, and every later tool call in the chat picks it up.
    runtime.context.access_token = token
    details = hrms_request(f"{AUTH_BASE_URL}/api/users/me", token)
    if details.get("error"):
        return "Logged in, but the HRMS record could not be read."
    runtime.context.profile = profile_from(details)
    return "Logged in. The employee is:\n" + runtime.context.profile


TOOLS = [
    login,
    search_policy,
    get_user_details,
    get_holidays,
    get_leave_history,
    get_team_members,
    get_leave_balance,
    apply_leave,
    cancel_leave,
    get_wfh_balance,
    get_wfh_history,
    apply_wfh,
    get_feedback_requests,
    get_feedback_form,
    save_feedback_form,
    list_feedback_files,
    submit_feedback_form,
]
