import json
import os
import urllib.error
import urllib.request
from functools import cache
from typing import Annotated
from urllib.parse import urlencode

from langchain.tools import ToolRuntime
from langchain_core.tools import InjectedToolArg, tool

from .chain import doc_label
from .config import (
    HRMS_TIMEOUT,
    MMR_FETCH_K,
    MMR_LAMBDA,
    RELEVANCE_THRESHOLD,
    TOKEN_LOGIN,
    TOP_K,
)
from .db import store
from .profile import profile_from, resolve_token


AUTH_BASE_URL = os.getenv("HRMS_AUTH_URL", "https://auth.dev2.autoscal.com")
ATTENDANCE_BASE_URL = os.getenv("HRMS_ATTENDANCE_URL", "https://attendence.dev2.autoscal.com")
JOB_BASE_URL = os.getenv("HRMS_JOB_URL", "https://job.dev2.autoscal.com")
# The HRMS rejects a request without its client id, alongside the bearer token.
CLIENT_ID = os.getenv("HRMS_CLIENT_ID", "02711038-26af-4022-8cad-d8e058bc8d89")
# Statuses that count as a current employee; the HRMS UI sends the same list.
ACTIVE_MEMBER_STATUSES = ["onboarded", "PreClearance", "Clearance", "InExit"]
# The regular "Work From Home" type, the only one the HRMS offers so far.
WFH_TYPE_ID = os.getenv("HRMS_WFH_TYPE_ID", "a44233a3-e99c-4410-b4a1-a19e63ddff48")


def hrms_request(url: str, access_token: str, body: dict | None = None) -> dict:
    """Call a JSON endpoint as the employee: GET, or POST when there is a body.
    Errors come back as a dict rather than raising, so the model can explain
    them instead of the loop crashing."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "x-client-id": CLIENT_ID,
    }
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=HRMS_TIMEOUT) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", "replace").strip()
        return {"error": f"HRMS returned {exc.code}", "detail": detail}
    except (urllib.error.URLError, OSError) as exc:
        return {"error": f"Cannot reach the HRMS: {exc}"}
    except json.JSONDecodeError:
        return {"error": "HRMS did not return JSON"}


@tool
def get_user_details(access_token: Annotated[str, InjectedToolArg]) -> dict:
    """Get the logged-in employee's own HRMS record: name, email, role, band,
    manager, joining date and similar personal details."""
    return hrms_request(f"{AUTH_BASE_URL}/api/users/me", access_token)


@tool
def get_holidays(access_token: Annotated[str, InjectedToolArg]) -> dict:
    """Get the company holiday list, including restricted holidays and their
    dates."""
    return hrms_request(f"{ATTENDANCE_BASE_URL}/api/holidays", access_token)


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
    # The ids stay here: apply_leave looks them up by name itself.
    return {
        "leave_types": [{k: v for k, v in t.items() if k != "id"}
                        for t in leave_types(data)],
        "upcoming_leaves": upcoming_leaves(data),
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
    """Submit a leave application for the employee. Only call this once they
    have clearly asked to apply and confirmed the details.

    Args:
        leave_type: Name of the leave type, such as "Casual Leave", as
            get_leave_balance lists it.
        start_date: First day of leave, YYYY-MM-DD.
        end_date: Last day of leave, YYYY-MM-DD. Same as start_date for one day.
        is_half_day: True for a half-day leave on a single date.
        people_to_notify: Ids of colleagues to notify, from get_team_members.
            Omit to notify nobody.
    """
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
    """Submit a work-from-home request for the employee. Only call this once
    they have clearly asked to apply and confirmed the details.

    Args:
        start_date: First WFH day, YYYY-MM-DD.
        end_date: Last WFH day, YYYY-MM-DD. Same as start_date for one day.
        is_half_day: True for a half-day WFH on a single date.
    """
    body = {
        "wfh_type_id": WFH_TYPE_ID,
        "start_date": start_date,
        "end_date": end_date,
        "is_half_day": is_half_day,
        "attachments": [],
    }
    return hrms_request(f"{ATTENDANCE_BASE_URL}/api/wfh/requests/", access_token, body)


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


@tool
def login(runtime: ToolRuntime) -> str:
    """Log the employee in to the HRMS by opening its login page in their
    browser. Call this when they ask to log in, or before any HRMS tool when
    they are not logged in yet."""
    try:
        token = resolve_token(TOKEN_LOGIN)
    except SystemExit as exc:  # the login flow exits on failure; a tool must not
        return f"Login failed: {exc}"
    # The session lives in the run context, never in a message: the model does
    # not see the token, and every later tool call in the chat picks it up.
    details = hrms_request(f"{AUTH_BASE_URL}/api/users/me", token)
    runtime.context.access_token = token
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
    get_wfh_balance,
    get_wfh_history,
    apply_wfh,
]
