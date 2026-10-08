import json
import os
import re
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


# Fields of a leave-dashboard entry worth showing the model: the balance and
# what makes it up. The rest is scheme configuration (see PROFILE_DROP).
BALANCE_RE = re.compile(r"balance|available|allocated|used|taken|remaining|booked|pending", re.I)


def leave_types(data) -> list[dict]:
    """The leave types in a leave-dashboard response, each as {id, name, ...}.

    An entry is a dict naming its type, either directly (`leave_type_id` and a
    name) or through a nested `leave_type` object. A matched entry is not walked
    into: its approval chains carry other employees' {id, name} records.
    """
    found: dict[str, dict] = {}

    def walk(node) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        nested = node.get("leave_type") if isinstance(node.get("leave_type"), dict) else {}
        type_id = node.get("leave_type_id") or nested.get("id")
        name = nested.get("name") or node.get("leave_type_name") or node.get("name")
        if type_id and name:
            entry = {"id": str(type_id), "name": str(name)}
            entry.update({k: v for k, v in node.items()
                          if BALANCE_RE.search(k) and not isinstance(v, (dict, list))})
            found.setdefault(entry["id"], entry)
            return
        for value in node.values():
            walk(value)

    walk(data)
    return list(found.values())


def match_leave_type(types: list[dict], wanted: str) -> dict | None:
    """The one type whose name is `wanted`, or the only one containing it:
    "casual" finds "Casual Leave". None when nothing or several match."""
    wanted = wanted.strip().lower()
    exact = [t for t in types if t["name"].lower() == wanted]
    if exact:
        return exact[0]
    partial = [t for t in types if wanted in t["name"].lower()]
    return partial[0] if len(partial) == 1 else None


def fetch_leave_types(access_token: str) -> list[dict] | dict:
    """The employee's leave types, or the HRMS error dict."""
    data = hrms_request(f"{ATTENDANCE_BASE_URL}/api/leaves/dashboard", access_token)
    if isinstance(data, dict) and data.get("error"):
        return data
    return leave_types(data)


@tool
def get_leave_balance(access_token: Annotated[str, InjectedToolArg]) -> dict:
    """Get the leave types the employee can apply for and the balance of each.
    Use the names it returns as apply_leave's leave_type."""
    types = fetch_leave_types(access_token)
    if isinstance(types, dict):
        return types
    # The ids stay here: apply_leave looks them up by name itself.
    return {"leave_types": [{k: v for k, v in t.items() if k != "id"} for t in types]}


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
    types = fetch_leave_types(access_token)
    if isinstance(types, dict):
        return types
    match = match_leave_type(types, leave_type)
    if match is None:
        return {"error": f"No single leave type matches {leave_type!r}. "
                         "Ask the employee which one they mean.",
                "leave_types": [t["name"] for t in types]}
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
