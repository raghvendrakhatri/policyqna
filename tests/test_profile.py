"""What the HRMS sends, and what is allowed to reach the prompt.

Every case here is a bug that actually happened against the live HRMS. They run
without Ollama, Postgres or a network, so they are cheap to keep green.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import rag  # noqa: E402


@pytest.fixture(autouse=True)
def sources(monkeypatch):
    """The labels the merged profile is built from, as .env configures them."""
    monkeypatch.setenv("HRMS_PROFILE_SOURCES", "api/users/me,wfh=x,leave=y,position=z")
    monkeypatch.delenv("HRMS_PROFILE_FIELDS", raising=False)


# ------------------------------------------------------------------- secrets


def test_key_material_never_reaches_the_prompt():
    """The tenant block of /api/users/me carries an RSA private key."""
    profile = rag.profile_from({"data": {"full_name": "A", "tenant": {
        "organization_name": "Spark Eighteen",
        "calendar_api_key": "305c300d06092a864886f70d0101010500034b00",
        "calendar_api_secret": "30820155020100300d06092a864886f70d010101",
    }}})
    assert "305c300d" not in profile
    assert "30820155" not in profile
    assert "Spark Eighteen" in profile


@pytest.mark.parametrize("field", [
    "access_token", "refresh_token", "google_refresh_token", "jiraAccessToken",
    "password", "api_key", "client_secret", "bank_account_number",
])
def test_credential_shaped_fields_are_dropped(field):
    profile = rag.profile_from({"full_name": "A", field: "SENTINEL-VALUE"})
    assert "SENTINEL-VALUE" not in profile


# --------------------------------------------------------- other people's data


def test_a_colleagues_record_collapses_to_their_name():
    """reporting_to arrives as the manager's entire HR record."""
    profile = rag.profile_from({"full_name": "A", "reporting_to": {
        "full_name": "Aayush Sharma", "mobile_no": "+917665352012",
        "date_of_birth": "1994-08-18", "marital_status": "Married",
        "blood_group": "B+", "current_ctc": 900000,
    }})
    assert "Aayush Sharma" in profile
    for private in ("917665352012", "1994-08-18", "Married", "B+", "900000"):
        assert private not in profile


def test_leave_approval_chains_do_not_leak_approvers():
    """Every leave type embeds its approvers' full records."""
    profile = rag.profile_from({"leave": {"leaves": {"leave_data": [{
        "leave_type": {"name": "Casual Leave"}, "balance": 2.5,
        "configuration": {"leave_approval": {"levels": [{"approval_assignees": [
            {"employee": {"full_name": "Manju Lakshmi", "date_of_birth": "1984-02-02",
                          "gender": "Female", "marital_status": "Divorced"}}]}]}},
    }]}}})
    for private in ("Manju", "1984-02-02", "Female", "Divorced"):
        assert private not in profile
    assert "2.5" in profile


def test_the_employees_own_private_data_is_dropped():
    profile = rag.profile_from({"full_name": "A", "date_of_birth": "2002-02-03",
                                "gender": "Male", "mobile_no": "+917987895418",
                                "address": [{"address_line": "106 Everest ashiyan"}]})
    for private in ("2002-02-03", "Male", "917987895418", "Everest"):
        assert private not in profile


# ------------------------------------------------------------------ collapsing


def test_a_lookup_collapses_to_its_name():
    profile = rag.profile_from({"department": {"id": "a3521c21", "name": "Technology"}})
    assert "- department: Technology" in profile


def test_a_record_with_figures_keeps_them():
    """A leave type is about its numbers; collapsing it to "Sick" loses the answer."""
    profile = rag.profile_from({"balances": {
        "sick": {"name": "Sick Leave", "total_allocated": 3.0, "balance": 2.0}}})
    assert "2.0" in profile and "3.0" in profile


def test_a_merged_sources_root_is_never_collapsed():
    """The position source's root has full_name, and collapsing it threw away
    the whole response - including the band."""
    profile = rag.profile_from({"full_name": "A", "position": {
        "full_name": "Raghvendra Khatri", "position_level": {"name": "SR1"},
        "department": {"name": "Technology"}}})
    assert "SR1" in profile


def test_a_nested_record_that_is_not_a_source_root_still_collapses():
    """Without this the previous test would pass just by never collapsing."""
    profile = rag.profile_from({"position": {"reporting_to": {
        "full_name": "Aayush Sharma", "date_of_birth": "1994-08-18"}}})
    assert "Aayush Sharma" in profile and "1994-08-18" not in profile


# ------------------------------------------------------------------- envelopes


def test_the_data_envelope_is_unwrapped():
    assert "SR1" in rag.profile_from(
        {"status": "success", "message": "loggedIn user", "data": {"level": "SR1"}})


def test_an_error_envelope_is_unwrapped_too():
    assert "13.0" in rag.profile_from(
        {"error": None, "data": {"balance_days": 13.0}, "display_message": {"level": "info"},
         "error_code": None})


def test_a_flat_response_is_left_alone():
    assert "SR1" in rag.profile_from({"level": "SR1", "a": 1, "b": 2, "c": 3, "d": 4})


# --------------------------------------------------------------- fitting a cap


def test_a_long_list_cannot_crowd_out_another_source():
    """The band was dropped twice because leave data filled the budget first."""
    # leave first, position last: the order .env actually configures, and the
    # order in which cutting the tail loses the band.
    merged = {
        "leave": {"leave_data": [
            {"leave_type": {"name": f"Leave Type Number {i}"}, "balance": float(i)}
            for i in range(15)]},
        "position": {"position_level": {"name": "SR1"}},
    }
    pruned = rag.prune_profile(merged, protect={"position", "leave"})
    rendered = rag.render_profile(pruned)
    assert "SR1" not in rendered[:500], "fixture too small to show the problem"
    for cap in (2000, 900, 500):
        fitted = rag.fit_profile(pruned, cap=cap)
        assert "SR1" in fitted, f"band lost at cap {cap}"
        assert len(fitted) <= cap + 80  # the trim marker costs a little


def test_a_profile_within_the_cap_is_untouched():
    pruned = rag.prune_profile({"full_name": "A", "level": "SR1"})
    assert "(...)" not in rag.fit_profile(pruned, cap=4000)


# ------------------------------------------------------------- source plumbing


@pytest.mark.parametrize("spec,expected", [
    ("api/users/me", ("me", "api/users/me", [])),
    ("wfh=https://a.com/api/wfh", ("wfh", "https://a.com/api/wfh", [])),
    ("p=https://a.com/x/{id}|level+dept", ("p", "https://a.com/x/{id}", ["level", "dept"])),
    ("https://a.com/api/balance", ("balance", "https://a.com/api/balance", [])),
])
def test_source_specs_parse(spec, expected):
    assert rag.split_source(spec) == expected


def test_a_placeholder_is_filled_from_an_earlier_response():
    known = rag.scalars({"data": {"team_member_id": "442c87de"}}, {})
    assert rag.fill("https://a.com/x/{team_member_id}", known) == "https://a.com/x/442c87de"


def test_an_unfillable_source_is_skipped_not_fatal():
    assert rag.fill("https://a.com/x/{nope}", {}) is None


def test_plain_http_to_a_remote_host_is_refused():
    with pytest.raises(SystemExit, match="plain HTTP"):
        rag.fetch_json("http://hrms.example.com/api/me", "token")


# --------------------------------------------------------------- the PA formula


def test_pa_score_matches_the_worked_example():
    """The example in knowledge/pa-calculation.md, which the model is shown."""
    result = rag.pa_score(
        [{"name": "Quality", "weight": 0.6, "client_avg": 4.5, "team_avg": 4.0},
         {"name": "Delivery", "weight": 0.4, "client_avg": None, "team_avg": 3.0}],
        client_weight=0.7, team_weight=0.3)
    assert round(result["final_score"], 4) == 59.4


def test_a_criterion_with_one_source_uses_only_that_source():
    result = rag.pa_score([{"name": "A", "weight": 1.0, "client_avg": None, "team_avg": 3.0}],
                          client_weight=0.7, team_weight=0.3)
    assert round(result["criteria"][0]["score"], 4) == 0.9
    assert result["criteria"][0]["sources"] == ["team"]


def test_a_criterion_with_no_feedback_scores_zero():
    result = rag.pa_score([{"name": "A", "weight": 1.0}], client_weight=0.7, team_weight=0.3)
    assert result["final_score"] == 0.0
    assert result["criteria"][0]["sources"] == []


def test_full_marks_from_both_sources_is_one_hundred():
    """Weights summing to 1 and top scores everywhere must reach the ceiling."""
    result = rag.pa_score(
        [{"name": "A", "weight": 0.5, "client_avg": 5.0, "team_avg": 5.0},
         {"name": "B", "weight": 0.5, "client_avg": 5.0, "team_avg": 5.0}],
        client_weight=0.6, team_weight=0.4)
    assert round(result["final_score"], 4) == 100.0


# ------------------------------------------------------- unconfigured records


def test_an_all_zero_leave_type_is_dropped():
    """The HRMS returns allocated 0 / used 0 / balance 0 for a leave type it has
    not been configured with. Left in, it reads as an entitlement of zero and is
    quoted over the handbook's actual grant."""
    profile = rag.profile_from({"leave": {"leave_data": [
        {"leave_type": {"name": "Sick Leave"}, "total_allocated": 0.0,
         "used": 0.0, "balance": 0.0},
        {"leave_type": {"name": "Casual Leave"}, "total_allocated": 6.0,
         "used": 3.5, "balance": 2.5}]}})
    assert "Sick Leave" not in profile
    assert "Casual Leave" in profile


def test_a_fully_spent_balance_is_kept():
    """Zero left is a real answer; zero allocated is missing data."""
    profile = rag.profile_from({"leave": {"leave_data": [
        {"leave_type": {"name": "Earned Leave"}, "total_allocated": 12.0,
         "used": 12.0, "balance": 0.0}]}})
    assert "Earned Leave" in profile


def test_all_zero_ignores_booleans():
    """False is not a zero measurement - a record of flags must not vanish."""
    assert not rag.all_zero({"name": "X", "is_paid": False, "is_active": False})
    assert rag.all_zero({"name": "X", "balance": 0, "used": 0.0})
    assert not rag.all_zero({"name": "X"})


# ----------------------------------------------------------------- attribution


def provenance(profile="", knowledge=None, docs=()):
    from langchain_core.documents import Document
    p = rag.Provenance()
    p.profile = profile
    p.knowledge = knowledge or {}
    p.docs = [Document(page_content=text, metadata=meta) for text, meta in docs]
    return p


def test_an_answer_about_the_person_is_credited_to_the_hrms():
    """It used to be credited to whichever policy page came back alongside."""
    source = provenance(
        profile="- full name: Raghvendra Khatri\n- position level: SR1",
        docs=[("Bronze 750 Silver 1200 Gold 1500 travel allowance table",
               {"source": "perks-and-benefits.md"})])
    assert rag.credit("You are Raghvendra Khatri, SR1.", source) == "the HRMS"


def test_a_figure_from_the_profile_beats_a_wordier_file():
    """A terse profile line used to lose to any file that discussed leave."""
    source = provenance(
        profile="- balance days: 13.0\n- total allocated days: 30.0",
        knowledge={"knowledge/sources-of-truth.md":
                   "The handbook is authoritative for entitlement, the profile "
                   "for days already used and days left as of the profile date."})
    assert rag.credit("You have 13.0 WFH days left, as of the profile date.",
                      source) == "the HRMS"


def test_a_policy_answer_is_credited_to_its_pages():
    source = provenance(docs=[
        ("The notice period shall be 3 months for all confirmed employees.",
         {"source": "policy.pdf", "page": 83}),
        ("Unrelated text about reimbursement invoices and GST.",
         {"source": "policy.pdf", "page": 12})])
    assert rag.credit("The notice period is 3 months.", source) == "policy.pdf p. 83"


def test_two_chunks_from_one_page_are_named_once():
    source = provenance(docs=[
        ("notice period 3 months clause one", {"source": "policy.pdf", "page": 83}),
        ("notice period 3 months clause two", {"source": "policy.pdf", "page": 83})])
    assert rag.credit("The notice period is 3 months.", source) == "policy.pdf p. 83"


def test_nothing_overlapping_is_credited_to_nothing():
    source = provenance(docs=[("wholly unrelated wording", {"source": "policy.pdf"})])
    assert rag.credit("qqqq zzzz", source) == ""
