"""Reading /api/team-feedback/requests/<id>/parse-excel/, from a trimmed copy
of a live response.

Runs without Ollama, Postgres or a network.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from rag import tools  # noqa: E402


def aspect(label, order, description, value=None):
    score = {"type": "number", "label": "Score", "order": 1, "isField": True,
             "maxValue": 5, "minValue": 1, "isRequired": True,
             "remarksCheckpoint": [1, 2, 5]}
    if value is not None:
        score["value"] = value
    return {"child": {"score": score,
                      "remarks": {"type": "string", "label": "Remarks", "order": 2,
                                  "isField": True}},
            "label": label, "order": order, "isField": False,
            "description": description}


TEMPLATE = {
    "kudos": {"type": "string", "label": "Kudos", "order": 5, "isField": True},
    "softSkills": {"label": "Soft Skills", "order": 2, "isField": False, "child": {
        "remarks": {"type": "string", "label": "Section Remarks", "order": 9, "isField": True},
        "diligence": aspect("Diligence", 5, "Attention to detail."),
        "punctuality": aspect("Punctuality/Availability", 1, "Availability."),
    }},
    "teamSkills": {"label": "Team Skills", "order": 1, "isField": False, "child": {
        "remarks": {"type": "string", "label": "Section Remarks", "order": 6, "isField": True},
        "teamSatisfaction": aspect("Team Satisfaction", 1, "Satisfaction."),
    }},
}


def receiver(first, last, email, form, editable=True):
    return {"receiver": {"id": "r-" + first, "firstName": first, "lastName": last,
                         "email": email, "presigned_profilePicture": "https://s3/..."},
            "sender": {"id": "me", "firstName": "Me"},
            "formFields": form, "isEditable": editable}


def test_sections_and_aspects_follow_the_form_order():
    outline = tools.feedback_outline(TEMPLATE)
    assert [s["name"] for s in outline] == ["Team Skills", "Soft Skills", "Kudos"]
    soft = outline[1]
    assert [a["name"] for a in soft["aspects"]] == ["Punctuality/Availability", "Diligence"]
    assert soft["section_remarks"] is True
    assert soft["aspects"][0] == {"name": "Punctuality/Availability",
                                  "description": "Availability.",
                                  "score_range": [1, 5],
                                  "remarks_for_scores": [1, 2, 5]}
    assert outline[2] == {"name": "Kudos", "free_text": True}


def test_colleagues_carry_only_who_they_are():
    filled = {**TEMPLATE, "teamSkills": {**TEMPLATE["teamSkills"], "child": {
        "teamSatisfaction": aspect("Team Satisfaction", 1, "", value=4)}}}
    colleagues = tools.feedback_colleagues([
        receiver("Sumit", "Lakra", "sumit@example.com", TEMPLATE),
        receiver("Cia", "", "cia@example.com", filled, editable=False),
    ])
    assert colleagues == [
        {"name": "Sumit Lakra", "email": "sumit@example.com"},
        {"name": "Cia", "email": "cia@example.com", "editable": False,
         "has_answers": True},
    ]


def test_tool_condenses_the_response(monkeypatch):
    monkeypatch.setattr(tools, "download_feedback_form", lambda token, rid: b"csv")
    monkeypatch.setattr(tools, "parse_feedback_form", lambda token, rid, csv: {
        "status": "success",
        "data": {"feedbackTemplate": {"form_fields": TEMPLATE},
                 "receivers": [receiver("Sumit", "Lakra", "s@example.com", TEMPLATE)],
                 "errors": {"unmatchedEmails": [], "missingFields": []}},
    })
    monkeypatch.setattr(tools, "fetch_feedback_requests", lambda token: [SUMMARY])
    result = tools.get_feedback_form.invoke({"access_token": "t"})
    assert result["request"]["project"] == "test" and "id" not in result["request"]
    assert result["colleagues"] == [{"name": "Sumit Lakra", "email": "s@example.com"}]
    assert len(result["sections"]) == 3
    assert "problems" not in result
    assert "presigned" not in str(result)


def test_download_errors_reach_the_model(monkeypatch):
    error = {"error": "HRMS returned 404", "detail": "not found"}
    monkeypatch.setattr(tools, "fetch_feedback_requests", lambda token: [SUMMARY])
    monkeypatch.setattr(tools, "download_feedback_form", lambda token, rid: error)
    assert tools.get_feedback_form.invoke({"access_token": "t"}) == error


def test_request_id_cannot_leave_its_path_segment():
    url = tools.feedback_url("../../users/me?", "export/")
    assert url.endswith("/requests/..%2F..%2Fusers%2Fme%3F/export/")


REQUEST = {
    "id": "req-1", "requester_id": "boss", "category": "performance",
    "status": "requested", "expiry_date": "2026-09-29T18:30:00.000Z",
    "feedback_for_date": "2026-09-15T00:00:00.000Z",
    "total_request": 3, "total_submission_count": 1,
    "form_fields": TEMPLATE,
    "feedback_template": {"id": "t", "name": "Project Team Feedback",
                          "form_fields": TEMPLATE},
    "project": {"id": "p", "name": "test"},
    "receivers": [{"id": "a", "team_member_id": "boss"},
                  {"id": "b", "team_member_id": "me"},
                  {"id": "c", "team_member_id": "peer"}],
    "submitters": [{"id": "d", "team_member_id": "me",
                    "team_member": {"full_name": "Me", "official_email": "me@x"}}],
}


def test_request_summary_drops_the_form_and_the_employee():
    summary = tools.feedback_request_summary(REQUEST, "me")
    assert summary["id"] == "req-1"
    assert (summary["project"], summary["form"]) == ("test", "Project Team Feedback")
    assert summary["feedback_for"] == "2026-09-15"
    assert summary["colleagues"] == 2
    assert summary["submissions"] == "1 of 3"
    assert summary["expired"] is True
    assert "form_fields" not in str(summary)


def test_requests_are_listed_for_the_employees_team_member_id(monkeypatch):
    urls = []

    def fake_request(url, token, body=None, method=None):
        urls.append(url)
        if url.endswith("/api/users/me"):
            return {"data": {"id": "user", "team_member_id": "me"}}
        return {"status": "success", "count": 1, "data": [REQUEST]}

    monkeypatch.setattr(tools, "hrms_request", fake_request)
    result = tools.get_feedback_requests.invoke({"access_token": "t"})
    assert "team_member_id=me" in urls[1] and "action=receivedRequests" in urls[1]
    assert [r["project"] for r in result["requests"]] == ["test"]
    assert "id" not in result["requests"][0]


SUMMARY = {"id": "req-1", "project": "test", "form": "Project Team Feedback"}
OTHER = {"id": "req-2", "project": "apollo", "form": "Project Team Feedback"}


def test_a_request_is_found_by_name_never_by_a_made_up_id(monkeypatch):
    monkeypatch.setattr(tools, "fetch_feedback_requests", lambda token: [SUMMARY, OTHER])
    assert tools.find_feedback_request("t", "Apollo")["id"] == "req-2"
    assert tools.find_feedback_request("t", "req-1")["id"] == "req-1"
    made_up = tools.find_feedback_request("t", "req_12345")
    assert "No feedback request matches" in made_up["error"]
    assert made_up["requests"] == [{"project": "test", "form": "Project Team Feedback"},
                                   {"project": "apollo", "form": "Project Team Feedback"}]
    assert "several" in tools.find_feedback_request("t", "")["error"]


def test_the_only_request_needs_no_name(monkeypatch):
    monkeypatch.setattr(tools, "fetch_feedback_requests", lambda token: [SUMMARY])
    assert tools.find_feedback_request("t", "")["id"] == "req-1"


def test_save_writes_the_csv_without_overwriting(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "FEEDBACK_DIR", str(tmp_path))
    monkeypatch.setattr(tools, "fetch_feedback_requests", lambda token: [SUMMARY])
    monkeypatch.setattr(tools, "download_feedback_form", lambda token, rid: b"a,b\n")
    result = tools.save_feedback_form.invoke({"access_token": "t"})
    assert open(result["saved_to"], "rb").read() == b"a,b\n"
    assert os.path.basename(result["saved_to"]).startswith("project-team-feedback_test_")


def test_cli_reads_a_score_row_labelled_after_its_aspect(tmp_path):
    from rag.feedback import parse_csv
    path = tmp_path / "form.csv"
    path.write_text("Fields,Sub-Category,Q2,Sumit\n"
                    ",,EMAIL,s@example.com\n"
                    "Soft Skills,Listening Skills,Score,\n"
                    ",,Remarks,\n"
                    ",Timeliness,Timeliness,\n"
                    ",,Remarks,\n"
                    ",Section Remarks,,\n"
                    "Kudos,Kudos,,\n")
    _, headers, sections = parse_csv(str(path))
    assert headers == ["Sumit"]
    soft = sections[0]
    assert [(s["name"], s["score_row"], s["remarks_row"]) for s in soft["subcats"]] == [
        ("Listening Skills", 2, 3), ("Timeliness", 4, 5)]
    assert soft["section_remarks_row"] == 6
    assert sections[1]["name"] == "Kudos"


def test_feedback_files_are_listed_newest_first(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "FEEDBACK_DIR", str(tmp_path))
    (tmp_path / "blank.csv").write_text("a")
    (tmp_path / "notes.txt").write_text("a")
    filled = tmp_path / "blank_filled_1.csv"
    filled.write_text("a")
    os.utime(filled, (2_000_000_000, 2_000_000_000))
    result = tools.list_feedback_files.invoke({})
    assert result["files"] == [{"file": "blank_filled_1.csv", "filled": True},
                               {"file": "blank.csv", "filled": False}]


def test_a_feedback_file_cannot_come_from_outside_the_folder(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "FEEDBACK_DIR", str(tmp_path))
    (tmp_path / "form.csv").write_text("a")
    (tmp_path / "form_filled_1.csv").write_text("a")
    assert tools.find_feedback_file("form.csv") == str(tmp_path / "form.csv")
    assert tools.find_feedback_file("filled") == str(tmp_path / "form_filled_1.csv")
    assert tools.find_feedback_file("/etc/../form.csv") == str(tmp_path / "form.csv")
    assert "No feedback file matches" in tools.find_feedback_file("../secrets.csv")["error"]
    assert "More than one" in tools.find_feedback_file("form")["error"]


def test_submit_uploads_the_picked_file(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "FEEDBACK_DIR", str(tmp_path))
    (tmp_path / "form_filled_1.csv").write_bytes(b"filled,csv\n")
    monkeypatch.setattr(tools, "fetch_feedback_requests", lambda token: [SUMMARY])
    sent = {}

    def fake_parse(token, rid, csv_bytes):
        sent.update(token=token, rid=rid, csv=csv_bytes)
        answered = {**TEMPLATE, "teamSkills": {**TEMPLATE["teamSkills"], "child": {
            "teamSatisfaction": aspect("Team Satisfaction", 1, "", value=4)}}}
        return {"status": "success", "data": {
            "receivers": [receiver("Sumit", "Lakra", "s@example.com", answered),
                          receiver("Cia", "", "c@example.com", TEMPLATE)],
            "errors": {"unmatchedEmails": ["x@example.com"], "missingFields": []}}}

    monkeypatch.setattr(tools, "parse_feedback_form", fake_parse)
    result = tools.submit_feedback_form.invoke(
        {"access_token": "t", "file": "form_filled_1.csv"})
    assert sent == {"token": "t", "rid": "req-1", "csv": b"filled,csv\n"}
    assert result["uploaded"] == "form_filled_1.csv"
    assert result["colleagues_with_answers"] == ["Sumit Lakra"]
    assert result["problems"] == {"unmatchedEmails": ["x@example.com"]}
    assert "id" not in result["request"]


def test_submit_needs_a_confirmation():
    from rag.agent import WRITE_TOOLS
    assert "submit_feedback_form" in WRITE_TOOLS
