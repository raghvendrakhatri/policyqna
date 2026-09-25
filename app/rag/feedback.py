"""Interactive feedback-form filler.

Reads a peer-feedback CSV (wide matrix: colleagues across the header, sections
and sub-categories down the rows), asks the reviewer for a per-section score
and a short brief per named colleague, and writes a copy with the cells
filled in. The local LLM only rephrases the brief into the section's tone;
scores and factual claims are the reviewer's own.

Nothing is written in place - output goes to `<orig>_filled.csv` next to the
input, so a re-run cannot destroy the blank form or a prior fill.
"""

import csv
import os
import re
import sys
from datetime import datetime
from difflib import get_close_matches

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from rich import box
from rich.panel import Panel
from rich.table import Table

from .guardrails import clean
from .models import llm
from .ui import console, err_console, waiting


# The three columns of the leading matrix. Everything to the right is one
# column per colleague.
FIXED_COLS = 3
SECTION_COL = 0
SUBCAT_COL = 1
KIND_COL = 2  # "Score", "Remarks", or empty for the header/kudos rows

# The header labels the CSV uses for the "one remark per section" cell and the
# free-form kudos row at the very end.
SECTION_REMARKS = "Section Remarks"
KUDOS = "Kudos"

SCORE_MIN, SCORE_MAX = 1, 5

SECTION_PHRASE_PROMPT = """You are drafting one reviewer's feedback about a colleague for a section.

Ground rules (all of them, not just the first one you notice):
- Draw only from the reviewer's note and the section's score. Do NOT invent
  events, projects, dates, or behaviour the note doesn't mention.
- Refer to the colleague as '{pronoun}' (never the name; the name is already
  above every column).
- Keep it plain and direct. No preamble, no praise sandwich, no closing line.
- A sub-category the note doesn't touch gets a short line consistent with the
  score - one clause, not a paragraph. Score 5 = exceptional, 4 = strong,
  3 = solid/expected, 2 = inconsistent, 1 = a real gap.

Output format (exactly this, nothing else):

[Section Remark]
<one or two sentences summarising the section>

[Sub-category Remarks]
- <sub-category>: <one line>
- <sub-category>: <one line>
...one line per sub-category listed below, in order...

Colleague: {name}
Section: {section} (score {score}/5)
Sub-categories:
{subcats_bullets}
Reviewer's note: {note}"""


def parse_csv(path: str) -> tuple[list[list[str]], list[str], list[dict]]:
    """Return (rows, colleague_headers, sections).

    `sections` is a list of {name, subcats: [{name, score_row, remarks_row}],
    section_remarks_row: int|None}.

    Kudos is modelled as a section whose `section_remarks_row` is its own row
    and whose subcats list is empty - keeps the downstream code uniform.
    """
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows or len(rows[0]) <= FIXED_COLS:
        sys.exit(f"{path} does not look like a feedback matrix.")
    headers = [h.strip() for h in rows[0][FIXED_COLS:]]

    sections: list[dict] = []
    current: dict | None = None
    last_subcat: dict | None = None
    for i, row in enumerate(rows):
        # Pad short rows so column access never IndexErrors.
        while len(row) < FIXED_COLS:
            row.append("")
        section = row[SECTION_COL].strip()
        subcat = row[SUBCAT_COL].strip()
        kind = row[KIND_COL].strip()
        if section and i > 0:  # a new section starts here
            current = {"name": section, "subcats": [], "section_remarks_row": None}
            if section == KUDOS:
                current["section_remarks_row"] = i
            sections.append(current)
            last_subcat = None
        if current is None:
            continue
        if kind.lower() == "score":
            last_subcat = {"name": subcat, "score_row": i, "remarks_row": None}
            current["subcats"].append(last_subcat)
        elif kind.lower() == "remarks" and last_subcat is not None:
            last_subcat["remarks_row"] = i
        elif subcat == SECTION_REMARKS:
            current["section_remarks_row"] = i
    return rows, headers, sections


def resolve_name(query: str, headers: list[str]) -> str | None:
    """Fuzzy match a typed name (or 1-based index) against the header.

    Returns None on no match or ambiguity - the caller reprompts.
    """
    q = query.strip()
    if not q:
        return None
    if q.isdigit():
        i = int(q) - 1
        if 0 <= i < len(headers):
            return headers[i]
        err_console.print(f"[yellow]{q} is out of range (1-{len(headers)}).[/yellow]")
        return None
    lower = q.lower()
    exact = [h for h in headers if h.lower() == lower]
    if exact:
        return exact[0]
    contains = [h for h in headers if lower in h.lower()]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        err_console.print(f"[yellow]'{q}' matches: {', '.join(contains)}."
                          " Type more of the name or its number.[/yellow]")
        return None
    close = get_close_matches(lower, [h.lower() for h in headers], n=1, cutoff=0.6)
    if close:
        return next(h for h in headers if h.lower() == close[0])
    err_console.print(f"[yellow]No colleague matches '{q}'.[/yellow]")
    return None


def show_roster(headers: list[str], filled: list[str]) -> None:
    """A numbered grid of colleagues. Filled ones are dimmed and marked."""
    table = Table(title="Colleagues", box=box.SIMPLE,
                  title_style="bold", header_style="bold cyan",
                  show_header=False, padding=(0, 2))
    per_col = 3  # three columns of names side by side
    cells_per_row = (len(headers) + per_col - 1) // per_col
    for _ in range(per_col):
        table.add_column()
    for r in range(cells_per_row):
        row_cells = []
        for c in range(per_col):
            i = c * cells_per_row + r
            if i >= len(headers):
                row_cells.append("")
                continue
            name = headers[i]
            marker = "[green]✓[/green]" if name in filled else " "
            style = "dim strike" if name in filled else ""
            row_cells.append(f"[{style}]{i+1:>2}. {name}[/{style}] {marker}" if style
                             else f"[dim]{i+1:>2}.[/dim] {name} {marker}")
        table.add_row(*row_cells)
    console.print()
    console.print(table)


def ask_score(section: str) -> int | None:
    """None means the reviewer skipped this section for this person."""
    while True:
        raw = console.input(
            f"  [cyan]{section}[/cyan] score "
            f"[dim]({SCORE_MIN}-{SCORE_MAX}, empty = skip):[/dim] "
        ).strip()
        if not raw:
            return None
        try:
            value = int(raw)
        except ValueError:
            err_console.print("[red]  ↳ needs a whole number.[/red]")
            continue
        if not SCORE_MIN <= value <= SCORE_MAX:
            err_console.print(f"[red]  ↳ must be between {SCORE_MIN} and {SCORE_MAX}.[/red]")
            continue
        return value


def parse_phrasing(text: str, subcat_names: list[str]) -> tuple[str, dict[str, str]]:
    """Split the model's `[Section Remark] / [Sub-category Remarks]` output.

    Missing sub-category lines come back empty; extra ones the model invented
    are dropped. Order isn't required - sub-cats are matched by name (case /
    whitespace-insensitive) against what the CSV defines.
    """
    body = clean(text)
    header_re = re.compile(r"\[\s*section remark\s*\]", re.I)
    subhdr_re = re.compile(r"\[\s*sub[- ]?category remarks\s*\]", re.I)

    section_part, subcat_part = body, ""
    m_sub = subhdr_re.search(body)
    if m_sub:
        section_part = body[:m_sub.start()]
        subcat_part = body[m_sub.end():]

    m_sec = header_re.search(section_part)
    section_remark = section_part[m_sec.end():].strip() if m_sec else section_part.strip()
    section_remark = section_remark.strip().strip('"').strip()

    lookup = {n.lower().strip(): n for n in subcat_names}
    subcat_remarks: dict[str, str] = {n: "" for n in subcat_names}
    for line in subcat_part.splitlines():
        line = line.strip()
        if not line or not line.lstrip("-*• ").strip():
            continue
        line = line.lstrip("-*• ").strip()
        if ":" not in line:
            continue
        label, _, remark = line.partition(":")
        matched = lookup.get(label.lower().strip())
        if matched and remark.strip():
            subcat_remarks[matched] = remark.strip().strip('"').strip()
    return section_remark, subcat_remarks


def phrase_section(model, section_name: str, subcat_names: list[str], name: str,
                   note: str, score: int, pronoun: str) -> tuple[str, dict[str, str]]:
    """One LLM call: returns (section_remark, {subcat_name: remark})."""
    bullets = "\n".join(f"- {n}" for n in subcat_names) or "- (none)"
    chain = (
        ChatPromptTemplate.from_messages([("human", SECTION_PHRASE_PROMPT)])
        | model
        | StrOutputParser()
    )
    with waiting(f"phrasing {section_name.lower()} for {name.split()[0]}"):
        raw = chain.invoke({
            "section": section_name,
            "subcats_bullets": bullets,
            "name": name,
            "note": note.strip() or "(no specific note)",
            "score": score,
            "pronoun": pronoun,
        })
    return parse_phrasing(raw, subcat_names)


def parse_selection(raw: str, headers: list[str]) -> list[str]:
    """Split a batch input into unique resolved names.

    Accepts `1,10,adit` and `1-5` ranges (inclusive). Unresolved tokens are
    warned about and skipped; the caller can still proceed with what parsed.
    """
    resolved: list[str] = []
    for token in [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]:
        if "-" in token and all(part.strip().isdigit() for part in token.split("-", 1)):
            lo, hi = (int(p) for p in token.split("-", 1))
            if lo > hi:
                lo, hi = hi, lo
            for i in range(lo, hi + 1):
                name = resolve_name(str(i), headers)
                if name and name not in resolved:
                    resolved.append(name)
            continue
        name = resolve_name(token, headers)
        if name and name not in resolved:
            resolved.append(name)
    return resolved


SCORE_FALLBACK = {
    5: "{Pronoun} is exceptional here.",
    4: "{Pronoun} is consistently strong here.",
    3: "{Pronoun} meets expectations here.",
    2: "{Pronoun} is inconsistent here.",
    1: "{Pronoun} has a real gap here.",
}


def fill_missing_subcats(subcat_remarks: dict[str, str], score: int, pronoun: str) -> None:
    """Backfill any empty sub-category remark with a short score-appropriate line.

    The LLM occasionally drops a bullet; leaving the cell blank looks like the
    reviewer forgot it. A one-clause default is better than a hole.
    """
    default = SCORE_FALLBACK.get(score, "{Pronoun} contributes here.").format(
        Pronoun=pronoun.capitalize()
    )
    for name, remark in subcat_remarks.items():
        if not remark.strip():
            subcat_remarks[name] = default


PRONOUNS = {"he", "she", "they"}

PRONOUN_INFER_PROMPT = """For a person with this Indian first name, which pronoun is the common default in professional writing?

Rules:
- Answer with exactly one word: he, she, or they. Nothing else.
- Commit to he or she when the name is commonly gendered in India, even
  though exceptions exist. Do NOT hedge to "they" just because a single
  counter-example is possible - only use "they" for names that are genuinely
  unisex in everyday Indian usage.
- Examples of commit-to-he: Adit, Tarun, Priyanshu, Ritik, Sanjeev, Rahul,
  Amit, Vikram, Arjun, Rohan.
- Examples of commit-to-she: Priya, Rashmi, Shikha, Asmita, Tejasri, Neha,
  Anjali, Kavya, Meera.
- Examples of unisex → they: Kiran, Anmol, Jyoti (context-dependent).

First name: {first_name}"""


# Names seen in this batch/run, so the same person's pronoun is looked up once.
_pronoun_cache: dict[str, str] = {}


def infer_pronoun(model, name: str) -> str:
    """Ask the model which pronoun fits the first name. Defaults to 'they' on
    any anomaly - unknown reply, empty reply, exception."""
    first = name.strip().split()[0].lower()
    if first in _pronoun_cache:
        return _pronoun_cache[first]
    chain = (
        ChatPromptTemplate.from_messages([("human", PRONOUN_INFER_PROMPT)])
        | model
        | StrOutputParser()
    )
    try:
        with waiting(f"inferring pronoun for {first.title()}"):
            raw = chain.invoke({"first_name": first.title()})
    except Exception:  # noqa: BLE001 - a mis-inferred pronoun must not block the batch
        raw = ""
    guess = clean(raw).strip().lower().rstrip(".").split()[0] if raw.strip() else ""
    pronoun = guess if guess in PRONOUNS else "they"
    _pronoun_cache[first] = pronoun
    return pronoun


def ask_shared_inputs(names: list[str], scored: list[dict]) -> dict | None:
    """Prompt scores + brief + pronoun + kudos once, applied to every name.

    Returns dict, or None when the reviewer skipped every section.
    """
    console.print()
    listing = ", ".join(names) if len(names) <= 6 else f"{len(names)} colleagues"
    console.print(Panel(f"[bold]{listing}[/bold]", border_style="magenta",
                        box=box.ROUNDED,
                        title=f"Batch · {len(names)} selected", title_align="left"))
    if len(names) > 6:
        console.print("[dim]  " + " · ".join(names) + "[/dim]")

    scores: dict = {}
    for section in scored:
        scores[section["name"]] = ask_score(section["name"])
    if not any(v is not None for v in scores.values()):
        console.print("[dim]No scores given - skipping.[/dim]")
        return None

    return {
        "scores": scores,
        "note": console.input(
            "  [cyan]Brief note[/cyan] "
            "[dim](one line; applied to all above):[/dim]\n  "
        ).strip(),
        "kudos": console.input(
            "  [cyan]Kudos[/cyan] [dim](one line, empty to skip):[/dim] "
        ).strip(),
    }


def build_answers(name: str, shared: dict, sections: list[dict], model) -> dict:
    """Per person, per section: {"score", "section_remark", "subcat_remarks"}.

    Kudos, when set, comes back under the KUDOS section with only "remark".
    """
    result: dict = {}
    note = shared["note"]
    pronoun = infer_pronoun(model, name)
    err_console.print(f"[dim]· {name.split()[0]} → {pronoun}[/dim]")
    scores = shared["scores"]
    for section in sections:
        if section["name"] == KUDOS:
            continue
        score = scores.get(section["name"])
        if score is None:
            continue
        subcat_names = [s["name"] for s in section["subcats"] if s["name"]]
        section_remark, subcat_remarks = phrase_section(
            model, section["name"], subcat_names, name, note, score, pronoun
        )
        fill_missing_subcats(subcat_remarks, score, pronoun)
        result[section["name"]] = {
            "score": score,
            "section_remark": section_remark,
            "subcat_remarks": subcat_remarks,
        }
    if shared["kudos"]:
        result[KUDOS] = {"remark": shared["kudos"]}
    return result


def apply_to_rows(rows: list[list[str]], headers: list[str], name: str,
                  answers: dict, sections: list[dict]) -> None:
    """Write one colleague's answers into the shared rows list, in place."""
    col = FIXED_COLS + headers.index(name)
    # Extend every row to include this column, so a sparse CSV doesn't lose the cell.
    width = col + 1
    for row in rows:
        while len(row) < width:
            row.append("")
    for section in sections:
        entry = answers.get(section["name"])
        if entry is None:
            continue
        if section["name"] == KUDOS:
            if entry.get("remark") and section["section_remarks_row"] is not None:
                rows[section["section_remarks_row"]][col] = entry["remark"]
            continue
        score = entry.get("score")
        if score is not None:
            for sub in section["subcats"]:
                rows[sub["score_row"]][col] = str(score)
        section_remark = entry.get("section_remark", "")
        if section_remark and section["section_remarks_row"] is not None:
            rows[section["section_remarks_row"]][col] = section_remark
        for sub in section["subcats"]:
            remark = entry.get("subcat_remarks", {}).get(sub["name"], "")
            if remark and sub["remarks_row"] is not None:
                rows[sub["remarks_row"]][col] = remark


def preview(rows: list[list[str]], headers: list[str], filled: list[str],
            sections: list[dict]) -> None:
    """A per-section score summary for every filled colleague."""
    if not filled:
        return
    table = Table(title="Preview", box=box.SIMPLE_HEAVY,
                  title_style="bold", header_style="bold cyan")
    table.add_column("Colleague")
    section_names = [s["name"] for s in sections if s["name"] != KUDOS]
    for name in section_names:
        table.add_column(name, justify="center")
    table.add_column("Kudos?", justify="center")
    for name in filled:
        col = FIXED_COLS + headers.index(name)
        cells = [name]
        for section in sections:
            if section["name"] == KUDOS:
                kudos = (rows[section["section_remarks_row"]][col]
                         if section["section_remarks_row"] is not None else "")
                cells.append("[green]✓[/green]" if kudos.strip() else "[dim]—[/dim]")
                continue
            first = section["subcats"][0]["score_row"] if section["subcats"] else None
            value = rows[first][col].strip() if first is not None else ""
            cells.append(value or "[dim]—[/dim]")
        table.add_row(*cells)
    console.print()
    console.print(table)


def output_path(path: str) -> str:
    stem, ext = os.path.splitext(path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_filled_{stamp}{ext or '.csv'}"


def cmd_feedback(path: str) -> int:
    if not os.path.exists(path):
        sys.exit(f"No such file: {path}")
    rows, headers, sections = parse_csv(path)
    scored_sections = [s['name'] for s in sections if s['name'] != KUDOS]
    console.print()
    console.print(Panel(
        f"[bold]{os.path.basename(path)}[/bold]\n"
        f"[dim]{len(headers)} colleagues · {len(scored_sections)} scored sections "
        f"({', '.join(scored_sections)})[/dim]",
        border_style="cyan", box=box.ROUNDED, title="Feedback form",
        title_align="left",
    ))

    model = llm()
    filled: list[str] = []
    scored = [s for s in sections if s["name"] != KUDOS]

    # Upfront shortlist: everyone else is skipped by default and never asked
    # about again. Reviewers rarely give feedback to all 29 colleagues; naming
    # the subset first keeps the rest of the flow scoped to what matters.
    show_roster(headers, filled=[])
    console.print()
    shortlist_raw = console.input(
        "[bold cyan]❯[/bold cyan] Colleagues you will give feedback to "
        "[dim](numbers/names, comma-separated or `1-5`):[/dim] "
    ).strip()
    shortlist = parse_selection(shortlist_raw, headers)
    if not shortlist:
        console.print("[dim]Nothing selected - not writing anything.[/dim]")
        return 0
    console.print(f"[green]Shortlist ({len(shortlist)}):[/green] "
                  f"{', '.join(shortlist)}")
    console.print(f"[dim]The other {len(headers) - len(shortlist)} colleague(s)"
                  " are skipped.[/dim]")

    # Loop over the shortlist in batches. Numbers in each round refer to the
    # remaining shortlist, not the original 29. The loop exits automatically
    # once every shortlist entry is filled.
    while True:
        remaining = [n for n in shortlist if n not in filled]
        if not remaining:
            console.print("\n[green]Everyone on the shortlist is filled.[/green]")
            break
        show_roster(remaining, filled=[])
        raw = console.input(
            "[bold cyan]❯[/bold cyan] Next batch "
            "[dim](numbers/names, comma-separated or `1-5`; empty to finish early):[/dim] "
        ).strip()
        if not raw:
            break
        names = parse_selection(raw, remaining)
        if not names:
            continue
        shared = ask_shared_inputs(names, scored)
        if shared is None:
            continue
        for name in names:
            answers = build_answers(name, shared, sections, model)
            apply_to_rows(rows, headers, name, answers, sections)
            if name not in filled:
                filled.append(name)

    if not filled:
        console.print("[dim]Nothing filled - not writing anything.[/dim]")
        return 0

    preview(rows, headers, filled, sections)
    console.print()
    confirm = console.input("[bold]Save?[/bold] [dim](y/N):[/dim] ").strip().lower()
    if confirm not in ("y", "yes"):
        console.print("[dim]Discarded. No file written.[/dim]")
        return 0

    out = output_path(path)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(rows)
    console.print(f"[green]Wrote[/green] {out} [dim]· {len(filled)} colleague(s)[/dim]")
    return 0
