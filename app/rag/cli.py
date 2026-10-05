"""Argument parsing and the cmd_* handlers each subcommand dispatches to."""

import argparse
import json
import os
import sys

import psycopg
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .chain import (
    Memory,
    Provenance,
    ask,
    build_chain,
    build_contextualizer,
    build_summariser,
    clean,
    looks_like_followup,
)
from .config import (
    COLLECTION,
    HISTORY_TURNS,
    REQUIRED_ENV,
    TOKEN_LOGIN,
    TOKEN_PROMPT,
    TOP_K,
    env,
)
from .db import counts_by_source, indexed, store
from .knowledge import knowledge_files, load_knowledge
from .profile import (
    fetch_profile,
    prune_profile,
    profile_from,
    profile_text,
    read_profile_file,
    resolve_token,
    unwrap_profile,
)
from .feedback import cmd_feedback
from .ingest import cmd_ingest
from .ui import chat_banner, console, err_console, waiting


def cmd_ask(question: str, k: int, token: str | None, profile_name: str | None,
            sources: bool) -> int:
    if not indexed():
        sys.exit("Nothing indexed yet - run `ingest` first.")
    provenance = Provenance() if sources else None
    profile = profile_text(token, profile_name)
    chain = build_chain(k, profile, provenance)
    ask(chain, question, provenance, profile)
    return 0


def cmd_chat(k: int, token: str | None, profile_name: str | None, sources: bool) -> int:
    if not indexed():
        sys.exit("Nothing indexed yet - run `ingest` first.")
    # Fetched once per session, not per question: a chat would otherwise hammer
    # the HRMS, and the balances should not shift underneath a conversation.
    profile = profile_text(token, profile_name)
    provenance = Provenance() if sources else None
    memory = Memory(summariser=build_summariser())
    chain = build_chain(k, profile, provenance, memory=memory)  # built once
    contextualize = build_contextualizer()

    console.print()
    console.print(chat_banner(profile, token, profile_name, sources, k))

    while True:
        try:
            console.print()
            question = console.input("[bold cyan]❯[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if not question:
            return 0

        standalone = question
        # Only rewrite when the question actually refers back. A standalone
        # question ("what is the reimbursement policy") already names its topic
        # and the rewriter would only pollute it with the asker's identity.
        if memory.history and looks_like_followup(question):
            recent = "\n".join(
                f"Q: {q}\nA: {a}" for q, a in memory.history[-HISTORY_TURNS:]
            )
            with waiting("reading your follow-up"):
                rewritten = contextualize.invoke(
                    {"conversation": f"{recent}\n\nFollow-up: {question}"}
                )
            standalone = clean(rewritten).strip() or question
            if standalone != question:
                err_console.print(f"[dim italic]· reading that as: {standalone}[/dim italic]")

        answer = ask(chain, standalone, provenance, profile)
        if answer:
            memory.remember(standalone, answer)


def cmd_stats() -> int:
    rows = counts_by_source()
    if rows is None:
        console.print("[yellow]No index yet — run `ingest` first.[/yellow]")
        return 0
    if not rows:
        console.print("[dim]Index is empty.[/dim]")
    else:
        table = Table(title="Indexed documents", box=box.SIMPLE_HEAVY,
                      title_style="bold", header_style="bold cyan")
        table.add_column("Source")
        table.add_column("Chunks", justify="right", style="green")
        for source, count in rows:
            table.add_row(source, f"{count}")
        console.print(table)
    files = knowledge_files()
    if files:
        total = len(load_knowledge())
        ktable = Table(
            title=f"knowledge/ · sent with every question · {total} chars total",
            box=box.SIMPLE, title_style="bold", header_style="bold cyan",
        )
        ktable.add_column("File")
        ktable.add_column("Size", justify="right", style="green")
        for path in files:
            ktable.add_row(os.path.basename(path), f"{os.path.getsize(path)} B")
        console.print()
        console.print(ktable)
    return 0


def cmd_reset() -> int:
    store().delete_collection()
    print(f"Dropped the '{COLLECTION}' collection.")
    return 0


def cmd_discover() -> int:
    """Find the HRMS endpoint that returns the employee's own record."""
    from . import hrms_login

    login_url = os.getenv("HRMS_LOGIN_URL") or os.getenv("HRMS_BASE_URL")
    if not login_url:
        sys.exit("Set HRMS_LOGIN_URL (or HRMS_BASE_URL) in .env first.")
    rows = hrms_login.discover(login_url, api_hint=os.getenv("HRMS_API_HINT", ""))
    if not rows:
        console.print("[yellow]No JSON API calls seen. Is HRMS_API_HINT too narrow?[/yellow]")
        return 0
    table = Table(title=f"{len(rows)} endpoint(s) seen", box=box.SIMPLE_HEAVY,
                  title_style="bold", header_style="bold cyan")
    table.add_column("Method", style="magenta")
    table.add_column("Status", justify="right")
    table.add_column("URL")
    table.add_column("Top-level keys", style="dim")
    for method, url, status_code, keys in rows:
        table.add_row(method, str(status_code), url, keys)
    console.print()
    console.print(table)
    console.print("[dim]Pick the one holding band, city tier and balances,"
                  " then set HRMS_BASE_URL and HRMS_PROFILE_SOURCES.[/dim]")
    return 0


# The scale feedback is given on, which is what finalScore is normalised by.
PA_SCALE = 5.0


def pa_score(criteria: list[dict], client_weight: float, team_weight: float,
             scale: float = PA_SCALE) -> dict:
    """The Performance Allowance score, computed rather than reasoned about.

    A model doing this arithmetic in tokens has been right every time so far,
    which is not the same as being reliable. Here it is once, in code:

        score      = client avg x client weight + team avg x team weight
                     (only the sources that have feedback)
        weightedPa = score x the criterion's own weight
        finalScore = sum(weightedPa) / scale x 100
    """
    rows = []
    for item in criteria:
        parts = []
        if item.get("client_avg") is not None:
            parts.append(("client", item["client_avg"] * client_weight))
        if item.get("team_avg") is not None:
            parts.append(("team", item["team_avg"] * team_weight))
        score = sum(value for _, value in parts)
        weighted = score * item["weight"]
        rows.append({"name": item.get("name", "?"), "weight": item["weight"],
                     "sources": [n for n, _ in parts], "score": score,
                     "weighted": weighted})
    total = sum(row["weighted"] for row in rows)
    return {"criteria": rows, "total": total, "final_score": total / scale * 100}


def cmd_pa(path: str) -> int:
    """Show the working, so the number can be checked rather than trusted."""
    with open(path, encoding="utf-8") as fh:
        data = unwrap_profile(json.load(fh))
    try:
        result = pa_score(data["criteria"], data["client_weight"], data["team_weight"])
    except (KeyError, TypeError) as exc:
        sys.exit(f"{path} needs client_weight, team_weight and criteria: {exc}")

    console.print()
    console.print(f"[dim]client weight[/dim] {data['client_weight']}   "
                  f"[dim]team weight[/dim] {data['team_weight']}")
    table = Table(box=box.SIMPLE, header_style="bold cyan")
    table.add_column("Criterion")
    table.add_column("Feedback")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Weighted", justify="right", style="green")
    for row in result["criteria"]:
        sources = "+".join(row["sources"]) or "[dim]no feedback[/dim]"
        table.add_row(row["name"], sources, f"{row['score']:.4g}",
                      f"{row['weight']}", f"{row['weighted']:.4g}")
    console.print(table)
    console.print(
        f"[dim]  sum of weightedPa =[/dim] {result['total']:.4g}"
    )
    console.print(Panel(
        Text.from_markup(
            f"finalScore = ({result['total']:.4g} / {PA_SCALE}) × 100 = "
            f"[bold green]{result['final_score']:.4g}[/bold green]"
        ),
        border_style="green", box=box.ROUNDED,
    ))
    return 0


def cmd_profile(token: str | None, name: str | None) -> int:
    """Show what the HRMS returns, and how much of the context it would cost."""
    raw = fetch_profile(resolve_token(token)) if token else read_profile_file(name or "")
    raw = unwrap_profile(raw)
    pruned = prune_profile(raw)
    table = Table(
        title=f"Top-level fields · {len(raw)} returned · {len(pruned)} kept",
        box=box.SIMPLE_HEAVY, title_style="bold", header_style="bold cyan",
    )
    table.add_column("")
    table.add_column("Field")
    table.add_column("Size", justify="right")
    for key in raw:
        size = len(json.dumps(raw[key]))
        if key in pruned:
            table.add_row("[green]●[/green]", key, f"[green]{size}[/green]")
        else:
            table.add_row("[red]✗[/red]", f"[dim]{key}[/dim]", f"[dim]{size}[/dim]")
    console.print()
    console.print(table)
    text = profile_from(raw)
    console.print()
    console.print(Panel(
        text or "[dim](empty)[/dim]",
        title=f"Rendered for the prompt · {len(text)} chars · ~{len(text) // 4} tokens",
        title_align="left", border_style="cyan", box=box.ROUNDED,
    ))
    console.print("[dim]Too big? Set HRMS_PROFILE_FIELDS to a comma-separated"
                  " list of the fields above.[/dim]")
    return 0


def chosen_token(args: argparse.Namespace) -> str | None:
    """--login beats --prompt-token beats whatever was passed or is in the env."""
    if args.login:
        return TOKEN_LOGIN
    return TOKEN_PROMPT if args.prompt_token else args.token


def add_profile_args(parser: argparse.ArgumentParser) -> None:
    """Personalise the answer. --login is the everyday path; --as reads a saved
    response for working offline."""
    # Deliberately not `--token` with an optional value: argparse would swallow
    # the question after a bare `--token` and treat it as the token.
    parser.add_argument(
        "--token",
        default=os.getenv("HRMS_TOKEN"),
        help="the employee's own HRMS token (or set HRMS_TOKEN in .env, which is gitignored)",
    )
    parser.add_argument(
        "--prompt-token",
        action="store_true",
        help="ask for the token on a hidden prompt, so it misses the shell history",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="open the HRMS login page in a browser and read the token from your session",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="leave off the documents and pages the answer was drawn from",
    )
    parser.add_argument("--as", dest="profile", help="a saved profile in profiles/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rag", description="Ask questions about local policy documents, answered by Ollama."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="chunk, embed and store a document")
    p.add_argument("path", nargs="?", default="data/policy.pdf")
    p.add_argument("--replace", action="store_true", help="rebuild instead of resuming")

    p = sub.add_parser("ask", help="answer one question")
    p.add_argument("question", nargs="+")
    p.add_argument("-k", type=int, default=TOP_K, help=f"chunks to retrieve (default {TOP_K})")
    add_profile_args(p)

    p = sub.add_parser("chat", help="interactive question loop")
    p.add_argument("-k", type=int, default=TOP_K)
    add_profile_args(p)

    sub.add_parser("discover", help="find the HRMS endpoint holding your record")

    p = sub.add_parser("pa", help="compute a PA score from a JSON file of feedback")
    p.add_argument("path", help="JSON with client_weight, team_weight and criteria")

    p = sub.add_parser("feedback", help="interactively fill a peer-feedback CSV")
    p.add_argument("path", help="the blank feedback CSV to fill")

    p = sub.add_parser("profile", help="show what the HRMS returns about you")
    add_profile_args(p)

    sub.add_parser("stats", help="show what is indexed")
    sub.add_parser("reset", help="drop the collection")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    missing = [n for n in REQUIRED_ENV if not os.getenv(n)]
    if missing:
        sys.exit(f"Missing env vars: {', '.join(missing)}. Copy .env.example to .env.")

    try:
        if args.command == "ingest":
            return cmd_ingest(args.path, args.replace)
        if args.command == "ask":
            return cmd_ask(" ".join(args.question), args.k, chosen_token(args),
                           args.profile, not args.no_sources)
        if args.command == "chat":
            return cmd_chat(args.k, chosen_token(args), args.profile, not args.no_sources)
        if args.command == "discover":
            return cmd_discover()
        if args.command == "pa":
            return cmd_pa(args.path)
        if args.command == "feedback":
            return cmd_feedback(args.path)
        if args.command == "profile":
            return cmd_profile(chosen_token(args), args.profile)
        if args.command == "stats":
            return cmd_stats()
        return cmd_reset()
    except psycopg.OperationalError as exc:
        sys.exit(f"Cannot reach Postgres: {exc}\nIs it up? `docker compose up -d postgres`")
    except Exception as exc:  # noqa: BLE001 - Ollama errors arrive in several shapes
        url = env("OLLAMA_BASE_URL")
        if not ollama_reachable(url):
            sys.exit(f"Cannot reach Ollama at {url}. Is `ollama serve` running?")
        raise SystemExit(f"{type(exc).__name__}: {exc}") from exc


def ollama_reachable(url: str) -> bool:
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"{url}/api/tags", timeout=3)
        return True
    except (urllib.error.URLError, OSError):
        return False
