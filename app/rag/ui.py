"""Rich consoles, spinner, refusal panel, chat banner.

One Console per stream. Rich auto-disables colours and boxes when the stream
is piped, so scripts that consume the output keep working unchanged.
"""

import contextlib
import os

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .config import MEMORY_WINDOW, env

console = Console()
err_console = Console(stderr=True)


# Rich's Status handles the spinner frames, elapsed time and line rewriting.
# We keep the retitle-on-status trick by holding the live handle here so a
# progress note can update its label instead of printing over the top of it.
_STATUS = None


@contextlib.contextmanager
def waiting(label: str):
    """A rich spinner on stderr. Rich no-ops on non-TTY, so a piped run stays clean."""
    global _STATUS
    with err_console.status(f"[cyan]{label}[/cyan]", spinner="dots") as handle:
        _STATUS = handle
        try:
            yield
        finally:
            _STATUS = None


def status(message: str) -> None:
    """Progress note. Retitles the active spinner if one is running."""
    if _STATUS is not None:
        _STATUS.update(f"[cyan]{message}[/cyan]")
    else:
        err_console.print(f"[dim]{message}[/dim]")


def refusal_panel(message: str, title: str = "Refused") -> Panel:
    return Panel(Text(message, style="bold red"), title=title, border_style="red",
                 box=box.ROUNDED)


def chat_banner(profile: str, token: str | None, profile_name: str | None,
                sources: bool, k: int) -> Panel:
    """Header panel shown once at the top of a chat session."""
    rows = [
        ("chat", f"[cyan]{env('CHAT_MODEL')}[/cyan]"),
        ("embed", f"[cyan]{env('EMBED_MODEL')}[/cyan]"),
    ]
    safety = os.getenv("SAFETY_MODEL")
    rows.append(("safety", f"[cyan]{safety}[/cyan]" if safety else "[dim]off[/dim]"))
    if profile and token:
        rows.append(("profile", "[green]live from HRMS[/green]"))
    elif profile:
        rows.append(("profile", f"[green]{profile_name} (offline)[/green]"))
    else:
        rows.append(("profile", "[dim]none — generic answers[/dim]"))
    rows.append(("sources", "[green]on[/green]" if sources else "[dim]off[/dim]"))
    rows.append(("memory", f"[green]on[/green] · last {MEMORY_WINDOW} turns + running summary · [dim]in RAM only[/dim]"))
    rows.append(("k", f"[cyan]{k}[/cyan] chunks per question"))
    body = Text()
    for label, value in rows:
        body.append(f"  {label:<8} ", style="dim")
        body.append_text(Text.from_markup(value))
        body.append("\n")
    body.append("\n  Ctrl-C or empty line to quit.", style="dim italic")
    return Panel(body, title="[bold]policyqa · chat[/bold]",
                 title_align="left", border_style="magenta", box=box.ROUNDED)
