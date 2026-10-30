"""Entry point: `uv run python app/main.py <command>`. See `--help`.

Runs the health check first (Postgres, Ollama, models) and only hands over to
the CLI once it passes.
"""

import sys

try:
    from .rag.cli import cmd_check, main  # python -m app
except ImportError:
    from rag.cli import cmd_check, main  # python app/main.py

# `check` is the health check itself, and help needs neither service up.
SKIP_CHECK = {"check", "-h", "--help"}


def run() -> int:
    if not SKIP_CHECK & set(sys.argv[1:2]):
        status = cmd_check()
        if status:
            return status
    return main()


if __name__ == "__main__":
    sys.exit(run())
