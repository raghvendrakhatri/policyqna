"""Kept so `python -m app.rag` still works; the entry point is app/main.py."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
