"""Entry point for `python -m app`; the same as `python app/main.py`."""

import sys

from .main import run

if __name__ == "__main__":
    sys.exit(run())
