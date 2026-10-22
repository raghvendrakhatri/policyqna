"""The HRMS login the employee asked the web UI to remember.

Kept as plain JSON in a file only its owner can read, outside git, so the
server can log in again by itself each time it starts. Logging out deletes it.
"""

import json
import os

from .config import _REPO_DIR

PATH = os.getenv("SAVED_LOGIN_FILE", os.path.join(_REPO_DIR, ".hrms_login.json"))


def save(email: str, password: str, subdomain: str) -> None:
    # Created 0600 rather than chmod-ed after, so it is never readable by others.
    fd = os.open(PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"email": email, "password": password, "subdomain": subdomain}, f)


def load() -> tuple[str, str, str] | None:
    """(email, password, subdomain), or None when nothing is saved or the
    file is unreadable."""
    try:
        with open(PATH) as f:
            saved = json.load(f)
        return saved["email"], saved["password"], saved["subdomain"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def forget() -> None:
    try:
        os.remove(PATH)
    except FileNotFoundError:
        pass
