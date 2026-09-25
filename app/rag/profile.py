"""HRMS profile: fetch, prune, render, fit into the prompt.

Nothing here names a specific HRMS field. It is all shape and size, so a
different HRMS needs no change here.
"""

import json
import os
import re
import sys

from .config import (
    HRMS_TIMEOUT,
    LOGIN_TIMEOUT,
    PROFILE_DIR,
    PROFILE_ENVELOPE,
    PROFILE_MAX_CHARS,
    PROFILE_MAX_LIST,
    PROFILE_MAX_VALUE_CHARS,
    PROFILE_SKIP,
    TOKEN_LOGIN,
    TOKEN_PROMPT,
)
from .ui import waiting


NAME_KEYS = ("name", "full_name", "title", "label")


def collapse_named(value: dict, kept: dict, depth: int, in_list: bool):
    """Reduce a nested record to its name, where the name is all it is worth.

    Two cases only:

    - a person, so `reporting_to` holding a colleague's entire HR record - their
      date of birth, their phone number - becomes just "Aayush Sharma";
    - a lookup with nothing left but its name, so `position: {id, name}` becomes
      "Backend Node Engineer".

    Anything else keeps its fields: a leave type of `{name: "Sick", balance: 3}`
    is about the 3, and collapsing it to "Sick" would throw the answer away.
    Never applied to a list item or to a merged source's root.
    """
    if depth == 0 or in_list:
        return None
    person = value.get("full_name")
    if isinstance(person, str) and person.strip():
        return person.strip()
    if kept and set(kept) <= set(NAME_KEYS):
        for key in NAME_KEYS:
            if isinstance(kept.get(key), str) and kept[key].strip():
                return kept[key].strip()
    return None


def all_zero(value) -> bool:
    """A record whose every number is zero, which says nothing.

    A leave type the HRMS has not configured comes back as allocated 0, used 0,
    balance 0. That is absence of data, but it reads as an entitlement of zero
    and the model will quote it over the handbook's actual grant. Dropping it
    lets the handbook answer, which is what it is authoritative for.

    A genuinely spent balance is not this: allocated 6, used 6, balance 0 still
    has a 6 in it, so it stays.
    """
    if not isinstance(value, dict):
        return False
    numbers = [v for v in value.values() if isinstance(v, (int, float))
               and not isinstance(v, bool)]
    return bool(numbers) and not any(numbers)


def prune_profile(value, depth: int = 0, in_list: bool = False, protect=()):
    """Strip an HRMS response down to what is worth prompt space.

    Drops credentials and bulk by key name, empty values, over-long strings and
    over-long lists. Still names no HRMS field: it is all shape and size, so a
    different HRMS needs no change here.
    """
    if isinstance(value, dict):
        kept = {}
        for key, item in value.items():
            if PROFILE_SKIP.search(key):
                continue
            # A merged source sits under its own label but is a root, not a
            # lookup: collapsing it would throw the whole response away.
            child = 0 if depth == 0 and key in protect else depth + 1
            item = prune_profile(item, child, protect=protect)
            if item not in (None, "", [], {}):
                kept[key] = item
        # Decided after pruning, so "is there anything here but a name?" is
        # asked of what survived, not of what the HRMS sent.
        named = collapse_named(value, kept, depth, in_list)
        return kept if named is None else named
    if isinstance(value, list):
        items = [prune_profile(v, depth + 1, in_list=True, protect=protect)
                 for v in value[:PROFILE_MAX_LIST]]
        return [v for v in items if v not in (None, "", [], {}) and not all_zero(v)]
    if isinstance(value, str) and len(value) > PROFILE_MAX_VALUE_CHARS:
        return value[:PROFILE_MAX_VALUE_CHARS] + "..."
    return value


def unwrap_profile(data: dict) -> dict:
    """Step past a {"status": ..., "message": ..., "data": {...}} envelope.

    Only when the wrapper holds nothing else of substance, so a response whose
    real fields sit at the top level is left alone.
    """
    while isinstance(data, dict):
        inner = next((k for k in PROFILE_ENVELOPE if isinstance(data.get(k), dict)), None)
        if not inner or len(data) > 4:
            return data
        data = data[inner]
    return data


def profile_from(data: dict) -> str:
    """Prune, optionally narrow to chosen fields, render, and cap the result."""
    data = unwrap_profile(data)
    wanted = [f.strip() for f in os.getenv("HRMS_PROFILE_FIELDS", "").split(",") if f.strip()]
    if wanted:
        data = {k: v for k, v in data.items() if k in wanted}
        missing = [f for f in wanted if f not in data]
        if missing:
            print(f"HRMS_PROFILE_FIELDS not in the response: {', '.join(missing)}",
                  file=sys.stderr)
    # Each merged source's label names a root that must survive pruning whole.
    labels = {split_source(s)[0] for s in profile_sources()}
    return fit_profile(prune_profile(data, protect=labels))


def fit_profile(data: dict, cap: int = PROFILE_MAX_CHARS) -> str:
    """Render within the budget, taking the space from whatever is largest.

    Cutting the tail would make the answer depend on the order sources happen to
    be configured in - one long list of leave types could push a band off the
    end. Instead every block gets an equal share, whatever is under its share
    keeps all of it, and only the blocks that are over get trimmed.
    """
    blocks = {key: render_profile({key: value}) for key, value in data.items()}
    if sum(len(b) for b in blocks.values()) <= cap:
        return "\n".join(blocks.values())

    over = dict(blocks)
    budget = cap
    while over:
        share = budget // len(over)
        small = {k: v for k, v in over.items() if len(v) <= share}
        if not small:
            break
        budget -= sum(len(v) for v in small.values())
        over = {k: v for k, v in over.items() if k not in small}
    share = budget // len(over) if over else 0

    trimmed = []
    for key, block in blocks.items():
        if key not in over:
            trimmed.append(block)
            continue
        kept, used = [], 0
        for line in block.splitlines():
            if used + len(line) > share:
                kept.append("  (...)")
                break
            kept.append(line)
            used += len(line) + 1
        trimmed.append("\n".join(kept))
        print(f"Profile: '{key}' was too long for the context and was trimmed.",
              file=sys.stderr)
    return "\n".join(trimmed)


def render_profile(data: dict, indent: int = 0) -> str:
    """JSON as an indented labelled list. Models read that far better than they
    read braces, and it stays generic, so a new HRMS field needs no code here."""
    pad = "  " * indent
    lines = []
    for key, value in data.items():
        label = key.replace("_", " ")
        if isinstance(value, dict):
            lines.append(f"{pad}- {label}:")
            lines.append(render_profile(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{pad}- {label}:")
            for item in value:
                lines.append(f"{pad}  - {flatten(item)}")
        else:
            lines.append(f"{pad}- {label}: {flatten(value)}")
    return "\n".join(lines)


def flatten(value) -> str:
    """One list item on one line: a dict becomes 'Quality: weight 0.6, ...'."""
    if isinstance(value, dict):
        name = value.get("name")
        rest = {k: v for k, v in value.items() if k != "name"}
        body = ", ".join(f"{k.replace('_', ' ')} {flatten(v)}" for k, v in rest.items())
        return f"{name}: {body}" if name else body
    return "none recorded" if value is None else str(value)


def profile_sources() -> list[str]:
    """Where the employee's details live. One HRMS service rarely holds all of
    it - identity in one, leave balances in another, payroll in a third."""
    raw = os.getenv("HRMS_PROFILE_SOURCES") or os.getenv("HRMS_PROFILE_PATH", "api/me")
    return [s.strip() for s in raw.split(",") if s.strip()]


def split_source(source: str) -> tuple[str, str, list[str]]:
    """`leave=https://.../balance|balance_days,band` -> label, url, fields.

    The label groups the fields in the prompt. The optional `|fields` list keeps
    only those keys from that response - worth pinning when a record is large
    and the field that matters sits at the end of it.
    """
    source, _, raw_fields = source.partition("|")
    fields = [f.strip() for f in raw_fields.split("+") if f.strip()]
    label, _, target = source.partition("=")
    if not target:  # no label given; name it after the last useful path segment
        target = label
        label = [p for p in target.rstrip("/").split("/") if p and "{" not in p][-1]
    return label.strip().replace("_", " "), target.strip(), fields


def absolute(target: str) -> str:
    if target.startswith("http://") or target.startswith("https://"):
        return target
    base = os.getenv("HRMS_BASE_URL")
    if not base:
        sys.exit("Set HRMS_BASE_URL in .env, or give each source a full URL.")
    return base.rstrip("/") + "/" + target.lstrip("/")


def scalars(data, into: dict) -> dict:
    """Flatten every scalar by key name, so a later URL can use {some_id}."""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                scalars(value, into)
            elif value not in (None, ""):
                into.setdefault(key, value)
    elif isinstance(data, list):
        for item in data:
            scalars(item, into)
    return into


def fill(url: str, known: dict) -> str | None:
    """Substitute {attendance_employee_id} and friends from what is known so
    far. Returns None when a placeholder cannot be filled."""
    for name in re.findall(r"\{(\w+)\}", url):
        if name not in known:
            print(f"Skipping {url}: no '{name}' in the earlier responses.", file=sys.stderr)
            return None
        url = url.replace("{" + name + "}", str(known[name]))
    return url


def fetch_profile(token: str) -> dict:
    """Read every configured source with the employee's own token and merge.

    The first source's fields sit at the top level; each later one is nested
    under its label. Whatever each returns is used as-is - no field is named
    here, so an HRMS that renames one needs no change.
    """
    merged: dict = {}
    known: dict = {}
    for index, source in enumerate(profile_sources()):
        label, target, fields = split_source(source)
        url = fill(absolute(target), known)
        if url is None:
            continue
        data = unwrap_profile(fetch_json(url, token))
        scalars(data, known)  # before narrowing: a later URL may need a dropped id
        if fields and isinstance(data, dict):
            missing = [f for f in fields if f not in data]
            if missing:
                print(f"{label}: no {', '.join(missing)} in {url}", file=sys.stderr)
            data = {k: v for k, v in data.items() if k in fields}
        if index == 0 and isinstance(data, dict):
            merged.update(data)
        else:
            merged[label] = data
    return merged


def fetch_json(url: str, token: str) -> dict:
    import urllib.error
    import urllib.request

    if not url.startswith("https://") and "//127.0.0.1" not in url and "//localhost" not in url:
        sys.exit(f"Refusing to send the HRMS token over plain HTTP to {url}. Use https://.")
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    # Some HRMS services want a header of their own alongside the session, such
    # as an x-client-id. HRMS_HEADERS carries them as "Name: value" pairs.
    for pair in os.getenv("HRMS_HEADERS", "").split(","):
        name, sep, value = pair.partition(":")
        if sep and name.strip():
            headers[name.strip()] = value.strip()
    request = urllib.request.Request(url, headers=headers)
    host = url.split("/")[2] if "//" in url else url
    try:
        with waiting(f"reading {host}"), urllib.request.urlopen(
            request, timeout=HRMS_TIMEOUT
        ) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # The body usually says exactly what is wrong, and every HRMS picks its
        # own code for a bad session - 401, 403, and 422 are all in use - so
        # quote the server rather than guessing from the number alone.
        try:
            detail = exc.read(500).decode("utf-8", "replace").strip()
        except Exception:  # noqa: BLE001 - a body is a nicety, not a requirement
            detail = ""
        hint = "\nLog in again with --login." if exc.code in (401, 403, 422) else ""
        # The token itself is never printed.
        sys.exit(f"HRMS returned {exc.code} for {url}.\n{detail}{hint}")
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(f"Cannot reach the HRMS at {url}: {exc}")
    except json.JSONDecodeError:
        sys.exit(f"HRMS did not return JSON from {url}. Is that source right?")


def read_profile_file(name: str) -> dict:
    """A saved response, for working with no HRMS reachable. PROFILE_DIR is
    gitignored: real employee data must not be committed."""
    path = os.path.join(PROFILE_DIR, f"{name.lower()}.json")
    if not os.path.exists(path):
        sys.exit(f"No profile file at {path}. Use --token to read from the HRMS.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def resolve_token(token: str | None) -> str | None:
    """Turn the --login / --prompt-token sentinels into an actual session."""
    if token == TOKEN_LOGIN:
        from . import hrms_login

        login_url = os.getenv("HRMS_LOGIN_URL") or os.getenv("HRMS_BASE_URL")
        if not login_url:
            sys.exit("Set HRMS_LOGIN_URL (or HRMS_BASE_URL) in .env to log in.")
        token = hrms_login.capture_token(
            login_url,
            api_hint=os.getenv("HRMS_API_HINT", ""),
            timeout=LOGIN_TIMEOUT,
        )
    if token == TOKEN_PROMPT:
        import getpass

        token = getpass.getpass("HRMS token (not echoed): ")
    # An empty token means HRMS_TOKEN is set but blank, or --token was passed
    # with nothing. Falling back to a generic answer there would look
    # personalised and quietly not be, so say so instead.
    if token is not None and not token.strip():
        sys.exit("--token (or HRMS_TOKEN) is empty. Omit it entirely for a generic answer.")
    return token


def profile_text(token: str | None, name: str | None) -> str:
    token = resolve_token(token)
    if token:
        return profile_from(fetch_profile(token))
    if name:
        return profile_from(read_profile_file(name))
    return ""
