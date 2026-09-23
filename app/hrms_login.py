"""Capture an HRMS access token by letting the employee log in themselves.

The HRMS cannot be changed, so there is no OAuth flow to use. Instead a real
browser opens on the real login page, the employee signs in as they always do -
SSO, MFA, whatever it takes - and the token their session is issued is read back
out. Nothing types their password, and nothing stores it.

Two ways to read the token, in order:

1. Watch the requests the HRMS frontend makes and take the `Authorization`
   header off one. This is the token the API actually accepts, whatever the
   frontend chose to keep it in - memory, a cookie, anywhere.
2. Failing that, scan localStorage and sessionStorage for a JWT.

Needs the optional browser extra:  uv sync --extra browser
                                   uv run playwright install chromium
"""

import re
import sys
import time

# A JWT: three base64url segments, and the header almost always starts `eyJ`.
JWT = re.compile(r"^eyJ[\w-]+\.[\w-]+\.[\w-]+$")
POLL_SECONDS = 0.5


def looks_like_jwt(value: object) -> bool:
    return isinstance(value, str) and bool(JWT.match(value.strip()))


def start_browser(playwright, headless: bool):
    """Opened but not yet navigated: the caller attaches its listeners first, so
    nothing that happens on the very first page load is missed."""
    browser = playwright.chromium.launch(headless=headless)
    context = browser.new_context()
    return browser, context, context.new_page()


def wait_for(context, page, done, timeout: int) -> None:
    """Poll until done() is true, the window is closed, or time runs out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if done() or page.is_closed() or not context.pages:
            return
        try:
            page.wait_for_timeout(int(POLL_SECONDS * 1000))
        except Exception:  # noqa: BLE001 - the window went away mid-wait
            return


def discover(login_url: str, api_hint: str = "", timeout: int = 600,
             headless: bool = False) -> list[tuple]:
    """Log which API calls the HRMS frontend makes, to find the profile endpoint.

    Only the shape is reported - method, URL, status and the top-level JSON keys.
    Values are deliberately not printed: the point is to identify an endpoint,
    not to spill somebody's salary into a terminal.
    """
    playwright_module = _import_playwright()
    seen: dict[str, tuple] = {}

    def on_response(response) -> None:
        if api_hint not in response.url or response.request.method != "GET":
            return
        if "json" not in (response.headers.get("content-type") or ""):
            return
        keys: object = "-"
        try:
            body = response.json()
            if isinstance(body, dict):
                keys = ", ".join(list(body)[:12]) or "(empty object)"
            elif isinstance(body, list):
                keys = f"[list of {len(body)}]"
        except Exception:  # noqa: BLE001 - streamed or already-consumed bodies
            keys = "(body unavailable)"
        seen.setdefault(response.url, (response.request.method, response.url,
                                       response.status, keys))

    with playwright_module() as playwright:
        browser, context, page = start_browser(playwright, headless)
        context.on("response", on_response)
        page.goto(login_url)
        print(
            f"A browser is open at {login_url}.\n"
            "Log in, then visit your profile and leave pages so their API calls "
            "are seen.\nClose the window when done.",
            file=sys.stderr,
        )
        wait_for(context, page, lambda: False, timeout)
        browser.close()

    return sorted(seen.values(), key=lambda row: row[1])


def _import_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit(
            "Browser login needs the optional extra:\n"
            "  uv sync --extra browser && uv run playwright install chromium"
        )
    return sync_playwright


def from_storage(context) -> str | None:
    """Last resort: any JWT sitting in web storage, longest first - a refresh
    token is usually the shorter of the two."""
    found = []
    for origin in context.storage_state().get("origins", []):
        for entry in origin.get("localStorage", []) + origin.get("sessionStorage", []):
            value = entry.get("value", "")
            if looks_like_jwt(value):
                found.append(value)
            elif value.startswith("{"):
                # Some frontends keep the whole auth response under one key.
                found.extend(re.findall(r'"(eyJ[\w-]+\.[\w-]+\.[\w-]+)"', value))
    return max(found, key=len) if found else None


def capture_token(login_url: str, api_hint: str = "", timeout: int = 300,
                  headless: bool = False) -> str:
    """Open the login page and return the token, or exit with why it failed."""
    playwright_module = _import_playwright()
    captured: list[str] = []

    def on_request(request) -> None:
        if captured:
            return
        header = request.headers.get("authorization", "")
        # api_hint keeps us from grabbing a bearer meant for a third party,
        # such as an analytics or error-reporting endpoint.
        if header.lower().startswith("bearer ") and api_hint in request.url:
            token = header.split(None, 1)[1].strip()
            if token:
                captured.append(token)

    with playwright_module() as playwright:
        browser, context, page = start_browser(playwright, headless)
        context.on("request", on_request)
        page.goto(login_url)
        print(
            f"A browser window is open at {login_url}.\n"
            "Log in there as you normally would; this waits for your session.",
            file=sys.stderr,
        )
        wait_for(context, page, lambda: bool(captured), timeout)

        token = captured[0] if captured else None
        if not token and not page.is_closed():
            token = from_storage(context)
        browser.close()

    if not token:
        sys.exit(
            "No token seen. Either the login did not finish, or the HRMS does "
            "not send a bearer token.\nCheck HRMS_API_HINT, or copy the token "
            "from devtools and use --prompt-token."
        )
    return token
