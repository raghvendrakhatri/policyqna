"""Health check: can we reach Postgres and Ollama? The app itself is rag.py."""

import os
import sys
import urllib.error
import urllib.request

import psycopg
from dotenv import load_dotenv

import rag


def ping_database() -> str:
    with psycopg.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT"),
        connect_timeout=5,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            return cur.fetchone()[0]


def ping_ollama(base_url: str) -> list[str]:
    with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as response:
        import json

        return [m["name"] for m in json.load(response).get("models", [])]


def main():
    load_dotenv()

    missing = [name for name in rag.REQUIRED_ENV if not os.getenv(name)]
    if missing:
        print(
            f"Missing env vars: {', '.join(missing)}. Copy .env.example to .env.",
            file=sys.stderr,
        )
        return 1

    try:
        version = ping_database()
    except psycopg.Error as exc:
        print(f"Database ping failed: {exc}", file=sys.stderr)
        return 1
    print(f"Database OK: {version}")

    base_url = os.getenv("OLLAMA_BASE_URL")
    try:
        models = ping_ollama(base_url)
    except (urllib.error.URLError, OSError) as exc:
        print(f"Ollama ping failed at {base_url}: {exc}", file=sys.stderr)
        return 1
    print(f"Ollama OK at {base_url}: {len(models)} model(s) available")

    for var in ("CHAT_MODEL", "EMBED_MODEL"):
        name = os.getenv(var)
        if name and name not in models:
            print(f"Warning: {var}={name} is not pulled. `ollama pull {name}`", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
