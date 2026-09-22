import os
import sys

import psycopg
from dotenv import load_dotenv


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


def main():
    load_dotenv()

    missing = [
        name
        for name in ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_PORT")
        if not os.getenv(name)
    ]
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
