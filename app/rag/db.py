"""Postgres + pgvector: connection string, store, raw SQL, index inspection."""

import os

from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

from .config import COLLECTION, env
from .models import embeddings


def dsn() -> str:
    """SQLAlchemy URL for the psycopg3 driver, which PGVector requires."""
    return (
        f"postgresql+psycopg://{env('POSTGRES_USER')}:{env('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{env('POSTGRES_PORT')}"
        f"/{env('POSTGRES_DB')}"
    )


def store() -> PGVector:
    return PGVector(
        embeddings=embeddings(),
        connection=dsn(),
        collection_name=COLLECTION,
        use_jsonb=True,
    )


# Scopes a raw statement to this collection, since the tables are shared.
IN_COLLECTION = (
    "collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = :collection)"
)


def sql(statement: str, **params):
    """PGVector has no delete-by-metadata or count, so those go through SQL."""
    params.setdefault("collection", COLLECTION)
    with create_engine(dsn()).begin() as conn:
        result = conn.execute(text(statement), params)
        return result.fetchall() if result.returns_rows else []


def counts_by_source() -> list[tuple[str, int]] | None:
    """None when the store has never been created."""
    exists = sql("SELECT to_regclass('public.langchain_pg_embedding')")[0][0]
    if exists is None:
        return None
    return [
        (row[0], row[1])
        for row in sql(
            "SELECT cmetadata->>'source', count(*) FROM langchain_pg_embedding"
            f" WHERE {IN_COLLECTION} GROUP BY 1 ORDER BY 1"
        )
    ]


def indexed() -> bool:
    return bool(counts_by_source())
