FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY app ./app
COPY knowledge ./knowledge

RUN uv sync --frozen

CMD ["uv", "run", "python", "app/main.py", "stats"]