FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

WORKDIR /srv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# Install dependencies first so this layer is cached between code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY backend ./backend

WORKDIR /srv/backend
RUN useradd --create-home app
USER app

EXPOSE 8000
# Hosts like Render set $PORT. --proxy-headers makes the client IP (used by rate limits) the real
# one behind the platform's load balancer.
CMD ["sh", "-c", "exec /srv/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
