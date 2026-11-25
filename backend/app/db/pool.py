from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.log import get_logger

log = get_logger("db")
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_pool: ConnectionPool | None = None


def apply_schema(database_url: str) -> None:
    """Create the extension and tables. Runs before the pool opens, because
    register_vector needs the `vector` type to exist already."""
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    log.info("schema applied")


def open_pool(database_url: str) -> ConnectionPool:
    global _pool
    _pool = ConnectionPool(
        database_url,
        min_size=1,
        max_size=10,
        configure=lambda conn: register_vector(conn),
        open=True,
    )
    _pool.wait(timeout=30)
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def get_pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("database pool is not open")
    return _pool
