"""
Low-level SQLite primitives: schema, meta, checksum, integrity, backup.
"""


import hashlib
from pathlib import Path
import shutil
import sqlite3

from ledger.errors import IntegrityError


_SCHEMA: str = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS categories (
    name TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS transactions (
    id TEXT PRIMARY KEY,
    date TEXT NOT NULL,
    cents INTEGER NOT NULL,
    statement_memo TEXT NOT NULL,
    account TEXT NOT NULL,
    user_memo TEXT NOT NULL,
    category TEXT NOT NULL,
    reconciled INTEGER NOT NULL,
    content_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_txn_content_hash ON transactions (content_hash);
"""


def connect(path: Path) -> sqlite3.Connection:
    conn: sqlite3.Connection = sqlite3.connect(str(path), isolation_level = None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


def read_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row: sqlite3.Row | None = conn.execute(
        "SELECT value FROM meta WHERE key = ?", (key,)
    ).fetchone()
    return row["value"] if (row is not None) else None


def write_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def compute_checksum(conn: sqlite3.Connection) -> str:
    parts: list[str] = []
    for row in conn.execute(
        "SELECT id, date, cents, statement_memo, account, user_memo, category, "
        "reconciled, content_hash FROM transactions ORDER BY id"
    ):
        parts.append("|".join([
            row["id"], row["date"], str(row["cents"]), row["statement_memo"], row["account"],
            row["user_memo"], row["category"], str(row["reconciled"]), row["content_hash"],
        ]))
    for row in conn.execute("SELECT name FROM categories ORDER BY name"):
        parts.append(f"cat:{row['name']}")
    blob: str = "\n".join(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def verify_integrity(conn: sqlite3.Connection) -> None:
    result: sqlite3.Row | None = conn.execute("PRAGMA integrity_check").fetchone()
    if (result is None or result[0] != "ok"):
        raise IntegrityError("account database failed PRAGMA integrity_check")
    stored: str | None = read_meta(conn, "content_checksum")
    actual: str = compute_checksum(conn)
    if (stored is not None and stored != actual):
        raise IntegrityError("account checksum mismatch; data may be corrupt (restore from .bak)")


def backup_file(src: Path, dst: Path) -> None:
    if (src.exists()):
        shutil.copyfile(src, dst)
