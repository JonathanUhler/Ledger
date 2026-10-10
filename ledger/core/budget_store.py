"""
A SQLite-backed global store for one-time (and future repeating) budgets.
"""


from contextlib import contextmanager
import hashlib
import sqlite3
from typing import Iterator

from ledger import config
from ledger.convert import convert_date_to_string, convert_string_to_date
from ledger.core import database
from ledger.errors import IntegrityError, InvalidBudgetError, MissingBudgetError
from ledger.models import Budget, BudgetKind
from ledger.version import APP_VERSION, BUDGET_MODEL_VERSION


_SCHEMA: str = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS budgets (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    goal_cents  INTEGER NOT NULL,
    category    TEXT NOT NULL,
    account     TEXT,
    percent_bps INTEGER NOT NULL,
    start_date  TEXT NOT NULL
);
"""

_EDITABLE_FIELDS: frozenset[str] = frozenset({
    "name", "kind", "goal_cents", "category", "account", "percent_bps", "start_date",
})


def _compute_checksum(conn: sqlite3.Connection) -> str:
    parts: list[str] = []
    for row in conn.execute(
        "SELECT id, name, kind, goal_cents, category, account, percent_bps, start_date "
        "FROM budgets ORDER BY id"
    ):
        parts.append("|".join([
            row["id"], row["name"], row["kind"], str(row["goal_cents"]), row["category"],
            row["account"] if (row["account"] is not None) else "",
            str(row["percent_bps"]), row["start_date"],
        ]))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _verify_integrity(conn: sqlite3.Connection) -> None:
    result: sqlite3.Row | None = conn.execute("PRAGMA integrity_check").fetchone()
    if (result is None or result[0] != "ok"):
        raise IntegrityError("budget store failed PRAGMA integrity_check")
    stored: str | None = database.read_meta(conn, "content_checksum")
    actual: str = _compute_checksum(conn)
    if (stored is not None and stored != actual):
        raise IntegrityError("budget store checksum mismatch; data may be corrupt "
                             "(restore from .bak)")


def _row_to_budget(row: sqlite3.Row) -> Budget:
    return Budget(
        id = row["id"],
        name = row["name"],
        kind = BudgetKind(row["kind"]),
        goal_cents = row["goal_cents"],
        category = row["category"],
        account = row["account"],
        percent_bps = row["percent_bps"],
        start_date = convert_string_to_date(row["start_date"]),
    )


class BudgetStore:

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn: sqlite3.Connection = conn

    @classmethod
    def open(cls) -> "BudgetStore":
        db_path = config.budget_db_path()
        is_new: bool = not db_path.exists()
        if (is_new):
            config.get_ledger_home().mkdir(parents = True, exist_ok = True)
        conn: sqlite3.Connection = database.connect(db_path)
        if (is_new):
            conn.executescript(_SCHEMA)
            database.write_meta(conn, "data_model_version", str(BUDGET_MODEL_VERSION))
            database.write_meta(conn, "app_version", APP_VERSION)
            database.write_meta(conn, "content_checksum", _compute_checksum(conn))
            return cls(conn)
        version: int = int(database.read_meta(conn, "data_model_version") or "0")
        if (version > BUDGET_MODEL_VERSION):
            conn.close()
            raise InvalidBudgetError(
                f"budget store uses data model v{version}, newer than this app "
                f"(v{BUDGET_MODEL_VERSION})"
            )
        _verify_integrity(conn)
        return cls(conn)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _mutate(self) -> Iterator[sqlite3.Connection]:
        database.backup_file(config.budget_db_path(), config.budget_backup_path())
        try:
            self._conn.execute("BEGIN")
            yield self._conn
            database.write_meta(self._conn, "content_checksum", _compute_checksum(self._conn))
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def list_budgets(self) -> list[Budget]:
        return [_row_to_budget(row)
                for row in self._conn.execute(
                    "SELECT * FROM budgets ORDER BY name ASC, id ASC")]

    def get_budget(self, budget_id: str) -> Budget:
        row: sqlite3.Row | None = self._conn.execute(
            "SELECT * FROM budgets WHERE id = ?", (budget_id,)).fetchone()
        if (row is None):
            raise MissingBudgetError(f"budget '{budget_id}' does not exist")
        return _row_to_budget(row)

    def add_budget(self, budget: Budget) -> None:
        with self._mutate() as conn:
            conn.execute(
                "INSERT INTO budgets (id, name, kind, goal_cents, category, account, "
                "percent_bps, start_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (budget.id, budget.name, budget.kind.value, budget.goal_cents, budget.category,
                 budget.account, budget.percent_bps, convert_date_to_string(budget.start_date)),
            )

    def edit_budget(self, budget_id: str, **fields: object) -> Budget:
        budget: Budget = self.get_budget(budget_id)
        for key, value in fields.items():
            if (key not in _EDITABLE_FIELDS):
                raise InvalidBudgetError(f"field '{key}' is not editable")
            setattr(budget, key, value)
        with self._mutate() as conn:
            conn.execute(
                "UPDATE budgets SET name = ?, kind = ?, goal_cents = ?, category = ?, "
                "account = ?, percent_bps = ?, start_date = ? WHERE id = ?",
                (budget.name, budget.kind.value, budget.goal_cents, budget.category,
                 budget.account, budget.percent_bps, convert_date_to_string(budget.start_date),
                 budget.id),
            )
        return budget

    def remove_budget(self, budget_id: str) -> None:
        self.get_budget(budget_id)   # raises MissingBudgetError if absent
        with self._mutate() as conn:
            conn.execute("DELETE FROM budgets WHERE id = ?", (budget_id,))
