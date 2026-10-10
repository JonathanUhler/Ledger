"""
A SQLite-backed store for a single account.
"""


from contextlib import contextmanager
import datetime
import sqlite3
from typing import Iterator

from ledger import config
from ledger.convert import convert_date_to_string, convert_string_to_date
from ledger.core import database
from ledger.errors import AccountFileError, AccountParseError, MissingAccountError
from ledger.models import AccountSummary, Transaction, compute_content_hash
from ledger.version import APP_VERSION, DATA_MODEL_VERSION


_EDITABLE_FIELDS: frozenset[str] = frozenset({
    "date", "cents", "statement_memo", "account", "user_memo", "category", "reconciled",
})
_HASH_FIELDS: frozenset[str] = frozenset({"date", "cents", "statement_memo", "account"})


def _row_to_transaction(row: sqlite3.Row) -> Transaction:
    return Transaction(
        id = row["id"],
        date = convert_string_to_date(row["date"]),
        cents = row["cents"],
        statement_memo = row["statement_memo"],
        account = row["account"],
        user_memo = row["user_memo"],
        category = row["category"],
        reconciled = bool(row["reconciled"]),
        content_hash = row["content_hash"],
    )


class Account:

    def __init__(self, name: str, conn: sqlite3.Connection) -> None:
        self._name: str = name
        self._conn: sqlite3.Connection = conn

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def create(cls, name: str, statement_type: str) -> "Account":
        config.validate_account_name(name)
        db_path = config.account_db_path(name)
        if (db_path.exists()):
            raise AccountFileError(f"account '{name}' already exists")
        config.get_accounts_dir().mkdir(parents = True, exist_ok = True)
        conn: sqlite3.Connection = database.connect(db_path)
        database.create_schema(conn)
        database.write_meta(conn, "data_model_version", str(DATA_MODEL_VERSION))
        database.write_meta(conn, "app_version", APP_VERSION)
        database.write_meta(conn, "account_name", name)
        database.write_meta(conn, "statement_type", statement_type)
        database.write_meta(conn, "content_checksum", database.compute_checksum(conn))
        return cls(name, conn)

    @classmethod
    def open(cls, name: str) -> "Account":
        config.validate_account_name(name)
        db_path = config.account_db_path(name)
        if (not db_path.exists()):
            raise MissingAccountError(f"account '{name}' does not exist")
        conn: sqlite3.Connection = database.connect(db_path)
        version: int = int(database.read_meta(conn, "data_model_version") or "0")
        if (version > DATA_MODEL_VERSION):
            raise AccountParseError(
                f"account '{name}' uses data model v{version}, newer than this app "
                f"(v{DATA_MODEL_VERSION})"
            )
        database.verify_integrity(conn)
        return cls(name, conn)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _mutate(self) -> Iterator[sqlite3.Connection]:
        database.backup_file(config.account_db_path(self._name),
                             config.account_backup_path(self._name))
        try:
            self._conn.execute("BEGIN")
            yield self._conn
            database.write_meta(self._conn, "content_checksum",
                                database.compute_checksum(self._conn))
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def get_statement_type(self) -> str:
        return database.read_meta(self._conn, "statement_type") or ""

    def set_statement_type(self, statement_type: str) -> None:
        with self._mutate() as conn:
            database.write_meta(conn, "statement_type", statement_type)

    def list_categories(self) -> list[str]:
        return [row["name"]
                for row in self._conn.execute("SELECT name FROM categories ORDER BY name")]

    def add_category(self, name: str) -> None:
        with self._mutate() as conn:
            conn.execute("INSERT OR IGNORE INTO categories (name) VALUES (?)", (name,))

    def list_transactions(self) -> list[Transaction]:
        return [_row_to_transaction(row)
                for row in self._conn.execute(
                    "SELECT * FROM transactions ORDER BY date ASC, id ASC")]

    def find_by_content_hash(self, content_hash: str) -> list[Transaction]:
        return [_row_to_transaction(row)
                for row in self._conn.execute(
                    "SELECT * FROM transactions WHERE content_hash = ?", (content_hash,))]

    def list_category_since(self, category: str, start: datetime.date) -> list[Transaction]:
        return [_row_to_transaction(row)
                for row in self._conn.execute(
                    "SELECT * FROM transactions WHERE category = ? AND date >= ? "
                    "ORDER BY date ASC, id ASC",
                    (category, convert_date_to_string(start)))]

    def add_transaction(self, txn: Transaction) -> None:
        self.add_many([txn])

    def add_many(self, txns: list[Transaction]) -> None:
        with self._mutate() as conn:
            for txn in txns:
                conn.execute(
                    "INSERT INTO transactions (id, date, cents, statement_memo, account, "
                    "user_memo, category, reconciled, content_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (txn.id, convert_date_to_string(txn.date), txn.cents, txn.statement_memo,
                     txn.account, txn.user_memo, txn.category, int(txn.reconciled),
                     txn.content_hash),
                )

    def edit_transaction(self, txn_id: str, **fields: object) -> Transaction:
        rows = self._conn.execute("SELECT * FROM transactions WHERE id = ?", (txn_id,)).fetchall()
        if (len(rows) == 0):
            raise AccountParseError(f"transaction '{txn_id}' does not exist")
        txn: Transaction = _row_to_transaction(rows[0])

        for key, value in fields.items():
            if (key not in _EDITABLE_FIELDS):
                raise AccountParseError(f"field '{key}' is not editable")
            setattr(txn, key, value)
        if (any(key in _HASH_FIELDS for key in fields)):
            txn.content_hash = compute_content_hash(txn.date, txn.cents, txn.statement_memo,
                                                    txn.account)

        with self._mutate() as conn:
            conn.execute(
                "UPDATE transactions SET date = ?, cents = ?, statement_memo = ?, account = ?, "
                "user_memo = ?, category = ?, reconciled = ?, content_hash = ? WHERE id = ?",
                (convert_date_to_string(txn.date), txn.cents, txn.statement_memo, txn.account,
                 txn.user_memo, txn.category, int(txn.reconciled), txn.content_hash, txn.id),
            )
        return txn

    def remove_transaction(self, txn_id: str) -> None:
        with self._mutate() as conn:
            conn.execute("DELETE FROM transactions WHERE id = ?", (txn_id,))

    def get_balance(self) -> int:
        row = self._conn.execute("SELECT COALESCE(SUM(cents), 0) AS total FROM transactions") \
            .fetchone()
        return row["total"]

    def summary(self) -> AccountSummary:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(cents), 0) AS total, "
            "MIN(date) AS earliest, MAX(date) AS latest FROM transactions"
        ).fetchone()
        earliest: datetime.date | None = \
            convert_string_to_date(row["earliest"]) if (row["earliest"] is not None) else None
        latest: datetime.date | None = \
            convert_string_to_date(row["latest"]) if (row["latest"] is not None) else None
        return AccountSummary(
            name = self._name,
            balance_cents = row["total"],
            count = row["n"],
            earliest = earliest,
            latest = latest,
            statement_type = self.get_statement_type(),
        )
