"""
Migrate a v1 YAML account to a v2 SQLite account, non-destructively.
"""


from dataclasses import dataclass

import yaml

from ledger import config
from ledger.convert import convert_string_to_date
from ledger.core.account import Account
from ledger.errors import MigrationError
from ledger.models import Transaction, new_transaction


STATEMENT_TYPE_MAP: dict[str, str] = {
    "WELLS_FARGO_CSV": "wells_fargo_csv",
    "CHARLES_SCHWAB_CSV": "charles_schwab_csv",
}

GRAMMAR_TEMPLATES: dict[str, dict] = {
    "wells_fargo_csv": {
        "name": "wells_fargo_csv",
        "kind": "csv",
        "csv": {"num_header_rows": 1, "date": {"index": 0, "format": "%m/%d/%Y"},
                "amount": {"index": 2}, "memo": {"index": 1}},
    },
    "charles_schwab_csv": {
        "name": "charles_schwab_csv",
        "kind": "csv",
        "csv": {"num_header_rows": 1, "date": {"index": 0, "format": "%m/%d/%Y"},
                "amount": {"index": 7}, "memo": {"index": 3}, "columns": {"action": 1}},
        "filter": {"all": [{"column": "action", "op": "in",
                            "values": ["MoneyLink Transfer", "Reinvest Dividend",
                                       "Credit Interest", "Short Term Cap Gain Reinvest"]}]},
        "transform": [{"when": {"column": "action", "op": "equals",
                                "value": "MoneyLink Transfer"},
                       "target": "date", "op": "split", "sep": " as of ", "take": 0}],
    },
}


@dataclass
class MigrationReport:
    account: str
    migrated: bool
    skipped: bool
    count: int
    v1_sum: int
    v2_sum: int
    message: str


def read_v1_account(name: str) -> tuple[dict, list[dict]]:
    account_dir = config.get_ledger_home() / name
    metadata_path = account_dir / f"{name}.metadata"
    transactions_path = account_dir / f"{name}.transactions"
    if (not metadata_path.exists() or not transactions_path.exists()):
        raise MigrationError(f"'{name}' is not a v1 account")
    try:
        metadata: dict = yaml.safe_load(metadata_path.read_text(encoding = "utf-8")) or {}
        transactions: list[dict] = \
            yaml.safe_load(transactions_path.read_text(encoding = "utf-8")) or []
    except yaml.YAMLError as error:
        raise MigrationError(f"cannot read v1 account '{name}': {error}") from error
    return metadata, transactions


def seed_grammar(statement_type_enum: str) -> str:
    grammar_name: str | None = STATEMENT_TYPE_MAP.get(statement_type_enum)
    if (grammar_name is None):
        raise MigrationError(f"unknown v1 statement type '{statement_type_enum}'")
    config.get_types_dir().mkdir(parents = True, exist_ok = True)
    path = config.type_path(grammar_name)
    if (not path.exists()):
        path.write_text(yaml.safe_dump(GRAMMAR_TEMPLATES[grammar_name], sort_keys = False),
                        encoding = "utf-8")
    return grammar_name


def _discard(name: str) -> None:
    for path in (config.account_db_path(name), config.account_backup_path(name)):
        if (path.exists()):
            path.unlink()


def migrate_account(name: str, force: bool = False) -> MigrationReport:
    metadata, v1_transactions = read_v1_account(name)

    db_path = config.account_db_path(name)
    if (db_path.exists()):
        if (not force):
            return MigrationReport(account = name, migrated = False, skipped = True, count = 0,
                                   v1_sum = 0, v2_sum = 0,
                                   message = f"v2 account '{name}' already exists; skipped")
        _discard(name)

    grammar_name: str = seed_grammar(metadata.get("statement_type"))
    v1_categories: list[str] = list(metadata.get("categories", []))

    account: Account = Account.create(name, grammar_name)
    try:
        for category in v1_categories:
            account.add_category(category)

        transactions: list[Transaction] = []
        for row in v1_transactions:
            transactions.append(new_transaction(
                date = convert_string_to_date(row["date"]),
                cents = int(row["cents"]),
                statement_memo = row.get("statement_memo") or "",
                account = name,
                user_memo = row.get("user_memo") or "",
                category = row.get("category") or "",
                reconciled = bool(row.get("reconciled")),
            ))
        account.add_many(transactions)

        v1_sum: int = sum(int(row["cents"]) for row in v1_transactions)
        v2_sum: int = account.get_balance()
        v2_count: int = account.summary().count
        v2_categories: set[str] = set(account.list_categories())
        problems: list[str] = []
        if (v2_count != len(v1_transactions)):
            problems.append(f"count {v2_count} != {len(v1_transactions)}")
        if (v2_sum != v1_sum):
            problems.append(f"sum {v2_sum} != {v1_sum}")
        if (v2_categories != set(v1_categories)):
            problems.append("category sets differ")
        if (len(problems) > 0):
            account.close()
            _discard(name)
            raise MigrationError(f"migration of '{name}' failed sanity checks: "
                                 + "; ".join(problems))
    except MigrationError:
        raise
    except Exception as error:
        account.close()
        _discard(name)
        raise MigrationError(f"migration of '{name}' failed: {error}") from error

    account.close()
    return MigrationReport(account = name, migrated = True, skipped = False,
                           count = len(v1_transactions), v1_sum = v1_sum, v2_sum = v2_sum,
                           message = f"migrated {len(v1_transactions)} transactions")
