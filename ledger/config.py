"""
Filesystem layout for ledger data, overridable via the LEDGER_HOME env var.
"""


import os
from pathlib import Path
import re

from ledger.errors import AccountParseError


_NAME_PATTERN: re.Pattern = re.compile(r"^[A-Za-z0-9 _-]+$")


def get_ledger_home() -> Path:
    override: str | None = os.environ.get("LEDGER_HOME")
    if (override is not None):
        return Path(override)
    return Path.home() / ".ledger"


def get_accounts_dir() -> Path:
    return get_ledger_home() / "accounts"


def get_types_dir() -> Path:
    return get_ledger_home() / "types"


def validate_account_name(name: str) -> None:
    if (not isinstance(name, str) or name.strip() == "" or not _NAME_PATTERN.match(name)):
        raise AccountParseError(f"invalid account name '{name}'")


def account_db_path(name: str) -> Path:
    validate_account_name(name)
    return get_accounts_dir() / f"{name}.db"


def account_backup_path(name: str) -> Path:
    validate_account_name(name)
    return get_accounts_dir() / f"{name}.db.bak"


def type_path(name: str) -> Path:
    return get_types_dir() / f"{name}.yaml"


def budget_db_path() -> Path:
    return get_ledger_home() / "budgets.db"


def budget_backup_path() -> Path:
    return get_ledger_home() / "budgets.db.bak"
