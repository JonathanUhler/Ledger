import pathlib

import pytest
import yaml

from ledger import config
from ledger.migrate import upgrade, v1v2
from ledger.core.account import Account
from ledger.errors import MigrationError


def _write_v1(home: pathlib.Path, name: str, statement_type: str,
              categories: list[str], txns: list[dict]) -> None:
    account_dir: pathlib.Path = home / name
    account_dir.mkdir()
    (account_dir / f"{name}.metadata").write_text(
        yaml.safe_dump({"statement_type": statement_type, "categories": categories}),
        encoding = "utf-8")
    (account_dir / f"{name}.transactions").write_text(yaml.safe_dump(txns), encoding = "utf-8")


def _v1_txn(date: str, cents: int, category: str) -> dict:
    return {"date": date, "cents": cents, "statement_memo": "M", "account": "checking",
            "user_memo": "u", "category": category, "reconciled": True}


def test_migrate_happy_path(ledger_home: pathlib.Path) -> None:
    _write_v1(ledger_home, "checking", "WELLS_FARGO_CSV", ["Food", "Rent"],
              [_v1_txn("2026-01-01", -100, "Food"), _v1_txn("2026-02-01", -200, "Rent")])
    reports = upgrade.upgrade()
    assert len(reports) == 1 and reports[0].migrated is True
    assert reports[0].count == 2 and reports[0].v1_sum == -300 == reports[0].v2_sum
    # v2 db exists, grammar seeded, v1 files untouched
    assert config.account_db_path("checking").exists()
    assert config.type_path("wells_fargo_csv").exists()
    assert (ledger_home / "checking" / "checking.transactions").exists()
    acct: Account = Account.open("checking")
    assert acct.get_balance() == -300
    assert acct.list_categories() == ["Food", "Rent"]
    acct.close()


def test_migrate_skips_existing_without_force(ledger_home: pathlib.Path) -> None:
    _write_v1(ledger_home, "checking", "WELLS_FARGO_CSV", [], [_v1_txn("2026-01-01", -100, "")])
    upgrade.upgrade()
    reports = upgrade.upgrade()               # second run
    assert reports[0].skipped is True and reports[0].migrated is False


def test_migrate_mismatch_discards(ledger_home: pathlib.Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    _write_v1(ledger_home, "checking", "WELLS_FARGO_CSV", [],
              [_v1_txn("2026-01-01", -100, ""), _v1_txn("2026-02-01", -200, "")])
    monkeypatch.setattr("ledger.core.account.Account.add_many", lambda self, txns: None)
    with pytest.raises(MigrationError):
        v1v2.migrate_account("checking")
    assert not config.account_db_path("checking").exists()    # discarded
