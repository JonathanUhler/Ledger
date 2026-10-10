import datetime

import pytest

from ledger import config
from ledger.core import database
from ledger.core.budget_store import BudgetStore
from ledger.errors import IntegrityError, InvalidBudgetError, MissingBudgetError
from ledger.models import BudgetKind, new_budget


def _budget(name: str = "Laptop", account: str | None = None):
    return new_budget(name = name, goal_cents = 100000, category = "Income/Paychecks",
                      percent_bps = 500, account = account,
                      start_date = datetime.date(2026, 1, 1))


def test_add_list_round_trips_kind_enum() -> None:
    store: BudgetStore = BudgetStore.open()
    budget = _budget()
    store.add_budget(budget)
    store.close()

    reopened: BudgetStore = BudgetStore.open()
    budgets = reopened.list_budgets()
    assert len(budgets) == 1
    assert budgets[0].name == "Laptop"
    assert budgets[0].kind is BudgetKind.ONE_TIME
    assert budgets[0].account is None
    assert budgets[0].start_date == datetime.date(2026, 1, 1)
    reopened.close()


def test_kind_stored_as_string_value() -> None:
    store: BudgetStore = BudgetStore.open()
    store.add_budget(_budget())
    store.close()
    conn = database.connect(config.budget_db_path())
    row = conn.execute("SELECT kind FROM budgets").fetchone()
    assert row["kind"] == "one_time"
    conn.close()


def test_get_missing_raises() -> None:
    store: BudgetStore = BudgetStore.open()
    with pytest.raises(MissingBudgetError):
        store.get_budget("does-not-exist")
    store.close()


def test_edit_updates_fields() -> None:
    store: BudgetStore = BudgetStore.open()
    budget = _budget()
    store.add_budget(budget)
    updated = store.edit_budget(budget.id, name = "Gaming PC", goal_cents = 200000,
                                account = "checking")
    assert updated.name == "Gaming PC"
    assert updated.goal_cents == 200000
    assert updated.account == "checking"
    store.close()
    assert BudgetStore.open().list_budgets()[0].name == "Gaming PC"


def test_remove_deletes_and_missing_raises() -> None:
    store: BudgetStore = BudgetStore.open()
    budget = _budget()
    store.add_budget(budget)
    store.remove_budget(budget.id)
    assert store.list_budgets() == []
    with pytest.raises(MissingBudgetError):
        store.remove_budget(budget.id)
    store.close()


def test_backup_written_on_mutation() -> None:
    store: BudgetStore = BudgetStore.open()
    store.add_budget(_budget())
    assert config.budget_backup_path().exists()
    store.close()


def test_checksum_detects_tamper() -> None:
    store: BudgetStore = BudgetStore.open()
    store.add_budget(_budget())
    store.close()
    # tamper directly without updating the stored checksum
    conn = database.connect(config.budget_db_path())
    conn.execute("UPDATE budgets SET goal_cents = 999999")
    conn.close()
    with pytest.raises(IntegrityError):
        BudgetStore.open()


def test_version_newer_than_app_raises() -> None:
    store: BudgetStore = BudgetStore.open()
    store.add_budget(_budget())
    store.close()
    # the version guard runs before the checksum check, so only the version meta needs changing
    conn = database.connect(config.budget_db_path())
    database.write_meta(conn, "data_model_version", "999")
    conn.close()
    with pytest.raises(InvalidBudgetError):
        BudgetStore.open()
