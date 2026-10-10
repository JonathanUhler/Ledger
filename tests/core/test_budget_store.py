import datetime

import pytest

from ledger import config
from ledger.core import database
from ledger.core.budget_store import BudgetStore
from ledger.errors import IntegrityError, InvalidBudgetError, MissingBudgetError
from ledger.models import BudgetKind, CategoryCap, new_budget


def _budget(name: str = "Laptop", account: str | None = None):
    return new_budget(name = name, goal_cents = 100000, category = "Income/Paychecks",
                      account = account,
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


def test_priority_round_trips() -> None:
    store: BudgetStore = BudgetStore.open()
    store.add_budget(new_budget(name = "A", goal_cents = 100000, category = "Income",
                                priority = 0,
                                start_date = datetime.date(2026, 1, 1)))
    store.add_budget(new_budget(name = "B", goal_cents = 100000, category = "Income",
                                priority = 1,
                                start_date = datetime.date(2026, 1, 1)))
    store.close()
    budgets = BudgetStore.open().list_budgets()
    assert [b.priority for b in budgets] == [0, 1]


def test_list_budgets_orders_by_category_then_priority() -> None:
    store: BudgetStore = BudgetStore.open()
    store.add_budget(new_budget(name = "Zeta", goal_cents = 1, category = "B", priority = 0, start_date = datetime.date(2026, 1, 1)))
    store.add_budget(new_budget(name = "Alpha", goal_cents = 1, category = "A", priority = 1, start_date = datetime.date(2026, 1, 1)))
    store.add_budget(new_budget(name = "Beta", goal_cents = 1, category = "A", priority = 0, start_date = datetime.date(2026, 1, 1)))
    names = [b.name for b in store.list_budgets()]
    assert names == ["Beta", "Alpha", "Zeta"]   # (A,0),(A,1),(B,0)
    store.close()


def test_set_priority_persists() -> None:
    store: BudgetStore = BudgetStore.open()
    budget = new_budget(name = "A", goal_cents = 100000, category = "Income", priority = 0, start_date = datetime.date(2026, 1, 1))
    store.add_budget(budget)
    store.set_priority(budget.id, 7)
    store.close()
    assert BudgetStore.open().list_budgets()[0].priority == 7


def test_cap_crud_and_default() -> None:
    store: BudgetStore = BudgetStore.open()
    assert store.get_cap("Income") is None            # unset -> None (service applies 100%)
    store.set_cap("Income", 3000)
    assert store.get_cap("Income") == 3000
    assert store.list_caps() == [CategoryCap(category = "Income", draw_bps = 3000)]
    store.clear_cap("Income")
    assert store.get_cap("Income") is None
    store.close()


def test_cap_edit_detected_by_checksum() -> None:
    store: BudgetStore = BudgetStore.open()
    store.set_cap("Income", 3000)
    store.close()
    conn = database.connect(config.budget_db_path())
    conn.execute("UPDATE category_caps SET draw_bps = 9999")
    conn.close()
    with pytest.raises(IntegrityError):
        BudgetStore.open()


def test_migrates_v1_store_to_v2() -> None:
    # Fabricate a v1 store: old schema (no priority / no category_caps), version = 1.
    config.get_ledger_home().mkdir(parents = True, exist_ok = True)
    conn = database.connect(config.budget_db_path())
    conn.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE budgets (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
            goal_cents INTEGER NOT NULL, category TEXT NOT NULL, account TEXT,
            percent_bps INTEGER NOT NULL, start_date TEXT NOT NULL);
    """)
    conn.execute("INSERT INTO budgets VALUES ('id2','Bravo','one_time',100,'Income',NULL,500,'2026-01-01')")
    conn.execute("INSERT INTO budgets VALUES ('id1','Alpha','one_time',100,'Income',NULL,500,'2026-01-01')")
    conn.execute("INSERT INTO meta VALUES ('data_model_version','1')")
    conn.commit()
    conn.close()

    store = BudgetStore.open()                 # triggers migration
    budgets = store.list_budgets()
    # priorities assigned per category by (name, id) order -> Alpha=0, Bravo=1
    by_name = {b.name: b.priority for b in budgets}
    assert by_name == {"Alpha": 0, "Bravo": 1}
    assert store.get_cap("Income") is None     # caps table created, empty
    store.close()
    # version was stamped to 2
    conn = database.connect(config.budget_db_path())
    assert database.read_meta(conn, "data_model_version") == "2"
    conn.close()
