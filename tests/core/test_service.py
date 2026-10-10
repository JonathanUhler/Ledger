import datetime
import pathlib

import pytest

from ledger import config
from ledger.core import reconcile, search
from ledger.core.service import Ledger
from ledger.models import new_transaction
from ledger.errors import InvalidCategoryError, UnsupportedStatementError, AccountParseError, \
    InvalidBudgetError, MissingBudgetError


WELLS_YAML: str = (
    "name: wells_fargo_csv\n"
    "kind: csv\n"
    "csv:\n"
    "  num_header_rows: 1\n"
    "  date: {index: 0, format: \"%m/%d/%Y\"}\n"
    "  amount: {index: 2}\n"
    "  memo: {index: 1}\n"
)


@pytest.fixture()
def ledger() -> Ledger:
    config.type_path("wells_fargo_csv").write_text(WELLS_YAML, encoding = "utf-8")
    return Ledger()


def _statement(tmp_path: pathlib.Path, text: str) -> str:
    path: pathlib.Path = tmp_path / "stmt.csv"
    path.write_text(text, encoding = "utf-8")
    return str(path)


def test_create_account_rejects_bad_name(ledger: Ledger) -> None:
    with pytest.raises(AccountParseError):
        ledger.create_account("../evil", "wells_fargo_csv")


def test_create_account_requires_existing_grammar(ledger: Ledger) -> None:
    with pytest.raises(UnsupportedStatementError):
        ledger.create_account("checking", "no_such_type")


def test_create_and_list_accounts(ledger: Ledger) -> None:
    ledger.create_account("checking", "wells_fargo_csv")
    summaries = ledger.list_accounts()
    assert [s.name for s in summaries] == ["checking"]
    assert summaries[0].statement_type == "wells_fargo_csv"


def test_add_category_requires_parent_unless_recursive(ledger: Ledger) -> None:
    ledger.create_account("checking", "wells_fargo_csv")
    with pytest.raises(InvalidCategoryError):
        ledger.add_category("checking", "Food/Coffee")       # no parent "Food"
    ledger.add_category("checking", "Food/Coffee", recursive = True)
    assert ledger.list_categories("checking") == ["Food", "Food/Coffee"]


def test_addall_and_import(ledger: Ledger) -> None:
    ledger.create_account("checking", "wells_fargo_csv")
    ledger.create_account("savings", "wells_fargo_csv")
    ledger.add_category_all("Rent")
    assert "Rent" in ledger.list_categories("checking")
    assert "Rent" in ledger.list_categories("savings")
    ledger.add_category("checking", "Food")
    count: int = ledger.import_categories("savings", "checking")
    assert count == 2 and "Food" in ledger.list_categories("savings")


def test_add_transaction_validates_category(ledger: Ledger) -> None:
    ledger.create_account("checking", "wells_fargo_csv")
    txn = new_transaction(date = datetime.date(2026, 9, 14), cents = -475,
                          statement_memo = "X", account = "checking", category = "Nope")
    with pytest.raises(InvalidCategoryError):
        ledger.add_transaction("checking", txn)
    ledger.add_category("checking", "Food")
    txn.category = "Food"
    ledger.add_transaction("checking", txn)
    assert ledger.search(["checking"], [search.unconditional()])[0].category == "Food"


def test_edit_validates_category(ledger: Ledger) -> None:
    ledger.create_account("checking", "wells_fargo_csv")
    ledger.add_category("checking", "Food")
    txn = new_transaction(date = datetime.date(2026, 9, 14), cents = -475,
                          statement_memo = "X", account = "checking", category = "Food")
    ledger.add_transaction("checking", txn)
    with pytest.raises(InvalidCategoryError):
        ledger.edit_transaction("checking", txn.id, category = "Ghost")


def test_adjust(ledger: Ledger) -> None:
    ledger.create_account("checking", "wells_fargo_csv")
    created = ledger.adjust("checking", 10000)
    assert created is not None and created.cents == 10000
    assert ledger.adjust("checking", 10000) is None       # already on target


def test_reconcile_then_reconcile_again_skip_all(ledger: Ledger, tmp_path: pathlib.Path) -> None:
    ledger.create_account("checking", "wells_fargo_csv")
    ledger.add_category("checking", "Food")
    path: str = _statement(tmp_path, "Date,Memo,Amount\n09/14/2026,COFFEE,-4.75\n"
                                     "09/14/2026,COFFEE,-4.75\n")   # two identical: cafe twice
    plan = ledger.plan_reconcile("checking", path)
    assert len(plan.proposals) == 2
    assert all(len(p.duplicates) == 0 for p in plan.proposals)       # not auto-dropped
    decisions = [reconcile.Decision(proposal = p, action = reconcile.Action.ADD,
                                    user_memo = "", category = "Food") for p in plan.proposals]
    result = ledger.commit_reconcile("checking", decisions)
    assert result.added == 2 and result.net_cents == -950

    plan2 = ledger.plan_reconcile("checking", path)                  # reconcile same file again
    assert all(len(p.duplicates) == 2 for p in plan2.proposals)      # now flagged as dupes
    skip_all = [reconcile.Decision(proposal = p, action = reconcile.Action.SKIP)
                for p in plan2.proposals]
    result2 = ledger.commit_reconcile("checking", skip_all)
    assert result2.added == 0 and result2.skipped == 2
    assert len(ledger.search(["checking"], [search.unconditional()])) == 2   # no doubling


def _seed_income(ledger: Ledger, account: str = "checking") -> None:
    ledger.create_account(account, "wells_fargo_csv")
    ledger.add_category(account, "Income/Paychecks", recursive = True)


def test_add_budget_persists_and_lists(ledger: Ledger) -> None:
    _seed_income(ledger)
    budget = ledger.add_budget(name = "Laptop", goal_cents = 100000,
                               category = "Income/Paychecks")
    listed = ledger.list_budgets()
    assert [b.id for b in listed] == [budget.id]
    assert listed[0].account is None


def test_list_budgets_empty_when_no_store(ledger: Ledger) -> None:
    assert ledger.list_budgets() == []


def test_add_budget_rejects_bad_values(ledger: Ledger) -> None:
    _seed_income(ledger)
    with pytest.raises(InvalidBudgetError):
        ledger.add_budget(name = "", goal_cents = 100000, category = "Income/Paychecks")
    with pytest.raises(InvalidBudgetError):
        ledger.add_budget(name = "X", goal_cents = 0, category = "Income/Paychecks")


def test_add_budget_validates_category_and_account(ledger: Ledger) -> None:
    _seed_income(ledger)
    with pytest.raises(InvalidBudgetError):
        ledger.add_budget(name = "X", goal_cents = 100000,
                          category = "Ghost")                # unknown category
    with pytest.raises(InvalidBudgetError):
        ledger.add_budget(name = "X", goal_cents = 100000,
                          category = "Income/Paychecks", account = "savings")                                   # unknown account


def test_edit_budget_clears_account_with_empty_string(ledger: Ledger) -> None:
    _seed_income(ledger)
    budget = ledger.add_budget(name = "Laptop", goal_cents = 100000,
                               category = "Income/Paychecks", account = "checking")
    ledger.edit_budget(budget.id, account = "")
    assert ledger.list_budgets()[0].account is None


def test_edit_unknown_budget_raises(ledger: Ledger) -> None:
    _seed_income(ledger)
    with pytest.raises(MissingBudgetError):
        ledger.edit_budget("nope", name = "X")


def test_remove_budget(ledger: Ledger) -> None:
    _seed_income(ledger)
    budget = ledger.add_budget(name = "Laptop", goal_cents = 100000,
                               category = "Income/Paychecks")
    ledger.remove_budget(budget.id)
    assert ledger.list_budgets() == []
    with pytest.raises(MissingBudgetError):
        ledger.remove_budget(budget.id)


def _income(ledger: Ledger, account: str, day: int, cents: int,
            category: str = "Income/Paychecks") -> None:
    ledger.add_transaction(account, new_transaction(
        date = datetime.date(2026, 1, day), cents = cents, statement_memo = "pay",
        account = account, category = category))


def test_progress_skips_missing_referenced_account(ledger: Ledger) -> None:
    _seed_income(ledger)
    budget = ledger.add_budget(name = "Laptop", goal_cents = 100000,
                               category = "Income/Paychecks", account = "checking", start_date = datetime.date(2026, 1, 1))
    # simulate the referenced account being deleted out from under the budget
    config.account_db_path("checking").unlink()
    progress = ledger.budget_progress(ledger.list_budgets()[0])
    assert progress.contributed_cents == 0 and progress.saved_cents == 0


def test_add_assigns_tail_priority(ledger: Ledger) -> None:
    _seed_income(ledger)
    a = ledger.add_budget(name = "A", goal_cents = 100000, category = "Income/Paychecks")
    b = ledger.add_budget(name = "B", goal_cents = 100000, category = "Income/Paychecks")
    by_id = {x.id: x.priority for x in ledger.list_budgets()}
    assert by_id[a.id] == 0 and by_id[b.id] == 1


def test_category_cap_service_crud_and_validation(ledger: Ledger) -> None:
    _seed_income(ledger)
    assert ledger.get_category_cap("Income/Paychecks") == 10000          # default 100%
    ledger.set_category_cap("Income/Paychecks", 3000)
    assert ledger.get_category_cap("Income/Paychecks") == 3000
    with pytest.raises(InvalidBudgetError):
        ledger.set_category_cap("Income/Paychecks", 0)                    # 0% rejected
    with pytest.raises(InvalidBudgetError):
        ledger.set_category_cap("Income/Paychecks", 10001)                # >100% rejected
    with pytest.raises(InvalidBudgetError):
        ledger.set_category_cap("Ghost", 3000)                           # unknown category
    ledger.clear_category_cap("Income/Paychecks")
    assert ledger.get_category_cap("Income/Paychecks") == 10000


def test_reorder_swaps_within_category(ledger: Ledger) -> None:
    _seed_income(ledger)
    a = ledger.add_budget(name = "A", goal_cents = 1, category = "Income/Paychecks")
    b = ledger.add_budget(name = "B", goal_cents = 1, category = "Income/Paychecks")
    ledger.reorder_budget(b.id, "up")
    order = [x.name for x in ledger.list_budgets() if x.category == "Income/Paychecks"]
    assert order == ["B", "A"]


def test_reorder_boundary_is_noop(ledger: Ledger) -> None:
    _seed_income(ledger)
    a = ledger.add_budget(name = "A", goal_cents = 1, category = "Income/Paychecks")
    b = ledger.add_budget(name = "B", goal_cents = 1, category = "Income/Paychecks")
    ledger.reorder_budget(a.id, "up")      # already at top
    ledger.reorder_budget(b.id, "down")    # already at bottom
    order = [x.name for x in ledger.list_budgets() if x.category == "Income/Paychecks"]
    assert order == ["A", "B"]
    with pytest.raises(InvalidBudgetError):
        ledger.reorder_budget(a.id, "sideways")
    with pytest.raises(MissingBudgetError):
        ledger.reorder_budget("nope", "up")


def test_edit_category_requeues_at_tail(ledger: Ledger) -> None:
    _seed_income(ledger)
    ledger.add_category("checking", "Income/Bonus", recursive = True)
    ledger.add_budget(name = "A", goal_cents = 1, category = "Income/Bonus")
    moved = ledger.add_budget(name = "B", goal_cents = 1, category = "Income/Paychecks")
    ledger.edit_budget(moved.id, category = "Income/Bonus")
    bonus = [x for x in ledger.list_budgets() if x.category == "Income/Bonus"]
    assert {x.name: x.priority for x in bonus} == {"A": 0, "B": 1}


def test_group_fills_sequentially_by_priority(ledger: Ledger) -> None:
    _seed_income(ledger)
    a = ledger.add_budget(name = "A", goal_cents = 30000, category = "Income/Paychecks",
                          start_date = datetime.date(2026, 1, 1))
    b = ledger.add_budget(name = "B", goal_cents = 20000, category = "Income/Paychecks",
                          start_date = datetime.date(2026, 1, 1))
    ledger.set_category_cap("Income/Paychecks", 10000)          # 100%
    _income(ledger, "checking", 5, 30000)                       # first $300 -> A fills, freezes
    assert ledger.budget_progress(a).is_frozen is True
    assert ledger.budget_progress(b).saved_cents == 0           # B not started yet
    _income(ledger, "checking", 6, 20000)                       # next $200 -> B fills
    assert ledger.budget_progress(b).is_met is True


def test_group_overfunded_cascades_within_cap(ledger: Ledger) -> None:
    _seed_income(ledger)
    a = ledger.add_budget(name = "A", goal_cents = 30000, category = "Income/Paychecks",
                          start_date = datetime.date(2026, 1, 1))
    b = ledger.add_budget(name = "B", goal_cents = 20000, category = "Income/Paychecks",
                          start_date = datetime.date(2026, 1, 1))
    ledger.set_category_cap("Income/Paychecks", 5000)           # 50%
    _income(ledger, "checking", 5, 100000)                      # $1000 income, pool = $500
    # pool $500: A takes $300 (freezes), cascades $200 to B (fills). Combined $500 == cap.
    assert ledger.budget_progress(a).saved_cents == 30000
    assert ledger.budget_progress(b).saved_cents == 20000


def test_group_cap_limits_combined_draw(ledger: Ledger) -> None:
    _seed_income(ledger)
    a = ledger.add_budget(name = "A", goal_cents = 1000000, category = "Income/Paychecks",
                          start_date = datetime.date(2026, 1, 1))
    ledger.set_category_cap("Income/Paychecks", 3000)           # 30%
    _income(ledger, "checking", 5, 100000)                      # $1000 -> at most $300 drawn
    assert ledger.budget_progress(a).saved_cents == 30000


def test_group_mixed_scope_shares_one_queue(ledger: Ledger) -> None:
    _seed_income(ledger, "checking")
    allb = ledger.add_budget(name = "All", goal_cents = 20000, category = "Income/Paychecks",
                             start_date = datetime.date(2026, 1, 1))
    chk = ledger.add_budget(name = "Chk", goal_cents = 20000, category = "Income/Paychecks",
                            account = "checking",
                            start_date = datetime.date(2026, 1, 1))
    ledger.set_category_cap("Income/Paychecks", 5000)           # 50%
    _income(ledger, "checking", 5, 100000)                      # $1000 checking, pool $500
    # priority order: All (0) then Chk (1). All takes $200 (freezes), Chk takes $200. Combined $400 <= $500.
    assert ledger.budget_progress(allb).saved_cents == 20000
    assert ledger.budget_progress(chk).saved_cents == 20000


def test_group_net_negative_reduces_first_active_only(ledger: Ledger) -> None:
    _seed_income(ledger)
    a = ledger.add_budget(name = "A", goal_cents = 100000, category = "Income/Paychecks",
                          start_date = datetime.date(2026, 1, 1))
    b = ledger.add_budget(name = "B", goal_cents = 100000, category = "Income/Paychecks",
                          start_date = datetime.date(2026, 1, 1))
    ledger.set_category_cap("Income/Paychecks", 10000)
    _income(ledger, "checking", 5, -50000)                      # -$500, pool -$500 -> A only
    assert ledger.budget_progress(a).saved_cents == -50000
    assert ledger.budget_progress(a).is_frozen is False
    assert ledger.budget_progress(b).saved_cents == 0


def test_group_frozen_is_immutable(ledger: Ledger) -> None:
    _seed_income(ledger)
    a = ledger.add_budget(name = "A", goal_cents = 30000, category = "Income/Paychecks",
                          start_date = datetime.date(2026, 1, 1))
    b = ledger.add_budget(name = "B", goal_cents = 1000000, category = "Income/Paychecks",
                          start_date = datetime.date(2026, 1, 1))
    ledger.set_category_cap("Income/Paychecks", 10000)
    _income(ledger, "checking", 5, 30000)                       # A freezes exactly
    _income(ledger, "checking", 6, 30000)                       # later income -> cascades to B only
    _income(ledger, "checking", 7, -100000)                     # later negative -> B only
    assert ledger.budget_progress(a).saved_cents == 30000       # A unchanged after freeze
    assert ledger.budget_progress(a).is_frozen is True


def test_group_start_date_eligibility(ledger: Ledger) -> None:
    _seed_income(ledger)
    a = ledger.add_budget(name = "A", goal_cents = 1000000, category = "Income/Paychecks",
                          start_date = datetime.date(2026, 1, 10))
    ledger.set_category_cap("Income/Paychecks", 10000)
    _income(ledger, "checking", 5, 100000)                      # before A's start -> ignored
    assert ledger.budget_progress(a).saved_cents == 0
    _income(ledger, "checking", 10, 100000)                     # on/after start -> counts
    assert ledger.budget_progress(a).saved_cents == 100000
