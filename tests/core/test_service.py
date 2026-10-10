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
                               category = "Income/Paychecks", percent_bps = 500)
    listed = ledger.list_budgets()
    assert [b.id for b in listed] == [budget.id]
    assert listed[0].account is None


def test_list_budgets_empty_when_no_store(ledger: Ledger) -> None:
    assert ledger.list_budgets() == []


def test_add_budget_rejects_bad_values(ledger: Ledger) -> None:
    _seed_income(ledger)
    with pytest.raises(InvalidBudgetError):
        ledger.add_budget(name = "", goal_cents = 100000,
                          category = "Income/Paychecks", percent_bps = 500)
    with pytest.raises(InvalidBudgetError):
        ledger.add_budget(name = "X", goal_cents = 0,
                          category = "Income/Paychecks", percent_bps = 500)
    with pytest.raises(InvalidBudgetError):
        ledger.add_budget(name = "X", goal_cents = 100000,
                          category = "Income/Paychecks", percent_bps = 0)       # 0% rejected
    with pytest.raises(InvalidBudgetError):
        ledger.add_budget(name = "X", goal_cents = 100000,
                          category = "Income/Paychecks", percent_bps = 10001)   # >100% rejected
    assert ledger.add_budget(name = "X", goal_cents = 100000,
                             category = "Income/Paychecks", percent_bps = 10000)   # 100% ok


def test_add_budget_validates_category_and_account(ledger: Ledger) -> None:
    _seed_income(ledger)
    with pytest.raises(InvalidBudgetError):
        ledger.add_budget(name = "X", goal_cents = 100000,
                          category = "Ghost", percent_bps = 500)                # unknown category
    with pytest.raises(InvalidBudgetError):
        ledger.add_budget(name = "X", goal_cents = 100000,
                          category = "Income/Paychecks", percent_bps = 500,
                          account = "savings")                                   # unknown account


def test_edit_budget_clears_account_with_empty_string(ledger: Ledger) -> None:
    _seed_income(ledger)
    budget = ledger.add_budget(name = "Laptop", goal_cents = 100000,
                               category = "Income/Paychecks", percent_bps = 500,
                               account = "checking")
    ledger.edit_budget(budget.id, account = "")
    assert ledger.list_budgets()[0].account is None


def test_edit_unknown_budget_raises(ledger: Ledger) -> None:
    _seed_income(ledger)
    with pytest.raises(MissingBudgetError):
        ledger.edit_budget("nope", name = "X")


def test_remove_budget(ledger: Ledger) -> None:
    _seed_income(ledger)
    budget = ledger.add_budget(name = "Laptop", goal_cents = 100000,
                               category = "Income/Paychecks", percent_bps = 500)
    ledger.remove_budget(budget.id)
    assert ledger.list_budgets() == []
    with pytest.raises(MissingBudgetError):
        ledger.remove_budget(budget.id)


def _income(ledger: Ledger, account: str, day: int, cents: int,
            category: str = "Income/Paychecks") -> None:
    ledger.add_transaction(account, new_transaction(
        date = datetime.date(2026, 1, day), cents = cents, statement_memo = "pay",
        account = account, category = category))


def test_progress_basic_percentage(ledger: Ledger) -> None:
    _seed_income(ledger)
    _income(ledger, "checking", 5, 1000000)                 # $10,000 income
    budget = ledger.add_budget(name = "Laptop", goal_cents = 100000,
                               category = "Income/Paychecks", percent_bps = 500,
                               start_date = datetime.date(2026, 1, 1))
    progress = ledger.budget_progress(budget)
    assert progress.saved_cents == 50000                    # 5% of $10,000 = $500
    assert progress.remaining_cents == 50000
    assert progress.is_met is False
    assert progress.is_frozen is False


def test_progress_across_all_accounts_vs_scoped(ledger: Ledger) -> None:
    _seed_income(ledger, "checking")
    ledger.create_account("savings", "wells_fargo_csv")
    ledger.add_category("savings", "Income/Paychecks", recursive = True)
    _income(ledger, "checking", 5, 1000000)
    _income(ledger, "savings", 5, 1000000)
    all_budget = ledger.add_budget(name = "All", goal_cents = 1000000,
                                   category = "Income/Paychecks", percent_bps = 500,
                                   start_date = datetime.date(2026, 1, 1))
    scoped = ledger.add_budget(name = "Scoped", goal_cents = 1000000,
                               category = "Income/Paychecks", percent_bps = 500,
                               account = "checking", start_date = datetime.date(2026, 1, 1))
    assert ledger.budget_progress(all_budget).contributed_cents == 2000000     # both accounts
    assert ledger.budget_progress(scoped).contributed_cents == 1000000         # checking only


def test_progress_start_date_boundary(ledger: Ledger) -> None:
    _seed_income(ledger)
    _income(ledger, "checking", 1, 1000000)                 # exactly on start date -> counts
    budget = ledger.add_budget(name = "B", goal_cents = 100000,
                               category = "Income/Paychecks", percent_bps = 500,
                               start_date = datetime.date(2026, 1, 1))
    assert ledger.budget_progress(budget).contributed_cents == 1000000
    ledger.edit_budget(budget.id, start_date = datetime.date(2026, 1, 2))   # now excludes day 1
    assert ledger.budget_progress(ledger.list_budgets()[0]).contributed_cents == 0


def test_progress_freezes_and_ignores_later_income(ledger: Ledger) -> None:
    _seed_income(ledger)
    _income(ledger, "checking", 5, 2000000)                 # 5% of $20,000 = $1,000 -> meets goal
    budget = ledger.add_budget(name = "Laptop", goal_cents = 100000,
                               category = "Income/Paychecks", percent_bps = 500,
                               start_date = datetime.date(2026, 1, 1))
    met = ledger.budget_progress(budget)
    assert met.is_met is True and met.is_frozen is True and met.saved_cents == 100000
    _income(ledger, "checking", 10, 2000000)                # more income, later
    _income(ledger, "checking", 11, -5000000)               # and a big negative, later
    frozen = ledger.budget_progress(ledger.list_budgets()[0])
    assert frozen.saved_cents == 100000                     # unchanged by later activity


def test_progress_overfunded_single_income(ledger: Ledger) -> None:
    _seed_income(ledger)
    _income(ledger, "checking", 5, 4000000)                 # 5% of $40,000 = $2,000 > $1,000 goal
    budget = ledger.add_budget(name = "Laptop", goal_cents = 100000,
                               category = "Income/Paychecks", percent_bps = 500,
                               start_date = datetime.date(2026, 1, 1))
    progress = ledger.budget_progress(budget)
    assert progress.saved_cents == 200000
    assert progress.remaining_cents == -100000
    assert progress.is_met is True and progress.is_frozen is True


def test_progress_net_negative_never_freezes(ledger: Ledger) -> None:
    _seed_income(ledger)
    _income(ledger, "checking", 5, -1000000)                # net-negative category
    budget = ledger.add_budget(name = "Laptop", goal_cents = 100000,
                               category = "Income/Paychecks", percent_bps = 500,
                               start_date = datetime.date(2026, 1, 1))
    progress = ledger.budget_progress(budget)
    assert progress.saved_cents == -50000
    assert progress.is_met is False and progress.is_frozen is False


def test_progress_skips_missing_referenced_account(ledger: Ledger) -> None:
    _seed_income(ledger)
    budget = ledger.add_budget(name = "Laptop", goal_cents = 100000,
                               category = "Income/Paychecks", percent_bps = 500,
                               account = "checking", start_date = datetime.date(2026, 1, 1))
    # simulate the referenced account being deleted out from under the budget
    config.account_db_path("checking").unlink()
    progress = ledger.budget_progress(ledger.list_budgets()[0])
    assert progress.contributed_cents == 0 and progress.saved_cents == 0
