import datetime
import pathlib

import pytest

from ledger import config
from ledger.core import reconcile, search
from ledger.core.service import Ledger
from ledger.models import new_transaction
from ledger.errors import InvalidCategoryError, UnsupportedStatementError, AccountParseError


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
