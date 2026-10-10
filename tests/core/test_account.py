import datetime

import pytest

from ledger.core.account import Account
from ledger.models import new_transaction
from ledger.errors import MissingAccountError, AccountFileError


def _txn(cents: int = -475, memo: str = "COFFEE", account: str = "checking"):
    return new_transaction(date = datetime.date(2026, 9, 14), cents = cents,
                           statement_memo = memo, account = account)


def test_create_open_and_missing() -> None:
    Account.create("checking", "wells_fargo_csv").close()
    acct: Account = Account.open("checking")
    assert acct.get_statement_type() == "wells_fargo_csv"
    acct.close()
    with pytest.raises(MissingAccountError):
        Account.open("nope")


def test_create_duplicate_raises() -> None:
    Account.create("checking", "wells_fargo_csv").close()
    with pytest.raises(AccountFileError):
        Account.create("checking", "wells_fargo_csv")


def test_categories_sorted_and_deduped() -> None:
    acct: Account = Account.create("checking", "wells_fargo_csv")
    acct.add_category("Food/Coffee")
    acct.add_category("Food")
    acct.add_category("Food")   # dedupe
    assert acct.list_categories() == ["Food", "Food/Coffee"]
    acct.close()


def test_add_balance_and_duplicates_both_stored() -> None:
    acct: Account = Account.create("checking", "wells_fargo_csv")
    t1 = _txn()
    t2 = _txn()   # identical content, different id
    acct.add_many([t1, t2])
    assert acct.get_balance() == -950
    assert len(acct.find_by_content_hash(t1.content_hash)) == 2
    acct.close()


def test_edit_recomputes_hash_and_persists() -> None:
    acct: Account = Account.create("checking", "wells_fargo_csv")
    t1 = _txn()
    old_hash: str = t1.content_hash
    acct.add_transaction(t1)
    updated = acct.edit_transaction(t1.id, cents = -500, user_memo = "fixed")
    assert updated.cents == -500
    assert updated.content_hash != old_hash          # hash recomputed
    assert updated.user_memo == "fixed"
    reopened = Account.open("checking")
    assert reopened.get_balance() == -500             # persisted
    reopened.close()
    acct.close()


def test_remove_and_summary() -> None:
    acct: Account = Account.create("checking", "wells_fargo_csv")
    t1 = _txn()
    acct.add_transaction(t1)
    acct.remove_transaction(t1.id)
    summary = acct.summary()
    assert summary.count == 0
    assert summary.balance_cents == 0
    assert summary.earliest is None
    acct.close()


def test_backup_created_on_mutation(ledger_home) -> None:
    acct: Account = Account.create("checking", "wells_fargo_csv")
    acct.add_transaction(_txn())
    assert (ledger_home / "accounts" / "checking.db.bak").exists()
    acct.close()
