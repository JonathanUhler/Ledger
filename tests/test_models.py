import datetime

from ledger import models


def test_content_hash_ignores_user_fields() -> None:
    a: models.Transaction = models.new_transaction(
        date = datetime.date(2026, 9, 14), cents = -475,
        statement_memo = "BLUE BOTTLE COFFEE", account = "checking",
        user_memo = "first", category = "Food/Coffee",
    )
    b: models.Transaction = models.new_transaction(
        date = datetime.date(2026, 9, 14), cents = -475,
        statement_memo = "blue  bottle coffee", account = "checking",
        user_memo = "second", category = "Uncategorized",
    )
    assert a.content_hash == b.content_hash   # normalized memo, user fields excluded
    assert a.id != b.id                        # but distinct surrogate identities


def test_different_amount_changes_hash() -> None:
    a: models.Transaction = models.new_transaction(
        date = datetime.date(2026, 9, 14), cents = -475,
        statement_memo = "X", account = "checking",
    )
    b: models.Transaction = models.new_transaction(
        date = datetime.date(2026, 9, 14), cents = -476,
        statement_memo = "X", account = "checking",
    )
    assert a.content_hash != b.content_hash


def test_new_transaction_defaults() -> None:
    txn: models.Transaction = models.new_transaction(
        date = datetime.date(2026, 1, 1), cents = 100, statement_memo = "m", account = "a",
    )
    assert txn.reconciled is False
    assert txn.user_memo == ""
    assert txn.category == ""
    assert len(txn.id) == 32
