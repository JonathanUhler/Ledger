import datetime

import pytest

from ledger.core import search
from ledger.models import new_transaction


def _txn(cents: int, category: str, day: int):
    t = new_transaction(date = datetime.date(2026, 9, day), cents = cents,
                        statement_memo = "m", account = "a")
    t.category = category
    return t


TXNS = [_txn(-475, "Food/Coffee", 14), _txn(-2000, "Rent", 1), _txn(5000, "Income/Salary", 15)]


def test_date_range() -> None:
    s = search.date_range(datetime.date(2026, 9, 10), datetime.date(2026, 9, 20))
    assert [t.cents for t in search.apply(TXNS, [s])] == [-475, 5000]


def test_amount_range() -> None:
    s = search.amount_range(-1000, 0)
    assert [t.category for t in search.apply(TXNS, [s])] == ["Food/Coffee"]


def test_category_regex_or_and_wildcard() -> None:
    s = search.category_regex(["Food.*", "Rent"])
    assert sorted(t.category for t in search.apply(TXNS, [s])) == ["Food/Coffee", "Rent"]


def test_category_regex_anchored() -> None:
    s = search.category_regex(["^Food$"])
    assert search.apply(TXNS, [s]) == []     # no exact "Food"


def test_conjunction() -> None:
    s1 = search.date_range(datetime.date(2026, 9, 1), datetime.date(2026, 9, 14))
    s2 = search.amount_range(-10000, 0)
    assert sorted(t.cents for t in search.apply(TXNS, [s1, s2])) == [-2000, -475]


def test_bad_pattern_raises() -> None:
    with pytest.raises(ValueError):
        search.category_regex(["("])
