import datetime

import pytest

from ledger.cli import render
from ledger.models import new_transaction
from ledger.errors import InvalidCategoryError


def _txn(cents: int, memo: str, category: str):
    t = new_transaction(date = datetime.date(2026, 9, 14), cents = cents,
                        statement_memo = memo, account = "checking")
    t.category = category
    t.user_memo = "note"
    return t


TXNS = [_txn(-475, "COFFEE", "Food/Coffee"), _txn(2000, "PAY", "Income")]


def test_markdown_has_header_and_rows() -> None:
    out: str = render.render(TXNS, fmt = "markdown")
    assert "| Date" in out and "Amount" in out
    assert "-$4.75" in out and "$20.00" in out


def test_csv_columns_subset() -> None:
    out: str = render.render(TXNS, columns = ["Amount", "Category"], fmt = "csv")
    lines: list[str] = out.splitlines()
    assert lines[0] == "Amount,Category"
    assert "-$4.75,Food/Coffee" in lines


def test_show_total() -> None:
    out: str = render.render(TXNS, columns = ["Amount"], fmt = "csv", show_total = True)
    assert "$15.25" in out   # -4.75 + 20.00


def test_invert_signs() -> None:
    out: str = render.render([_txn(-475, "C", "Food")], columns = ["Amount"], fmt = "csv",
                             invert_signs = True)
    assert "$4.75" in out and "-$4.75" not in out


def test_unknown_column_raises() -> None:
    with pytest.raises(InvalidCategoryError):
        render.render(TXNS, columns = ["Bogus"], fmt = "csv")
