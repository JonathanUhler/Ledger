import datetime

import pytest

from ledger import convert
from ledger.errors import AmountConversionError, DateConversionError


@pytest.mark.parametrize("text,cents", [
    ("$1,234.56", 123456),
    ("-$4.75", -475),
    ("4.75", 475),
    ("-0.09", -9),
    ("$0.00", 0),
])
def test_amount_to_cents(text: str, cents: int) -> None:
    assert convert.convert_amount_to_cents(text) == cents


@pytest.mark.parametrize("cents,text", [
    (123456, "$1234.56"),
    (-475, "-$4.75"),
    (0, "$0.00"),
    (-9, "-$0.09"),
])
def test_cents_to_amount(cents: int, text: str) -> None:
    assert convert.convert_cents_to_amount(cents) == text


@pytest.mark.parametrize("bad", ["(4.75)", "", "abc", "$"])
def test_bad_amount_raises(bad: str) -> None:
    with pytest.raises(AmountConversionError):
        convert.convert_amount_to_cents(bad)


def test_date_round_trip() -> None:
    date: datetime.date = convert.convert_string_to_date("2026-09-14")
    assert date == datetime.date(2026, 9, 14)
    assert convert.convert_date_to_string(date) == "2026-09-14"


def test_date_custom_format() -> None:
    date: datetime.date = convert.convert_string_to_date("09/14/2026", date_format = "%m/%d/%Y")
    assert date == datetime.date(2026, 9, 14)


def test_bad_date_raises() -> None:
    with pytest.raises(DateConversionError):
        convert.convert_string_to_date("not-a-date")


def test_normalize_string() -> None:
    assert convert.normalize_string("  Blue   Bottle\tCOFFEE ") == "blue bottle coffee"
    assert convert.normalize_string(None) == ""
