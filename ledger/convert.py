"""
Conversions between the ledger's canonical types and human/statement strings.
"""


import datetime
from decimal import Decimal, InvalidOperation
import re

from ledger.errors import AmountConversionError, DateConversionError


def normalize_string(string: str) -> str:
    if (string is None):
        return ""
    return re.sub(r"\s+", " ", string).strip().lower()


def convert_amount_to_cents(amount: str) -> int:
    cleaned: str = amount.replace("$", "").replace(",", "").strip()
    if (cleaned == ""):
        raise AmountConversionError(f"cannot convert dollar amount '{amount}'")
    try:
        return int(Decimal(cleaned) * 100)
    except (InvalidOperation, ValueError) as error:
        raise AmountConversionError(f"cannot convert dollar amount '{amount}'") from error


def convert_cents_to_amount(cents: int) -> str:
    sign: str = "-" if (cents < 0) else ""
    cents = abs(cents)
    integral_part: int = cents // 100
    fraction_part: int = cents % 100
    return f"{sign}${integral_part}.{fraction_part:02}"


def convert_string_to_date(string: str, date_format: str = "%Y-%m-%d") -> datetime.date:
    try:
        return datetime.datetime.strptime(string, date_format).date()
    except (ValueError, TypeError) as error:
        raise DateConversionError(f"cannot convert date '{string}'") from error


def convert_date_to_string(date: datetime.date, date_format: str = "%Y-%m-%d") -> str:
    if (not isinstance(date, datetime.date)):
        return str(date)
    return date.strftime(date_format)
