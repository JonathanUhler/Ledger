"""
Composable, pure transaction searchers (predicates).
"""


import datetime
import re
from typing import Callable

from ledger.models import Transaction


Searcher = Callable[[Transaction], bool]


def unconditional() -> Searcher:
    def searcher(_: Transaction) -> bool:
        return True
    return searcher


def date_range(earliest: datetime.date, latest: datetime.date) -> Searcher:
    def searcher(txn: Transaction) -> bool:
        return earliest <= txn.date <= latest
    return searcher


def amount_range(minimum: int, maximum: int) -> Searcher:
    def searcher(txn: Transaction) -> bool:
        return minimum <= txn.cents <= maximum
    return searcher


def category_regex(patterns: list[str]) -> Searcher:
    try:
        compiled: list[re.Pattern] = [re.compile(pattern) for pattern in patterns]
    except re.error as error:
        raise ValueError(f"invalid category pattern: {error}") from error

    def searcher(txn: Transaction) -> bool:
        return any(pattern.search(txn.category or "") is not None for pattern in compiled)
    return searcher


def apply(transactions: list[Transaction], searchers: list[Searcher]) -> list[Transaction]:
    return [txn for txn in transactions if all(searcher(txn) for searcher in searchers)]
