"""
Core data structures for the ledger.
"""


from dataclasses import dataclass
import datetime
from enum import Enum
import hashlib
import uuid

from ledger.convert import convert_date_to_string, normalize_string


@dataclass
class Transaction:
    id: str
    date: datetime.date
    cents: int
    statement_memo: str
    account: str
    user_memo: str
    category: str
    reconciled: bool
    content_hash: str


    def content_key(self) -> str:
        date_string: str = convert_date_to_string(self.date)
        memo: str = normalize_string(self.statement_memo)
        return f"{date_string}|{self.cents}|{memo}|{self.account}"


def compute_content_hash(date: datetime.date, cents: int, statement_memo: str, account: str) -> str:
    date_string: str = convert_date_to_string(date)
    memo: str = normalize_string(statement_memo)
    key: str = f"{date_string}|{cents}|{memo}|{account}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def new_transaction(date: datetime.date,
                    cents: int,
                    statement_memo: str,
                    account: str = "",
                    user_memo: str = "",
                    category: str = "",
                    reconciled: bool = False) -> Transaction:
    return Transaction(
        id = uuid.uuid4().hex,
        date = date,
        cents = cents,
        statement_memo = statement_memo,
        account = account,
        user_memo = user_memo,
        category = category,
        reconciled = reconciled,
        content_hash = compute_content_hash(date, cents, statement_memo, account),
    )


@dataclass
class AccountSummary:
    name: str
    balance_cents: int
    count: int
    earliest: datetime.date | None
    latest: datetime.date | None
    statement_type: str


class BudgetKind(Enum):
    ONE_TIME = "one_time"
    # REPEATING = "repeating"  # added when periodic budgets land


@dataclass
class Budget:
    id: str
    name: str
    kind: BudgetKind
    goal_cents: int
    category: str
    account: str | None
    percent_bps: int
    start_date: datetime.date


@dataclass
class BudgetProgress:
    saved_cents: int
    goal_cents: int
    remaining_cents: int
    contributed_cents: int
    is_met: bool
    is_frozen: bool


def new_budget(name: str,
               goal_cents: int,
               category: str,
               percent_bps: int,
               account: str | None = None,
               start_date: datetime.date | None = None,
               kind: BudgetKind = BudgetKind.ONE_TIME) -> Budget:
    return Budget(
        id = uuid.uuid4().hex,
        name = name,
        kind = kind,
        goal_cents = goal_cents,
        category = category,
        account = account,
        percent_bps = percent_bps,
        start_date = start_date if (start_date is not None) else datetime.date.today(),
    )
