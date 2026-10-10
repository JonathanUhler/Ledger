"""
Core data structures for the ledger.
"""


from dataclasses import dataclass
import datetime
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
