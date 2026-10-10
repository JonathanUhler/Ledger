"""
Two-phase reconciliation data types and plan construction.
"""


from dataclasses import dataclass, field
from enum import Enum

from ledger.models import Transaction


class Action(Enum):
    ADD = "add"
    SKIP = "skip"


@dataclass
class Proposal:
    transaction: Transaction
    duplicates: list[Transaction] = field(default_factory = list)


@dataclass
class ReconcilePlan:
    proposals: list[Proposal]


@dataclass
class Decision:
    proposal: Proposal
    action: Action
    user_memo: str = ""
    category: str = ""


@dataclass
class ReconcileResult:
    added: int
    skipped: int
    net_cents: int


def build_plan(incoming: list[Transaction],
               existing_by_hash: dict[str, list[Transaction]]) -> ReconcilePlan:
    proposals: list[Proposal] = []
    for txn in incoming:
        duplicates: list[Transaction] = list(existing_by_hash.get(txn.content_hash, []))
        proposals.append(Proposal(transaction = txn, duplicates = duplicates))
    return ReconcilePlan(proposals = proposals)
