import datetime

from ledger.core import reconcile
from ledger.models import new_transaction


def _txn(memo: str = "COFFEE", cents: int = -475):
    return new_transaction(date = datetime.date(2026, 9, 14), cents = cents,
                           statement_memo = memo, account = "checking")


def test_two_identical_incoming_against_empty_account_not_flagged() -> None:
    a, b = _txn(), _txn()                      # identical content, distinct ids
    plan = reconcile.build_plan([a, b], existing_by_hash = {})
    assert len(plan.proposals) == 2
    assert plan.proposals[0].duplicates == []  # both addable: the cafe-twice case
    assert plan.proposals[1].duplicates == []


def test_incoming_matching_existing_is_flagged() -> None:
    existing = _txn()
    incoming = _txn()                          # same content_hash as existing
    plan = reconcile.build_plan([incoming], {existing.content_hash: [existing]})
    assert len(plan.proposals[0].duplicates) == 1
    assert plan.proposals[0].duplicates[0].id == existing.id
