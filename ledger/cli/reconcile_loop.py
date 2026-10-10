"""
The CLI's interactive reconciliation loop over a ReconcilePlan.
"""


from typing import Callable

from ledger.convert import convert_cents_to_amount, convert_date_to_string
from ledger.core.reconcile import Action, Decision, ReconcileResult
from ledger.core.service import Ledger


def run_cli_reconcile(ledger: Ledger, account: str, statement_path: str,
                      input_fn: Callable[[str], str] = input,
                      output_fn: Callable[[str], None] = print) -> ReconcileResult:
    plan = ledger.plan_reconcile(account, statement_path)
    categories: list[str] = ledger.list_categories(account)
    decisions: list[Decision] = []

    total: int = len(plan.proposals)
    for index, proposal in enumerate(plan.proposals):
        txn = proposal.transaction
        output_fn("-" * 80)
        output_fn(f"({index + 1}/{total})  {convert_date_to_string(txn.date)}  "
                  f"{convert_cents_to_amount(txn.cents)}  {txn.statement_memo}")

        if (len(proposal.duplicates) > 0):
            output_fn(f"  WARNING: possible duplicate of {len(proposal.duplicates)} existing "
                      "transaction(s):")
            for existing in proposal.duplicates:
                output_fn(f"    {convert_date_to_string(existing.date)}  "
                          f"{convert_cents_to_amount(existing.cents)}  "
                          f"{existing.statement_memo}  ({existing.category})")
            keep: str = input_fn("  Keep this transaction anyway? [y/N]: ")
            if (keep.strip().lower() != "y"):
                decisions.append(Decision(proposal = proposal, action = Action.SKIP))
                continue

        user_memo: str = input_fn("  Memo: ")
        category: str = ""
        while (category not in categories):
            category = input_fn("  Category: ")
        decisions.append(Decision(proposal = proposal, action = Action.ADD,
                                  user_memo = user_memo, category = category))

    result: ReconcileResult = ledger.commit_reconcile(account, decisions)
    output_fn("-" * 80)
    output_fn(f"Added {result.added} ({result.skipped} skipped). "
              f"Net: {convert_cents_to_amount(result.net_cents)}")
    return result
