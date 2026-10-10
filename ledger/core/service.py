"""
The UI-agnostic Ledger service: the single API both the CLI and TUI call.
"""


import datetime

from ledger import config
from ledger.core import reconcile as reconcile_module
from ledger.core import search as search_module
from ledger.core.account import Account
from ledger.errors import InvalidCategoryError
from ledger.models import AccountSummary, Transaction, compute_content_hash, new_transaction
from ledger.statements import csv_grammar, grammar


class Ledger:

    def all_account_names(self) -> list[str]:
        accounts_dir = config.get_accounts_dir()
        if (not accounts_dir.exists()):
            return []
        return sorted(path.stem for path in accounts_dir.glob("*.db"))

    def list_accounts(self) -> list[AccountSummary]:
        summaries: list[AccountSummary] = []
        for name in self.all_account_names():
            account: Account = Account.open(name)
            try:
                summaries.append(account.summary())
            finally:
                account.close()
        return summaries

    def create_account(self, name: str, statement_type: str) -> None:
        config.validate_account_name(name)
        grammar.load_grammar(statement_type)   # raises UnsupportedStatementError if missing
        Account.create(name, statement_type).close()

    def list_categories(self, account: str) -> list[str]:
        acct: Account = Account.open(account)
        try:
            return acct.list_categories()
        finally:
            acct.close()

    def add_category(self, account: str, name: str, recursive: bool = False) -> None:
        acct: Account = Account.open(account)
        try:
            self._add_category_to(acct, name, recursive)
        finally:
            acct.close()

    def add_category_all(self, name: str, recursive: bool = False) -> None:
        for account_name in self.all_account_names():
            acct: Account = Account.open(account_name)
            try:
                self._add_category_to(acct, name, recursive)
            finally:
                acct.close()

    def _add_category_to(self, acct: Account, name: str, recursive: bool) -> None:
        components: list[str] = name.split("/")
        if (name.strip() == "" or any(part == "" for part in components)):
            raise InvalidCategoryError(f"invalid category name '{name}'")
        existing: list[str] = acct.list_categories()
        for i in range(len(components), 0, -1):
            category: str = "/".join(components[:i])
            parent: str | None = None if (i == 1) else "/".join(components[:i - 1])
            if (parent is None or parent in existing):
                acct.add_category(category)
                return
            if (recursive):
                acct.add_category(category)
            else:
                raise InvalidCategoryError(
                    f"parent category '{parent}' for '{category}' does not exist"
                )

    def import_categories(self, dest: str, src: str) -> int:
        source: Account = Account.open(src)
        try:
            categories: list[str] = source.list_categories()
        finally:
            source.close()
        destination: Account = Account.open(dest)
        try:
            for category in categories:
                destination.add_category(category)
        finally:
            destination.close()
        return len(categories)

    def _require_category(self, acct: Account, category: str) -> None:
        if (category not in acct.list_categories()):
            raise InvalidCategoryError(f"category '{category}' does not exist")

    def add_transaction(self, account: str, txn: Transaction) -> Transaction:
        acct: Account = Account.open(account)
        try:
            self._require_category(acct, txn.category)
            txn.account = account
            txn.content_hash = compute_content_hash(txn.date, txn.cents, txn.statement_memo,
                                                    account)
            acct.add_transaction(txn)
            return txn
        finally:
            acct.close()

    def edit_transaction(self, account: str, txn_id: str, **fields: object) -> Transaction:
        acct: Account = Account.open(account)
        try:
            if ("category" in fields):
                self._require_category(acct, str(fields["category"]))
            return acct.edit_transaction(txn_id, **fields)
        finally:
            acct.close()

    def remove_transaction(self, account: str, txn_id: str) -> None:
        acct: Account = Account.open(account)
        try:
            acct.remove_transaction(txn_id)
        finally:
            acct.close()

    def adjust(self, account: str, target_cents: int) -> Transaction | None:
        acct: Account = Account.open(account)
        try:
            delta: int = target_cents - acct.get_balance()
            if (delta == 0):
                return None
            if ("Adjustment" not in acct.list_categories()):
                acct.add_category("Adjustment")
            txn: Transaction = new_transaction(
                date = datetime.date.today(), cents = delta,
                statement_memo = "Balance adjustment", account = account,
                user_memo = "Balance adjustment", category = "Adjustment", reconciled = True,
            )
            acct.add_transaction(txn)
            return txn
        finally:
            acct.close()

    def search(self, accounts: list[str],
               searchers: list[search_module.Searcher]) -> list[Transaction]:
        names: list[str] = accounts if (len(accounts) > 0) else self.all_account_names()
        results: list[Transaction] = []
        for name in names:
            acct: Account = Account.open(name)
            try:
                results += search_module.apply(acct.list_transactions(), searchers)
            finally:
                acct.close()
        return results

    def plan_reconcile(self, account: str, statement_path: str) -> reconcile_module.ReconcilePlan:
        acct: Account = Account.open(account)
        try:
            parser: grammar.Grammar = grammar.load_grammar(acct.get_statement_type())
            incoming: list[Transaction] = csv_grammar.parse_statement(parser, statement_path)
            for txn in incoming:
                txn.account = account
                txn.content_hash = compute_content_hash(txn.date, txn.cents, txn.statement_memo,
                                                        account)
            existing_by_hash: dict[str, list[Transaction]] = {}
            for txn in incoming:
                if (txn.content_hash not in existing_by_hash):
                    existing_by_hash[txn.content_hash] = acct.find_by_content_hash(txn.content_hash)
            return reconcile_module.build_plan(incoming, existing_by_hash)
        finally:
            acct.close()

    def commit_reconcile(self, account: str,
                         decisions: list[reconcile_module.Decision]
                         ) -> reconcile_module.ReconcileResult:
        acct: Account = Account.open(account)
        try:
            valid: set[str] = set(acct.list_categories())
            to_add: list[Transaction] = []
            skipped: int = 0
            for decision in decisions:
                if (decision.action is reconcile_module.Action.SKIP):
                    skipped += 1
                    continue
                txn: Transaction = decision.proposal.transaction
                txn.user_memo = decision.user_memo
                txn.category = decision.category
                txn.reconciled = True
                txn.account = account
                if (txn.category not in valid):
                    raise InvalidCategoryError(f"category '{txn.category}' does not exist")
                to_add.append(txn)
            acct.add_many(to_add)
            return reconcile_module.ReconcileResult(
                added = len(to_add),
                skipped = skipped,
                net_cents = sum(txn.cents for txn in to_add),
            )
        finally:
            acct.close()
