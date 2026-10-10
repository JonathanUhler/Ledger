"""
The UI-agnostic Ledger service: the single API both the CLI and TUI call.
"""


import datetime

from ledger import config
from ledger.core import reconcile as reconcile_module
from ledger.core import search as search_module
from ledger.core.account import Account
from ledger.core.budget_store import BudgetStore
from ledger.convert import round_half_up
from ledger.errors import InvalidCategoryError, InvalidBudgetError
from ledger.models import AccountSummary, Budget, BudgetProgress, CategoryCap, \
    Transaction, compute_content_hash, new_budget, new_transaction
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

    def list_budgets(self) -> list[Budget]:
        if (not config.budget_db_path().exists()):
            return []
        store: BudgetStore = BudgetStore.open()
        try:
            return store.list_budgets()
        finally:
            store.close()

    def _validate_budget(self, name: str, goal_cents: int, category: str,
                         account: str | None) -> None:
        if (name.strip() == ""):
            raise InvalidBudgetError("budget name must not be empty")
        if (goal_cents <= 0):
            raise InvalidBudgetError("budget goal must be a positive amount")
        if (account is not None and account != ""):
            if (account not in self.all_account_names()):
                raise InvalidBudgetError(f"account '{account}' does not exist")
            if (category not in self.list_categories(account)):
                raise InvalidBudgetError(
                    f"category '{category}' does not exist in account '{account}'")
            return
        for existing in self.all_account_names():
            if (category in self.list_categories(existing)):
                return
        raise InvalidBudgetError(f"category '{category}' does not exist in any account")

    def add_budget(self, name: str, goal_cents: int, category: str,
                   start_date: datetime.date | None = None, account: str | None = None) -> Budget:
        account = account if (account) else None
        self._validate_budget(name, goal_cents, category, account)
        store: BudgetStore = BudgetStore.open()
        try:
            existing: list[int] = [b.priority for b in store.list_budgets()
                                   if (b.category == category)]
            priority: int = (max(existing) + 1) if (existing) else 0
            budget: Budget = new_budget(name = name, goal_cents = goal_cents, category = category,
                                        account = account, start_date = start_date,
                                        priority = priority)
            store.add_budget(budget)
        finally:
            store.close()
        return budget

    def edit_budget(self, budget_id: str, **fields: object) -> Budget:
        store: BudgetStore = BudgetStore.open()
        try:
            current: Budget = store.get_budget(budget_id)
            name = str(fields.get("name", current.name))
            goal_cents = int(fields.get("goal_cents", current.goal_cents))
            category = str(fields.get("category", current.category))
            raw_account = fields.get("account", current.account)
            account = str(raw_account) if (raw_account) else None
            self._validate_budget(name, goal_cents, category, account)
            apply_fields: dict[str, object] = dict(fields)
            if ("account" in apply_fields):
                apply_fields["account"] = account
            if (category != current.category):
                existing: list[int] = [b.priority for b in store.list_budgets()
                                       if (b.category == category)]
                apply_fields["priority"] = (max(existing) + 1) if (existing) else 0
            return store.edit_budget(budget_id, **apply_fields)
        finally:
            store.close()

    def remove_budget(self, budget_id: str) -> None:
        store: BudgetStore = BudgetStore.open()
        try:
            store.remove_budget(budget_id)
        finally:
            store.close()

    def get_category_cap(self, category: str) -> int:
        store: BudgetStore = BudgetStore.open()
        try:
            stored: int | None = store.get_cap(category)
        finally:
            store.close()
        return stored if (stored is not None) else 10000

    def set_category_cap(self, category: str, draw_bps: int) -> None:
        if (not (0 < draw_bps <= 10000)):
            raise InvalidBudgetError("cap must be greater than 0% and at most 100%")
        found: bool = any(category in self.list_categories(name)
                          for name in self.all_account_names())
        if (not found):
            raise InvalidBudgetError(f"category '{category}' does not exist in any account")
        store: BudgetStore = BudgetStore.open()
        try:
            store.set_cap(category, draw_bps)
        finally:
            store.close()

    def clear_category_cap(self, category: str) -> None:
        store: BudgetStore = BudgetStore.open()
        try:
            store.clear_cap(category)
        finally:
            store.close()

    def list_category_caps(self) -> list[CategoryCap]:
        store: BudgetStore = BudgetStore.open()
        try:
            return store.list_caps()
        finally:
            store.close()

    def reorder_budget(self, budget_id: str, direction: str) -> None:
        if (direction not in ("up", "down")):
            raise InvalidBudgetError(f"direction must be 'up' or 'down', not '{direction}'")
        store: BudgetStore = BudgetStore.open()
        try:
            target: Budget = store.get_budget(budget_id)
            siblings: list[Budget] = [b for b in store.list_budgets()
                                      if (b.category == target.category)]
            siblings.sort(key = lambda b: (b.priority, b.id))
            index: int = next(i for i, b in enumerate(siblings) if (b.id == target.id))
            swap: int = index - 1 if (direction == "up") else index + 1
            if (swap < 0 or swap >= len(siblings)):
                return
            other: Budget = siblings[swap]
            store.set_priority(target.id, other.priority)
            store.set_priority(other.id, target.priority)
        finally:
            store.close()

    def _allocate_category(self, category: str
                           ) -> tuple[dict[str, int], dict[str, int], dict[str, bool]]:
        store: BudgetStore = BudgetStore.open()
        try:
            budgets: list[Budget] = [b for b in store.list_budgets() if (b.category == category)]
            cap_stored: int | None = store.get_cap(category)
        finally:
            store.close()
        cap_bps: int = cap_stored if (cap_stored is not None) else 10000
        budgets.sort(key = lambda b: (b.priority, b.id))
        if (len(budgets) == 0):
            return {}, {}, {}

        earliest: datetime.date = min(b.start_date for b in budgets)
        contributions: list[Transaction] = []
        for name in self.all_account_names():
            acct: Account = Account.open(name)
            try:
                contributions += acct.list_category_since(category, earliest)
            finally:
                acct.close()
        contributions.sort(key = lambda txn: (txn.date, txn.id))

        saved: dict[str, int] = {b.id: 0 for b in budgets}
        seen: dict[str, int] = {b.id: 0 for b in budgets}
        frozen: dict[str, bool] = {b.id: False for b in budgets}

        for txn in contributions:
            pool: int = round_half_up(cap_bps * txn.cents, 10000)
            eligible: list[Budget] = [
                b for b in budgets
                if (not frozen[b.id]
                    and (b.account is None or b.account == txn.account)
                    and txn.date >= b.start_date)
            ]
            for b in eligible:
                seen[b.id] += txn.cents
            if (pool >= 0):
                remaining: int = pool
                for b in eligible:
                    if (remaining <= 0):
                        break
                    need: int = b.goal_cents - saved[b.id]
                    if (need <= 0):
                        frozen[b.id] = True
                        continue
                    take: int = min(remaining, need)
                    saved[b.id] += take
                    remaining -= take
                    if (saved[b.id] >= b.goal_cents):
                        frozen[b.id] = True
            elif (len(eligible) > 0):
                saved[eligible[0].id] += pool
        return saved, seen, frozen

    def budget_progress(self, budget: Budget) -> BudgetProgress:
        saved, seen, frozen = self._allocate_category(budget.category)
        amount: int = saved.get(budget.id, 0)
        return BudgetProgress(
            saved_cents = amount,
            goal_cents = budget.goal_cents,
            remaining_cents = budget.goal_cents - amount,
            contributed_cents = seen.get(budget.id, 0),
            is_met = amount >= budget.goal_cents,
            is_frozen = frozen.get(budget.id, False),
        )
