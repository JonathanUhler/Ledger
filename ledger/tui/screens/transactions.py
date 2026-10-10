"""
Transaction view, edit, remove, and reconcile screen.
"""


from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Label

from ledger.convert import convert_amount_to_cents, convert_cents_to_amount, \
    convert_date_to_string, convert_string_to_date
from ledger.core import search as search_module
from ledger.core.reconcile import Action, Decision
from ledger.core.service import Ledger
from ledger.errors import LedgerError
from ledger.models import Transaction


class ReconcileModal(ModalScreen):

    def __init__(self, account: str, statement_path: str, origin: "TransactionsScreen") -> None:
        super().__init__()
        self._account: str = account
        self._origin: "TransactionsScreen" = origin
        self._ledger: Ledger = Ledger()
        self._plan = self._ledger.plan_reconcile(account, statement_path)
        self._decisions: list[Decision] = []
        self._index: int = 0

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("", id = "rc_info"),
            Label("", id = "rc_dupe"),
            Input(placeholder = "memo", id = "rc_memo"),
            Input(placeholder = "category", id = "rc_category"),
            Horizontal(Button("Add", id = "rc_add"), Button("Skip", id = "rc_skip"),
                       Button("Cancel", id = "rc_cancel")),
            id = "rc_box",
        )

    def on_mount(self) -> None:
        self._show()

    def _show(self) -> None:
        if (self._index >= len(self._plan.proposals)):
            self._finish()
            return
        proposal = self._plan.proposals[self._index]
        txn = proposal.transaction
        self.query_one("#rc_info", Label).update(
            f"({self._index + 1}/{len(self._plan.proposals)}) "
            f"{convert_date_to_string(txn.date)}  {convert_cents_to_amount(txn.cents)}  "
            f"{txn.statement_memo}")
        self.query_one("#rc_dupe", Label).update(
            "WARNING: possible duplicate" if (len(proposal.duplicates) > 0) else "")
        self.query_one("#rc_memo", Input).value = ""
        self.query_one("#rc_category", Input).value = ""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if (event.button.id == "rc_cancel"):
            self.app.pop_screen()
            return
        proposal = self._plan.proposals[self._index]
        if (event.button.id == "rc_add"):
            self._decisions.append(Decision(
                proposal = proposal, action = Action.ADD,
                user_memo = self.query_one("#rc_memo", Input).value,
                category = self.query_one("#rc_category", Input).value))
        else:
            self._decisions.append(Decision(proposal = proposal, action = Action.SKIP))
        self._index += 1
        self._show()

    def _finish(self) -> None:
        try:
            self._ledger.commit_reconcile(self._account, self._decisions)
        except LedgerError:
            pass
        self.app.pop_screen()
        self._origin.reload()


class EditModal(ModalScreen):

    DEFAULT_CSS: str = """
    EditModal {
        align: center middle;
    }
    EditModal #ed_box {
        width: 72;
        height: auto;
        padding: 1 2;
        border: thick $panel;
        background: $surface;
    }
    """

    def __init__(self, account: str, txn: Transaction, origin: "TransactionsScreen") -> None:
        super().__init__()
        self._account: str = account
        self._txn: Transaction = txn
        self._origin: "TransactionsScreen" = origin
        self._ledger: Ledger = Ledger()

    def compose(self) -> ComposeResult:
        txn: Transaction = self._txn
        yield Vertical(
            Label(f"Edit transaction {txn.id[:8]}  ({self._account})"),
            Input(value = convert_date_to_string(txn.date), placeholder = "date (YYYY-MM-DD)",
                  id = "ed_date"),
            Input(value = convert_cents_to_amount(txn.cents), placeholder = "amount",
                  id = "ed_amount"),
            Input(value = txn.statement_memo, placeholder = "description", id = "ed_memo"),
            Input(value = txn.user_memo, placeholder = "memo", id = "ed_user_memo"),
            Input(value = txn.category, placeholder = "category", id = "ed_category"),
            Horizontal(Button("Save", id = "ed_save"), Button("Cancel", id = "ed_cancel")),
            Label("", id = "ed_status"),
            id = "ed_box",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if (event.button.id == "ed_cancel"):
            self.app.pop_screen()
            return
        if (event.button.id != "ed_save"):
            return
        try:
            fields: dict[str, object] = {
                "date": convert_string_to_date(self.query_one("#ed_date", Input).value.strip()),
                "cents": convert_amount_to_cents(self.query_one("#ed_amount", Input).value.strip()),
                "statement_memo": self.query_one("#ed_memo", Input).value,
                "user_memo": self.query_one("#ed_user_memo", Input).value,
                "category": self.query_one("#ed_category", Input).value.strip(),
            }
            self._ledger.edit_transaction(self._account, self._txn.id, **fields)
        except LedgerError as error:
            self.query_one("#ed_status", Label).update(f"error: {error}")
            return
        self.app.pop_screen()
        self._origin.reload()


class TransactionsScreen(Vertical):

    DEFAULT_CSS: str = """
    TransactionsScreen #tx_buttons {
        height: auto;
        margin-top: 1;
        margin-bottom: 1;
    }
    TransactionsScreen #tx_table {
        height: 1fr;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._by_id: dict[str, Transaction] = {}

    def compose(self) -> ComposeResult:
        yield Input(placeholder = "account(s), space separated", id = "tx_accounts")
        yield Input(placeholder = "category regex (optional)", id = "tx_category")
        yield Input(placeholder = "statement path (for reconcile)", id = "tx_statement")
        yield Horizontal(
            Button("Load", id = "tx_load"),
            Button("Edit", id = "tx_edit"),
            Button("Remove", id = "tx_remove"),
            Button("Reconcile", id = "tx_reconcile"),
            id = "tx_buttons",
        )
        yield DataTable(id = "tx_table")
        yield Label("", id = "tx_status")

    def on_mount(self) -> None:
        table: DataTable = self.query_one("#tx_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("id", "Date", "Amount", "Account", "Category", "Description", "Memo")

    def _accounts(self) -> list[str]:
        return self.query_one("#tx_accounts", Input).value.split()

    def _selected_transaction(self) -> Transaction | None:
        table: DataTable = self.query_one("#tx_table", DataTable)
        if (table.row_count == 0):
            return None
        cell_key = table.coordinate_to_cell_key(table.cursor_coordinate)
        return self._by_id.get(cell_key.row_key.value)

    def reload(self) -> None:
        searchers: list[search_module.Searcher] = [search_module.unconditional()]
        pattern: str = self.query_one("#tx_category", Input).value.strip()
        if (pattern != ""):
            searchers.append(search_module.category_regex([pattern]))
        table: DataTable = self.query_one("#tx_table", DataTable)
        table.clear()
        self._by_id = {}
        transactions = Ledger().search(self._accounts(), searchers)
        transactions.sort(key = lambda txn: convert_date_to_string(txn.date), reverse = True)
        for txn in transactions:
            self._by_id[txn.id] = txn
            table.add_row(txn.id, convert_date_to_string(txn.date),
                          convert_cents_to_amount(txn.cents), txn.account, txn.category,
                          txn.statement_memo, txn.user_memo, key = txn.id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        status: Label = self.query_one("#tx_status", Label)
        try:
            if (event.button.id == "tx_load"):
                self.reload()
            elif (event.button.id == "tx_reconcile"):
                accounts: list[str] = self._accounts()
                path: str = self.query_one("#tx_statement", Input).value.strip()
                if (len(accounts) > 0 and path != ""):
                    self.app.push_screen(ReconcileModal(accounts[0], path, self))
            elif (event.button.id == "tx_edit"):
                txn: Transaction | None = self._selected_transaction()
                if (txn is not None):
                    self.app.push_screen(EditModal(txn.account, txn, self))
                else:
                    status.update("Load transactions and select a row first")
            elif (event.button.id == "tx_remove"):
                selected: Transaction | None = self._selected_transaction()
                if (selected is not None):
                    Ledger().remove_transaction(selected.account, selected.id)
                    self.reload()
                else:
                    status.update("Load transactions and select a row first")
        except LedgerError as error:
            status.update(f"error: {error}")
