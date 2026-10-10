"""
Accounts summary screen.
"""


from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Label

from ledger.convert import convert_cents_to_amount
from ledger.core.service import Ledger


class AccountsScreen(Vertical):

    def compose(self) -> ComposeResult:
        yield DataTable(id = "accounts_table")
        yield Label("", id = "accounts_total")

    def on_mount(self) -> None:
        self.reload()

    def reload(self) -> None:
        table: DataTable = self.query_one("#accounts_table", DataTable)
        table.clear(columns = True)
        table.add_columns("Account", "Balance", "Txns", "Earliest", "Latest", "Type")
        total: int = 0
        for summary in Ledger().list_accounts():
            table.add_row(summary.name, convert_cents_to_amount(summary.balance_cents),
                          str(summary.count), str(summary.earliest or "N/A"),
                          str(summary.latest or "N/A"), summary.statement_type)
            total += summary.balance_cents
        self.query_one("#accounts_total", Label).update(
            f"Total: {convert_cents_to_amount(total)}")
