"""
The Textual TUI application shell.
"""


from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, TabbedContent, TabPane

from ledger.tui.screens.accounts import AccountsScreen
from ledger.tui.screens.categories import CategoriesScreen
from ledger.tui.screens.create import CreateScreen
from ledger.tui.screens.transactions import TransactionsScreen
from ledger.tui.screens.upgrade import UpgradeScreen


class LedgerApp(App):

    TITLE: str = "Ledger"
    BINDINGS: list = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Accounts", id = "tab_accounts"):
                yield AccountsScreen()
            with TabPane("Categories", id = "tab_categories"):
                yield CategoriesScreen()
            with TabPane("Create", id = "tab_create"):
                yield CreateScreen()
            with TabPane("Transactions", id = "tab_transactions"):
                yield TransactionsScreen()
            with TabPane("Upgrade", id = "tab_upgrade"):
                yield UpgradeScreen()
        yield Footer()


def run_tui() -> None:
    LedgerApp().run()
