"""
Category management screen.
"""


from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label

from ledger.core.service import Ledger
from ledger.errors import LedgerError


class CategoriesScreen(Vertical):

    DEFAULT_CSS: str = """
    CategoriesScreen #cat_buttons {
        height: auto;
        margin-top: 1;
        margin-bottom: 1;
    }
    CategoriesScreen #cat_table {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield Input(placeholder = "account", id = "cat_account")
        yield Input(placeholder = "new category (a/b/c)", id = "cat_name")
        yield Horizontal(Button("Load", id = "cat_load"), Button("Add", id = "cat_add"),
                         Button("Add to all", id = "cat_addall"), id = "cat_buttons")
        yield DataTable(id = "cat_table")
        yield Label("", id = "cat_status")

    def on_mount(self) -> None:
        self.query_one("#cat_table", DataTable).add_columns("Category")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        ledger: Ledger = Ledger()
        account: str = self.query_one("#cat_account", Input).value.strip()
        name: str = self.query_one("#cat_name", Input).value.strip()
        status: Label = self.query_one("#cat_status", Label)
        try:
            if (event.button.id == "cat_add"):
                ledger.add_category(account, name, recursive = True)
            elif (event.button.id == "cat_addall"):
                ledger.add_category_all(name, recursive = True)
            table: DataTable = self.query_one("#cat_table", DataTable)
            table.clear()
            for category in ledger.list_categories(account):
                table.add_row(category)
            status.update("OK")
        except LedgerError as error:
            status.update(f"error: {error}")
