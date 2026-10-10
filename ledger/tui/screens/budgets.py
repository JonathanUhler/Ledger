"""
Budget management screen: a table of budgets grouped by category with inline progress bars,
per-category draw caps, and priority reordering.
"""


from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Label

from ledger.convert import convert_amount_to_cents, convert_bps_to_percent, \
    convert_cents_to_amount, convert_percent_to_bps
from ledger.core.service import Ledger
from ledger.errors import LedgerError
from ledger.models import Budget, BudgetProgress


_BAR_WIDTH: int = 10


def _progress_cell(budget: Budget, progress: BudgetProgress) -> str:
    pct: int = 0 if (budget.goal_cents == 0) \
        else round(progress.saved_cents * 100 / budget.goal_cents)
    clamped: int = max(0, min(100, pct))
    filled: int = clamped // (100 // _BAR_WIDTH)
    bar: str = ("█" * filled) + ("░" * (_BAR_WIDTH - filled))
    marker: str = "FROZEN" if (progress.is_frozen) else "in progress"
    return (f"{convert_cents_to_amount(progress.saved_cents)} / "
            f"{convert_cents_to_amount(budget.goal_cents)} {bar} {pct}% {marker}")


class BudgetsScreen(Vertical):

    DEFAULT_CSS: str = """
    BudgetsScreen .budget_row {
        height: auto;
    }
    BudgetsScreen .budget_row Input {
        width: 1fr;
    }
    BudgetsScreen .budget_buttons {
        height: auto;
    }
    BudgetsScreen #budget_buttons {
        margin-top: 1;
    }
    BudgetsScreen #budget_buttons2 {
        margin-bottom: 1;
    }
    BudgetsScreen #budget_table {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield Horizontal(
            Input(placeholder = "name", id = "budget_name"),
            Input(placeholder = "goal (e.g. 1000.00)", id = "budget_goal"),
            classes = "budget_row",
        )
        yield Horizontal(
            Input(placeholder = "income category (a/b)", id = "budget_category"),
            Input(placeholder = "account (blank = all accounts)", id = "budget_account"),
            classes = "budget_row",
        )
        yield Horizontal(
            Input(placeholder = "id (for edit / remove)", id = "budget_id"),
            Input(placeholder = "cap % for category (e.g. 30)", id = "budget_cap"),
            classes = "budget_row",
        )
        yield Horizontal(
            Button("Load", id = "budget_load"), Button("Add", id = "budget_add"),
            Button("Edit", id = "budget_edit"), Button("Remove", id = "budget_remove"),
            id = "budget_buttons", classes = "budget_buttons",
        )
        yield Horizontal(
            Button("Up", id = "budget_up"), Button("Down", id = "budget_down"),
            Button("Set Cap", id = "budget_setcap"), Button("Clear Cap", id = "budget_clearcap"),
            id = "budget_buttons2", classes = "budget_buttons",
        )
        yield DataTable(id = "budget_table")
        yield Label("", id = "budget_status")

    def on_mount(self) -> None:
        table: DataTable = self.query_one("#budget_table", DataTable)
        table.cursor_type = "row"
        table.add_columns("id", "#", "Name", "Progress", "Account", "Category @ cap")

    def _selected_id(self) -> str | None:
        table: DataTable = self.query_one("#budget_table", DataTable)
        if (table.row_count == 0):
            return None
        cell_key = table.coordinate_to_cell_key(table.cursor_coordinate)
        return cell_key.row_key.value

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        ledger: Ledger = Ledger()
        status: Label = self.query_one("#budget_status", Label)
        try:
            if (event.button.id == "budget_add"):
                account: str = self.query_one("#budget_account", Input).value.strip()
                ledger.add_budget(
                    name = self.query_one("#budget_name", Input).value.strip(),
                    goal_cents = convert_amount_to_cents(
                        self.query_one("#budget_goal", Input).value),
                    category = self.query_one("#budget_category", Input).value.strip(),
                    start_date = None,
                    account = account if (account != "") else None,
                )
            elif (event.button.id == "budget_edit"):
                ledger.edit_budget(self.query_one("#budget_id", Input).value.strip(),
                                   **self._edit_fields())
            elif (event.button.id == "budget_remove"):
                ledger.remove_budget(self.query_one("#budget_id", Input).value.strip())
            elif (event.button.id in ("budget_up", "budget_down")):
                selected: str | None = self._selected_id()
                if (selected is None):
                    status.update("Load budgets and select a row first")
                    return
                ledger.reorder_budget(selected,
                                      "up" if (event.button.id == "budget_up") else "down")
            elif (event.button.id == "budget_setcap"):
                ledger.set_category_cap(
                    self.query_one("#budget_category", Input).value.strip(),
                    convert_percent_to_bps(self.query_one("#budget_cap", Input).value))
            elif (event.button.id == "budget_clearcap"):
                ledger.clear_category_cap(
                    self.query_one("#budget_category", Input).value.strip())
            self._reload(ledger)
            status.update("OK")
        except LedgerError as error:
            status.update(f"error: {error}")

    def _edit_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {}
        name: str = self.query_one("#budget_name", Input).value.strip()
        goal: str = self.query_one("#budget_goal", Input).value.strip()
        category: str = self.query_one("#budget_category", Input).value.strip()
        account: str = self.query_one("#budget_account", Input).value.strip()
        if (name != ""):
            fields["name"] = name
        if (goal != ""):
            fields["goal_cents"] = convert_amount_to_cents(goal)
        if (category != ""):
            fields["category"] = category
        if (account != ""):
            fields["account"] = account
        return fields

    def _reload(self, ledger: Ledger) -> None:
        table: DataTable = self.query_one("#budget_table", DataTable)
        table.clear()
        position: int = 0
        current_category: str | None = None
        for budget in ledger.list_budgets():          # already ordered (category, priority, id)
            if (budget.category != current_category):
                current_category = budget.category
                position = 0
            position += 1
            progress: BudgetProgress = ledger.budget_progress(budget)
            scope: str = budget.account if (budget.account) else "all accounts"
            cap: str = convert_bps_to_percent(ledger.get_category_cap(budget.category))
            table.add_row(
                budget.id,
                str(position),
                budget.name,
                _progress_cell(budget, progress),
                scope,
                f"{budget.category} @ {cap}",
                key = budget.id,
            )
