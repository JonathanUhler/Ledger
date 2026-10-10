"""
Budget management screen with a progress bar per budget.
"""


from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, ProgressBar

from ledger.convert import convert_amount_to_cents, convert_bps_to_percent, \
    convert_cents_to_amount, convert_percent_to_bps
from ledger.core.service import Ledger
from ledger.errors import LedgerError


class BudgetsScreen(Vertical):

    DEFAULT_CSS: str = """
    BudgetsScreen .budget_row {
        height: auto;
    }
    BudgetsScreen .budget_row Input {
        width: 1fr;
    }
    BudgetsScreen #budget_buttons {
        height: auto;
        margin-top: 1;
        margin-bottom: 1;
    }
    BudgetsScreen #budget_list {
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
            Input(placeholder = "percent (e.g. 5)", id = "budget_percent"),
            classes = "budget_row",
        )
        yield Horizontal(
            Input(placeholder = "account (blank = all accounts)", id = "budget_account"),
            Input(placeholder = "id (for edit / remove)", id = "budget_id"),
            classes = "budget_row",
        )
        yield Horizontal(
            Button("Load", id = "budget_load"), Button("Add", id = "budget_add"),
            Button("Edit", id = "budget_edit"), Button("Remove", id = "budget_remove"),
            id = "budget_buttons",
        )
        yield VerticalScroll(id = "budget_list")
        yield Label("", id = "budget_status")

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
                    percent_bps = convert_percent_to_bps(
                        self.query_one("#budget_percent", Input).value),
                    start_date = None,
                    account = account if (account != "") else None,
                )
            elif (event.button.id == "budget_edit"):
                ledger.edit_budget(self.query_one("#budget_id", Input).value.strip(),
                                   **self._edit_fields())
            elif (event.button.id == "budget_remove"):
                ledger.remove_budget(self.query_one("#budget_id", Input).value.strip())
            await self._reload(ledger)
            status.update("OK")
        except LedgerError as error:
            status.update(f"error: {error}")

    def _edit_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {}
        name: str = self.query_one("#budget_name", Input).value.strip()
        goal: str = self.query_one("#budget_goal", Input).value.strip()
        category: str = self.query_one("#budget_category", Input).value.strip()
        percent: str = self.query_one("#budget_percent", Input).value.strip()
        account: str = self.query_one("#budget_account", Input).value.strip()
        if (name != ""):
            fields["name"] = name
        if (goal != ""):
            fields["goal_cents"] = convert_amount_to_cents(goal)
        if (category != ""):
            fields["category"] = category
        if (percent != ""):
            fields["percent_bps"] = convert_percent_to_bps(percent)
        if (account != ""):
            fields["account"] = account
        return fields

    async def _reload(self, ledger: Ledger) -> None:
        container: VerticalScroll = self.query_one("#budget_list", VerticalScroll)
        await container.remove_children()
        for budget in ledger.list_budgets():
            progress = ledger.budget_progress(budget)
            pct: int = 0 if (budget.goal_cents == 0) \
                else max(0, min(100, round(progress.saved_cents * 100 / budget.goal_cents)))
            marker: str = "FROZEN" if (progress.is_frozen) else "in progress"
            text: str = (f"{budget.name} [{budget.id}] — "
                         f"{convert_cents_to_amount(progress.saved_cents)} / "
                         f"{convert_cents_to_amount(budget.goal_cents)} "
                         f"({budget.category} @ {convert_bps_to_percent(budget.percent_bps)}) "
                         f"{marker}")
            await container.mount(Label(text))
            bar: ProgressBar = ProgressBar(total = 100, show_eta = False)
            await container.mount(bar)
            bar.update(progress = pct)
