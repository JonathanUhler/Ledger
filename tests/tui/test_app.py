import asyncio
import datetime

from textual.css.scalar import Unit
from textual.widgets import DataTable, Input, TabbedContent

from ledger import config
from ledger.core import search
from ledger.core.service import Ledger
from ledger.models import new_transaction
from ledger.tui.app import LedgerApp
from ledger.tui.screens.transactions import TransactionsScreen


WELLS_YAML: str = (
    "name: wells_fargo_csv\nkind: csv\n"
    "csv:\n  num_header_rows: 1\n  date: {index: 0, format: \"%m/%d/%Y\"}\n"
    "  amount: {index: 2}\n  memo: {index: 1}\n"
)


def _seed_account() -> Ledger:
    config.type_path("wells_fargo_csv").write_text(WELLS_YAML, encoding = "utf-8")
    ledger: Ledger = Ledger()
    ledger.create_account("checking", "wells_fargo_csv")
    return ledger


def test_app_boots_and_accounts_tab_populated() -> None:
    _seed_account()

    async def scenario() -> None:
        app: LedgerApp = LedgerApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            table: DataTable = app.query_one("#accounts_table", DataTable)
            assert table.row_count == 1

    asyncio.run(scenario())


def test_transactions_table_is_height_constrained() -> None:
    _seed_account()

    async def scenario() -> None:
        app: LedgerApp = LedgerApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            table: DataTable = app.query_one("#tx_table", DataTable)
            # A fractional height makes the table scroll internally instead of
            # overflowing and overlapping the button row above it.
            assert table.styles.height.unit == Unit.FRACTION

    asyncio.run(scenario())


def test_edit_button_updates_transaction() -> None:
    ledger: Ledger = _seed_account()
    ledger.add_category("checking", "Food")
    ledger.add_category("checking", "Rent")
    txn = new_transaction(date = datetime.date(2026, 9, 14), cents = -475,
                          statement_memo = "BLUE BOTTLE", account = "checking", category = "Food")
    ledger.add_transaction("checking", txn)

    async def scenario() -> None:
        app: LedgerApp = LedgerApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(TabbedContent).active = "tab_transactions"
            await pilot.pause()
            app.query_one("#tx_accounts", Input).value = "checking"
            await pilot.click("#tx_load")
            await pilot.pause()
            await pilot.click("#tx_edit")
            await pilot.pause()
            app.screen.query_one("#ed_category", Input).value = "Rent"
            await pilot.click("#ed_save")
            await pilot.pause()
            stored = ledger.search(["checking"], [search.unconditional()])
            assert len(stored) == 1
            assert stored[0].category == "Rent"

    asyncio.run(scenario())


def test_budgets_tab_lists_in_table() -> None:
    ledger: Ledger = _seed_account()
    ledger.add_category("checking", "Income/Paychecks", recursive = True)
    ledger.add_transaction("checking", new_transaction(
        date = datetime.date(2026, 1, 5), cents = 2000000, statement_memo = "pay",
        account = "checking", category = "Income/Paychecks"))
    budget = ledger.add_budget(name = "Laptop", goal_cents = 100000, category = "Income/Paychecks",
                               start_date = datetime.date(2026, 1, 1))

    async def scenario() -> None:
        app: LedgerApp = LedgerApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(TabbedContent).active = "tab_budgets"
            await pilot.pause()
            await pilot.click("#budget_load")
            await pilot.pause()
            table: DataTable = app.query_one("#budget_table", DataTable)
            # A fractional height keeps the table from overflowing onto the buttons.
            assert table.styles.height.unit == Unit.FRACTION
            assert table.row_count == 1
            row = table.get_row_at(0)
            assert row[0] == budget.id                   # id
            assert row[1] == "1"                         # order within category
            assert row[2] == "Laptop"                    # name
            # default cap 100% -> $20,000 income fills the $1,000 goal; frozen at 100%.
            assert "█" in row[3]                          # progress bar glyph
            assert "100%" in row[3]
            assert "FROZEN" in row[3]
            assert row[4] == "all accounts"              # account scope
            assert row[5] == "Income/Paychecks @ 100%"   # category @ cap (default)

    asyncio.run(scenario())


def test_categories_table_is_height_constrained() -> None:
    _seed_account()

    async def scenario() -> None:
        app: LedgerApp = LedgerApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(TabbedContent).active = "tab_categories"
            await pilot.pause()
            table: DataTable = app.query_one("#cat_table", DataTable)
            # Matches the transactions table: a fractional height makes the table
            # scroll internally instead of covering the buttons below it.
            assert table.styles.height.unit == Unit.FRACTION

    asyncio.run(scenario())


def test_budgets_tab_add_button_creates_budget() -> None:
    ledger: Ledger = _seed_account()
    ledger.add_category("checking", "Income/Paychecks", recursive = True)

    async def scenario() -> None:
        app: LedgerApp = LedgerApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(TabbedContent).active = "tab_budgets"
            await pilot.pause()
            app.query_one("#budget_name", Input).value = "Laptop"
            app.query_one("#budget_goal", Input).value = "1000.00"
            app.query_one("#budget_category", Input).value = "Income/Paychecks"
            await pilot.click("#budget_add")
            await pilot.pause()
            assert [b.name for b in ledger.list_budgets()] == ["Laptop"]

    asyncio.run(scenario())


def test_budgets_tab_set_cap_button() -> None:
    ledger: Ledger = _seed_account()
    ledger.add_category("checking", "Income/Paychecks", recursive = True)
    ledger.add_budget(name = "Laptop", goal_cents = 100000, category = "Income/Paychecks",
                      start_date = datetime.date(2026, 1, 1))

    async def scenario() -> None:
        app: LedgerApp = LedgerApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(TabbedContent).active = "tab_budgets"
            await pilot.pause()
            app.query_one("#budget_category", Input).value = "Income/Paychecks"
            app.query_one("#budget_cap", Input).value = "30"
            await pilot.click("#budget_setcap")
            await pilot.pause()
            assert ledger.get_category_cap("Income/Paychecks") == 3000
            table: DataTable = app.query_one("#budget_table", DataTable)
            assert table.get_row_at(0)[5] == "Income/Paychecks @ 30%"

    asyncio.run(scenario())


def test_budgets_tab_reorder_selected_row() -> None:
    ledger: Ledger = _seed_account()
    ledger.add_category("checking", "Income/Paychecks", recursive = True)
    ledger.add_budget(name = "A", goal_cents = 100000, category = "Income/Paychecks")
    ledger.add_budget(name = "B", goal_cents = 100000, category = "Income/Paychecks")

    async def scenario() -> None:
        app: LedgerApp = LedgerApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(TabbedContent).active = "tab_budgets"
            await pilot.pause()
            await pilot.click("#budget_load")
            await pilot.pause()
            table: DataTable = app.query_one("#budget_table", DataTable)
            table.move_cursor(row = 1)                 # select "B" (second row)
            await pilot.click("#budget_up")
            await pilot.pause()
            assert [b.name for b in ledger.list_budgets()] == ["B", "A"]
            assert table.get_row_at(0)[2] == "B"

    asyncio.run(scenario())
