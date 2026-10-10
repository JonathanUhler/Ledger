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
