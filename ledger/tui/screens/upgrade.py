"""
v1 -> v2 upgrade screen.
"""


from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from ledger.convert import convert_cents_to_amount
from ledger.migrate import upgrade as upgrade_module


class UpgradeScreen(Vertical):

    def compose(self) -> ComposeResult:
        yield Button("Scan & upgrade v1 accounts", id = "up_btn")
        yield Static("", id = "up_out")

    def on_button_pressed(self, _event: Button.Pressed) -> None:
        reports = upgrade_module.upgrade()
        lines: list[str] = []
        for report in reports:
            state: str = "migrated" if (report.migrated) \
                else ("skipped" if (report.skipped) else "FAILED")
            lines.append(f"{report.account}: {state} — {report.message}")
            if (report.migrated):
                lines.append(f"  count={report.count}  "
                             f"v1={convert_cents_to_amount(report.v1_sum)}  "
                             f"v2={convert_cents_to_amount(report.v2_sum)}")
        self.query_one("#up_out", Static).update("\n".join(lines) or "No v1 accounts found")
