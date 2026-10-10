"""
Account and statement-type creation screen.
"""


import yaml
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Input, Label

from ledger import config
from ledger.core.service import Ledger
from ledger.errors import LedgerError
from ledger.statements import grammar


class CreateScreen(Vertical):

    def compose(self) -> ComposeResult:
        yield Label("New account")
        yield Input(placeholder = "account name", id = "ca_name")
        yield Input(placeholder = "statement type", id = "ca_type")
        yield Button("Create account", id = "ca_btn")
        yield Label("New statement type (CSV)")
        yield Input(placeholder = "type name", id = "ct_name")
        yield Input(placeholder = "date index", id = "ct_date_index")
        yield Input(placeholder = "date format (e.g. %m/%d/%Y)", id = "ct_date_format")
        yield Input(placeholder = "amount index", id = "ct_amount_index")
        yield Input(placeholder = "memo index", id = "ct_memo_index")
        yield Input(placeholder = "header rows", id = "ct_header")
        yield Button("Create type", id = "ct_btn")
        yield Label("", id = "create_status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        status: Label = self.query_one("#create_status", Label)
        try:
            if (event.button.id == "ca_btn"):
                Ledger().create_account(self.query_one("#ca_name", Input).value.strip(),
                                        self.query_one("#ca_type", Input).value.strip())
                status.update("Account created")
            elif (event.button.id == "ct_btn"):
                name: str = self.query_one("#ct_name", Input).value.strip()
                data: dict = {
                    "name": name, "kind": "csv",
                    "csv": {
                        "num_header_rows": int(self.query_one("#ct_header", Input).value or "1"),
                        "date": {"index": int(self.query_one("#ct_date_index", Input).value),
                                 "format": self.query_one("#ct_date_format", Input).value.strip()},
                        "amount": {"index": int(self.query_one("#ct_amount_index", Input).value)},
                        "memo": {"index": int(self.query_one("#ct_memo_index", Input).value)},
                    },
                }
                grammar.load_grammar_from_dict(data, name)
                config.get_types_dir().mkdir(parents = True, exist_ok = True)
                config.type_path(name).write_text(yaml.safe_dump(data, sort_keys = False),
                                                  encoding = "utf-8")
                status.update(f"Type created. Edit {config.type_path(name)} for filter rules.")
        except (LedgerError, ValueError) as error:
            status.update(f"error: {error}")
