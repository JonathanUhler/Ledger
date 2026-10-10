import pathlib

import pytest

from ledger import config
from ledger.cli.reconcile_loop import run_cli_reconcile
from ledger.core.service import Ledger


WELLS_YAML: str = (
    "name: wells_fargo_csv\nkind: csv\n"
    "csv:\n  num_header_rows: 1\n  date: {index: 0, format: \"%m/%d/%Y\"}\n"
    "  amount: {index: 2}\n  memo: {index: 1}\n"
)


@pytest.fixture()
def prepared(tmp_path: pathlib.Path) -> tuple[Ledger, str]:
    config.type_path("wells_fargo_csv").write_text(WELLS_YAML, encoding = "utf-8")
    ledger: Ledger = Ledger()
    ledger.create_account("checking", "wells_fargo_csv")
    ledger.add_category("checking", "Food")
    path: pathlib.Path = tmp_path / "s.csv"
    path.write_text("Date,Memo,Amount\n09/14/2026,COFFEE,-4.75\n09/15/2026,TEA,-3.00\n",
                    encoding = "utf-8")
    return ledger, str(path)


def test_loop_adds_with_scripted_input(prepared: tuple[Ledger, str]) -> None:
    ledger, path = prepared
    answers = iter(["my coffee", "Food", "my tea", "Food"])
    result = run_cli_reconcile(ledger, "checking", path,
                               input_fn = lambda _prompt: next(answers),
                               output_fn = lambda _line: None)
    assert result.added == 2 and result.net_cents == -775


def test_loop_reprompts_invalid_category(prepared: tuple[Ledger, str]) -> None:
    ledger, path = prepared
    answers = iter(["m", "Ghost", "Food", "m2", "Food"])   # first category rejected
    result = run_cli_reconcile(ledger, "checking", path,
                               input_fn = lambda _prompt: next(answers),
                               output_fn = lambda _line: None)
    assert result.added == 2
