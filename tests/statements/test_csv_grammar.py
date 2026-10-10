import datetime
import pathlib

import pytest

from ledger.statements import csv_grammar, grammar
from ledger.errors import StatementParseError


WELLS: grammar.Grammar = grammar.load_grammar_from_dict(
    {"name": "wells_fargo_csv", "kind": "csv",
     "csv": {"num_header_rows": 1, "date": {"index": 0, "format": "%m/%d/%Y"},
             "amount": {"index": 2}, "memo": {"index": 1}}},
    "wells_fargo_csv",
)

SCHWAB: grammar.Grammar = grammar.load_grammar_from_dict(
    {"name": "charles_schwab_csv", "kind": "csv",
     "csv": {"num_header_rows": 1, "date": {"index": 0, "format": "%m/%d/%Y"},
             "amount": {"index": 7}, "memo": {"index": 3}, "columns": {"action": 1}},
     "filter": {"all": [{"column": "action", "op": "in",
                         "values": ["MoneyLink Transfer", "Credit Interest"]}]},
     "transform": [{"when": {"column": "action", "op": "equals", "value": "MoneyLink Transfer"},
                    "target": "date", "op": "split", "sep": " as of ", "take": 0}]},
    "charles_schwab_csv",
)


def _write(tmp_path: pathlib.Path, name: str, text: str) -> str:
    path: pathlib.Path = tmp_path / name
    path.write_text(text, encoding = "utf-8")
    return str(path)


def test_parse_wells(tmp_path: pathlib.Path) -> None:
    path: str = _write(tmp_path, "w.csv",
                       "Date,Memo,Amount\n09/14/2026,BLUE BOTTLE,-4.75\n09/15/2026,PAYROLL,2000.00\n")
    txns = csv_grammar.parse_statement(WELLS, path)
    assert len(txns) == 2
    assert txns[0].date == datetime.date(2026, 9, 14)
    assert txns[0].cents == -475
    assert txns[0].statement_memo == "BLUE BOTTLE"


def test_parse_schwab_filters_and_rewrites_date(tmp_path: pathlib.Path) -> None:
    path: str = _write(
        tmp_path, "s.csv",
        "Date,Action,A,Memo,B,C,D,Amount\n"
        "09/14/2026 as of 09/13/2026,MoneyLink Transfer,,XFER,,,,100.00\n"
        "09/14/2026,Buy,,STOCK,,,,-50.00\n"
        "09/16/2026,Credit Interest,,INT,,,,0.12\n",
    )
    txns = csv_grammar.parse_statement(SCHWAB, path)
    assert len(txns) == 2                              # "Buy" row filtered out
    assert txns[0].date == datetime.date(2026, 9, 14)  # " as of ..." stripped
    assert txns[1].statement_memo == "INT"


def test_header_only_file_yields_nothing(tmp_path: pathlib.Path) -> None:
    path: str = _write(tmp_path, "empty.csv", "Date,Memo,Amount\n")
    assert csv_grammar.parse_statement(WELLS, path) == []


def test_short_row_raises_parse_error(tmp_path: pathlib.Path) -> None:
    path: str = _write(tmp_path, "short.csv", "Date,Memo,Amount\n09/14/2026,ONLYMEMO\n")
    with pytest.raises(StatementParseError):
        csv_grammar.parse_statement(WELLS, path)


def test_bad_amount_raises_parse_error(tmp_path: pathlib.Path) -> None:
    path: str = _write(tmp_path, "bad.csv", "Date,Memo,Amount\n09/14/2026,X,(4.75)\n")
    with pytest.raises(StatementParseError):
        csv_grammar.parse_statement(WELLS, path)
