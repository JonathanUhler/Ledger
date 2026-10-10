import pytest

from ledger.statements import grammar
from ledger.errors import GrammarError


WELLS: dict = {
    "name": "wells_fargo_csv", "kind": "csv",
    "csv": {"num_header_rows": 1, "date": {"index": 0, "format": "%m/%d/%Y"},
            "amount": {"index": 2}, "memo": {"index": 1}},
}

SCHWAB: dict = {
    "name": "charles_schwab_csv", "kind": "csv",
    "csv": {"num_header_rows": 1, "date": {"index": 0, "format": "%m/%d/%Y"},
            "amount": {"index": 7}, "memo": {"index": 3}, "columns": {"action": 1}},
    "filter": {"all": [{"column": "action", "op": "in",
                        "values": ["MoneyLink Transfer", "Credit Interest"]}]},
    "transform": [{"when": {"column": "action", "op": "equals", "value": "MoneyLink Transfer"},
                   "target": "date", "op": "split", "sep": " as of ", "take": 0}],
}


def test_load_wells() -> None:
    g: grammar.Grammar = grammar.load_grammar_from_dict(WELLS, "wells_fargo_csv")
    assert g.csv.amount_index == 2
    assert g.csv.date_format == "%m/%d/%Y"
    assert g.filter is None


def test_load_schwab_resolves_named_target() -> None:
    g: grammar.Grammar = grammar.load_grammar_from_dict(SCHWAB, "charles_schwab_csv")
    assert g.csv.columns["action"] == 1
    assert g.transform[0]["target"] == "date"   # "date" is a reserved alias (= date index)


def test_bad_kind_raises() -> None:
    with pytest.raises(GrammarError):
        grammar.load_grammar_from_dict({"name": "x", "kind": "pdf", "csv": {}}, "x")


def test_bad_regex_raises_at_load() -> None:
    bad: dict = {**WELLS, "filter": {"column": 1, "op": "matches", "value": "("}}
    with pytest.raises(GrammarError):
        grammar.load_grammar_from_dict(bad, "x")


def test_negative_index_raises() -> None:
    bad: dict = {"name": "x", "kind": "csv",
                 "csv": {"num_header_rows": 0, "date": {"index": -1, "format": "%Y-%m-%d"},
                         "amount": {"index": 2}, "memo": {"index": 1}}}
    with pytest.raises(GrammarError):
        grammar.load_grammar_from_dict(bad, "x")


def test_unknown_named_column_raises() -> None:
    bad: dict = {**WELLS, "filter": {"column": "nope", "op": "equals", "value": "x"}}
    with pytest.raises(GrammarError):
        grammar.load_grammar_from_dict(bad, "x")
