import pytest

from ledger.statements import rules
from ledger.errors import GrammarError, StatementParseError


COLS: dict[str, int] = {"action": 1}


def test_condition_in() -> None:
    cond: dict = {"column": "action", "op": "in", "values": ["MoneyLink Transfer", "Credit Interest"]}
    assert rules.match_condition(cond, ["2026-01-01", "Credit Interest"], COLS) is True
    assert rules.match_condition(cond, ["2026-01-01", "Buy"], COLS) is False


def test_condition_matches_regex() -> None:
    cond: dict = {"column": 0, "op": "matches", "value": r"^2026-"}
    assert rules.match_condition(cond, ["2026-09-14"], {}) is True
    assert rules.match_condition(cond, ["2025-09-14"], {}) is False


def test_filter_all_any_not() -> None:
    node: dict = {"all": [
        {"column": 0, "op": "not_empty"},
        {"not": {"column": 1, "op": "equals", "value": "skip"}},
    ]}
    assert rules.match_filter(node, ["x", "keep"], {}) is True
    assert rules.match_filter(node, ["x", "skip"], {}) is False
    assert rules.match_filter(node, ["", "keep"], {}) is False


def test_transform_split() -> None:
    row: list[str] = ["2026-09-14 as of 09/13", "MoneyLink Transfer"]
    transform: dict = {
        "when": {"column": 1, "op": "equals", "value": "MoneyLink Transfer"},
        "target": 0, "op": "split", "sep": " as of ", "take": 0,
    }
    rules.apply_transform(transform, row, COLS)
    assert row[0] == "2026-09-14"


def test_transform_skipped_when_guard_false() -> None:
    row: list[str] = ["2026-09-14 as of 09/13", "Buy"]
    transform: dict = {
        "when": {"column": 1, "op": "equals", "value": "MoneyLink Transfer"},
        "target": 0, "op": "split", "sep": " as of ", "take": 0,
    }
    rules.apply_transform(transform, row, COLS)
    assert row[0] == "2026-09-14 as of 09/13"


def test_unknown_op_raises_grammar_error() -> None:
    with pytest.raises(GrammarError):
        rules.match_condition({"column": 0, "op": "bogus"}, ["x"], {})


def test_split_take_out_of_range_raises_parse_error() -> None:
    transform: dict = {"target": 0, "op": "split", "sep": ",", "take": 5}
    with pytest.raises(StatementParseError):
        rules.apply_transform(transform, ["a,b"], {})
