"""
Grammar files: the declarative description of how to parse a statement into transactions.
"""


from dataclasses import dataclass, field
import re

import yaml

from ledger import config
from ledger.errors import GrammarError, UnsupportedStatementError
from ledger.statements import rules


@dataclass
class CsvSpec:
    num_header_rows: int
    date_index: int
    date_format: str
    amount_index: int
    memo_index: int
    columns: dict[str, int] = field(default_factory = dict)


@dataclass
class Grammar:
    name: str
    kind: str
    csv: CsvSpec
    filter: dict | None = None
    transform: list[dict] = field(default_factory = list)


    def resolver_columns(self) -> dict[str, int]:
        """
        All name->index aliases usable in rules, including the reserved field names.
        """

        resolved: dict[str, int] = {
            "date": self.csv.date_index,
            "amount": self.csv.amount_index,
            "memo": self.csv.memo_index,
        }
        resolved.update(self.csv.columns)
        return resolved


def _require_int(value: object, label: str) -> int:
    if (not isinstance(value, int) or isinstance(value, bool)):
        raise GrammarError(f"{label} must be an integer")
    return value


def load_grammar_from_dict(data: dict, name: str) -> Grammar:
    if (not isinstance(data, dict)):
        raise GrammarError("grammar must be a mapping")
    kind: str = data.get("kind")
    if (kind != "csv"):
        raise GrammarError(f"unsupported grammar kind '{kind}' (only 'csv' is supported)")

    csv_data: dict = data.get("csv")
    if (not isinstance(csv_data, dict)):
        raise GrammarError("grammar is missing a 'csv' block")

    try:
        date_block: dict = csv_data["date"]
        amount_block: dict = csv_data["amount"]
        memo_block: dict = csv_data["memo"]
        spec: CsvSpec = CsvSpec(
            num_header_rows = _require_int(csv_data.get("num_header_rows", 0), "num_header_rows"),
            date_index = _require_int(date_block["index"], "date.index"),
            date_format = str(date_block["format"]),
            amount_index = _require_int(amount_block["index"], "amount.index"),
            memo_index = _require_int(memo_block["index"], "memo.index"),
            columns = {str(key): _require_int(value, f"columns.{key}")
                       for key, value in dict(csv_data.get("columns", {})).items()},
        )
    except KeyError as error:
        raise GrammarError(f"grammar is missing required field {error}") from error

    grammar: Grammar = Grammar(
        name = name,
        kind = kind,
        csv = spec,
        filter = data.get("filter"),
        transform = list(data.get("transform", [])),
    )
    validate_grammar(grammar)
    return grammar


def load_grammar(name: str) -> Grammar:
    path = config.type_path(name)
    if (not path.exists()):
        raise UnsupportedStatementError(f"statement type '{name}' does not exist")
    try:
        data: dict = yaml.safe_load(path.read_text(encoding = "utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise GrammarError(f"cannot read grammar '{name}': {error}") from error
    return load_grammar_from_dict(data, name)


def _validate_condition(cond: dict, columns: dict[str, int]) -> None:
    if (not isinstance(cond, dict) or "op" not in cond):
        raise GrammarError(f"invalid condition {cond!r}")
    if (cond["op"] not in rules.CONDITION_OPS):
        raise GrammarError(f"unknown condition op '{cond['op']}'")
    rules.resolve_column(cond.get("column"), columns)
    if (cond["op"] == "matches"):
        try:
            re.compile(str(cond.get("value")))
        except re.error as error:
            raise GrammarError(f"invalid regex '{cond.get('value')}': {error}") from error


def _validate_filter(node: dict, columns: dict[str, int]) -> None:
    if (not isinstance(node, dict)):
        raise GrammarError("filter node must be a mapping")
    if ("all" in node or "any" in node):
        children: list = node.get("all", node.get("any"))
        if (not isinstance(children, list)):
            raise GrammarError("'all'/'any' must contain a list")
        for child in children:
            _validate_filter(child, columns)
    elif ("not" in node):
        _validate_filter(node["not"], columns)
    elif ("column" in node):
        _validate_condition(node, columns)
    else:
        raise GrammarError(f"invalid filter node {node!r}")


def validate_grammar(grammar: Grammar) -> None:
    for label, index in (("date", grammar.csv.date_index), ("amount", grammar.csv.amount_index),
                         ("memo", grammar.csv.memo_index)):
        if (index < 0):
            raise GrammarError(f"{label}.index must be non-negative")
    if (grammar.csv.num_header_rows < 0):
        raise GrammarError("num_header_rows must be non-negative")

    columns: dict[str, int] = grammar.resolver_columns()
    if (grammar.filter is not None):
        _validate_filter(grammar.filter, columns)

    for transform in grammar.transform:
        if (not isinstance(transform, dict) or transform.get("op") not in rules.TRANSFORM_OPS):
            raise GrammarError(f"invalid transform {transform!r}")
        rules.resolve_column(transform.get("target"), columns)
        if ("when" in transform):
            _validate_condition(transform["when"], columns)
        if (transform["op"] == "replace"):
            try:
                re.compile(str(transform.get("pattern")))
            except re.error as error:
                raise GrammarError(f"invalid regex '{transform.get('pattern')}': {error}") \
                    from error
