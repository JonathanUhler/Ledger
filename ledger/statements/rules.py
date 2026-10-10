"""
A small, safe, declarative rule engine for statement grammars (no code execution).
"""


import re

from ledger.errors import GrammarError, StatementParseError


CONDITION_OPS: frozenset[str] = frozenset({
    "equals", "not_equals", "in", "not_in", "contains", "matches", "empty", "not_empty",
})
TRANSFORM_OPS: frozenset[str] = frozenset({"split", "replace", "strip"})


def resolve_column(ref: int | str, columns: dict[str, int]) -> int:
    if (isinstance(ref, bool)):
        raise GrammarError(f"invalid column reference '{ref}'")
    if (isinstance(ref, int)):
        return ref
    if (isinstance(ref, str) and ref in columns):
        return columns[ref]
    raise GrammarError(f"unknown column reference '{ref}'")


def _cell(row: list[str], ref: int | str, columns: dict[str, int]) -> str:
    index: int = resolve_column(ref, columns)
    if (index < 0 or index >= len(row)):
        return ""
    return row[index]


def match_condition(cond: dict, row: list[str], columns: dict[str, int]) -> bool:
    op: str = cond.get("op")
    if (op not in CONDITION_OPS):
        raise GrammarError(f"unknown condition op '{op}'")
    cell: str = _cell(row, cond.get("column"), columns)

    if (op == "equals"):
        return cell == str(cond.get("value"))
    if (op == "not_equals"):
        return cell != str(cond.get("value"))
    if (op == "in"):
        return cell in list(cond.get("values", []))
    if (op == "not_in"):
        return cell not in list(cond.get("values", []))
    if (op == "contains"):
        return str(cond.get("value")) in cell
    if (op == "matches"):
        return re.search(str(cond.get("value")), cell) is not None
    if (op == "empty"):
        return cell.strip() == ""
    # op == "not_empty"
    return cell.strip() != ""


def match_filter(node: dict, row: list[str], columns: dict[str, int]) -> bool:
    if (not isinstance(node, dict)):
        raise GrammarError("filter node must be a mapping")
    if ("all" in node):
        return all(match_filter(child, row, columns) for child in node["all"])
    if ("any" in node):
        return any(match_filter(child, row, columns) for child in node["any"])
    if ("not" in node):
        return not match_filter(node["not"], row, columns)
    if ("column" in node):
        return match_condition(node, row, columns)
    raise GrammarError(f"invalid filter node {node!r}")


def apply_transform(transform: dict, row: list[str], columns: dict[str, int]) -> None:
    when: dict | None = transform.get("when")
    if (when is not None and not match_condition(when, row, columns)):
        return

    op: str = transform.get("op")
    if (op not in TRANSFORM_OPS):
        raise GrammarError(f"unknown transform op '{op}'")

    index: int = resolve_column(transform.get("target"), columns)
    if (index < 0 or index >= len(row)):
        raise StatementParseError(f"transform target column {index} is out of range")
    cell: str = row[index]

    if (op == "split"):
        parts: list[str] = cell.split(transform.get("sep", ""))
        take: int = int(transform.get("take", 0))
        if (take < 0 or take >= len(parts)):
            raise StatementParseError(f"split take index {take} out of range for '{cell}'")
        row[index] = parts[take]
    elif (op == "replace"):
        row[index] = re.sub(str(transform.get("pattern")), str(transform.get("replacement", "")),
                            cell)
    else:  # op == "strip"
        row[index] = cell.strip()
