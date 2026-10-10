"""
Parse a CSV statement file into transactions using a grammar.
"""


import csv

from ledger import convert
from ledger.errors import (
    AmountConversionError, DateConversionError, StatementFileError, StatementParseError
)
from ledger.models import Transaction, new_transaction
from ledger.statements import rules
from ledger.statements.grammar import Grammar


def _row_to_transaction(grammar: Grammar, row: list[str], columns: dict[str, int]) -> Transaction:
    try:
        date = convert.convert_string_to_date(row[grammar.csv.date_index],
                                              date_format = grammar.csv.date_format)
        cents: int = convert.convert_amount_to_cents(row[grammar.csv.amount_index])
        statement_memo: str = row[grammar.csv.memo_index]
    except IndexError as error:
        raise StatementParseError("statement row is missing fields") from error
    except (AmountConversionError, DateConversionError) as error:
        raise StatementParseError(f"cannot parse statement row: {error}") from error
    return new_transaction(date = date, cents = cents, statement_memo = statement_memo)


def parse_statement(grammar: Grammar, statement_path: str) -> list[Transaction]:
    try:
        with open(statement_path, "r", encoding = "utf-8", newline = "") as handle:
            rows: list[list[str]] = list(csv.reader(handle))
    except OSError as error:
        raise StatementFileError(f"cannot open statement: {error}") from error

    if (len(rows) >= grammar.csv.num_header_rows):
        rows = rows[grammar.csv.num_header_rows:]

    columns: dict[str, int] = grammar.resolver_columns()
    transactions: list[Transaction] = []
    for row in rows:
        if (grammar.filter is not None and not rules.match_filter(grammar.filter, row, columns)):
            continue
        for transform in grammar.transform:
            rules.apply_transform(transform, row, columns)
        transactions.append(_row_to_transaction(grammar, row, columns))
    return transactions
