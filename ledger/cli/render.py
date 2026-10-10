"""
Render transactions as CSV or Markdown tables for `transactions view`.
"""


from ledger.convert import convert_cents_to_amount, convert_date_to_string
from ledger.errors import InvalidCategoryError
from ledger.models import Transaction


DEFAULT_COLUMNS: list[str] = ["Date", "Amount", "Account", "Category", "Description", "Memo"]


def _fields(txn: Transaction, invert_signs: bool) -> dict[str, str]:
    cents: int = -txn.cents if (invert_signs) else txn.cents
    return {
        "Date": convert_date_to_string(txn.date),
        "Amount": convert_cents_to_amount(cents),
        "Account": txn.account,
        "Category": txn.category,
        "Description": txn.statement_memo,
        "Memo": txn.user_memo,
    }


def render(transactions: list[Transaction], columns: list[str] | None = None,
           fmt: str = "markdown", show_total: bool = False, show_filters: bool = False,
           filters: list[str] | None = None, invert_signs: bool = False) -> str:
    columns = columns if (columns is not None) else DEFAULT_COLUMNS
    for column in columns:
        if (column not in DEFAULT_COLUMNS):
            raise InvalidCategoryError(f"column '{column}' is not supported")

    rows: list[dict[str, str]] = [_fields(txn, invert_signs) for txn in transactions]
    if (show_total):
        total_cents: int = sum((-txn.cents if (invert_signs) else txn.cents)
                               for txn in transactions)
        total_row: dict[str, str] = {column: "" for column in DEFAULT_COLUMNS}
        total_row["Date"] = "-- TOTAL --"
        total_row["Amount"] = convert_cents_to_amount(total_cents)
        rows.append(total_row)

    lines: list[str] = []
    if (fmt == "csv"):
        if (show_filters and filters is not None):
            lines.append("# " + ", ".join(filters))
        lines.append(",".join(columns))
        for row in rows:
            lines.append(",".join(row[column] for column in columns))
    elif (fmt == "markdown"):
        widths: dict[str, int] = {column: len(column) for column in columns}
        for row in rows:
            for column in columns:
                widths[column] = max(widths[column], len(row[column]))
        if (show_filters and filters is not None):
            lines.append("# " + ", ".join(filters))
        lines.append("| " + " | ".join(column.ljust(widths[column]) for column in columns) + " |")
        lines.append("| " + " | ".join("-" * widths[column] for column in columns) + " |")
        for row in rows:
            lines.append("| " + " | ".join(row[column].ljust(widths[column])
                                           for column in columns) + " |")
    else:
        raise InvalidCategoryError(f"unknown format '{fmt}'")
    return "\n".join(lines)
