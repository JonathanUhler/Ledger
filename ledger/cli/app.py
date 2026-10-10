"""
The headless CLI: an argparse command tree over the Ledger service.
"""


from argparse import ArgumentParser, Namespace
import sys

import yaml

from ledger import config
from ledger.convert import convert_amount_to_cents, convert_cents_to_amount, \
    convert_bps_to_percent, convert_date_to_string, convert_percent_to_bps, convert_string_to_date
from ledger.core import search as search_module
from ledger.core.service import Ledger
from ledger.cli import render
from ledger.cli.reconcile_loop import run_cli_reconcile
from ledger.errors import LedgerError
from ledger.models import new_transaction
from ledger.migrate import upgrade as upgrade_module
from ledger.statements import grammar


def cmd_accounts(ledger: Ledger, _args: Namespace) -> int:
    summaries = ledger.list_accounts()
    total: int = 0
    for summary in summaries:
        print(f"Account: {summary.name}")
        print(f"  Net balance:            {convert_cents_to_amount(summary.balance_cents)}")
        print(f"  Number of transactions: {summary.count}")
        print(f"  Latest transaction:     {summary.latest or 'N/A'}")
        print(f"  Earliest transaction:   {summary.earliest or 'N/A'}")
        print(f"  Statement type:         {summary.statement_type}")
        print()
        total += summary.balance_cents
    print("----------------------------------------")
    print(f"Number of accounts: {len(summaries)}")
    print(f"Total net balance:  {convert_cents_to_amount(total)}")
    return 0


def cmd_categories_add(ledger: Ledger, args: Namespace) -> int:
    ledger.add_category(args.account, args.name, recursive = args.recursive)
    print(f"Added category '{args.name}' to '{args.account}'")
    return 0


def cmd_categories_addall(ledger: Ledger, args: Namespace) -> int:
    ledger.add_category_all(args.name, recursive = args.recursive)
    print(f"Added category '{args.name}' to all accounts")
    return 0


def cmd_categories_import(ledger: Ledger, args: Namespace) -> int:
    count: int = ledger.import_categories(args.dest, args.src)
    print(f"Imported {count} categories from '{args.src}' to '{args.dest}'")
    return 0


def cmd_categories_list(ledger: Ledger, args: Namespace) -> int:
    for category in ledger.list_categories(args.account):
        print(f"  {category}")
    return 0


def cmd_create_account(ledger: Ledger, args: Namespace) -> int:
    ledger.create_account(args.name, args.statement_type)
    print(f"Created account '{args.name}'")
    return 0


def cmd_create_type(_ledger: Ledger, args: Namespace) -> int:
    columns: dict[str, int] = {}
    for pair in args.column:
        key, _, value = pair.partition("=")
        columns[key] = int(value)
    data: dict = {
        "name": args.name,
        "kind": args.kind,
        "csv": {
            "num_header_rows": args.header_rows,
            "date": {"index": args.date_index, "format": args.date_format},
            "amount": {"index": args.amount_index},
            "memo": {"index": args.memo_index},
            "columns": columns,
        },
    }
    grammar.load_grammar_from_dict(data, args.name)   # validate before writing
    config.get_types_dir().mkdir(parents = True, exist_ok = True)
    config.type_path(args.name).write_text(yaml.safe_dump(data, sort_keys = False),
                                           encoding = "utf-8")
    print(f"Created statement type '{args.name}'. Edit {config.type_path(args.name)} "
          "to add filter/transform rules.")
    return 0


def cmd_txn_add(ledger: Ledger, args: Namespace) -> int:
    txn = new_transaction(date = args.date, cents = args.amount, statement_memo = args.memo,
                          account = args.account, user_memo = args.user_memo or "",
                          category = args.category, reconciled = True)
    ledger.add_transaction(args.account, txn)
    print(f"Added transaction {txn.id}")
    return 0


def cmd_txn_edit(ledger: Ledger, args: Namespace) -> int:
    fields: dict[str, object] = {}
    if (args.date is not None):
        fields["date"] = args.date
    if (args.amount is not None):
        fields["cents"] = args.amount
    if (args.memo is not None):
        fields["statement_memo"] = args.memo
    if (args.user_memo is not None):
        fields["user_memo"] = args.user_memo
    if (args.category is not None):
        fields["category"] = args.category
    ledger.edit_transaction(args.account, args.id, **fields)
    print(f"Edited transaction {args.id}")
    return 0


def cmd_txn_remove(ledger: Ledger, args: Namespace) -> int:
    ledger.remove_transaction(args.account, args.id)
    print(f"Removed transaction {args.id}")
    return 0


def cmd_txn_adjust(ledger: Ledger, args: Namespace) -> int:
    created = ledger.adjust(args.account, convert_amount_to_cents(args.target_amount))
    if (created is None):
        print("Account balance already on target; no adjustment added")
    else:
        print(f"Added adjustment for {convert_cents_to_amount(created.cents)}")
    return 0


def cmd_txn_reconcile(ledger: Ledger, args: Namespace) -> int:
    try:
        import readline

        categories: list[str] = ledger.list_categories(args.account)

        def completer(text: str, state: int) -> str | None:
            matches: list[str] = [c for c in categories if (text.lower() in c.lower())]
            return matches[state] if (state < len(matches)) else None

        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")
    except Exception:   # pylint: disable=broad-exception-caught
        pass
    run_cli_reconcile(ledger, args.account, args.statement)
    return 0


def cmd_txn_view(ledger: Ledger, args: Namespace) -> int:
    searchers: list[search_module.Searcher] = []
    filters: list[str] = []
    if (args.all):
        searchers.append(search_module.unconditional())
        filters.append("all")
    if (args.by_dates):
        searchers.append(search_module.date_range(args.by_dates[0], args.by_dates[1]))
        filters.append(f"dates {args.by_dates[0]}..{args.by_dates[1]}")
    if (args.by_amounts):
        searchers.append(search_module.amount_range(args.by_amounts[0], args.by_amounts[1]))
        filters.append(f"amounts {args.by_amounts[0]}..{args.by_amounts[1]}")
    if (args.by_category):
        searchers.append(search_module.category_regex(args.by_category))
        filters.append("categories " + "|".join(args.by_category))
    if (len(searchers) == 0):
        searchers.append(search_module.unconditional())

    transactions = ledger.search(args.account, searchers)
    transactions.sort(key = lambda txn: (convert_date_to_string(txn.date), txn.id), reverse = True)
    print(render.render(transactions, columns = args.show_columns, fmt = args.format,
                        show_total = args.show_total, show_filters = args.show_filters,
                        filters = filters, invert_signs = args.invert_signs))
    return 0


def cmd_upgrade(_ledger: Ledger, args: Namespace) -> int:
    reports = upgrade_module.upgrade(names = args.account, force = args.force)
    if (len(reports) == 0):
        print("No v1 accounts found to upgrade")
        return 0
    failures: int = 0
    for report in reports:
        status: str = "migrated" if (report.migrated) \
            else ("skipped" if (report.skipped) else "FAILED")
        print(f"{report.account}: {status} — {report.message}")
        if (report.migrated):
            print(f"  count={report.count}  v1_sum={convert_cents_to_amount(report.v1_sum)}  "
                  f"v2_sum={convert_cents_to_amount(report.v2_sum)}")
        if (not report.migrated and not report.skipped):
            failures += 1
    return 1 if (failures > 0) else 0


def cmd_budget_add(ledger: Ledger, args: Namespace) -> int:
    budget = ledger.add_budget(name = args.name, goal_cents = args.goal, category = args.category,
                               start_date = args.start, account = args.account)
    print(f"Added budget {budget.id}")
    return 0


def cmd_budget_edit(ledger: Ledger, args: Namespace) -> int:
    fields: dict[str, object] = {}
    if (args.name is not None):
        fields["name"] = args.name
    if (args.goal is not None):
        fields["goal_cents"] = args.goal
    if (args.category is not None):
        fields["category"] = args.category
    if (args.account is not None):
        fields["account"] = args.account
    if (args.start is not None):
        fields["start_date"] = args.start
    ledger.edit_budget(args.id, **fields)
    print(f"Edited budget {args.id}")
    return 0


def cmd_budget_remove(ledger: Ledger, args: Namespace) -> int:
    ledger.remove_budget(args.id)
    print(f"Removed budget {args.id}")
    return 0


def cmd_budget_view(ledger: Ledger, args: Namespace) -> int:
    budgets = ledger.list_budgets()
    if (len(budgets) == 0):
        print("No budgets")
        return 0
    by_category: dict[str, list] = {}
    for budget in budgets:
        by_category.setdefault(budget.category, []).append(budget)
    for category in sorted(by_category):
        cap_bps: int = ledger.get_category_cap(category)
        print(f"{category} — cap {convert_bps_to_percent(cap_bps)}")
        for position, budget in enumerate(by_category[category], start = 1):
            progress = ledger.budget_progress(budget)
            scope: str = budget.account if (budget.account) else "all accounts"
            status: str = "FROZEN" if (progress.is_frozen) \
                else ("MET" if (progress.is_met) else "in progress")
            pct: int = 0 if (budget.goal_cents == 0) \
                else round(progress.saved_cents * 100 / budget.goal_cents)
            print(f"  {position}. {budget.name}  [{budget.id}]  {status}")
            print(f"       {convert_cents_to_amount(progress.saved_cents)} / "
                  f"{convert_cents_to_amount(budget.goal_cents)} ({pct}%)  "
                  f"remaining {convert_cents_to_amount(progress.remaining_cents)}  ({scope})")
        print()
    return 0


def cmd_budget_move(ledger: Ledger, args: Namespace) -> int:
    ledger.reorder_budget(args.id, args.direction)
    print(f"Moved budget {args.id} {args.direction}")
    return 0


def cmd_budget_setcap(ledger: Ledger, args: Namespace) -> int:
    ledger.set_category_cap(args.category, args.percent)
    print(f"Set cap for {args.category}")
    return 0


def cmd_budget_clearcap(ledger: Ledger, args: Namespace) -> int:
    ledger.clear_category_cap(args.category)
    print(f"Cleared cap for {args.category}")
    return 0


def _require_subcommand(_ledger: Ledger, _args: Namespace) -> int:
    print("error: a subcommand is required", file = sys.stderr)
    return 1


def build_parser() -> ArgumentParser:
    parser: ArgumentParser = ArgumentParser(prog = "ledger",
                                            description = "A lean local financial ledger.")
    parser.add_argument("-v", "--verbose", action = "store_true", help = "verbose output")
    commands = parser.add_subparsers(dest = "command")

    accounts_parser = commands.add_parser("accounts", help = "summarize all accounts")
    accounts_parser.set_defaults(handler = cmd_accounts)

    categories_parser = commands.add_parser("categories", help = "manage categories")
    categories_parser.set_defaults(handler = _require_subcommand)
    categories_sub = categories_parser.add_subparsers(dest = "subcommand")
    cat_add = categories_sub.add_parser("add")
    cat_add.add_argument("account")
    cat_add.add_argument("name")
    cat_add.add_argument("--recursive", action = "store_true")
    cat_add.set_defaults(handler = cmd_categories_add)
    cat_addall = categories_sub.add_parser("addall")
    cat_addall.add_argument("name")
    cat_addall.add_argument("--recursive", action = "store_true")
    cat_addall.set_defaults(handler = cmd_categories_addall)
    cat_import = categories_sub.add_parser("import")
    cat_import.add_argument("dest")
    cat_import.add_argument("src")
    cat_import.set_defaults(handler = cmd_categories_import)
    cat_list = categories_sub.add_parser("list")
    cat_list.add_argument("account")
    cat_list.set_defaults(handler = cmd_categories_list)

    create_parser = commands.add_parser("create", help = "create accounts and types")
    create_parser.set_defaults(handler = _require_subcommand)
    create_sub = create_parser.add_subparsers(dest = "subcommand")
    create_account = create_sub.add_parser("account")
    create_account.add_argument("name")
    create_account.add_argument("statement_type")
    create_account.set_defaults(handler = cmd_create_account)
    create_type = create_sub.add_parser("type")
    create_type.add_argument("name")
    create_type.add_argument("--kind", default = "csv", choices = ["csv"])
    create_type.add_argument("--date-index", dest = "date_index", type = int, required = True)
    create_type.add_argument("--date-format", dest = "date_format", required = True)
    create_type.add_argument("--amount-index", dest = "amount_index", type = int, required = True)
    create_type.add_argument("--memo-index", dest = "memo_index", type = int, required = True)
    create_type.add_argument("--header-rows", dest = "header_rows", type = int, default = 1)
    create_type.add_argument("--column", action = "append", default = [], metavar = "NAME=IDX")
    create_type.set_defaults(handler = cmd_create_type)

    txn_parser = commands.add_parser("transactions", help = "view, add, and update transactions")
    txn_parser.set_defaults(handler = _require_subcommand)
    txn_sub = txn_parser.add_subparsers(dest = "subcommand")
    txn_add = txn_sub.add_parser("add")
    txn_add.add_argument("account")
    txn_add.add_argument("--date", type = convert_string_to_date, required = True)
    txn_add.add_argument("--amount", type = convert_amount_to_cents, required = True)
    txn_add.add_argument("--memo", required = True)
    txn_add.add_argument("--category", required = True)
    txn_add.add_argument("--user-memo", dest = "user_memo", default = "")
    txn_add.set_defaults(handler = cmd_txn_add)
    txn_edit = txn_sub.add_parser("edit")
    txn_edit.add_argument("account")
    txn_edit.add_argument("id")
    txn_edit.add_argument("--date", type = convert_string_to_date, default = None)
    txn_edit.add_argument("--amount", type = convert_amount_to_cents, default = None)
    txn_edit.add_argument("--memo", default = None)
    txn_edit.add_argument("--user-memo", dest = "user_memo", default = None)
    txn_edit.add_argument("--category", default = None)
    txn_edit.set_defaults(handler = cmd_txn_edit)
    txn_remove = txn_sub.add_parser("remove")
    txn_remove.add_argument("account")
    txn_remove.add_argument("id")
    txn_remove.set_defaults(handler = cmd_txn_remove)
    txn_adjust = txn_sub.add_parser("adjust")
    txn_adjust.add_argument("account")
    txn_adjust.add_argument("target_amount")
    txn_adjust.set_defaults(handler = cmd_txn_adjust)
    txn_reconcile = txn_sub.add_parser("reconcile")
    txn_reconcile.add_argument("account")
    txn_reconcile.add_argument("statement")
    txn_reconcile.set_defaults(handler = cmd_txn_reconcile)
    txn_view = txn_sub.add_parser("view")
    txn_view.add_argument("account", nargs = "*", default = [])
    txn_view.add_argument("--all", action = "store_true")
    txn_view.add_argument("--by_dates", nargs = 2, type = convert_string_to_date,
                          metavar = "YYYY-MM-DD")
    txn_view.add_argument("--by_amounts", nargs = 2, type = convert_amount_to_cents,
                          metavar = "AMOUNT")
    txn_view.add_argument("--by_category", nargs = "+", metavar = "PATTERN")
    txn_view.add_argument("--format", choices = ["csv", "markdown"], default = "markdown")
    txn_view.add_argument("--show_columns", nargs = "+", default = None)
    txn_view.add_argument("--show_total", action = "store_true")
    txn_view.add_argument("--show_filters", action = "store_true")
    txn_view.add_argument("--invert_signs", action = "store_true")
    txn_view.set_defaults(handler = cmd_txn_view)

    budget_parser = commands.add_parser("budget", help = "manage savings budgets")
    budget_parser.set_defaults(handler = _require_subcommand)
    budget_sub = budget_parser.add_subparsers(dest = "subcommand")
    budget_add = budget_sub.add_parser("add")
    budget_add.add_argument("name")
    budget_add.add_argument("--goal", type = convert_amount_to_cents, required = True)
    budget_add.add_argument("--category", required = True)
    budget_add.add_argument("--account", default = None)
    budget_add.add_argument("--start", type = convert_string_to_date, default = None,
                            metavar = "YYYY-MM-DD")
    budget_add.set_defaults(handler = cmd_budget_add)
    budget_edit = budget_sub.add_parser("edit")
    budget_edit.add_argument("id")
    budget_edit.add_argument("--name", default = None)
    budget_edit.add_argument("--goal", type = convert_amount_to_cents, default = None)
    budget_edit.add_argument("--category", default = None)
    budget_edit.add_argument("--account", default = None)
    budget_edit.add_argument("--start", type = convert_string_to_date, default = None,
                             metavar = "YYYY-MM-DD")
    budget_edit.set_defaults(handler = cmd_budget_edit)
    budget_remove = budget_sub.add_parser("remove")
    budget_remove.add_argument("id")
    budget_remove.set_defaults(handler = cmd_budget_remove)
    budget_view = budget_sub.add_parser("view")
    budget_view.set_defaults(handler = cmd_budget_view)
    budget_move = budget_sub.add_parser("move")
    budget_move.add_argument("id")
    budget_move.add_argument("direction", choices = ["up", "down"])
    budget_move.set_defaults(handler = cmd_budget_move)
    budget_setcap = budget_sub.add_parser("setcap")
    budget_setcap.add_argument("category")
    budget_setcap.add_argument("percent", type = convert_percent_to_bps)
    budget_setcap.set_defaults(handler = cmd_budget_setcap)
    budget_clearcap = budget_sub.add_parser("clearcap")
    budget_clearcap.add_argument("category")
    budget_clearcap.set_defaults(handler = cmd_budget_clearcap)

    upgrade_parser = commands.add_parser("upgrade", help = "upgrade v1 accounts to v2")
    upgrade_parser.add_argument("account", nargs = "*", default = [])
    upgrade_parser.add_argument("--force", action = "store_true")
    upgrade_parser.set_defaults(handler = cmd_upgrade)

    return parser


def main(argv: list[str]) -> int:
    parser: ArgumentParser = build_parser()
    try:
        args: Namespace = parser.parse_args(argv)
    except SystemExit as exit_error:
        return 1 if (exit_error.code not in (0, None)) else 0
    if (getattr(args, "handler", None) is None):
        parser.print_usage()
        return 1
    ledger: Ledger = Ledger()
    try:
        return args.handler(ledger, args)
    except LedgerError as error:
        print(f"error: {error}", file = sys.stderr)
        return 1
