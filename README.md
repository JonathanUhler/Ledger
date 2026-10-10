# Ledger

A simple CLI- and TUI-based financial ledger for reconciling and storing bank-statement transactions.

## Installation

The core ledger is installed as a Python library, usually in a virtual environment:

```
python3 -m venv ./venv/
source ./venv/bin/activate
pip install -e .
```

## Usage

Once installed, the ledger can be opened as follows:

- `ledger`: launch a TUI session
- `ledger <command> ...`: run a headless CLI command

The available commands are as follows, see `--help` on any command for more information.

```
accounts: List a summary of all accounts

categories  add: Add a new category to a specific account
            addall: Add a new category to all accounts
            import: Import all categories from one account to another
            list: List the categories associated with an account

create  account: Create a new account with no data
        type: Create a parsing grammar for a new statement file type

transactions  add: Manually add a new transaction to an account
              edit: Manually edit an existing transaction
              remove: Manually delete an existing transaction
              adjust: Manually update the balance of an account with a dummy transaction
              reconcile: Add and reconcile all transactions from a bank statement file
              view: View and analyze transactions from one or more accounts

upgrade: Migrate transaction data from older versions of the ledger app
```

## Data

By default, all data used by the ledger application is created at `~/.ledger/`. This location can
be overriden by setting the `LEDGER_HOME` environment variable to the preferred path.

The following files are created when the ledger is used:

- `accounts/<name>.db`: one SQLite database per account (plus a single `.db.bak` backup file)
- `types/<name>.yaml`: parsing grammar descriptions for statement files

The current data model version is `2`.

## Statement Grammars

Statement grammar files describe how the ledger should parse the information in bank statements to
extract the transaction details that are actually stored in the SQLite databases.

Currently, only grammar files for CSV-based statements are supported. Arbitrary code execution is
not allowed, so statement files must conform to some basic assumptions. A minimal grammar might
look like this:

```yaml
name: wells_fargo_csv
kind: csv
csv:
  num_header_rows: 1
  date:   { index: 0, format: "%m/%d/%Y" }
  amount: { index: 2 }
  memo:   { index: 1 }
```

More complex grammars might require transformations and filters:

```yaml
name: charles_schwab_csv
kind: csv
csv:
  num_header_rows: 1
  date:   { index: 0, format: "%m/%d/%Y" }
  amount: { index: 7 }
  memo:   { index: 3 }
  columns:
    action: 1
filter:
  all:
    - { column: action, op: in, values: ["MoneyLink Transfer", "Reinvest Dividend",
                                          "Credit Interest", "Short Term Cap Gain Reinvest"] }
transform:
  - when: { column: action, op: equals, value: "MoneyLink Transfer" }
    target: date
    op: split
    sep: " as of "
    take: 0
```
