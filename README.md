# Ledger
A basic command line ledger for reconciling and storing CSV transaction information.

The motivation for this program comes from a dissatisfaction with closed source financial software
like Quicken. For most use cases, ledger software really doesn't need to have that many features, so
the purpose of this program is to provide a nice interface for organizing transaction data locally.

## Usage
The ledger utility is divided into different sub-commands that operate on data organized into
one or more accounts.

The top level commands are:

| Command        | Purpose                                                                        |
|----------------|--------------------------------------------------------------------------------|
| `accounts`     | Lists all the existing accounts created with the ledger utility.               |
| `adjust`       | Adds a special adjustment transaction to an account to reach a target balance. |
| `category`     | Manages transaction categories for an account.                                 |
| `create`       | Creates a new account with no transactions.                                    |
| `reconcile`    | Reconciles transactions in a statement file to an account.                     |
| `transactions` | Exports transaction data in an account in one of several formats.              |

See the help pages in each of these sub-commands for more information on their usage.

## Data Organization
Accounts are created in a `.ledger` directory in the user's home. Each account gets its own data
folder named after the account, which means that account names must be unique.

The structure of an account folder is:

- `<account>.metadata`: Contains information about the account. Currently this is a YAML file with
  the fields `statement_type` for the account statement type enumeration and `categories` for the
  list of supported categories.
- `<account>.transactions`: Contains `Transaction` objects serialized as CSV data.
