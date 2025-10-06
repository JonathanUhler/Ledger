"""
A basic command line ledger for reconciling and storing CSV transaction information.

Author: Jonathan Uhler
"""


from argparse import ArgumentParser, Namespace
import bisect
import csv
from dataclasses import dataclass
import datetime
from enum import Enum
import readline
import logging
from logging import Formatter, Handler, Logger, StreamHandler
import os
from pathlib import Path
import sys
from typing import Callable, Final
import yaml


####################################################################################################
# LOGGING
####################################################################################################


log: Logger = logging.getLogger("run_backup")


def initialize_logger(log_level: int) -> None:
    """
    Initializes the global logger object with the specified log level.

    Args:
     log_level (int): The minimum log level (inclusive) to print to standard output.
    """

    format_str: str = "[%(levelname)s] %(message)s"
    formatter: Formatter = Formatter(format_str)

    stdout_handler: Handler = StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)

    log.setLevel(logging.DEBUG)
    log.addHandler(stdout_handler)

    log.debug(f"Logger initialized with stdout level {log_level}")


####################################################################################################
# CUSTOM EXCEPTIONS
####################################################################################################


class AmountConversionError(ValueError):
    """
    An amount of money (as a string) is invalid and cannot be converted to an integer.
    """
    pass


class DateConversionError(ValueError):
    """
    A date (as a string) is formatted incorrectly and cannot be converted to an integer.
    """
    pass


class StatementFileError(OSError):
    """
    A file I/O error occured when performing operations on a transaction statement file.
    """
    pass


class UnsupportedStatementError(ValueError):
    """
    The type/format of a transaction statement is not supported.
    """
    pass


class StatementParseError(ValueError):
    """
    A transaction statement contained invalid data and could not be parsed.
    """
    pass


class MissingAccountError(FileNotFoundError):
    """
    A specified account does not exist.
    """
    pass


class AccountFileError(OSError):
    """
    A file I/O error occured when performing operations on an account file.
    """
    pass


class AccountParseError(ValueError):
    """
    An account file contained invalid data and could not be parsed.
    """
    pass


class InvalidCategoryError(ValueError):
    """
    A transaction category does not exist or is formatted incorrectly.
    """
    pass


####################################################################################################
# TRANSACTION ABSTRACTIONS
####################################################################################################


@dataclass
class Transaction:
    """
    A structure that holds information about a single financial transaction.

    Attributes:
     date (datetime.date): The date that the transaction took place.
     cents (int):          The signed transaction amount, in cents.
     statement_memo (str): The memo provided by the transaction statement file.
     user_memo (str):      The memo provided by the user during reconciliation.
     category (str):       The category provided by the user during reconciliation.
     reconciled (bool):    Whether the transactions has been reconciled.
    """

    date: datetime.date
    cents: int
    statement_memo: str

    user_memo: str
    category: str
    reconciled: bool


    def __hash__(self) -> int:
        """
        Returns a hash code for this transaction, used to determine set membership and equality.

        The hash code is determined by the date, amount, and statement memo, but not any of the
        user-defined fields.

        Returns:
         int: Hash code for this transaction.
        """

        return hash(f"{self.date}_{self.cents}_{self.statement_memo}")


    def __eq__(self, other: 'Transaction') -> bool:
        """
        Determines whether this transactions is equal to the other transaction.

        Equality is determined based on the transaction hash code. See `__hash__` for more
        information.

        Args:
         other (Transaction): The other transaction to compare.

        Returns:
         bool: Whether this transaction is equal to the other transaction.
        """

        return hash(self) == hash(other)


####################################################################################################
# FINANCIAL TYPE SYSTEM
####################################################################################################


def convert_amount_to_cents(amount: str) -> int:
    """
    Converts an amount (as a string) to cents (as an integer).

    The expected amount format is:

      [-|+]?\$XXX,XXX,XXX.YY

    Where Xs are integers for the dollar portion of the amount and Ys are integers for the cents
    portion of the amount. Dollars and cents must be separated by a period, but the dollar sign
    and commas are optional.

    Args:
     amount (str): The amount to convert to cents.

    Returns:
     int: The amount in cents

    Raises:
     AmountConversionError: If the amount string cannot be converted
    """

    amount = amount.replace("$", "")
    amount = amount.replace(",", "")
    try:
        return int(float(amount) * 100)
    except ValueError as value_error:
        raise AmountConversionError(f"cannot convert dollar amount '{amount}'") from value_error


def convert_cents_to_amount(cents: int) -> str:
    """
    Converts a number of cents (as an integer) to an amount (as a string).

    The returned amount will be of the format:

      [-|+]?\$XXXXXXXXX.YY

    See `convert_amount_to_cents` for more information about the format.

    Args:
     cents (int): The number of cents.

    Returns:
     str: The cents as a dollar string
    """

    sign: bool = "-" if cents < 0 else ""
    cents = abs(cents)
    integral_part: int = cents // 100
    fraction_part: int = cents % 100

    integral_string: str = f"{integral_part}"
    fraction_string: str = f"{fraction_part:02}"

    return f"{sign}${integral_string}.{fraction_string}"


def convert_string_to_date(string: str, date_format: str = "%Y-%m-%d") -> datetime.date:
    """
    Converts a date (as a string) of the specified format to a date object.

    Args:
     string (str):      The string to convert.
     date_format (str): The date format for Python's datetime library.

    Returns:
     datetime.date: The date string as a date object.

    Raises:
     DateConversionError: If the date string cannot be converted.
    """

    try:
        return datetime.datetime.strptime(string, date_format).date()
    except ValueError as value_error:
        raise DateConversionError(f"cannot convert date '{string}'") from value_error


def convert_date_to_string(date: datetime.date, date_format: str = "%Y-%m-%d") -> str:
    """
    Converts a date object to a string in the specified format.

    Args:
     date (datetime.date): The date object to convert.
     date_format (str):    The date format for Python's datetime library.

    Returns:
     str: The date object as a string.
    """
    if (not isinstance(date, datetime.date)):
        return date
    return date.strftime(date_format)


####################################################################################################
# STATEMENT FILE PARSING
####################################################################################################


class Statement:
    """
    A class to hold transaction data loaded from a statement file.

    Attributes:
     col_map (ColumnMap): A description of the columns in the statement file.
     rows (list):         The raw transaction date loaded from the statement file.
    """

    class Type(Enum):
        """
        Enumeration of the supported statement file types.

        Attributes:
         WELLS_FARGO_CSV (str):    Wells Fargo CSV file.
         CHARLES_SCHWAB_CSV (str): Charles Schwab CSV file.
        """

        # Wells Fargo statement types
        WELLS_FARGO_CSV: str    = "Wells Fargo (CSV)"
        # Charles Schwab statement types
        CHARLES_SCHWAB_CSV: str = "Charles Schwab (CSV)"


    @dataclass
    class ColumnMap:
        """
        A struct that describes the columns in the statement file.

        Attributes:
         num_header_rows (int): The number of header rows at the start of the file to exclude.
         date_index (int):      The zero-aligned column index of the transaction date.
         date_format (str):     The format of dates in the file for Python's datetime library.
         amount_index (int):    The zero-aligned column index of the transaction amount.
         memo_index (int):      The zero-aligned column index of the transaction memo/description.
        """

        num_header_rows: int
        date_index: int
        date_format: str
        amount_index: int
        memo_index: int


    # A list of pre-defined maps for all supported transaction statement types
    COLUMN_MAPS: Final = {
        Type.WELLS_FARGO_CSV: ColumnMap(num_header_rows = 0,
                                        date_index = 0,
                                        date_format = "%m/%d/%Y",
                                        amount_index = 1,
                                        memo_index = 4),
        Type.CHARLES_SCHWAB_CSV: ColumnMap(num_header_rows = 1,
                                           date_index = 0,
                                           date_format = "%m/%d/%Y",
                                           amount_index = 7,
                                           memo_index = 3)
    }


    def __init__(self, statement_path: str, statement_type: Type):
        """
        Constructs a new statement object from the provided statement file of the specified type.

        Args:
         statement_path (str):  The path to the statement file.
         statement_type (Type): The type of the statement file.

        Raises:
         UnsupportedStatementError: If no column map exists for the specified statement type.
         StatementFileError:        If an I/O error occurs while loading the statement data.
        """

        self.col_map: Statement.ColumnMap = Statement.COLUMN_MAPS.get(statement_type)
        if (self.col_map is None):
            raise UnsupportedStatementError(f"unsupported type '{statement_type.value}'")

        try:
            with open(statement_path, "r", encoding = "utf-8", newline = "") as csv_file:
                csv_reader = csv.reader(csv_file)
                self.rows: list = list(csv_reader)
                if (len(self.rows) >= self.col_map.num_header_rows):
                    self.rows = self.rows[self.col_map.num_header_rows:]
        except OSError as os_error:
            raise StatementFileError(f"cannot open statement: {os_error}") from os_error


    def _get_transaction_from_row(self, row: list) -> Transaction:
        """
        Converts a row of raw transaction data to a transaction structure.

        Args:
         row (list): The row to convert.

        Returns:
         Transaction: The converted transaction.

        Raises:
         StatementParseError: If any of the data in the row is missing or cannot be parsed.
        """

        try:
            date: datetime.date = convert_string_to_date(row[self.col_map.date_index],
                                                         date_format = self.col_map.date_format)
            cents: int = convert_amount_to_cents(row[self.col_map.amount_index])
            statement_memo: str = row[self.col_map.memo_index]
        except (AmountConversionError, DateConversionError) as convert_error:
            raise StatementParseError(f"cannot parse statement: {convert_error}") from convert_error
        except IndexError as index_error:
            raise StatementParseError("statement row is missing fields") from index_error

        return Transaction(date = date,
                           cents = cents,
                           statement_memo = statement_memo,
                           user_memo = None,
                           category = None,
                           reconciled = False)


    def get_transactions(self) -> list:
        """
        Returns a list of all transactions in the statement file.

        Returns:
         list: All the transactions in the statement file.
        """

        transactions: list = []
        for row in self.rows:
            transactions.append(self._get_transaction_from_row(row))
        return transactions


####################################################################################################
# ACCOUNT FILE PARSING
####################################################################################################


class Account:
    """
    Manages data about a specific account.

    Attributes:
     is_new (bool):           Whether this object's construction created the account.
     name (str):              The name of the account.
     path (str):              The path to the account data directory.
     metadata_path (str):     The path to the .metadata file for this account.
     transactions_path (str): The path to the .transactions file for this account.
     metadata (dict):         The account metadata.
     transactions (list):     The list of transactions logged in the account.
    """


    # Path to store account data directories
    ACCOUNTS_PATH: Final = os.path.abspath(os.path.join(str(Path.home()), ".ledger"))


    def __init__(self, account_name: str, create_if_not_exists: bool = False):
        """
        Constructs a new account manager object.

        Args:
         account_name (str):          The name of the account (used to find its data directory).
         create_if_not_exists (bool): Whether to create the account data directory and files if
                                      they don't already exist.

        Raises:
         MissingAccountError: If the account name does not exist and `create_if_not_exists` is
                              false.
        """

        self.is_new = False
        self.name = account_name
        self.path = os.path.join(Account.ACCOUNTS_PATH, account_name)
        self.metadata_path = os.path.join(self.path, f"{account_name}.metadata")
        self.transactions_path = os.path.join(self.path, f"{account_name}.transactions")
        self.metadata: dict = {}
        self.transactions: list = []

        if (not os.path.isdir(self.path)):
            if (create_if_not_exists):
                self._create_account()
                self.is_new = True
            else:
                raise MissingAccountError(f"account '{account_name}' does not exist")

        self._load_metadata_file()
        self._load_transactions_file()


    def _create_account(self) -> None:
        """
        Creates the directory tree for a new empty account.

        Raises:
         AccountFileError: If any I/O operation cannot be completed while creating the account.
        """

        log.debug(f"Attempting to create new account files at {self.path}")
        try:
            os.makedirs(self.path)
            log.debug(f"Made account directory at {self.path}")

            with open(self.metadata_path, "w", encoding = "utf-8") as f:
                yaml.dump({"statement_type": None, "categories": []}, f)
            log.debug(f"Made account metadata file at {self.metadata_path}")

            with open(self.transactions_path, "w", encoding = "utf-8") as _:
                pass
            log.debug(f"Made account transactions file at {self.transactions_path}")
        except OSError as os_error:
            raise AccountFileError(f"cannot create account: {os_error}") from os_error


    def _load_metadata_file(self) -> None:
        """
        Loads the content of the .metadata file into the metadata dictionary.

        Raises:
         AccountFileError: If the .metadata file cannot be loaded.
        """

        log.debug(f"Attempting to open metadata file at {self.metadata_path}")
        try:
            with open(self.metadata_path, "r", encoding = "utf-8") as f:
                self.metadata = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as os_error:
            raise AccountFileError(f"cannot open metadata file: {os_error}") from os_error


    def _save_metadata_file(self) -> None:
        """
        Saves the contents of the metadata dictionary to the .metadata file.

        Raises:
         AccountFileError: If the metadata cannot be saved.
        """

        log.debug(f"Attempting to save metadata file to {self.metadata_path}")
        try:
            with open(self.metadata_path, "w", encoding = "utf-8") as f:
                yaml.dump(self.metadata, f)
        except (OSError, yaml.YAMLError) as os_error:
            raise AccountFileError(f"cannot save metadata file: {os_error}") from os_error


    def get_statement_type(self) -> Statement.Type:
        """
        Returns the statement type used by transactions in this account.

        Returns:
         Statement.Type: The statement type used by this account.

        Raises:
         AccountParseError: If the .metadata file does not contain a statement type.
        """

        try:
            return Statement.Type[self.metadata["statement_type"]]
        except KeyError as key_error:
            raise AccountParseError(
                "account metadata has a missing or invalid statement type"
            ) from key_error


    def set_statement_type(self, statement_type: Statement.Type) -> None:
        """
        Sets the statement type for this account.

        Args:
         statement_type (Statement.Type): The new statement type.
        """

        self.metadata["statement_type"] = statement_type.name
        self._save_metadata_file()


    def get_categories(self) -> list:
        """
        Gets the list of all transaction categories supported by this account.

        Returns:
         list: The supported transaction categories.

        Raises:
         AccountParseError: If the .metadata file does not contain a categories list.
        """

        try:
            return self.metadata["categories"]
        except KeyError as key_error:
            raise AccountParseError(
                "account metadata has a missing category list"
            ) from key_error


    def add_category(self, category: str) -> None:
        """
        Adds a new transaction category to this account.

        If the provided category already exists, no action is taken. The new category is added such
        that the list of categories remains sorted.

        Args:
         category (str): The new category to add.

        Raises:
         AccountParseError: If the .metadata file does not contain a categories list.
        """

        try:
            if (category in self.metadata["categories"]):
                return
            bisect.insort(self.metadata["categories"], category)
        except KeyError as key_error:
            raise AccountParseError(
                "account metadata has a missing category list"
            ) from key_error

        self._save_metadata_file()


    def _load_transactions_file(self) -> None:
        """
        Loads the contents of the .transactions file into the transactions list.

        Raises:
         AccountFileError: If the .transactions file cannot be loaded.
        """

        log.debug(f"Attempting to open transactions file at {self.transactions_path}")
        try:
            with open(self.transactions_path, "r", encoding = "utf-8", newline = "") as f:
                csv_reader = csv.reader(f)
                rows: list = list(csv_reader)
                log.debug(f"Read {len(rows)} rows from transactions file")

                self._set_transactions_from_csv(rows)
                log.debug("Successfully loaded account transaction data")
        except OSError as os_error:
            raise AccountFileError(f"cannot open transactions file: {os_error}") from os_error


    def _save_transactions_file(self) -> None:
        """
        Saves the transactions list to the .transactions file.

        Raises:
         AccountFileError: If the transaction data cannot be saved.
        """

        log.debug(f"Attempting to save transactions file to {self.transactions_path}")
        try:
            with open(self.transactions_path, "w", encoding = "utf-8", newline = "") as f:
                csv_writer = csv.writer(f)
                rows: list = self._get_transactions_as_csv()
                log.debug(f"Collected {len(rows)} rows for writing")

                csv_writer.writerows(rows)
                log.debug("Successfully wrote account transaction data")
        except OSError as os_error:
            raise AccountFileError(f"cannot save transactions file: {os_error}") from os_error


    def _set_transactions_from_csv(self, rows: list) -> None:
        """
        Converts all the provided rows to transaction objects and adds them to the transactions list
        in chronological order.

        Args:
         rows (list): The list of rows to add as transactions.

        Raises:
         AccountParseError: If any row cannot be parsed.
        """

        for row in rows:
            try:
                transaction: Transaction = Transaction(
                    date = convert_string_to_date(row[0]),
                    cents = int(row[1]),
                    statement_memo = row[2],
                    user_memo = row[3],
                    category = row[4],
                    reconciled = bool(row[5])
                )
            except IndexError as index_error:
                raise AccountParseError("transaction in account is missing data") from index_error
            except DateConversionError as convert_error:
                raise AccountParseError(
                    f"transaction in account has invalid date: {row[0]}"
                ) from convert_error
            except ValueError as value_error:
                raise AccountParseError(
                    f"transaction in account has invalid amount: {row[1]}"
                ) from value_error

            self._insert_transaction_by_date(transaction)


    def _get_transactions_as_csv(self) -> list:
        """
        Converts the transactions list to a list of CSV rows.

        Returns:
         list: The list of CSV rows.
        """

        rows: list = []
        for transaction in self.transactions:
            row: list = [
                convert_date_to_string(transaction.date),
                str(transaction.cents),
                transaction.statement_memo,
                transaction.user_memo,
                transaction.category,
                str(transaction.reconciled)
            ]
            rows.append(row)
        return rows


    def _insert_transaction_by_date(self, transaction: Transaction) -> None:
        """
        Adds a transaction to the account transactions list in chronological order.

        Note that this method does not save the transactions file.
        """

        for i, other_transaction in enumerate(self.transactions):
            if (transaction.date > other_transaction.date):
                self.transactions.insert(i, transaction)
                return
        self.transactions.append(transaction)


    def add_transaction(self, transaction: Transaction,
                        reconciler: Callable = None,
                        force: bool = False) -> bool:
        """
        Adds a single transaction to this account and updates the account data files.

        Args:
         transaction (Transaction): The transaction to add.
         reconciler (Callable):     A function which will be called with this account class and
                                    the transaction to get transaction data from the user.
         force (bool):              If true, the transaction will be added to the account even if
                                    it already exists (as determined by the hash code). If false
                                    (default) duplicate/existing transactions will be ignored.

        Returns:
         bool: Whether the transaction was added.
        """

        if (transaction in self.transactions and not force):
            log.debug("Not adding transaction; transaction exists and force is disabled")
            return False

        def reconcile_completer(text: str, state: int) -> str:
            matches: list = [c for c in self.get_categories() if text.lower() in c.lower()]
            return matches[state] if state < len(matches) else None

        if (reconciler is not None and not transaction.reconciled):
            log.debug("Reconciling transaction")
            readline.set_completer(reconcile_completer)
            readline.set_completer_delims(
                readline.get_completer_delims().replace("/", "").replace(" ", "")
            )
            readline.parse_and_bind("tab: complete")
            reconciler(self, transaction)
            transaction.reconciled = True

        self._insert_transaction_by_date(transaction)
        self._save_transactions_file()
        return True


    def add_statement(self, statement: Statement,
                      reconciler: Callable = None,
                      force: bool = False) -> (int, int, int):
        """
        Adds all transactions in a statement to this account.

        Args:
         statement (Statement): The statement to add transactions from.
         reconciler (Callable): A function which will be called with this account class and
                                each transaction to get transaction data from the user.
         force (bool):          If true, the transaction will be added to the account even if
                                it already exists (as determined by the hash code). If false
                                (default) duplicate/existing transactions will be ignored.

        Returns:
         int: The number of new transactions added.
         int: The number of transactions that were skipped (de-duplicated).
         int: The net amount in cents of all the transactions that were added.
        """

        progress_width: int = 80
        print("-" * progress_width)

        transactions: list = statement.get_transactions()
        log.info(f"Found {len(transactions)} transactions in statement")
        num_added: int = 0
        sum_added: int = 0
        for i, transaction in enumerate(transactions):
            added: bool = self.add_transaction(transaction, reconciler = reconciler, force = force)
            if (added):
                num_added += 1
                sum_added += transaction.cents
                progress_percent: float = (i + 1) / len(transactions)
                progress_filled: int = int(progress_percent * progress_width)
                progress_blank: int = progress_width - progress_filled
                print("-" * progress_width)
                print(f"[{'*'*progress_filled}{' '*progress_blank}] {progress_percent*100:.0f}%")
        return num_added, len(transactions) - num_added, sum_added


    def search_transactions(self, searchers: list) -> list:
        """
        Returns a filtered view of the transactions in this account.

        Args:
         searchers (list): A list of callables, which each take a transaction object and return
                           a boolean indicating whether the transaction passes the filter criteria.
                           Each transaction will only be returned if all the searchers return true.

        Returns:
         list: The list of all the transactions that meet all the searcher criteria.
        """

        filtered: list = []
        for transaction in self.transactions:
            if (all(searcher(transaction) for searcher in searchers)):
                filtered.append(transaction)
        return filtered


    def get_balance(self) -> int:
        """
        Returns the net amount in cents of all the transactions in this account.

        Returns:
         int: The amount in cents of the transactions in this account.
        """

        balance: int = 0
        for transaction in self.transactions:
            balance += transaction.cents
        return balance



####################################################################################################
# DEPENDENCY INJECTION CALLABLES
####################################################################################################


class Reconciler:
    """
    A container to hold different factories for reconciler functions.
    """

    @staticmethod
    def get_no_reconciler() -> Callable: # pylint: disable=useless-return
        """
        Gets a reconciler function that does nothing (does not set any user data in transactions
        and does not mark the transaction reconciled, but allows it to be added to the account).
        """

        log.debug("Created no reconciler")
        return None


    @staticmethod
    def get_default_reconciler() -> Callable:
        """
        Gets a reconciler function that prompts the user for a memo and category.
        """

        def default_reconciler(account: Account, transaction: Transaction) -> None:
            print("Reconcile Transaction:")
            print(f"  {transaction.date}  " +
                  f"{convert_cents_to_amount(transaction.cents)}  " +
                  f"{transaction.statement_memo}")
            transaction.user_memo = input("Memo: ")
            while (transaction.category not in account.get_categories()):
                transaction.category = input("Category: ")
        log.debug("Created default reconciler")
        return default_reconciler


class Searcher:
    """
    A container to hold different factories for searcher functions.
    """

    @staticmethod
    def get_unconditional_searcher() -> Callable:
        """
        Gets a searcher that never filters out a transaction.
        """

        def unconditional_searcher(_: Transaction) -> bool:
            return True
        return unconditional_searcher


    @staticmethod
    def get_date_range_searcher(earliest: datetime.date, latest: datetime.date) -> Callable:
        """
        Gets a searcher that filters for transactions between the provided dates (inclusive).

        Args:
         earliest (datetime.date): The earliest date to allow, inclusive.
         latest (datetime.date):   The latest date to allow, inclusive.
        """

        def date_range_searcher(transaction: Transaction) -> bool:
            return transaction.date >= earliest and transaction.date <= latest
        log.debug(f"Created date range searcher for {earliest} to {latest}")
        return date_range_searcher


    @staticmethod
    def get_amount_range_searcher(minimum: int, maximum: int) -> Callable:
        """
        Gets a searcher that filters for transactions between the provided amounts (inclusive).

        Args:
         minimum (int): The minimum signed transaction amount in cents to allow, inclusive.
         maximum (int): The maximum signed transaction amount in cents to allow, inclusive.
        """

        def amount_range_searcher(transaction: Transaction) -> bool:
            return transaction.cents >= minimum and transaction.cents <= maximum
        log.debug(f"Created amount range searcher for {minimum} to {maximum}")
        return amount_range_searcher


    @staticmethod
    def get_category_searcher(top_category: str) -> Callable:
        """
        Gets a searcher that filters for transactions in the provided category and any of its
        subcategories.

        Args:
         top_category (str): The highest parent category to allow. Any subcategories are also
                             allowed.
        """

        def category_searcher(transaction: Transaction) -> bool:
            return transaction.category.startswith(top_category)
        log.debug(f"Created category searcher for '{top_category}'")
        return category_searcher


####################################################################################################
# COMMAND LINE INTERFACE
####################################################################################################


def process_accounts(_: Namespace):
    """
    Process the 'ledger accounts' command.
    """

    if (not os.path.exists(Account.ACCOUNTS_PATH)):
        log.info(f"No accounts exist")
        return

    folders: str = [name for name in os.listdir(Account.ACCOUNTS_PATH)
                    if os.path.isdir(os.path.join(Account.ACCOUNTS_PATH, name))]

    num_accounts: int = 0
    total_net_cents: int = 0

    for folder in folders:
        try:
            account: Account = Account(folder)
        except (MissingAccountError, AccountFileError):
            continue

        transactions: list = account.search_transactions([Searcher.get_unconditional_searcher()])
        net_cents: int = account.get_balance()
        num_transactions: int = len(transactions)
        latest_date: datetime.date = transactions[0].date if num_transactions > 0 else 'N/A'
        earliest_date: datetime.date = transactions[-1].date if num_transactions > 0 else 'N/A'

        print(f"Account: {account.name}")
        print(f"  Net balance:            {convert_cents_to_amount(net_cents)}")
        print(f"  Number of transactions: {num_transactions}")
        print(f"  Latest transaction:     {latest_date}")
        print(f"  Earliest transaction:   {earliest_date}")
        print(f"  Statement type:         {account.get_statement_type().value}")
        print()

        num_accounts += 1
        total_net_cents += net_cents

    print("----------------------------------------")
    print(f"Number of accounts: {num_accounts}")
    print(f"Total net balance:  {convert_cents_to_amount(total_net_cents)}")


def process_adjust(args: Namespace):
    """
    Process the 'ledger adjust' command.

    Args:
     args (Namespace): The command line arguments.
    """

    account_name: str = args.account
    account: Account = Account(account_name)

    target_cents: int = convert_amount_to_cents(args.target_amount)
    current_cents: int = account.get_balance()
    adjustment: int = target_cents - current_cents

    log.debug(f"Target account balance is:  {target_cents} cents")
    log.debug(f"Current account balance is: {current_cents} cents")
    log.debug(f"Adjustment is:              {adjustment} cents")

    if (adjustment == 0):
        log.warning(f"Account balance is already ${args.target_amount}, not adding adjustment")
        return

    transaction: Transaction = Transaction(
        date = datetime.date.today(),
        cents = adjustment,
        statement_memo = "Balance adjustment",
        user_memo = "Balance adjustment",
        category = "Adjustment",
        reconciled = True
    )
    account.add_transaction(transaction)
    log.info(f"Added adjustment for {convert_cents_to_amount(adjustment)}")


def process_category(args: Namespace):
    """
    Process the 'ledger category' command.

    Args:
     args (Namespace): The command line arguments.
    """

    category_command: str = args.category_command
    account_name: str = args.account
    account: Account = Account(account_name)

    def process_category_add():
        """
        Process the 'ledger category add' command.
        """

        components: list = args.name.split("/")
        num_components: int = len(components)
        if (num_components == 0):
            raise AccountParseError("invalid blank category name")
        for i in range(num_components, 0, -1):
            category: str = "/".join(components[:i])
            parent: str = None if i == 1 else "/".join(components[:i - 1])
            if (parent is None or parent in account.get_categories()):
                account.add_category(category)
                break
            if (args.recursive):
                account.add_category(category)
            else:
                raise InvalidCategoryError(
                    f"parent category '{parent}' for category '{category}' does not exist"
                )
        log.info(f"Successfully added category '{args.name}'")

    def process_category_import():
        """
        Process the 'ledger category import' command.
        """

        other_account_name: str = args.other_account
        other_account: Account = Account(other_account_name)
        other_categories: list = other_account.get_categories()
        for category in other_categories:
            account.add_category(category)
        log.info(f"Successfully added {len(other_categories)} categories " +
                 f"from '{other_account_name}' to '{account_name}'")

    def process_category_list():
        """
        Process the 'ledger category list' command.
        """

        categories: list = account.get_categories()
        for category in categories:
            print(f"  {category}")
        log.info(f"Account has {len(categories)} categories")

    if (category_command == "add"):
        process_category_add()
    elif (category_command == "import"):
        process_category_import()
    elif (category_command == "list"):
        process_category_list()
    elif (category_command is None):
        log.error("No category sub-command specified.")
        sys.exit(1)
    else:
        log.critical(f"Unsupported category command '{category_command}'; this is a logic error")
        sys.exit(1)


def process_create(args: Namespace):
    """
    Process the 'ledger create' command.

    Args:
     args (Namespace): The command line arguments.
    """

    account_name: str = args.account
    account: Account = Account(account_name, create_if_not_exists = True)

    if (not account.is_new):
        log.warning(f"Account '{account_name}' already exists")
        return

    statement_type: Statement.Type = Statement.Type[args.statement_type]
    account.set_statement_type(statement_type)

    log.info(f"Successfully created account '{account_name}'")


def process_reconcile(args: Namespace):
    """
    Process the 'ledger reconcile' command.

    Args:
     args (Namespace): The command line arguments.
    """

    account_name: str = args.account
    statement_path: str = args.statement

    account: Account = Account(account_name)
    statement: Statement = Statement(statement_path, account.get_statement_type())

    added, skipped, cents = \
        account.add_statement(statement, reconciler = Reconciler.get_default_reconciler())

    log.info(f"Reconciliation of '{statement_path}' is complete")
    log.info(f"New transactions added: {added} ({skipped} skipped during deduplication)")
    log.info(f"Net amount added: {convert_cents_to_amount(cents)}")


def process_transactions(args: Namespace):
    """
    Process the 'ledger transactions' command.

    Args:
     args (Namespace): The command line arguments.
    """

    account_name: str = args.account
    account: Account = Account(account_name)

    searcher_comments: list = []
    searchers: list = []
    if (args.all):
        searcher_comments.append("all transactions")
        searchers.append(Searcher.get_unconditional_searcher())
    if (args.by_dates):
        searcher_comments.append(f"dates: {args.by_dates}")
        searchers.append(Searcher.get_date_range_searcher(*args.by_dates))
    if (args.by_amounts):
        searcher_comments.append(f"cents: {args.by_amounts}")
        searchers.append(Searcher.get_amount_range_searcher(*args.by_amounts))
    if (args.by_category):
        searcher_comments.append(f"categories: {args.by_category}")
        searchers.append(Searcher.get_category_searcher(*args.by_category))

    transactions: list = account.search_transactions(searchers)
    if (args.show_total):
        transactions.append(Transaction(
            date = "-- TOTAL --",
            cents = sum(transaction.cents for transaction in transactions),
            statement_memo = "-- TOTAL --",
            user_memo = "-- TOTAL --",
            category = "-- TOTAL --",
            reconciled = False
        ))

    widest_date: int = 0
    widest_amount: int = 0
    widest_category: int = 0
    widest_statement_memo: int = 0
    widest_user_memo: int = 0

    for transaction in transactions:
        widest_date = max(widest_date, len(convert_date_to_string(transaction.date)))
        widest_amount = max(widest_amount, len(convert_cents_to_amount(transaction.cents)))
        widest_category = max(widest_category, len(transaction.category))
        widest_statement_memo = max(widest_statement_memo, len(transaction.statement_memo))
        widest_user_memo = max(widest_user_memo, len(transaction.user_memo))

    if (args.format == "csv"):
        if (args.show_filters):
            print(",".join(searcher_comment for searcher_comment in searcher_comments))
        print("Date,Amount,Category,Description,Memo")
    elif (args.format == "markdown"):
        if (args.show_filters):
            print("# " + ", ".join(searcher_comment for searcher_comment in searcher_comments))
        print(f"| {'Date'.ljust(widest_date)} | " +
              f"{'Amount'.ljust(widest_amount)} | " +
              f"{'Category'.ljust(widest_category)} | " +
              f"{'Description'.ljust(widest_statement_memo)} | " +
              f"{'Memo'.ljust(widest_user_memo)} |")
        print("| " + " | ".join(["-" * widest_date,
                                 "-" * widest_amount,
                                 "-" * widest_category,
                                 "-" * widest_statement_memo,
                                 "-" * widest_user_memo]) + " |")
    else:
        log.critical(f"Unknown transaction format '{args.format}'; this is a logic error")
        sys.exit(1)

    for transaction in transactions:
        if (args.invert_signs):
            transaction.cents = -transaction.cents

        if (args.format == "csv"):
            print(f"{convert_date_to_string(transaction.date)}," +
                  f"{convert_cents_to_amount(transaction.cents)}," +
                  f"{transaction.category}," +
                  f"{transaction.statement_memo}," +
                  f"{transaction.user_memo}")
        elif (args.format == "markdown"):
            print(f"| {convert_date_to_string(transaction.date).rjust(widest_date)} | " +
                  f"{convert_cents_to_amount(transaction.cents).rjust(widest_amount)} | " +
                  f"{transaction.category.ljust(widest_category)} | " +
                  f"{transaction.statement_memo.ljust(widest_statement_memo)} | " +
                  f"{transaction.user_memo.ljust(widest_user_memo)} |")
        else:
            log.critical(f"Unknown transactions format '{args.format}'; this is a logic error")
            sys.exit(1)


def add_accounts_parser(subparsers: list) -> None:
    """
    Adds the argument subparser for 'ledger accounts'.

    Args:
     subparsers (list): The list of subparsers to add to.
    """

    accounts_parser: ArgumentParser = subparsers.add_parser( # pylint: disable=unused-variable
        "accounts", description = "List all existing accounts."
    )


def add_adjust_parser(subparsers: list) -> None:
    """
    Adds the argument subparser for 'ledger adjust'.

    Args:
     subparsers (list): The list of subparsers to add to.
    """

    adjust_parser: ArgumentParser = subparsers.add_parser(
        "adjust", description = "Perform a balance adjustment on an account."
    )
    adjust_parser.add_argument("account", type = str,
                               help = "the name of the account to adjust")
    adjust_parser.add_argument("target_amount", type = str,
                               help = "the desired account balance; an ajustment transaction will" +
                               " be added to reach this balance")


def add_category_parser(subparsers: list) -> None:
    """
    Adds the argument subparser for 'ledger category'.

    Args:
     subparsers (list): The list of subparsers to add to.
    """

    category_parser: ArgumentParser = subparsers.add_parser(
        "category", description = "Manage transactions categories."
    )
    category_parser.add_argument("account", type = str,
                                 help = "the name of the account to manage categories for")

    category_subparsers: list = category_parser.add_subparsers(dest = "category_command")

    add_parser: ArgumentParser = category_subparsers.add_parser(
        "add", description = "add a new category or categories"
    )
    add_parser.add_argument("name", type = str,
                            help = "the name of the new category. Subcategory names must be" +
                            " separated by forward slashes")
    add_parser.add_argument("--recursive", action = "store_true",
                            help = "if one or more parent categories do not already exists, add" +
                            " them recursively")

    import_parser: ArgumentParser = category_subparsers.add_parser(
        "import", description = "Import categories from an existing account."
    )
    import_parser.add_argument("other_account", help = "the name of the account to import from")

    list_parser: ArgumentParser = category_subparsers.add_parser( # pylint: disable=unused-variable
        "list", description = "List the structure of all available categories."
    )


def add_create_parser(subparsers: list) -> None:
    """
    Adds the argument subparser for 'ledger create'.

    Args:
     subparsers (list): The list of subparsers to add to.
    """

    create_parser: ArgumentParser = subparsers.add_parser(
        "create", description = "Create a new account with no transactions."
    )
    create_parser.add_argument("account", type = str,
                               help = "the name of the account to create")
    create_parser.add_argument("statement_type", type = str,
                               choices = [statement_type.name for statement_type in Statement.Type],
                               help = "the type of statements this account is configured to read")


def add_reconcile_parser(subparsers: list) -> None:
    """
    Adds the argument subparser for 'ledger reconcile'.

    Args:
     subparsers (list): The list of subparsers to add to.
    """

    reconcile_parser: ArgumentParser = subparsers.add_parser(
        "reconcile", description = "Add and reconcile transactions from a statement file."
    )
    reconcile_parser.add_argument("account", type = str,
                                  help = "the name of the account to add transactions to")
    reconcile_parser.add_argument("statement", type = str,
                                  help = "the path to the statement file to reconcile")


def add_transactions_parser(subparsers: list) -> None:
    """
    Adds the argument subparser for 'ledger transactions'.

    Args:
     subparsers (list): The list of subparsers to add to.
    """

    transactions_parser: ArgumentParser = subparsers.add_parser(
        "transactions", description = "Analyze transactions in an account."
    )
    transactions_parser.add_argument("account", type = str,
                                     help = "the name of the account to view transactions in")
    transactions_parser.add_argument("--all", action = "store_true",
                                     help = "show all transactions in the account")
    transactions_parser.add_argument("--by-dates", nargs = 2, type = convert_string_to_date,
                                     metavar = "YYYY-MM-DD",
                                     help = "show transactions in the date range, inclusive")
    transactions_parser.add_argument("--by-amounts", nargs = 2, type = convert_amount_to_cents,
                                     metavar = "AMOUNT",
                                     help = "show transactions in the amount range, inclusive")
    transactions_parser.add_argument("--by-category", nargs = 1, type = str,
                                     metavar = "CATEGORY",
                                     help = "show transactions in the category or a subcategory")
    transactions_parser.add_argument("--format", type = str,
                                     choices = {"csv", "markdown"}, default = "markdown",
                                     help = "the name of the account to view transactions in")
    transactions_parser.add_argument("--show_total", action = "store_true",
                                     help = "add a final row with the sum of all transactions")
    transactions_parser.add_argument("--show_filters", action = "store_true",
                                     help = "add a comment showing the filters used")
    transactions_parser.add_argument("--invert_signs", action = "store_true",
                                     help = "invert the signs of all transactions")


def main() -> None:
    """
    Main command line entry point.
    """

    parser: ArgumentParser = ArgumentParser(
        description = "A simple ledger tool to parse and reconcile transaction history data."
    )
    parser.add_argument("-v", "--verbose", action = "store_true",
                        help = "print verbose debug messages about program execution")

    subparsers: list = parser.add_subparsers(dest = "command")
    add_accounts_parser(subparsers)
    add_adjust_parser(subparsers)
    add_category_parser(subparsers)
    add_create_parser(subparsers)
    add_reconcile_parser(subparsers)
    add_transactions_parser(subparsers)

    args: Namespace = parser.parse_args()

    initialize_logger(logging.DEBUG if args.verbose else logging.INFO)

    command: str = args.command
    try:
        if (command == "accounts"):
            process_accounts(args)
        elif (command == "adjust"):
            process_adjust(args)
        elif (command == "category"):
            process_category(args)
        elif (command == "create"):
            process_create(args)
        elif (command == "reconcile"):
            process_reconcile(args)
        elif (command == "transactions"):
            process_transactions(args)
        elif (command is None):
            parser.print_usage()
            sys.exit(1)
        else:
            log.critical(f"Unsupported command '{command}; this is a logic error'")
            sys.exit(1)
    except (StatementFileError, UnsupportedStatementError, StatementParseError) as statement_error:
        log.critical(f"Statement error: {statement_error}")
        sys.exit(1)
    except (MissingAccountError,
            AccountFileError,
            AccountParseError,
            InvalidCategoryError) as account_error:
        log.critical(f"Account error: {account_error}")
        sys.exit(1)
    except Exception as uncaught_error: # pylint: disable=broad-exception-caught
        log.critical(f"Uncaught error: {uncaught_error}")
        sys.exit(1)


if (__name__ == "__main__"):
    main()
