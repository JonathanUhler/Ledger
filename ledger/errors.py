"""
Exceptions related to the ledger operations.
"""


class LedgerError(Exception):
    """
    Base class for all ledger errors.
    """


class AmountConversionError(LedgerError, ValueError):
    """
    An amount string is invalid and cannot be converted to cents.
    """


class DateConversionError(LedgerError, ValueError):
    """
    A date string is formatted incorrectly and cannot be converted.
    """


class StatementFileError(LedgerError, OSError):
    """
    A file I/O error occurred on a transaction statement file.
    """


class UnsupportedStatementError(LedgerError, ValueError):
    """
    The type/format of a statement is not supported.
    """


class StatementParseError(LedgerError, ValueError):
    """
    A statement contained invalid data and could not be parsed.
    """


class MissingAccountError(LedgerError, FileNotFoundError):
    """
    A specified account does not exist.
    """


class AccountFileError(LedgerError, OSError):
    """
    A file I/O or database error occurred on an account.
    """


class AccountParseError(LedgerError, ValueError):
    """
    Account data was missing or invalid.
    """


class InvalidCategoryError(LedgerError, ValueError):
    """
    A category does not exist or is formatted incorrectly.
    """


class GrammarError(LedgerError, ValueError):
    """
    A statement grammar file is missing or malformed.
    """


class IntegrityError(LedgerError):
    """
    An account database failed an integrity or checksum check.
    """


class MigrationError(LedgerError):
    """
    A data-model migration failed a sanity check.
    """
