import pytest

from ledger import config
from ledger.cli import app


WELLS_YAML: str = (
    "name: wells_fargo_csv\nkind: csv\n"
    "csv:\n  num_header_rows: 1\n  date: {index: 0, format: \"%m/%d/%Y\"}\n"
    "  amount: {index: 2}\n  memo: {index: 1}\n"
)


def test_accounts_empty(capsys: pytest.CaptureFixture) -> None:
    assert app.main(["accounts"]) == 0
    assert "Number of accounts: 0" in capsys.readouterr().out


def test_create_type_then_account_then_accounts(capsys: pytest.CaptureFixture) -> None:
    assert app.main(["create", "type", "wells_fargo_csv", "--kind", "csv",
                     "--date-index", "0", "--date-format", "%m/%d/%Y",
                     "--amount-index", "2", "--memo-index", "1", "--header-rows", "1"]) == 0
    assert config.type_path("wells_fargo_csv").exists()
    assert app.main(["create", "account", "checking", "wells_fargo_csv"]) == 0
    capsys.readouterr()
    assert app.main(["accounts"]) == 0
    assert "checking" in capsys.readouterr().out


def test_categories_add_and_list(capsys: pytest.CaptureFixture) -> None:
    config.type_path("wells_fargo_csv").write_text(WELLS_YAML, encoding = "utf-8")
    app.main(["create", "account", "checking", "wells_fargo_csv"])
    assert app.main(["categories", "add", "checking", "Food"]) == 0
    assert app.main(["categories", "list", "checking"]) == 0
    assert "Food" in capsys.readouterr().out


def test_view_and_adjust(capsys: pytest.CaptureFixture) -> None:
    config.type_path("wells_fargo_csv").write_text(WELLS_YAML, encoding = "utf-8")
    app.main(["create", "account", "checking", "wells_fargo_csv"])
    assert app.main(["transactions", "adjust", "checking", "100.00"]) == 0
    capsys.readouterr()
    assert app.main(["transactions", "view", "checking", "--all", "--format", "csv"]) == 0
    assert "$100.00" in capsys.readouterr().out


def test_missing_account_returns_error(capsys: pytest.CaptureFixture) -> None:
    assert app.main(["categories", "list", "ghost"]) == 1


def test_no_subcommand_prints_usage() -> None:
    assert app.main(["categories"]) == 1
