import pytest

from ledger import config
from ledger.cli import app
from ledger.core.service import Ledger


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


def _make_account_with_income() -> None:
    config.type_path("wells_fargo_csv").write_text(WELLS_YAML, encoding = "utf-8")
    app.main(["create", "account", "checking", "wells_fargo_csv"])
    app.main(["categories", "add", "checking", "Income", "--recursive"])
    app.main(["categories", "add", "checking", "Income/Paychecks", "--recursive"])
    app.main(["transactions", "add", "checking", "--date", "2026-01-05",
              "--amount", "20000.00", "--memo", "pay", "--category", "Income/Paychecks"])


def test_budget_add_and_view(capsys: pytest.CaptureFixture) -> None:
    _make_account_with_income()
    assert app.main(["budget", "add", "Laptop", "--goal", "300.00",
                     "--category", "Income/Paychecks", "--start", "2026-01-01"]) == 0
    assert app.main(["budget", "setcap", "Income/Paychecks", "50"]) == 0
    capsys.readouterr()
    assert app.main(["budget", "view"]) == 0
    out: str = capsys.readouterr().out
    assert "Income/Paychecks" in out
    assert "cap 50%" in out
    assert "Laptop" in out
    assert "$300.00" in out


def test_budget_edit_and_remove(capsys: pytest.CaptureFixture) -> None:
    _make_account_with_income()
    app.main(["budget", "add", "Laptop", "--goal", "1000.00",
              "--category", "Income/Paychecks", "--start", "2026-01-01"])
    budget_id: str = Ledger().list_budgets()[0].id
    assert app.main(["budget", "edit", budget_id, "--name", "Gaming PC"]) == 0
    assert Ledger().list_budgets()[0].name == "Gaming PC"
    assert app.main(["budget", "remove", budget_id]) == 0
    assert Ledger().list_budgets() == []


def test_budget_view_empty(capsys: pytest.CaptureFixture) -> None:
    assert app.main(["budget", "view"]) == 0
    assert "No budgets" in capsys.readouterr().out


def test_budget_add_rejects_percent_flag() -> None:
    _make_account_with_income()
    # --percent is no longer a recognised option -> argparse error -> exit 1
    assert app.main(["budget", "add", "Laptop", "--goal", "300.00",
                     "--category", "Income/Paychecks", "--percent", "5"]) == 1


def test_budget_move_reorders(capsys: pytest.CaptureFixture) -> None:
    _make_account_with_income()
    app.main(["budget", "add", "A", "--goal", "100.00", "--category", "Income/Paychecks"])
    app.main(["budget", "add", "B", "--goal", "100.00", "--category", "Income/Paychecks"])
    second_id: str = Ledger().list_budgets()[1].id
    assert app.main(["budget", "move", second_id, "up"]) == 0
    assert Ledger().list_budgets()[0].id == second_id


def test_budget_setcap_and_clearcap() -> None:
    _make_account_with_income()
    assert app.main(["budget", "setcap", "Income/Paychecks", "30"]) == 0
    assert Ledger().get_category_cap("Income/Paychecks") == 3000
    assert app.main(["budget", "clearcap", "Income/Paychecks"]) == 0
    assert Ledger().get_category_cap("Income/Paychecks") == 10000


def test_budget_setcap_bad_value_returns_error() -> None:
    _make_account_with_income()
    assert app.main(["budget", "setcap", "Income/Paychecks", "150"]) == 1   # >100%


def test_budget_edit_unknown_id_returns_error() -> None:
    _make_account_with_income()
    assert app.main(["budget", "edit", "nope", "--name", "X"]) == 1
