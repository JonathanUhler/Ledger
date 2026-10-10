import pathlib

import pytest

from ledger import config
from ledger.errors import AccountParseError


def test_ledger_home_honors_env(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.setenv("LEDGER_HOME", str(tmp_path))
    assert config.get_ledger_home() == tmp_path
    assert config.get_accounts_dir() == tmp_path / "accounts"
    assert config.get_types_dir() == tmp_path / "types"


def test_account_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.setenv("LEDGER_HOME", str(tmp_path))
    assert config.account_db_path("checking") == tmp_path / "accounts" / "checking.db"
    assert config.account_backup_path("checking") == tmp_path / "accounts" / "checking.db.bak"
    assert config.type_path("wells_fargo_csv") == tmp_path / "types" / "wells_fargo_csv.yaml"


@pytest.mark.parametrize("bad", ["", "a/b", "../x", "x\\y", "..", "   "])
def test_invalid_account_names_rejected(bad: str) -> None:
    with pytest.raises(AccountParseError):
        config.validate_account_name(bad)


@pytest.mark.parametrize("good", ["checking", "My Account", "acct_1", "a-b"])
def test_valid_account_names_accepted(good: str) -> None:
    config.validate_account_name(good)
