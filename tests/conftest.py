import pathlib

import pytest


@pytest.fixture(autouse = True)
def ledger_home(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> pathlib.Path:
    home: pathlib.Path = tmp_path / "ledger_home"
    (home / "accounts").mkdir(parents = True)
    (home / "types").mkdir(parents = True)
    monkeypatch.setenv("LEDGER_HOME", str(home))
    return home
