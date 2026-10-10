import pathlib

import pytest

from ledger.core import database
from ledger.errors import IntegrityError


def _fresh(path: pathlib.Path):
    conn = database.connect(path)
    database.create_schema(conn)
    return conn


def test_meta_round_trip(tmp_path: pathlib.Path) -> None:
    conn = _fresh(tmp_path / "a.db")
    assert database.read_meta(conn, "x") is None
    database.write_meta(conn, "x", "1")
    database.write_meta(conn, "x", "2")   # upsert
    assert database.read_meta(conn, "x") == "2"


def test_checksum_changes_with_data(tmp_path: pathlib.Path) -> None:
    conn = _fresh(tmp_path / "a.db")
    before: str = database.compute_checksum(conn)
    conn.execute("INSERT INTO transactions (id, date, cents, statement_memo, account, "
                 "user_memo, category, reconciled, content_hash) "
                 "VALUES ('i1','2026-01-01',100,'m','acct','','',0,'h')")
    assert database.compute_checksum(conn) != before


def test_verify_integrity_ok_then_tamper(tmp_path: pathlib.Path) -> None:
    conn = _fresh(tmp_path / "a.db")
    database.write_meta(conn, "content_checksum", database.compute_checksum(conn))
    database.verify_integrity(conn)   # no raise
    # tamper: insert a row without updating the stored checksum
    conn.execute("INSERT INTO transactions (id, date, cents, statement_memo, account, "
                 "user_memo, category, reconciled, content_hash) "
                 "VALUES ('i1','2026-01-01',100,'m','acct','','',0,'h')")
    with pytest.raises(IntegrityError):
        database.verify_integrity(conn)


def test_backup_file(tmp_path: pathlib.Path) -> None:
    src: pathlib.Path = tmp_path / "a.db"
    src.write_bytes(b"hello")
    dst: pathlib.Path = tmp_path / "a.db.bak"
    database.backup_file(src, dst)
    assert dst.read_bytes() == b"hello"
