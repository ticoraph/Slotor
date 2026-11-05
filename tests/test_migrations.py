from __future__ import annotations

import sqlite3

import pytest

from slotor.db import Migration, apply_migrations, connect, current_version, list_migrations
from slotor.db.migrate import MigrationError


def test_packaged_migrations_are_contiguous():
    versions = [m.version for m in list_migrations()]
    assert versions == list(range(1, len(versions) + 1))


def test_apply_creates_schema_and_sets_version(conn):
    assert current_version(conn) == len(list_migrations())
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for expected in (
        "source",
        "catalog_version",
        "piece_type",
        "piece_set",
        "inventory",
        "solution",
    ):
        assert expected in tables
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_apply_is_idempotent(conn):
    before = current_version(conn)
    assert apply_migrations(conn) == before


def test_foreign_keys_are_enforced(conn):
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO piece_type(catalog_version_id, slug, category, name) "
            "VALUES (999, 'x', 'straight', 'x')"
        )


def _version_id(conn) -> int:
    return conn.execute(
        "INSERT INTO catalog_version(name, content_sha256) VALUES ('t', 'h')"
    ).lastrowid


@pytest.mark.parametrize(
    "columns, values",
    [
        ("category", "'banana'"),  # unknown category
        ("category, level_in, level_out", "'ramp_up', 1, 0"),  # ramp_up must go 0 -> 1
        ("category, level_in", "'straight', 2"),  # only two levels (D4)
        ("category, radius_mm", "'curve', -5"),  # positive radius
        ("category, radius_ref", "'curve', 'middle'"),  # radius_ref vocabulary (D20)
    ],
)
def test_piece_constraints_reject_invalid_rows(conn, columns, values):
    vid = _version_id(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            f"INSERT INTO piece_type(catalog_version_id, slug, name, {columns}) "
            f"VALUES ({vid}, 's', 'n', {values})"
        )


def test_slug_is_unique_per_catalog_version(conn):
    vid = _version_id(conn)
    sql = (
        "INSERT INTO piece_type(catalog_version_id, slug, category, name) "
        f"VALUES ({vid}, 'dup', 'straight', 'n')"
    )
    conn.execute(sql)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql)


def test_failed_migration_rolls_back_and_keeps_version():
    connection = connect(":memory:")
    good = Migration(1, "0001_ok.sql", "CREATE TABLE a(x INTEGER) STRICT;")
    bad = Migration(
        2, "0002_bad.sql", "CREATE TABLE b(x INTEGER) STRICT; CREATE TABLE a(y INTEGER);"
    )
    with pytest.raises(MigrationError):
        apply_migrations(connection, [good, bad])
    assert current_version(connection) == 1
    names = {
        r["name"] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert "a" in names
    assert "b" not in names  # the failed migration did not leave a half-applied table


def test_database_newer_than_code_is_refused():
    connection = connect(":memory:")
    connection.execute("PRAGMA user_version = 5")
    with pytest.raises(MigrationError):
        apply_migrations(connection, [Migration(1, "0001_ok.sql", "SELECT 1;")])


def test_gap_in_versions_is_refused():
    connection = connect(":memory:")
    with pytest.raises(MigrationError):
        apply_migrations(connection, [Migration(2, "0002_x.sql", "SELECT 1;")])
