"""Connection helper and numbered-migration runner (D18).

Migrations are ``NNNN_name.sql`` files in ``slotor/db/migrations``. The schema version is stored in
``PRAGMA user_version``. A migration file is never edited once applied: schema changes are
new files.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

MIN_SQLITE_VERSION = (3, 37, 0)  # STRICT tables
_MIGRATION_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")


class MigrationError(RuntimeError):
    """Raised when migration files are inconsistent or a migration fails."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open a connection with foreign keys enforced and rows addressable by column name."""
    if sqlite3.sqlite_version_info < MIN_SQLITE_VERSION:
        raise MigrationError(
            f"SQLite >= {'.'.join(map(str, MIN_SQLITE_VERSION))} required, "
            f"found {sqlite3.sqlite_version}"
        )
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")
    return conn


def list_migrations() -> list[Migration]:
    """Return the packaged migrations sorted by version; versions must be 1..N without gaps."""
    found: list[Migration] = []
    for entry in resources.files("slotor.db").joinpath("migrations").iterdir():
        match = _MIGRATION_NAME.match(entry.name)
        if match:
            found.append(Migration(int(match[1]), entry.name, entry.read_text(encoding="utf-8")))
    found.sort(key=lambda m: m.version)
    _check_sequence(found)
    return found


def _check_sequence(migrations: list[Migration]) -> None:
    for expected, migration in enumerate(migrations, start=1):
        if migration.version != expected:
            raise MigrationError(
                f"migration versions must be 1..N without gaps or duplicates; "
                f"expected {expected:04d}, found {migration.name}"
            )


def current_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def apply_migrations(
    conn: sqlite3.Connection, migrations: Iterable[Migration] | None = None
) -> int:
    """Apply pending migrations, each in its own transaction. Returns the resulting version."""
    pending_source = list(migrations) if migrations is not None else list_migrations()
    pending_source.sort(key=lambda m: m.version)
    _check_sequence(pending_source)

    version = current_version(conn)
    if version > len(pending_source):
        raise MigrationError(
            f"database is at version {version}, newer than the "
            f"{len(pending_source)} known migrations"
        )
    for migration in pending_source:
        if migration.version <= version:
            continue
        script = f"BEGIN;\n{migration.sql}\nPRAGMA user_version = {migration.version};\nCOMMIT;"
        try:
            conn.executescript(script)
        except sqlite3.Error as exc:
            if conn.in_transaction:
                conn.rollback()
            raise MigrationError(f"migration {migration.name} failed: {exc}") from exc
        version = current_version(conn)
        if version != migration.version:
            raise MigrationError(
                f"migration {migration.name} left user_version={version}, "
                f"expected {migration.version}"
            )
    return version
