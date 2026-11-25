"""Schema migrations: numbered SQL files, each applied in one transaction."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from mediasift.core.errors import LibraryError, MigrationError, SchemaTooNew
from mediasift.db._util import now

_FILE_NAME = re.compile(r"(\d{4})_(\w+)\.sql")
_COMMENTS = re.compile(r"(\s+|--[^\n]*(\n|$)|/\*.*?\*/)+", re.DOTALL)
# These would end the migration's transaction half-way.
_FORBIDDEN = {"BEGIN", "COMMIT", "END", "ROLLBACK", "SAVEPOINT", "RELEASE", "ATTACH", "DETACH"}

_CREATE_LOG = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  version    INTEGER PRIMARY KEY,
  name       TEXT NOT NULL,
  checksum   TEXT NOT NULL,
  applied_at TEXT NOT NULL
) STRICT
"""


@dataclass(frozen=True, slots=True)
class Migration:
    """One schema step."""

    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        """Hash of the SQL, the same whatever the line endings of the checkout."""
        text = self.sql.replace("\r\n", "\n").replace("\r", "\n")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_migrations() -> tuple[Migration, ...]:
    """Migrations shipped with the package, in order."""
    found = []
    for item in (resources.files("mediasift.db") / "migrations").iterdir():
        match = _FILE_NAME.fullmatch(item.name)
        if match:
            sql = item.read_text(encoding="utf-8")
            found.append(Migration(int(match[1]), match[2], sql))
    found.sort(key=lambda m: m.version)
    _check_numbering(found)
    return tuple(found)


def verify(conn: sqlite3.Connection, migrations: Sequence[Migration]) -> int:
    """Check what was applied against what is known. Return the number applied."""
    applied = _applied(conn)
    newest = migrations[-1].version if migrations else 0
    if applied and applied[-1][0] > newest:
        raise SchemaTooNew(
            f"the library has schema version {applied[-1][0]}, this version of MediaSift "
            f"knows up to {newest}; please update MediaSift"
        )
    for index, (version, name, checksum) in enumerate(applied):
        known = migrations[index]
        if version != known.version:
            raise MigrationError(f"migration {known.version} is missing from the library's history")
        if checksum != known.checksum:
            raise MigrationError(
                f"migration {version} ({name}) is not the one this library was built with"
            )
    return len(applied)


def upgrade(conn: sqlite3.Connection, migrations: Sequence[Migration], backups: Path) -> None:
    """Apply what is missing. An existing library is backed up first."""
    _check_numbering(migrations)
    done = verify(conn, migrations)
    pending = migrations[done:]
    if not pending:
        return
    if done:
        backup(conn, backups, migrations[done - 1].version)
    for migration in pending:
        _apply(conn, migration)


def backup(conn: sqlite3.Connection, folder: Path, version: int) -> Path:
    """Write a compact copy of the database and flush it to disk."""
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    target = folder / f"library-v{version}-{stamp}.db"
    try:
        conn.execute("VACUUM INTO ?", (str(target),))
        with target.open("rb+") as file:
            os.fsync(file.fileno())
    except (sqlite3.Error, OSError) as exc:
        with suppress(OSError):
            target.unlink(missing_ok=True)
        raise MigrationError(f"cannot back up the library, so it was not upgraded: {exc}") from exc
    return target


def split_statements(sql: str) -> Iterator[str]:
    """Split a script into statements.

    Not ``executescript``, because that commits on its own and would break atomicity.
    """
    buffer = ""
    for piece in sql.split(";")[:-1]:
        buffer += piece + ";"
        if sqlite3.complete_statement(buffer):
            yield buffer.strip()
            buffer = ""
    buffer += sql.rsplit(";", 1)[-1]
    if _first_word(buffer):
        yield buffer.strip()


def _apply(conn: sqlite3.Connection, migration: Migration) -> None:
    # Foreign keys are off while tables may be rebuilt, and checked before the commit.
    # The pragma does nothing inside a transaction, so it has to stand outside.
    conn.execute("PRAGMA foreign_keys = OFF")
    try:
        conn.execute("BEGIN IMMEDIATE")
        _run(conn, migration)
        conn.execute("COMMIT")
    except sqlite3.Error as exc:
        raise MigrationError(f"migration {migration.version} failed: {exc}") from exc
    finally:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        conn.execute("PRAGMA foreign_keys = ON")


def _run(conn: sqlite3.Connection, migration: Migration) -> None:
    conn.execute(_CREATE_LOG)
    for statement in split_statements(migration.sql):
        if _first_word(statement) in _FORBIDDEN:
            raise MigrationError(f"migration {migration.version} must not control transactions")
        conn.execute(statement)
    broken = conn.execute("PRAGMA foreign_key_check").fetchone()
    if broken is not None:
        raise MigrationError(f"migration {migration.version} breaks references in {broken[0]}")
    conn.execute(
        "INSERT INTO schema_migrations (version, name, checksum, applied_at) VALUES (?, ?, ?, ?)",
        (migration.version, migration.name, migration.checksum, now()),
    )


def _applied(conn: sqlite3.Connection) -> list[tuple[int, str, str]]:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "schema_migrations" in tables:
        return conn.execute(
            "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
        ).fetchall()
    if tables:
        raise LibraryError("the database in the library folder is not a MediaSift library")
    return []


def _check_numbering(migrations: Sequence[Migration]) -> None:
    for expected, migration in enumerate(migrations, start=1):
        if migration.version != expected:
            raise MigrationError(f"migration {expected} is missing")


def _first_word(sql: str) -> str:
    start = _COMMENTS.match(sql)
    word = re.match(r"\w+", sql[start.end() if start else 0 :])
    return word[0].upper() if word else ""
