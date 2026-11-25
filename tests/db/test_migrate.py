from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mediasift.core.errors import ExitCode, LibraryError, MigrationError, SchemaTooNew
from mediasift.db import migrate
from mediasift.db.connection import BACKUP_FOLDER, DATABASE_FILE, Database
from mediasift.db.migrate import Migration, load_migrations, split_statements

ONE = Migration(1, "one", "CREATE TABLE a (x); INSERT INTO a VALUES ('kept');")
TWO = Migration(2, "two", "CREATE TABLE b (x);")
THREE = Migration(3, "three", "CREATE TABLE c (x);")

PLANNED_TABLES = {
    "schema_migrations",
    "roots",
    "files",
    "features",
    "metadata",
    "scans",
    "scan_issues",
    "match_runs",
    "match_groups",
    "match_members",
    "decisions",
    "op_sessions",
    "operations",
}


def tables(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        sql = "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        return {row[0] for row in conn.execute(sql)}
    finally:
        conn.close()


def backups(library: Path) -> list[Path]:
    return sorted((library / BACKUP_FOLDER).glob("*")) if (library / BACKUP_FOLDER).exists() else []


def test_from_empty(library: Path) -> None:
    Database.open(library).close()
    assert tables(library / DATABASE_FILE) == PLANNED_TABLES
    assert backups(library) == []
    Database.open(library).close()  # nothing to do
    assert backups(library) == []


def test_shipped_migrations_are_numbered_and_recorded(db: Database) -> None:
    shipped = load_migrations()
    assert [m.version for m in shipped] == list(range(1, len(shipped) + 1))
    recorded = db.execute("SELECT version, name, checksum FROM schema_migrations").fetchall()
    assert recorded == [(m.version, m.name, m.checksum) for m in shipped]


def test_upgrade_backs_up_first(library: Path) -> None:
    Database.open(library, migrations=[ONE]).close()
    Database.open(library, migrations=[ONE, TWO, THREE]).close()
    assert tables(library / DATABASE_FILE) == {"schema_migrations", "a", "b", "c"}
    (saved,) = backups(library)
    assert tables(saved) == {"schema_migrations", "a"}
    conn = sqlite3.connect(saved)
    assert conn.execute("SELECT x FROM a").fetchall() == [("kept",)]
    assert conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    conn.close()


def test_no_upgrade_without_a_backup(library: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    Database.open(library, migrations=[ONE]).close()
    (library / BACKUP_FOLDER).mkdir()

    def broken(fd: int) -> None:
        raise OSError(5, "disk failure")

    monkeypatch.setattr(migrate.os, "fsync", broken)
    with pytest.raises(MigrationError, match="not upgraded"):
        Database.open(library, migrations=[ONE, TWO])
    assert tables(library / DATABASE_FILE) == {"schema_migrations", "a"}
    assert backups(library) == []  # no half-written backup left


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE b (x); INSERT INTO a VALUES ('lost'); INSERT INTO nowhere VALUES (1);",
        "CREATE TABLE b (x); INSERT INTO a VALUES ('lost'); SELEKT 1",
        "CREATE TABLE b (x); DELETE FROM a; COMMIT; INSERT INTO nowhere VALUES (1);",
        "CREATE TABLE b (x); DELETE FROM a; /* x */ -- y\n end;",
        "CREATE TABLE b (x); DROP TABLE a; ROLLBACK;",
        "CREATE TABLE b (id PRIMARY KEY); CREATE TABLE d (r REFERENCES b (id));"
        "INSERT INTO d VALUES (9);",
    ],
)
def test_failing_migration_changes_nothing(library: Path, sql: str) -> None:
    Database.open(library, migrations=[ONE]).close()
    bad = Migration(3, "bad", sql)
    with pytest.raises(MigrationError, match="migration 3"):
        Database.open(library, migrations=[ONE, TWO, bad])
    # The good step before it stays, the bad one leaves nothing.
    assert tables(library / DATABASE_FILE) == {"schema_migrations", "a", "b"}
    conn = sqlite3.connect(library / DATABASE_FILE)
    assert conn.execute("SELECT x FROM a").fetchall() == [("kept",)]
    assert conn.execute("SELECT max(version) FROM schema_migrations").fetchone() == (2,)
    conn.close()
    with Database.open(library, migrations=[ONE, TWO, THREE]) as db:  # recovers
        assert db.execute("PRAGMA foreign_keys").fetchone() == (1,)


def test_failing_first_migration_leaves_an_empty_database(library: Path) -> None:
    bad = Migration(1, "bad", "CREATE TABLE a (x); SELEKT 1;")
    with pytest.raises(MigrationError):
        Database.open(library, migrations=[bad])
    assert tables(library / DATABASE_FILE) == set()
    Database.open(library, migrations=[ONE]).close()


def test_rebuilding_a_referenced_table(library: Path) -> None:
    first = Migration(
        1,
        "one",
        "CREATE TABLE p (id INTEGER PRIMARY KEY);"
        "CREATE TABLE c (p REFERENCES p (id) ON DELETE CASCADE);"
        "INSERT INTO p VALUES (1); INSERT INTO c VALUES (1);",
    )
    second = Migration(
        2,
        "two",
        "CREATE TABLE q (id INTEGER PRIMARY KEY, extra);"
        "INSERT INTO q SELECT id, 'new' FROM p; DROP TABLE p; ALTER TABLE q RENAME TO p;",
    )
    Database.open(library, migrations=[first]).close()
    with Database.open(library, migrations=[first, second]) as db:
        assert db.execute("SELECT * FROM p").fetchall() == [(1, "new")]
        assert db.execute("SELECT * FROM c").fetchall() == [(1,)]


def test_newer_schema_is_refused(library: Path) -> None:
    Database.open(library, migrations=[ONE, TWO]).close()
    for write in (True, False):
        with pytest.raises(SchemaTooNew, match=r"version 2.*up to 1") as caught:
            Database.open(library, write=write, migrations=[ONE])
        assert caught.value.exit_code is ExitCode.BUSY
    assert backups(library) == []


def test_older_schema_is_not_migrated_by_a_reader(library: Path) -> None:
    Database.open(library, migrations=[ONE]).close()
    with pytest.raises(LibraryError, match="older version"):
        Database.open(library, write=False, migrations=[ONE, TWO])
    assert tables(library / DATABASE_FILE) == {"schema_migrations", "a"}


def test_changed_migration_is_refused(library: Path) -> None:
    Database.open(library, migrations=[ONE, TWO]).close()
    altered = Migration(1, "one", ONE.sql + " CREATE TABLE sneaky (x);")
    for write in (True, False):
        with pytest.raises(MigrationError, match="migration 1"):
            Database.open(library, write=write, migrations=[altered, TWO])


def test_gap_in_the_history_is_refused(library: Path) -> None:
    Database.open(library, migrations=[ONE, TWO]).close()
    conn = sqlite3.connect(library / DATABASE_FILE)
    conn.execute("DELETE FROM schema_migrations WHERE version = 1")
    conn.commit()
    conn.close()
    with pytest.raises(MigrationError, match="migration 1 is missing"):
        Database.open(library, migrations=[ONE, TWO])


def test_gap_in_the_shipped_migrations_is_refused(library: Path) -> None:
    with pytest.raises(MigrationError, match="migration 2 is missing"):
        Database.open(library, migrations=[ONE, THREE])


def test_checksum_ignores_line_endings_only() -> None:
    unix = Migration(1, "x", "SELECT 1;\nSELECT 2;\n")
    assert Migration(1, "x", unix.sql.replace("\n", "\r\n")).checksum == unix.checksum
    assert Migration(1, "x", unix.sql.replace("2", "3")).checksum != unix.checksum


def test_split_statements() -> None:
    sql = """
    -- a comment; with a semicolon
    CREATE TABLE t (x TEXT DEFAULT 'a;b'); INSERT INTO t VALUES ('it''s; fine');
    /* another; one */
    CREATE TRIGGER g AFTER INSERT ON t BEGIN
      DELETE FROM t; SELECT 1;
    END;
    SELECT 2 -- no semicolon at the end
    """
    parts = list(split_statements(sql))
    assert len(parts) == 4
    assert parts[1] == "INSERT INTO t VALUES ('it''s; fine');"
    assert parts[2].endswith("END;")
    assert list(split_statements("-- nothing; here\n/* at all; */ ")) == []


# --- what the schema guarantees ---


def test_operation_log_survives_its_files(db: Database) -> None:
    db.execute(
        "INSERT INTO roots (id, path, display, path_key, case_sensitive)"
        " VALUES (1, x'61', 'a', x'61', 1)"
    )
    db.execute(
        "INSERT INTO files (id, root_id, rel_path, rel_display, rel_key, ext, media_type,"
        " size, mtime_ns, status) VALUES (5, 1, x'62', 'b', x'62', '.jpg', 'image', 1, 1,"
        " 'present')"
    )
    db.execute("INSERT INTO op_sessions (id, kind, state, created_at) VALUES (1, 'x', 'y', 'z')")
    db.execute(
        "INSERT INTO operations (session_id, seq, action, backend, state, file_id, src,"
        " started_at) VALUES (1, 1, 'remove', 'quarantine', 'done', 5, x'62', 'now')"
    )
    db.execute("DELETE FROM roots")
    assert db.execute("SELECT count(*) FROM files").fetchone() == (0,)
    assert db.execute("SELECT file_id, src FROM operations").fetchall() == [(5, b"b")]
    for table in ("operations", "op_sessions"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute(f"DELETE FROM {table}")


def test_file_ids_are_never_reused(db: Database) -> None:
    def add() -> int:
        cursor = db.execute(
            "INSERT INTO files (root_id, rel_path, rel_display, rel_key, ext, media_type, size,"
            " mtime_ns, status) VALUES (1, x'62', 'b', x'62', '', 'image', 1, 1, 'present')"
        )
        return cursor.lastrowid or 0

    db.execute(
        "INSERT INTO roots (id, path, display, path_key, case_sensitive)"
        " VALUES (1, x'61', 'a', x'61', 1)"
    )
    first = add()
    db.execute("DELETE FROM files")
    assert add() > first
