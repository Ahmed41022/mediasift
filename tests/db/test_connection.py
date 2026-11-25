from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

import mediasift
from mediasift.core.errors import ExitCode, LibraryBusy, LibraryError
from mediasift.db.connection import DATABASE_FILE, LOCK_FILE, Database

HOLDER = """
import os, sys
from mediasift.db.connection import Database
db = Database.open(sys.argv[1])
print(os.getpid(), flush=True)
sys.stdin.read()
"""


@contextmanager
def other_process(library: Path) -> Iterator[tuple[subprocess.Popen[str], str]]:
    """Another process that holds the library open for writing, and its id.

    The id comes from the process itself: in a virtual environment on Windows, Popen's is
    that of a launcher.
    """
    env = {**os.environ, "PYTHONPATH": str(Path(mediasift.__file__).parents[1])}
    with subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(library)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    ) as process:
        try:
            assert process.stdout is not None
            pid = process.stdout.readline().strip()
            assert pid.isdigit()
            yield process, pid
        finally:
            process.kill()


def roots(db: Database) -> list[str]:
    return [row[0] for row in db.execute("SELECT display FROM roots ORDER BY display")]


def add(db: Database, name: str) -> None:
    db.execute(
        "INSERT INTO roots (path, display, path_key, case_sensitive) VALUES (?, ?, ?, 1)",
        (name.encode(), name, name.encode()),
    )


# --- opening ---


def test_creates_the_library(tmp_path: Path) -> None:
    folder = tmp_path / "new" / "مكتبة"
    with Database.open(folder) as db:
        assert db.writable
        assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        assert db.execute("PRAGMA foreign_keys").fetchone() == (1,)
        assert db.execute("PRAGMA synchronous").fetchone() == (2,)
        assert db.execute("PRAGMA busy_timeout").fetchone()[0] >= 1000
    assert (folder / DATABASE_FILE).is_file()


def test_folder_that_is_a_file(tmp_path: Path) -> None:
    (tmp_path / "file").write_bytes(b"")
    with pytest.raises(LibraryError, match="library folder"):
        Database.open(tmp_path / "file")


def test_not_a_database(library: Path) -> None:
    library.mkdir()
    (library / DATABASE_FILE).write_bytes(b"this is not SQLite" * 100)
    for write in (True, False):
        with pytest.raises(LibraryError):
            Database.open(library, write=write)
    (library / DATABASE_FILE).unlink()  # the failed opens let go of the file and the lock
    Database.open(library).close()


def test_foreign_database_is_left_alone(library: Path) -> None:
    library.mkdir()
    conn = sqlite3.connect(library / DATABASE_FILE)
    conn.execute("CREATE TABLE mine (x)")
    conn.close()
    with pytest.raises(LibraryError, match="not a MediaSift library"):
        Database.open(library)
    conn = sqlite3.connect(library / DATABASE_FILE)
    assert conn.execute("SELECT name FROM sqlite_master").fetchall() == [("mine",)]
    conn.close()


def test_close_twice(library: Path) -> None:
    db = Database.open(library)
    db.close()
    db.close()
    with pytest.raises(sqlite3.ProgrammingError):
        db.execute("SELECT 1")


# --- readers ---


def test_reader_needs_a_library(library: Path) -> None:
    with pytest.raises(LibraryError, match="there is no library"):
        Database.open(library, write=False)
    assert not library.exists()
    library.mkdir()
    (library / DATABASE_FILE).write_bytes(b"")
    with pytest.raises(LibraryError, match="there is no library"):
        Database.open(library, write=False)


def test_reader_after_the_writer_closed(library: Path) -> None:
    with Database.open(library) as db, db.transaction():
        add(db, "a")
    with Database.open(library, write=False) as reader:
        assert not reader.writable
        assert roots(reader) == ["a"]


def test_reader_beside_a_writer(library: Path) -> None:
    with Database.open(library) as writer, Database.open(library, write=False) as reader:
        with writer.transaction():
            add(writer, "a")
            assert roots(reader) == []
        assert roots(reader) == ["a"]


def test_reader_cannot_write(library: Path) -> None:
    Database.open(library).close()
    with Database.open(library, write=False) as reader:
        with pytest.raises(LibraryError, match="reading only"), reader.transaction():
            pass
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            add(reader, "a")


# --- one writer ---


def test_second_writer_in_the_same_process(library: Path) -> None:
    with Database.open(library), pytest.raises(LibraryBusy):
        Database.open(library)
    Database.open(library).close()


def test_second_writer_in_another_process(library: Path) -> None:
    with other_process(library) as (process, pid):
        with pytest.raises(LibraryBusy, match=f"process {pid}") as caught:
            Database.open(library)
        assert caught.value.exit_code is ExitCode.BUSY
        Database.open(library, write=False).close()

        process.kill()  # a crash
        process.wait()
        deadline = time.monotonic() + 10
        while True:
            try:
                Database.open(library).close()
                break
            except LibraryBusy:
                assert time.monotonic() < deadline, "the lock outlived its process"
                time.sleep(0.05)
    assert (library / LOCK_FILE).exists()


def test_failed_open_releases_the_lock(library: Path) -> None:
    library.mkdir()
    (library / DATABASE_FILE).mkdir()
    with pytest.raises(LibraryError):
        Database.open(library)
    (library / DATABASE_FILE).rmdir()
    Database.open(library).close()


def test_other_threads_are_refused(db: Database) -> None:
    errors: list[BaseException] = []

    def use() -> None:
        try:
            db.execute("SELECT 1")
        except sqlite3.ProgrammingError as exc:
            errors.append(exc)

    thread = threading.Thread(target=use)
    thread.start()
    thread.join()
    assert len(errors) == 1


# --- transactions ---


def test_commit(db: Database, library: Path) -> None:
    with db.transaction():
        assert db.in_transaction
        add(db, "a")
    assert not db.in_transaction
    db.close()
    with Database.open(library, write=False) as reader:
        assert roots(reader) == ["a"]


def test_rollback_on_any_exception(db: Database) -> None:
    for error in (ValueError, KeyboardInterrupt, GeneratorExit):
        with pytest.raises(error), db.transaction():
            add(db, "a")
            raise error
    assert roots(db) == []
    with db.transaction():
        add(db, "b")
    assert roots(db) == ["b"]


def test_statements_outside_a_transaction_commit_at_once(db: Database, library: Path) -> None:
    add(db, "a")
    with Database.open(library, write=False) as reader:
        assert roots(reader) == ["a"]


def test_inner_failure_undoes_the_inner_block_only(db: Database) -> None:
    with db.transaction():
        add(db, "a")
        with pytest.raises(ValueError), db.transaction():
            add(db, "b")
            with db.transaction():
                add(db, "c")
            raise ValueError
        with db.transaction():
            add(db, "d")
    assert roots(db) == ["a", "d"]


def test_outer_failure_undoes_inner_blocks(db: Database) -> None:
    with pytest.raises(ValueError), db.transaction():
        with db.transaction():
            add(db, "a")
        raise ValueError
    assert roots(db) == []


def test_dropped_transaction_never_turns_into_autocommit(db: Database) -> None:
    """A full disk makes SQLite roll back by itself; later statements must not slip through."""
    with pytest.raises(LibraryError, match="rolled back"), db.transaction():
        add(db, "a")
        db._conn.execute("ROLLBACK")
        add(db, "b")
    assert roots(db) == []
    with pytest.raises(LibraryError, match="rolled back"), db.transaction():
        add(db, "a")
        db._conn.execute("ROLLBACK")
    assert roots(db) == []
    with db.transaction():
        add(db, "c")
    assert roots(db) == ["c"]


def test_full_disk_is_a_library_error(db: Database) -> None:
    pages = db.execute("PRAGMA page_count").fetchone()[0]
    db.execute(f"PRAGMA max_page_count = {pages}")
    with pytest.raises(LibraryError, match="full"), db.transaction():
        for number in range(1000):
            add(db, f"{number:0>500}")
    assert roots(db) == []


def test_locked_by_another_program_is_busy(db: Database, library: Path) -> None:
    db.execute("PRAGMA busy_timeout = 50")
    other = sqlite3.connect(library / DATABASE_FILE, isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    with pytest.raises(LibraryBusy), db.transaction():
        add(db, "a")
    other.close()
    with db.transaction():
        add(db, "b")
    assert roots(db) == ["b"]


def test_programming_errors_stay_visible(db: Database) -> None:
    with pytest.raises(sqlite3.OperationalError, match="syntax"), db.transaction():
        db.execute("SELEKT 1")
    with pytest.raises(sqlite3.IntegrityError), db.transaction():
        add(db, "a")
        add(db, "a")
    assert roots(db) == []
