"""Opening the library: lock, connection, transactions."""

from __future__ import annotations

import os
import sqlite3
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import quote

from mediasift.core.errors import LibraryBusy, LibraryError
from mediasift.db import migrate
from mediasift.db.migrate import Migration

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

DATABASE_FILE = "library.db"
LOCK_FILE = "library.lock"
BACKUP_FOLDER = "backups"
BUSY_TIMEOUT_MS = 10_000
MIN_SQLITE = (3, 37, 0)  # STRICT tables

# Far beyond the text in the lock file, which Windows would otherwise make unreadable.
_LOCK_OFFSET = 1 << 30

_BUSY_CODES = {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}
_STORAGE_CODES = {
    sqlite3.SQLITE_READONLY,
    sqlite3.SQLITE_IOERR,
    sqlite3.SQLITE_CORRUPT,
    sqlite3.SQLITE_FULL,
    sqlite3.SQLITE_CANTOPEN,
    sqlite3.SQLITE_NOTADB,
}

Params = Sequence[Any] | dict[str, Any]


class _WriterLock:
    """Operating-system lock on a file. A crashed process cannot leave it behind.

    The file is never deleted: removing a lock file lets two processes hold "the" lock.
    """

    def __init__(self, path: Path) -> None:
        try:
            self._fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as exc:
            raise LibraryError(f"cannot open the lock file {path}: {exc}") from exc
        try:
            self._acquire(path)
            os.ftruncate(self._fd, 0)
            os.write(self._fd, f"{os.getpid()}\n".encode())
        except BaseException:
            os.close(self._fd)
            raise

    def _acquire(self, path: Path) -> None:
        try:
            if sys.platform == "win32":
                os.lseek(self._fd, _LOCK_OFFSET, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                os.lseek(self._fd, 0, os.SEEK_SET)
            else:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError) as exc:
            os.lseek(self._fd, 0, os.SEEK_SET)
            owner = os.read(self._fd, 32).decode("ascii", "replace").strip()
            who = f" (process {owner})" if owner.isdigit() else ""
            raise LibraryBusy(f"another process{who} is using the library {path.parent}") from exc
        except OSError as exc:
            raise LibraryError(f"cannot lock {path}: {exc}") from exc

    def release(self) -> None:
        """Closing the file releases the lock."""
        if self._fd >= 0:
            fd, self._fd = self._fd, -1
            os.close(fd)


class Database:
    """One connection to the library index. Use it from the thread that opened it.

    There is one writer per library, held by the lock file. The writer commits with
    ``synchronous=FULL``, so the operation log survives a power cut without a second
    writing connection, which could only deadlock against this one; scans commit in
    batches and do not notice. Readers take no lock and never migrate.
    """

    def __init__(
        self, conn: sqlite3.Connection, directory: Path, lock: _WriterLock | None = None
    ) -> None:
        self._conn = conn
        self._lock = lock
        self._depth = 0
        self.directory = directory

    @classmethod
    def open(
        cls,
        directory: str | os.PathLike[str],
        *,
        write: bool = True,
        migrations: Sequence[Migration] | None = None,
    ) -> Database:
        """Open the library in ``directory``.

        Writing creates and upgrades it and raises LibraryBusy when another process writes.
        Reading needs an existing library with the current schema.
        """
        if sqlite3.sqlite_version_info < MIN_SQLITE:
            raise LibraryError(f"SQLite {sqlite3.sqlite_version} is too old, 3.37 is needed")
        folder = Path(os.path.abspath(directory))  # noqa: PTH100
        steps = migrate.load_migrations() if migrations is None else migrations
        return cls._open_writer(folder, steps) if write else cls._open_reader(folder, steps)

    @classmethod
    def _open_writer(cls, folder: Path, steps: Sequence[Migration]) -> Database:
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise LibraryError(f"cannot create the library folder {folder}: {exc}") from exc
        lock = _WriterLock(folder / LOCK_FILE)
        try:
            conn = _connect(folder / DATABASE_FILE, write=True)
            try:
                with _as_library_errors():
                    _configure_writer(conn)
                    migrate.upgrade(conn, steps, folder / BACKUP_FOLDER)
            except BaseException:
                conn.close()
                raise
        except BaseException:
            lock.release()
            raise
        return cls(conn, folder, lock)

    @classmethod
    def _open_reader(cls, folder: Path, steps: Sequence[Migration]) -> Database:
        path = folder / DATABASE_FILE
        if not path.is_file():
            raise LibraryError(f"there is no library in {folder}")
        conn = _connect(path, write=False)
        try:
            with _as_library_errors():
                conn.execute("PRAGMA foreign_keys = ON")
                done = migrate.verify(conn, steps)
            if not done:
                raise LibraryError(f"there is no library in {folder}")
            if done < len(steps):
                raise LibraryError(
                    "the library was written by an older version of MediaSift; "
                    "run a scan to upgrade it"
                )
        except BaseException:
            conn.close()
            raise
        return cls(conn, folder)

    @property
    def writable(self) -> bool:
        """False for readers."""
        return self._lock is not None

    @property
    def in_transaction(self) -> bool:
        """True inside ``transaction()``."""
        return self._depth > 0

    def execute(self, sql: str, params: Params = ()) -> sqlite3.Cursor:
        """Run one statement."""
        self._check_transaction()
        return self._conn.execute(sql, params)

    def executemany(self, sql: str, rows: Iterable[Params]) -> int:
        """Run one statement for many rows. Return the number of rows changed."""
        self._check_transaction()
        return self._conn.executemany(sql, rows).rowcount

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit everything in the block, or nothing of it.

        Nested blocks are savepoints: an inner failure that is caught undoes the inner
        block only, and nothing is on disk before the outermost block ends.
        """
        if not self.writable:
            raise LibraryError("the library is open for reading only")
        name = f"level_{self._depth}"
        with _as_library_errors():
            self._check_transaction()
            self._conn.execute(f"SAVEPOINT {name}" if self._depth else "BEGIN IMMEDIATE")
            self._depth += 1
            try:
                yield
                self._check_transaction()
                self._conn.execute(f"RELEASE {name}" if self._depth > 1 else "COMMIT")
            except BaseException:
                # A full disk makes SQLite roll everything back by itself.
                if self._conn.in_transaction:
                    self._conn.execute(f"ROLLBACK TO {name}" if self._depth > 1 else "ROLLBACK")
                raise
            finally:
                self._depth -= 1

    def _check_transaction(self) -> None:
        """Refuse to go on in autocommit mode after SQLite dropped the transaction."""
        if self._depth and not self._conn.in_transaction:
            raise LibraryError("the transaction was rolled back by the database")

    def close(self) -> None:
        """Close the connection, then release the lock. Safe to call twice."""
        try:
            self._conn.close()
        finally:
            if self._lock is not None:
                self._lock.release()

    def __enter__(self) -> Database:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()


def _connect(path: Path, *, write: bool) -> sqlite3.Connection:
    # isolation_level=None: Python starts no transactions of its own.
    with _as_library_errors():
        if write:
            conn = sqlite3.connect(path, isolation_level=None)
        else:
            # Four slashes for UNC paths, three for drive letters, as SQLite expects.
            location = quote(os.fsencode(path.as_posix()))
            prefix = "" if location.startswith("/") else "/"
            uri = f"file://{prefix}{location}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, isolation_level=None)
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        return conn


def _configure_writer(conn: sqlite3.Connection) -> None:
    mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    if mode != "wal":
        raise LibraryError("the library folder does not support SQLite's WAL; use a local disk")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA foreign_keys = ON")


@contextmanager
def _as_library_errors() -> Iterator[None]:
    """Turn conditions a user can meet into LibraryError. Programming errors stay as they are."""
    try:
        yield
    except sqlite3.DatabaseError as exc:
        code = getattr(exc, "sqlite_errorcode", 0) & 0xFF
        if code in _BUSY_CODES:
            raise LibraryBusy(f"the library is locked by another program: {exc}") from exc
        if code in _STORAGE_CODES:
            raise LibraryError(f"the library cannot be used: {exc}") from exc
        raise
