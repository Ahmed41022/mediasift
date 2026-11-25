"""Scans and the problems they met."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from mediasift.core import paths
from mediasift.core.errors import LibraryError
from mediasift.core.paths import StrPath
from mediasift.db._util import clean_text, dump_json, load_json, now
from mediasift.db.connection import Database

_COLUMNS = "id, started_at, finished_at, state, options, counters"


class ScanState(StrEnum):
    """How a scan stands. ``interrupted`` means the process died."""

    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class ScanRecord:
    """One scan."""

    id: int
    started_at: str
    finished_at: str | None
    state: ScanState
    options: Mapping[str, Any]
    counters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ScanIssue:
    """A problem with one path. ``kind`` is a short stable code."""

    path: StrPath
    kind: str
    message: str = ""
    root_id: int | None = None


class ScansRepo:
    """Rows of the scans and scan_issues tables."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def start(self, options: Mapping[str, Any] | None = None) -> int:
        """Begin a scan and return its id.

        Only one process writes, so a scan still marked as running belongs to one that died.
        """
        with self._db.transaction():
            self._db.execute(
                "UPDATE scans SET state = 'interrupted', finished_at = ? WHERE state = 'running'",
                (now(),),
            )
            cursor = self._db.execute(
                "INSERT INTO scans (started_at, state, options) VALUES (?, 'running', ?)",
                (now(), dump_json(options or {})),
            )
        return cursor.lastrowid or 0

    def set_counters(self, scan_id: int, counters: Mapping[str, Any]) -> None:
        """Save progress counters of a scan."""
        with self._db.transaction():
            self._update("counters = ?", (dump_json(counters), scan_id))

    def finish(
        self, scan_id: int, state: ScanState, counters: Mapping[str, Any] | None = None
    ) -> None:
        """End a scan."""
        with self._db.transaction():
            self._update("state = ?, finished_at = ?", (state.value, now(), scan_id))
            if counters is not None:
                self.set_counters(scan_id, counters)

    def _update(self, assignments: str, params: tuple[Any, ...]) -> None:
        cursor = self._db.execute(f"UPDATE scans SET {assignments} WHERE id = ?", params)
        if cursor.rowcount != 1:
            raise LibraryError(f"there is no scan with number {params[-1]}")

    def get(self, scan_id: int) -> ScanRecord:
        """A scan by id."""
        found = self._records("WHERE id = ?", (scan_id,))
        if not found:
            raise LibraryError(f"there is no scan with number {scan_id}")
        return found[0]

    def latest(self) -> ScanRecord | None:
        """The newest scan, if any."""
        found = self._records("ORDER BY id DESC LIMIT 1", ())
        return found[0] if found else None

    def _records(self, rest: str, params: tuple[Any, ...]) -> list[ScanRecord]:
        rows = self._db.execute(f"SELECT {_COLUMNS} FROM scans {rest}", params).fetchall()
        return [
            ScanRecord(
                id=scan_id,
                started_at=started,
                finished_at=finished,
                state=ScanState(state),
                options=load_json(options),
                counters=load_json(counters),
            )
            for scan_id, started, finished, state, options, counters in rows
        ]

    def add_issues(self, scan_id: int, issues: Iterable[ScanIssue]) -> int:
        """Record problems."""
        with self._db.transaction():
            return self._db.executemany(
                "INSERT INTO scan_issues (scan_id, root_id, path, kind, message)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    (
                        scan_id,
                        issue.root_id,
                        paths.path_to_bytes(issue.path),
                        issue.kind,
                        clean_text(issue.message),
                    )
                    for issue in issues
                ),
            )

    def issues(self, scan_id: int, *, limit: int = 1000, offset: int = 0) -> list[ScanIssue]:
        """Problems of a scan, in the order they were met."""
        rows = self._db.execute(
            "SELECT path, kind, message, root_id FROM scan_issues WHERE scan_id = ?"
            " ORDER BY id LIMIT ? OFFSET ?",
            (scan_id, limit, offset),
        ).fetchall()
        return [ScanIssue(_path(raw), kind, text, root) for raw, kind, text, root in rows]

    def issue_counts(self, scan_id: int) -> dict[str, int]:
        """Number of problems by kind."""
        rows = self._db.execute(
            "SELECT kind, count(*) FROM scan_issues WHERE scan_id = ? GROUP BY kind ORDER BY kind",
            (scan_id,),
        ).fetchall()
        return dict(rows)


def _path(raw: bytes) -> Path:
    return paths.bytes_to_path(raw)
