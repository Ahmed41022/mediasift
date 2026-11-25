"""Indexed files.

Reads that can be long are generators that fetch one page at a time, ordered by id. Nothing
stays open between pages, so a caller may write while it iterates.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mediasift.core import paths
from mediasift.core.errors import LibraryError
from mediasift.core.paths import StrPath
from mediasift.core.types import FileRecord, FileSignature, FileStatus, MediaType
from mediasift.db._util import PAGE_SIZE, chunked, marks, to_signed, to_unsigned
from mediasift.db.connection import Database
from mediasift.db.roots import RootsRepo

RECORD_COLUMNS = (
    "f.id, f.root_id, f.rel_path, f.media_type, f.size, f.mtime_ns, f.dev, f.inode, f.nlink,"
    " f.status"
)
# Condition on files "f": nothing outside enabled roots is ever matched or touched.
ENABLED = "EXISTS (SELECT 1 FROM roots r WHERE r.id = f.root_id AND r.enabled = 1)"

_UPSERT = """
INSERT INTO files (root_id, rel_path, rel_display, rel_key, ext, media_type, size, mtime_ns,
                   dev, inode, nlink, status, first_scan_id, last_scan_id)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (root_id, rel_path) DO UPDATE SET
  media_type = excluded.media_type, size = excluded.size, mtime_ns = excluded.mtime_ns,
  dev = excluded.dev, inode = excluded.inode, nlink = excluded.nlink,
  status = excluded.status, last_scan_id = excluded.last_scan_id
"""


@dataclass(frozen=True, slots=True)
class FoundFile:
    """A file as the scanner sees it."""

    root_id: int
    rel_path: StrPath
    media_type: MediaType
    signature: FileSignature
    status: FileStatus = FileStatus.PRESENT


@dataclass(frozen=True, slots=True)
class FileMove:
    """A known file found at another place."""

    file_id: int
    root_id: int
    rel_path: StrPath
    signature: FileSignature


@dataclass(frozen=True, slots=True)
class TypeStats:
    """How many files of one media type and status there are."""

    media_type: MediaType
    status: FileStatus
    count: int
    bytes: int


class RecordBuilder:
    """Turns rows of RECORD_COLUMNS into FileRecord, looking root folders up once."""

    def __init__(self, db: Database) -> None:
        self._roots = RootsRepo(db)
        self._paths: dict[int, Path] = {}

    def __call__(self, row: tuple[Any, ...]) -> FileRecord:
        file_id, root_id, raw, media_type, size, mtime_ns, dev, inode, nlink, status = row
        root = self._paths.get(root_id)
        if root is None:
            root = self._paths[root_id] = self._roots.get(root_id).path
        rel_path = paths.bytes_to_path(raw)
        return FileRecord(
            id=file_id,
            root_id=root_id,
            path=root / rel_path,
            rel_path=rel_path,
            media_type=MediaType(media_type),
            signature=FileSignature(size, mtime_ns, to_unsigned(dev), to_unsigned(inode), nlink),
            status=FileStatus(status),
        )


class FilesRepo:
    """Rows of the files table."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._roots = RootsRepo(db)

    # --- reading ---

    def get(self, file_id: int) -> FileRecord:
        """A file by id."""
        found = self.get_many([file_id])
        if not found:
            raise LibraryError(f"there is no file with number {file_id} in the library")
        return found[file_id]

    def get_many(self, file_ids: Iterable[int]) -> dict[int, FileRecord]:
        """Files by id. Unknown ids are left out."""
        build = RecordBuilder(self._db)
        found = {}
        for ids in chunked(file_ids):
            for row in self._select(f"f.id IN ({marks(len(ids))})", ids):
                found[row[0]] = build(row)
        return found

    def find_by_paths(self, root_id: int, rel_paths: Iterable[StrPath]) -> dict[str, FileRecord]:
        """Known files of a root at these places, whatever their status.

        Keyed by the exact text of ``rel_path``: Path objects would mix up names that differ
        by case, which Windows allows in some folders.
        """
        build = RecordBuilder(self._db)
        found = {}
        for chunk in chunked(paths.path_to_bytes(Path(rel)) for rel in rel_paths):
            where = f"f.root_id = ? AND f.rel_path IN ({marks(len(chunk))})"
            for row in self._select(where, [root_id, *chunk]):
                record = build(row)
                found[str(record.rel_path)] = record
        return found

    def find_by_identity(self, dev: int, inode: int) -> list[FileRecord]:
        """Known files with this identity: hard links, or where a moved file used to be."""
        if not inode:
            return []
        where = "f.inode <> 0 AND f.dev = ? AND f.inode = ?"
        return self._records(where, [to_signed(dev), to_signed(inode)])

    def find_by_size_mtime(self, size: int, mtime_ns: int) -> list[FileRecord]:
        """Known files that may be a file moved from another volume."""
        return self._records("f.size = ? AND f.mtime_ns = ?", [size, mtime_ns])

    def iter_root(self, root_id: int, *, page_size: int = PAGE_SIZE) -> Iterator[FileRecord]:
        """All known files of a root, whatever their status."""
        yield from self.iter_where("f.root_id = :root", {"root": root_id}, page_size=page_size)

    def iter_present(
        self,
        media_types: Collection[MediaType] | None = None,
        *,
        page_size: int = PAGE_SIZE,
    ) -> Iterator[FileRecord]:
        """Present files of enabled roots, of the given media types."""
        where, params = media_filter(media_types)
        where = f"f.status = 'present' AND {ENABLED}{where}"
        yield from self.iter_where(where, params, page_size=page_size)

    def iter_where(
        self, where: str, params: dict[str, Any], *, page_size: int = PAGE_SIZE
    ) -> Iterator[FileRecord]:
        """Files matching a condition, for the other repositories."""
        build = RecordBuilder(self._db)
        # NOT INDEXED: walk by id. Any other index would need a sort for every page.
        sql = (
            f"SELECT {RECORD_COLUMNS} FROM files f NOT INDEXED"
            f" WHERE f.id > :after AND ({where}) ORDER BY f.id LIMIT :limit"
        )
        after = 0
        while True:
            page = {**params, "after": after, "limit": page_size}
            rows = self._db.execute(sql, page).fetchall()
            for row in rows:
                yield build(row)
            if len(rows) < page_size:
                return
            after = rows[-1][0]

    def stats(self, *, enabled_only: bool = True) -> list[TypeStats]:
        """Counts and bytes by media type and status."""
        where = f" WHERE {ENABLED}" if enabled_only else ""
        rows = self._db.execute(
            "SELECT f.media_type, f.status, count(*), sum(f.size) FROM files f"
            f"{where} GROUP BY f.media_type, f.status ORDER BY f.media_type, f.status"
        ).fetchall()
        return [TypeStats(MediaType(m), FileStatus(s), n, size) for m, s, n, size in rows]

    # --- writing ---

    def upsert_many(self, files: Iterable[FoundFile], scan_id: int) -> int:
        """Add files, or update the ones already known at the same place.

        Cached features of a changed file are not deleted: they stop being valid because
        they carry the size and time they were computed from.
        """
        sensitive = {root.id: root.case_sensitive for root in self._roots.list()}
        with self._db.transaction():
            return self._db.executemany(
                _UPSERT, (self._upsert_row(file, scan_id, sensitive) for file in files)
            )

    @staticmethod
    def _upsert_row(file: FoundFile, scan_id: int, sensitive: dict[int, bool]) -> tuple[Any, ...]:
        if file.root_id not in sensitive:
            raise LibraryError(f"there is no folder with number {file.root_id} in the library")
        place = _place(file.rel_path, sensitive[file.root_id])
        sig = file.signature
        return (
            file.root_id,
            *place,
            file.media_type.value,
            *_signature_row(sig),
            file.status.value,
            scan_id,
            scan_id,
        )

    def mark_seen(
        self,
        file_ids: Iterable[int],
        scan_id: int,
        status: FileStatus = FileStatus.PRESENT,
    ) -> int:
        """Record that these files were found unchanged by a scan."""
        with self._db.transaction():
            return self._db.executemany(
                "UPDATE files SET last_scan_id = ?, status = ? WHERE id = ?",
                ((scan_id, status.value, file_id) for file_id in file_ids),
            )

    def set_status(self, file_ids: Iterable[int], status: FileStatus) -> int:
        """Change the status without touching when the files were last seen."""
        with self._db.transaction():
            return self._db.executemany(
                "UPDATE files SET status = ? WHERE id = ?",
                ((status.value, file_id) for file_id in file_ids),
            )

    def mark_missing(
        self, root_id: int, scan_id: int, *, unread_folders: Iterable[StrPath] = ()
    ) -> int:
        """Mark the files of a root that this scan did not see as missing.

        Call it only after a complete walk of a reachable root. Folders that could not be
        listed go in ``unread_folders`` (relative to the root): nothing is known about what
        is below them, so their files keep their status.
        """
        unread = {os.path.normpath(folder) for folder in unread_folders}
        if "." in unread:
            return 0
        where = "f.root_id = :root AND f.status <> 'missing' AND f.last_scan_id IS NOT :scan"
        gone = (
            record.id
            for record in self.iter_where(where, {"root": root_id, "scan": scan_id})
            if not _below(record.rel_path, unread)
        )
        with self._db.transaction():
            return sum(self.set_status(ids, FileStatus.MISSING) for ids in chunked(gone))

    def move(self, move: FileMove, scan_id: int) -> None:
        """Move one row. See ``move_many``."""
        self.move_many([move], scan_id)

    def move_many(self, moves: Collection[FileMove], scan_id: int) -> None:
        """Give known files a new place, keeping their id and so features and decisions.

        All or nothing, and in no particular order: swaps and chains (a to b, b to c) work.
        A missing file that used to be at a target place is forgotten; a target still held
        by another file is an error.
        """
        sensitive = {root.id: root.case_sensitive for root in self._roots.list()}
        with self._db.transaction():
            # Out of the way first. No real path holds a NUL, so these cannot collide.
            self._db.executemany(
                "UPDATE files SET rel_path = ? WHERE id = ?",
                ((b"\0moving %d" % move.file_id, move.file_id) for move in moves),
            )
            for move in moves:
                self._place_moved(move, scan_id, sensitive)

    def _place_moved(self, move: FileMove, scan_id: int, sensitive: dict[int, bool]) -> None:
        if move.root_id not in sensitive:
            raise LibraryError(f"there is no folder with number {move.root_id} in the library")
        raw, display, key, ext = _place(move.rel_path, sensitive[move.root_id])
        self._db.execute(
            "DELETE FROM files WHERE root_id = ? AND rel_path = ? AND status = 'missing'",
            (move.root_id, raw),
        )
        holder = self._db.execute(
            "SELECT id FROM files WHERE root_id = ? AND rel_path = ?", (move.root_id, raw)
        ).fetchone()
        if holder is not None:
            raise LibraryError(f"{display} is already a known file")
        cursor = self._db.execute(
            "UPDATE files SET root_id = ?, rel_path = ?, rel_display = ?, rel_key = ?, ext = ?,"
            " size = ?, mtime_ns = ?, dev = ?, inode = ?, nlink = ?,"
            " status = 'present', last_scan_id = ? WHERE id = ?",
            (
                move.root_id,
                raw,
                display,
                key,
                ext,
                *_signature_row(move.signature),
                scan_id,
                move.file_id,
            ),
        )
        if cursor.rowcount != 1:
            raise LibraryError(f"there is no file with number {move.file_id} in the library")

    # --- helpers ---

    def _select(self, where: str, params: list[Any]) -> list[tuple[Any, ...]]:
        return self._db.execute(
            f"SELECT {RECORD_COLUMNS} FROM files f WHERE {where} ORDER BY f.id", params
        ).fetchall()

    def _records(self, where: str, params: list[Any]) -> list[FileRecord]:
        build = RecordBuilder(self._db)
        return [build(row) for row in self._select(where, params)]


def media_filter(media_types: Collection[MediaType] | None) -> tuple[str, dict[str, Any]]:
    """Condition and parameters for a media type filter. None means no filter."""
    if media_types is None:
        return "", {}
    params = {f"media{n}": media.value for n, media in enumerate(sorted(media_types))}
    names = ", ".join(f":{name}" for name in params)
    return f" AND f.media_type IN ({names})", params


def _place(rel_path: StrPath, case_sensitive: bool) -> tuple[bytes, str, bytes, str]:
    """Columns that follow from a relative path: bytes, display, key, extension."""
    rel = Path(rel_path)
    if rel.is_absolute() or rel.drive or rel.root or not rel.parts or ".." in rel.parts:
        raise LibraryError(f"not a path inside a folder: {paths.display_path(rel)}")
    display = paths.display_path(rel)
    key = paths.path_key(rel, case_sensitive=case_sensitive)
    return paths.path_to_bytes(rel), display, key, paths.display_path(rel.suffix.lower())


def _signature_row(sig: FileSignature) -> tuple[int, int, int, int, int]:
    return sig.size, sig.mtime_ns, to_signed(sig.dev), to_signed(sig.inode), sig.nlink


def _below(rel_path: Path, folders: set[str]) -> bool:
    return bool(folders) and any(str(parent) in folders for parent in rel_path.parents)
