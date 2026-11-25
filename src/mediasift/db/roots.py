"""The folders the user chose."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from mediasift.core import paths
from mediasift.core.errors import LibraryError, SafetyError
from mediasift.core.paths import StrPath
from mediasift.core.types import Root
from mediasift.db.connection import Database

_COLUMNS = "id, path, case_sensitive, read_only, prefer_rank, enabled, volume_id"


class RootsRepo:
    """Roots never overlap, so every file belongs to exactly one."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, path: StrPath, *, read_only: bool = False, prefer_rank: int = 0) -> Root:
        """Register a folder under its real path, with links resolved.

        Refuses a folder that contains, lies in, or is an existing root, disabled ones
        included. Use ``find`` first when adding an existing root should be harmless.
        """
        real = self._real_folder(path)
        for other in self.list():
            if paths.same_location(real, other.path):
                raise SafetyError(
                    f"{paths.display_path(real)} is already in the library", code="root_exists"
                )
            if paths.is_within(real, other.path) or paths.is_within(other.path, real):
                raise SafetyError(
                    f"{paths.display_path(real)} overlaps {paths.display_path(other.path)}, "
                    "which is already in the library",
                    code="root_overlap",
                )
        sensitive = paths.is_case_sensitive(real)
        row = (
            paths.path_to_bytes(real),
            paths.display_path(real),
            paths.path_key(real, case_sensitive=sensitive),
            paths.volume_id(real),
            sensitive,
            read_only,
            prefer_rank,
        )
        with self._db.transaction():
            try:
                cursor = self._db.execute(
                    "INSERT INTO roots (path, display, path_key, volume_id, case_sensitive,"
                    " read_only, prefer_rank) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    row,
                )
            except sqlite3.IntegrityError as exc:
                raise SafetyError(
                    f"{paths.display_path(real)} cannot be told apart from a folder that is "
                    "already in the library",
                    code="root_exists",
                ) from exc
        return self.get(cursor.lastrowid or 0)

    @staticmethod
    def _real_folder(path: StrPath) -> Path:
        try:
            real = paths.resolve(path)
            is_folder = Path(paths.fs_path(real)).is_dir()
        except (OSError, ValueError) as exc:
            raise LibraryError(f"cannot use {paths.display_path(path)}: {exc}") from exc
        if not is_folder:
            raise LibraryError(f"{paths.display_path(path)} is not a folder")
        if paths.is_link_like(real):
            raise LibraryError(f"{paths.display_path(path)} is a link that cannot be resolved")
        return real

    def list(self, *, enabled_only: bool = False) -> list[Root]:
        """Roots, preferred ones first."""
        where = " WHERE enabled = 1" if enabled_only else ""
        rows = self._db.execute(
            f"SELECT {_COLUMNS} FROM roots{where} ORDER BY prefer_rank DESC, id"
        ).fetchall()
        return [_root(row) for row in rows]

    def get(self, root_id: int) -> Root:
        """A root by id."""
        row = self._db.execute(f"SELECT {_COLUMNS} FROM roots WHERE id = ?", (root_id,)).fetchone()
        if row is None:
            raise LibraryError(f"there is no folder with number {root_id} in the library")
        return _root(row)

    def find(self, path: StrPath) -> Root | None:
        """The root that is this folder, if any."""
        for root in self.list():
            if paths.same_location(path, root.path):
                return root
        return None

    def find_containing(self, path: StrPath) -> Root | None:
        """The root that is, or contains, this path."""
        for root in self.list():
            if paths.same_location(path, root.path) or paths.is_within(path, root.path):
                return root
        return None

    def set_enabled(self, root_id: int, enabled: bool) -> Root:
        """Files of disabled roots are neither scanned, matched nor touched."""
        return self._set(root_id, "enabled", enabled)

    def set_read_only(self, root_id: int, read_only: bool) -> Root:
        """Files of read-only roots are matched, never acted upon."""
        return self._set(root_id, "read_only", read_only)

    def set_prefer_rank(self, root_id: int, rank: int) -> Root:
        """A higher rank makes the keeper choose this root's files."""
        return self._set(root_id, "prefer_rank", rank)

    def set_volume_id(self, root_id: int, volume_id: str) -> Root:
        """Record the volume the root is now known to live on."""
        return self._set(root_id, "volume_id", volume_id)

    def _set(self, root_id: int, column: str, value: object) -> Root:
        with self._db.transaction():
            self._db.execute(f"UPDATE roots SET {column} = ? WHERE id = ?", (value, root_id))
            return self.get(root_id)


def _root(row: tuple[Any, ...]) -> Root:
    root_id, path, case_sensitive, read_only, prefer_rank, enabled, volume = row
    return Root(
        id=root_id,
        path=paths.bytes_to_path(path),
        case_sensitive=bool(case_sensitive),
        read_only=bool(read_only),
        prefer_rank=prefer_rank,
        enabled=bool(enabled),
        volume_id=volume,
    )
