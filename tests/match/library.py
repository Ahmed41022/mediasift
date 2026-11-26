"""Index folders without the scanner: rows are written with FilesRepo.upsert_many."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from mediasift.core import paths
from mediasift.core.types import FileRecord, MatchKind, Root, media_type_for
from mediasift.db.connection import Database
from mediasift.db.files import FilesRepo, FoundFile
from mediasift.db.matches import MatchesRepo
from mediasift.db.roots import RootsRepo
from mediasift.db.scans import ScansRepo

# A group as a test compares it: relative paths of the members, and the bytes it frees.
Group = tuple[frozenset[str], int]


class Library:
    """A library index over folders below a test's tmp_path."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.files = FilesRepo(db)
        self.scan_id = ScansRepo(db).start()

    def add_root(self, folder: Path) -> Root:
        folder.mkdir(parents=True, exist_ok=True)
        return RootsRepo(self.db).add(folder)

    def index(self, root: Root) -> dict[str, FileRecord]:
        """Index the media files below a root as they are now, by relative path text."""
        top = paths.fs_path(root.path)
        found = []
        for folder, _, names in os.walk(top):
            for name in names:
                full = os.path.join(folder, name)
                st = os.lstat(full)
                media = media_type_for(name)
                if media is None or paths.is_link_like(st):
                    continue
                rel = os.path.relpath(full, top)
                found.append(FoundFile(root.id, rel, media, paths.signature(st)))
        self.files.upsert_many(found, self.scan_id)
        return self.records(root)

    def refresh(self, root: Root, rel_paths: Iterable[str]) -> None:
        """Index these files again, as a rescan would after they changed."""
        found = []
        for rel in rel_paths:
            media = media_type_for(rel)
            assert media is not None
            found.append(FoundFile(root.id, rel, media, paths.signature(root.path / rel)))
        self.files.upsert_many(found, self.scan_id)

    def records(self, root: Root) -> dict[str, FileRecord]:
        return {_rel(r.rel_path): r for r in self.files.iter_root(root.id)}

    def groups(self) -> set[Group]:
        """Stored exact groups."""
        found = set()
        for group in MatchesRepo(self.db).iter_groups(MatchKind.EXACT):
            records = self.files.get_many(member.file_id for member in group.members)
            rels = frozenset(_rel(record.rel_path) for record in records.values())
            found.add((rels, group.reclaimable_bytes))
        return found


def _rel(path: Path) -> str:
    return path.as_posix()


def rel_of(path: Path, root: Path) -> str:
    """Relative path text of a file the media kit wrote."""
    return path.relative_to(root).as_posix()
