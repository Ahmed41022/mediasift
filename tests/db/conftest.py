from __future__ import annotations

import zlib
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from mediasift.core.types import FileRecord, FileSignature, MediaType, Root
from mediasift.db.connection import Database
from mediasift.db.files import FilesRepo, FoundFile
from mediasift.db.roots import RootsRepo
from mediasift.db.scans import ScansRepo

AddFiles = Callable[..., list[FileRecord]]


@pytest.fixture
def library(tmp_path: Path) -> Path:
    return tmp_path / "library"


@pytest.fixture
def db(library: Path) -> Iterator[Database]:
    with Database.open(library) as database:
        yield database


@pytest.fixture
def make_root(db: Database, tmp_path: Path) -> Callable[[str], Root]:
    def make(name: str) -> Root:
        (tmp_path / name).mkdir(exist_ok=True)
        return RootsRepo(db).add(tmp_path / name)

    return make


@pytest.fixture
def root(make_root: Callable[[str], Root]) -> Root:
    return make_root("photos")


@pytest.fixture
def scan_id(db: Database) -> int:
    return ScansRepo(db).start()


@pytest.fixture
def add_files(db: Database, root: Root, scan_id: int) -> AddFiles:
    """Index files that need not exist on disk: ``add_files("a.jpg", ("b.jpg", 10, 99))``."""

    def add(*specs: str | tuple[str, int, int], to: Root | None = None) -> list[FileRecord]:
        target = to or root
        found = []
        for spec in specs:
            name, size, mtime = (spec, 100, 1000) if isinstance(spec, str) else spec
            sig = FileSignature(size, mtime, 7, (target.id << 32) + zlib.crc32(name.encode()))
            kind = MediaType.VIDEO if name.endswith(".mp4") else MediaType.IMAGE
            found.append(FoundFile(target.id, name, kind, sig))
        files = FilesRepo(db)
        files.upsert_many(found, scan_id)
        known = files.find_by_paths(target.id, [item.rel_path for item in found])
        return [known[str(Path(item.rel_path))] for item in found]

    return add
