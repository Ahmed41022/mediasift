from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from mediasift.core.config import JobSettings, ScanSettings, Settings
from mediasift.core.types import FileRecord, FileStatus, Root
from mediasift.db.connection import Database
from mediasift.db.files import FilesRepo
from mediasift.db.roots import RootsRepo
from mediasift.jobs.context import JobContext
from mediasift.scan.scanner import Scanner, ScanResult

MakeRoot = Callable[[str], Root]


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    with Database.open(tmp_path / "library") as database:
        yield database


@pytest.fixture
def make_root(db: Database, tmp_path: Path) -> MakeRoot:
    def make(name: str) -> Root:
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
        return RootsRepo(db).add(tmp_path / name)

    return make


@pytest.fixture
def root(make_root: MakeRoot) -> Root:
    return make_root("photos")


def settings(batch_size: int = 200, **scan: object) -> Settings:
    return Settings(scan=ScanSettings(**scan), jobs=JobSettings(batch_size=batch_size))  # type: ignore[arg-type]


def run(
    db: Database,
    roots: list[Root] | None = None,
    config: Settings | None = None,
    job: JobContext | None = None,
) -> ScanResult:
    return Scanner(db, config or settings()).scan(roots, job)


def records(db: Database) -> dict[Path, FileRecord]:
    """Every indexed file by its absolute path."""
    files = FilesRepo(db)
    found: dict[Path, FileRecord] = {}
    for root in RootsRepo(db).list():
        for record in files.iter_root(root.id):
            found[record.path] = record
    return found


def present(db: Database) -> dict[Path, int]:
    """Ids of present files by absolute path."""
    return {p: r.id for p, r in records(db).items() if r.status is FileStatus.PRESENT}


def write(path: Path, data: bytes, mtime_ns: int | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if mtime_ns is not None:
        os.utime(path, ns=(mtime_ns, mtime_ns))
    return path
