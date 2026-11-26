from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from mediasift.actions import checks
from mediasift.actions.backends.base import PlannedRemoval, RemovalRequest
from mediasift.actions.checks import KnownFile, VerifiedFile
from mediasift.core import paths
from mediasift.core.types import FileRecord, MediaType, Root

WINDOWS = sys.platform == "win32"
POSIX = not WINDOWS
ARABIC = "صورة العائلة.jpg"


def digest(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=32).digest()


def write(path: Path, data: bytes = b"photo") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def make_symlink(link: Path, target: Path, *, folder: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=folder)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("the system does not allow creating symlinks")
        raise


def make_folder_link(link: Path, target: Path) -> None:
    """A junction on Windows, which needs no privilege; a symlink elsewhere."""
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


_EXTRA = (
    "st_atime",
    "st_mtime",
    "st_ctime",
    "st_atime_ns",
    "st_mtime_ns",
    "st_ctime_ns",
    "st_file_attributes",
    "st_reparse_tag",
    "st_blocks",
    "st_blksize",
)


def on_other_device(st: os.stat_result) -> os.stat_result:
    """The same stat as if the file were on another volume."""
    fields = list(st)[:10]
    fields[2] = st.st_dev + 1
    extra = {name: getattr(st, name) for name in _EXTRA if hasattr(st, name)}
    return os.stat_result(fields, extra)


def set_read_only(path: Path, read_only: bool = True) -> None:
    os.chmod(path, 0o444 if read_only else 0o644)


@dataclass
class Library:
    """A root with files the way the index would record them."""

    root: Root
    next_id: int = 1

    @property
    def folder(self) -> Path:
        return self.root.path

    def add(self, rel: str, data: bytes = b"photo", mtime_ns: int | None = None) -> KnownFile:
        path = write(self.folder / rel, data)
        if mtime_ns is not None:
            os.utime(path, ns=(mtime_ns, mtime_ns))
        record = FileRecord(
            id=self.next_id,
            root_id=self.root.id,
            path=path,
            rel_path=Path(rel),
            media_type=MediaType.IMAGE,
            signature=paths.signature(path),
        )
        self.next_id += 1
        return KnownFile(record, digest(data))

    def with_root(self, **changes: object) -> Library:
        return Library(replace(self.root, **changes), self.next_id)


def make_library(folder: Path, root_id: int = 1) -> Library:
    folder.mkdir(parents=True, exist_ok=True)
    real = paths.resolve(folder)
    return Library(Root(id=root_id, path=real, case_sensitive=paths.is_case_sensitive(real)))


@pytest.fixture
def lib(tmp_path: Path) -> Library:
    return make_library(tmp_path / "photos")


def verified(item: KnownFile) -> VerifiedFile:
    path = item.record.path
    return VerifiedFile(path, paths.signature(path), item.full_hash)


def request_for(planned: PlannedRemoval, keeper: KnownFile) -> RemovalRequest:
    return RemovalRequest(planned, paths.signature(planned.src), verified(keeper))


def checked_request(
    planned: PlannedRemoval, item: KnownFile, keeper: KnownFile, root: Root
) -> RemovalRequest:
    result = checks.check_item(item, [keeper], [root])
    return RemovalRequest(planned, result.file.signature, result.keeper)
