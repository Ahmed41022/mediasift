"""The Recycle Bin backend against a fake shell and bin in tmp_path.

Only ``test_real_recycle_bin`` uses the real one; it is marked ``realbin``.
"""

from __future__ import annotations

import os
import struct
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mediasift.actions.backends.base import Location, RestoreRequest, StoredFile
from mediasift.actions.backends.recycle_bin import (
    BIN_FOLDER,
    RECORDS_FOLDER,
    RecycleBinBackend,
    Volume,
    is_shell_safe,
    parse_record,
)
from mediasift.actions.checks import KnownFile
from mediasift.core.errors import ActionError, SafetyError
from mediasift.core.paths import QUARANTINE_FOLDER
from tests.actions.conftest import Library, checked_request, make_library, write

USER = "S-1-5-21-1000"


def record_bytes(original: str, *, version: int = 2, size: int = 5, ticks: int = 0) -> bytes:
    head = struct.pack("<qqQ", version, size, ticks)
    text = (original + "\0").encode("utf-16-le", "surrogatepass")
    if version == 1:
        return head + text.ljust(520, b"\0")
    return head + struct.pack("<I", len(text) // 2) + text


class FakeBin:
    """A shell that does what it is told, and the bin it writes to."""

    def __init__(self, volume: Path) -> None:
        self.volume = volume
        self.folder = volume / BIN_FOLDER / USER
        self.folder.mkdir(parents=True)
        self.mode = "bin"
        self.count = 0
        self.sent: list[str] = []

    def info(self, path: Path) -> Volume:
        return Volume(self.volume, fixed=True, hard_links=True)

    def put(self, path: Path, original: str, ticks: int = 10**17) -> Path:
        self.count += 1
        content = self.folder / f"$R{self.count:06d}{path.suffix}"
        os.rename(path, content)
        index = self.folder / ("$I" + content.name[2:])
        index.write_bytes(record_bytes(original, size=content.stat().st_size, ticks=ticks))
        return content

    def send(self, path: str) -> None:
        self.sent.append(path)
        if self.mode == "bin":
            self.put(Path(path), path)
        elif self.mode == "nuke":  # a full or disabled bin deletes for good
            os.unlink(path)
        elif self.mode == "fail":
            raise OSError(5, "Access is denied")
        elif self.mode == "elsewhere":
            self.put(Path(path), "C:\\somewhere\\else.jpg")
        elif self.mode == "nuke_and_fail":
            os.unlink(path)
            raise OSError(0x78, "shell error")

    def contents(self) -> list[Path]:
        return sorted(self.folder.glob("$R*"))


@pytest.fixture
def fake(tmp_path: Path) -> FakeBin:
    return FakeBin(tmp_path / "volume")


@pytest.fixture
def vlib(tmp_path: Path, fake: FakeBin) -> Library:
    return make_library(tmp_path / "volume" / "photos")


def backend(fake: FakeBin, **more: object) -> RecycleBinBackend:
    return RecycleBinBackend(send=fake.send, volume=fake.info, delay=0, **more)  # type: ignore[arg-type]


def code_of(call: Callable[[], object]) -> str:
    with pytest.raises(SafetyError) as caught:
        call()
    return caught.value.code


def remove(bin_: RecycleBinBackend, lib: Library, item: KnownFile, keeper: KnownFile) -> StoredFile:
    planned = bin_.plan_destination(item.record.path, lib.root, "s1")
    return bin_.remove(checked_request(planned, item, keeper, lib.root))


def leftovers(lib: Library) -> list[str]:
    session = lib.folder / QUARANTINE_FOLDER / "s1"
    return [p.name for p in session.rglob("*") if p.is_file()] if session.exists() else []


# --- the $I record ---


def test_parse_version_2() -> None:
    record = parse_record(
        record_bytes("C:\\Photos\\صورة.jpg", size=1234, ticks=133_000_000_000_000_000)
    )
    assert (record.version, record.size, record.original) == (2, 1234, "C:\\Photos\\صورة.jpg")
    assert record.deleted_at == datetime(
        2022, 6, 18, 4, 26, 40, tzinfo=UTC
    )  # 1655526400 in Unix time


def test_parse_version_1() -> None:
    record = parse_record(record_bytes("D:\\a.jpg", version=1, size=7))
    assert (record.version, record.size, record.original) == (1, 7, "D:\\a.jpg")
    assert record.deleted_at == datetime(1601, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x02" + b"\0" * 22,
        record_bytes("C:\\a", version=3),
        record_bytes("C:\\a.jpg")[:-4],
        record_bytes("C:\\a", version=1)[:100],
        struct.pack("<qqQI", 2, 0, 0, 0),
        record_bytes(""),
        struct.pack("<qqQ", 2, 0, 2**64 - 1) + struct.pack("<I", 2) + "a\0".encode("utf-16-le"),
    ],
)
def test_parse_refuses_garbage(data: bytes) -> None:
    with pytest.raises(ValueError):
        parse_record(data)


@pytest.mark.parametrize(
    ("path", "safe"),
    [
        ("C:\\Photos\\a.jpg", True),
        ("C:\\Photos\\صورة العائلة.jpg", True),
        ("C:\\Photos\\photo.", False),
        ("C:\\Photos\\photo ", False),
        ("C:\\Photos\\nul.jpg", False),
        ("C:\\Photos\\COM1", False),
        ("C:\\con\\a.jpg", False),
        ("C:\\Photos\\a*.jpg", False),
        ("C:\\" + "a" * 300, False),
        ("C:\\Photos\\console.jpg", True),
    ],
)
def test_shell_safe_names(path: str, safe: bool) -> None:
    assert is_shell_safe(path) is safe


# --- removing ---


def test_remove_and_restore(fake: FakeBin, vlib: Library) -> None:
    bin_ = backend(fake)
    item, keeper = vlib.add("2024/a.jpg", b"mine"), vlib.add("keep.jpg", b"mine")
    mtime = item.record.signature.mtime_ns
    stored = remove(bin_, vlib, item, keeper)
    assert fake.sent == [str(item.record.path)]
    assert stored.dst == fake.contents()[0]
    assert stored.dst.read_bytes() == b"mine"
    assert stored.dst.stat().st_nlink == 1  # the guard was released
    assert leftovers(vlib) == []
    assert not item.record.path.exists()
    request = RestoreRequest(
        stored, item.record.path, vlib.root, item.record.signature, item.full_hash
    )
    restored = bin_.restore(request)
    assert restored.path == item.record.path and not restored.renamed
    assert item.record.path.read_bytes() == b"mine"
    assert item.record.path.stat().st_mtime_ns == mtime
    assert fake.contents() == [] and list(fake.folder.iterdir()) == []
    records = vlib.folder / QUARANTINE_FOLDER / RECORDS_FOLDER
    assert [p.name.startswith("$I") for p in records.iterdir()] == [True]


@pytest.mark.parametrize("mode", ["nuke", "nuke_and_fail"])
def test_bin_that_deletes_for_good(fake: FakeBin, vlib: Library, mode: str) -> None:
    fake.mode = mode
    item, keeper = vlib.add("a.jpg", b"mine"), vlib.add("keep.jpg", b"mine")
    with pytest.raises(ActionError, match="put back"):
        remove(backend(fake), vlib, item, keeper)
    assert item.record.path.read_bytes() == b"mine"
    assert item.record.path.stat().st_mtime_ns == item.record.signature.mtime_ns
    assert leftovers(vlib) == []


def test_shell_refuses(fake: FakeBin, vlib: Library) -> None:
    fake.mode = "fail"
    item, keeper = vlib.add("a.jpg"), vlib.add("keep.jpg")
    with pytest.raises(ActionError, match="untouched"):
        remove(backend(fake), vlib, item, keeper)
    assert item.record.path.stat().st_nlink == 1
    assert leftovers(vlib) == []


def test_shell_does_nothing(fake: FakeBin, vlib: Library) -> None:
    fake.mode = "ignore"
    item, keeper = vlib.add("a.jpg"), vlib.add("keep.jpg")
    with pytest.raises(ActionError, match="untouched"):
        remove(backend(fake), vlib, item, keeper)
    assert item.record.path.exists() and leftovers(vlib) == []


def test_entry_with_another_path_is_not_trusted(fake: FakeBin, vlib: Library) -> None:
    fake.mode = "elsewhere"
    item, keeper = vlib.add("a.jpg", b"mine"), vlib.add("keep.jpg", b"mine")
    with pytest.raises(ActionError, match="put back"):
        remove(backend(fake), vlib, item, keeper)
    assert item.record.path.read_bytes() == b"mine"
    # The file now has two names, one in the bin: nothing is lost.
    assert leftovers(vlib) == []


def test_entry_of_another_file_with_the_same_path(fake: FakeBin, vlib: Library) -> None:
    item, keeper = vlib.add("a.jpg", b"mine"), vlib.add("keep.jpg", b"mine")
    other = write(vlib.folder / "other.jpg", b"mine")
    older = fake.put(other, str(item.record.path), ticks=10**18)  # newer date, other file
    stored = remove(backend(fake), vlib, item, keeper)
    assert stored.dst != older
    assert stored.dst.read_bytes() == b"mine"
    assert len(fake.contents()) == 2


def test_two_entries_of_the_same_file(fake: FakeBin, vlib: Library) -> None:
    """An entry that was there before belongs to an earlier deletion, whatever its date says."""
    item, keeper = vlib.add("a.jpg"), vlib.add("keep.jpg")
    old_link = vlib.folder / "old-link.jpg"
    os.link(item.record.path, old_link)
    earlier = fake.put(old_link, str(item.record.path), ticks=10**18)
    stored = remove(backend(fake), vlib, item, keeper)
    assert stored.dst != earlier
    assert stored.dst.exists() and earlier.exists()


def test_new_entry_of_another_file_is_not_trusted(fake: FakeBin, vlib: Library) -> None:
    item, keeper = vlib.add("a.jpg", b"mine"), vlib.add("keep.jpg", b"mine")

    def send(path: str) -> None:  # deletes for good, and a stranger shows up under that path
        os.unlink(path)
        fake.put(write(vlib.folder / "decoy.jpg", b"fake"), path)
        keeper.record.path.unlink()  # so that a wrongly accepted entry would be moved back

    bin_ = RecycleBinBackend(send=send, volume=fake.info, delay=0)
    with pytest.raises(ActionError, match="put back"):
        remove(bin_, vlib, item, keeper)
    assert item.record.path.read_bytes() == b"mine"
    assert item.record.path.stat().st_mtime_ns == item.record.signature.mtime_ns


def test_keeper_lost_while_sending(fake: FakeBin, vlib: Library) -> None:
    item, keeper = vlib.add("a.jpg", b"mine"), vlib.add("keep.jpg", b"mine")

    def send(path: str) -> None:
        fake.send(path)
        keeper.record.path.unlink()

    bin_ = RecycleBinBackend(send=send, volume=fake.info, delay=0)
    assert code_of(lambda: remove(bin_, vlib, item, keeper)) == "keeper_missing"
    assert item.record.path.read_bytes() == b"mine"
    assert item.record.path.stat().st_nlink == 1
    assert fake.contents() == [] and leftovers(vlib) == []


def test_name_taken_after_a_lost_send(fake: FakeBin, vlib: Library) -> None:
    item, keeper = vlib.add("a.jpg", b"mine"), vlib.add("keep.jpg", b"mine")

    def send(path: str) -> None:
        os.unlink(path)
        write(Path(path), b"newcomer")

    bin_ = RecycleBinBackend(send=send, volume=fake.info, delay=0)
    planned = bin_.plan_destination(item.record.path, vlib.root, "s1")
    with pytest.raises(ActionError) as caught:
        bin_.remove(checked_request(planned, item, keeper, vlib.root))
    assert str(planned.dst) in str(caught.value)
    assert planned.dst.read_bytes() == b"mine"
    assert item.record.path.read_bytes() == b"newcomer"


@pytest.mark.parametrize(
    ("volume", "code"),
    [
        (Volume(Path("X:\\"), fixed=False, hard_links=True), "unsupported"),
        (Volume(Path("X:\\"), fixed=True, hard_links=False), "unsupported"),
    ],
)
def test_unsuitable_drives(fake: FakeBin, vlib: Library, volume: Volume, code: str) -> None:
    item, keeper = vlib.add("a.jpg"), vlib.add("keep.jpg")
    bin_ = RecycleBinBackend(send=fake.send, volume=lambda path: volume)
    assert code_of(lambda: remove(bin_, vlib, item, keeper)) == code
    assert fake.sent == [] and item.record.path.exists()


def test_no_hard_link_after_all(
    fake: FakeBin, vlib: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    item, keeper = vlib.add("a.jpg"), vlib.add("keep.jpg")

    def no_link(*args: object, **kwargs: object) -> None:
        raise OSError(1, "not supported")

    import mediasift.actions.backends.recycle_bin as module

    monkeypatch.setattr(module.os, "link", no_link)
    assert code_of(lambda: remove(backend(fake), vlib, item, keeper)) == "unsupported"
    assert fake.sent == [] and item.record.path.exists()


def test_names_the_shell_would_misread(fake: FakeBin, vlib: Library) -> None:
    if sys.platform != "win32":
        pytest.skip("such names behave differently only on Windows")
    from mediasift.core import paths

    path = vlib.folder / "photo."
    Path(paths.fs_path(path)).write_bytes(b"photo")
    assert code_of(lambda: backend(fake).check(path, vlib.root)) == "unsupported"


# --- restoring ---


def test_restore_when_bin_entry_changed(fake: FakeBin, vlib: Library) -> None:
    bin_ = backend(fake)
    item, keeper = vlib.add("a.jpg", b"mine"), vlib.add("keep.jpg", b"mine")
    stored = remove(bin_, vlib, item, keeper)
    stored.dst.write_bytes(b"changed!")
    request = RestoreRequest(
        stored, item.record.path, vlib.root, item.record.signature, item.full_hash
    )
    assert code_of(lambda: bin_.restore(request)) == "stored_changed"
    stored.dst.unlink()
    assert code_of(lambda: bin_.restore(request)) == "stored_missing"


def test_restore_refuses_files_outside_the_bin(
    fake: FakeBin, vlib: Library, tmp_path: Path
) -> None:
    item = vlib.add("a.jpg")
    stray = write(tmp_path / "volume" / "$R123.jpg")
    request = RestoreRequest(StoredFile(stray), item.record.path, vlib.root, item.record.signature)
    assert code_of(lambda: backend(fake).restore(request)) == "outside_root"


def test_restore_beside_a_newcomer(fake: FakeBin, vlib: Library) -> None:
    bin_ = backend(fake)
    item, keeper = vlib.add("a.jpg", b"mine"), vlib.add("keep.jpg", b"mine")
    stored = remove(bin_, vlib, item, keeper)
    write(item.record.path, b"newcomer")
    request = RestoreRequest(
        stored, item.record.path, vlib.root, item.record.signature, item.full_hash
    )
    restored = bin_.restore(request)
    assert restored.renamed and restored.path.name == "a (restored 1).jpg"
    assert restored.path.read_bytes() == b"mine"


# --- recovery ---


def test_locate_all_states(fake: FakeBin, vlib: Library) -> None:
    bin_ = backend(fake)
    item = vlib.add("a.jpg")
    planned = bin_.plan_destination(item.record.path, vlib.root, "s1")
    expected = item.record.signature
    assert bin_.locate(planned, expected).location is Location.SOURCE
    planned.dst.parent.mkdir(parents=True)
    os.link(item.record.path, planned.dst)  # crashed after the guard was made
    assert bin_.locate(planned, expected).location is Location.BOTH
    assert bin_.tidy(planned, expected)
    assert bin_.locate(planned, expected).location is Location.SOURCE
    assert not planned.dst.exists()

    os.link(item.record.path, planned.dst)
    fake.send(str(item.record.path))  # crashed before the guard was released
    found = bin_.locate(planned, expected)
    assert found.location is Location.DESTINATION
    assert found.stored is not None and found.stored.dst == fake.contents()[0]
    assert bin_.tidy(planned, expected)
    assert not planned.dst.exists() and fake.contents()[0].exists()

    fake.contents()[0].unlink()
    assert bin_.locate(planned, expected).location is Location.NEITHER


def test_locate_finds_the_guard_alone(fake: FakeBin, vlib: Library) -> None:
    bin_ = backend(fake)
    item = vlib.add("a.jpg")
    planned = bin_.plan_destination(item.record.path, vlib.root, "s1")
    planned.dst.parent.mkdir(parents=True)
    os.link(item.record.path, planned.dst)
    item.record.path.unlink()  # the bin deleted it for good, then the crash
    found = bin_.locate(planned, item.record.signature)
    assert found.location is Location.DESTINATION
    assert found.stored is not None and found.stored.dst == planned.dst
    request = RestoreRequest(
        found.stored, item.record.path, vlib.root, item.record.signature, item.full_hash
    )
    assert bin_.restore(request).path == item.record.path
    assert item.record.path.read_bytes() == b"photo"


@pytest.mark.realbin
@pytest.mark.skipif(sys.platform != "win32", reason="the Recycle Bin is Windows'")
def test_real_recycle_bin(tmp_path: Path) -> None:
    """Sends one temporary file to the real Recycle Bin and restores it."""
    lib = make_library(tmp_path / "photos")
    item, keeper = lib.add("mediasift-test.jpg", b"temporary"), lib.add("keep.jpg", b"temporary")
    bin_ = RecycleBinBackend()
    stored = remove(bin_, lib, item, keeper)
    try:
        assert not item.record.path.exists()
        assert stored.dst.name.startswith("$R")
    finally:
        request = RestoreRequest(
            stored, item.record.path, lib.root, item.record.signature, item.full_hash
        )
        restored = bin_.restore(request)
    assert restored.path.read_bytes() == b"temporary"
