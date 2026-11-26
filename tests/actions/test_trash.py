"""The freedesktop.org Trash, with the home Trash and volume top folders in tmp_path."""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pytest

from mediasift.actions import fsops
from mediasift.actions.backends import trash as trash_module
from mediasift.actions.backends.base import Location, RestoreRequest, StoredFile
from mediasift.actions.backends.trash import TrashBackend, info_text, recorded_value
from mediasift.actions.checks import KnownFile
from mediasift.core.errors import SafetyError
from tests.actions.conftest import (
    ARABIC,
    Library,
    checked_request,
    make_library,
    on_other_device,
    write,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the Trash is Linux'")

WHEN = datetime(2026, 11, 26, 8, 30, 5)


def code_of(call: Callable[[], object]) -> str:
    with pytest.raises(SafetyError) as caught:
        call()
    return caught.value.code


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home" / ".local" / "share" / "Trash"


@pytest.fixture
def tlib(tmp_path: Path) -> Library:
    return make_library(tmp_path / "volume" / "photos")


def home_backend(home: Path, **more: object) -> TrashBackend:
    return TrashBackend(home_trash=home, now=lambda: WHEN, **more)  # type: ignore[arg-type]


@pytest.fixture
def volume_trash(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., TrashBackend]:
    """Backends that see the home Trash on another volume than the photos."""
    real = os.stat
    home_root = tmp_path / "home"
    home_root.mkdir(exist_ok=True)

    def fake_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        st = real(path, *args, **kwargs)  # type: ignore[arg-type]
        return on_other_device(st) if Path(os.fsdecode(path)) == home_root else st  # type: ignore[arg-type]

    monkeypatch.setattr(trash_module.os, "stat", fake_stat)

    def make(**more: object) -> TrashBackend:
        return home_backend(home, topdir=lambda path: tmp_path / "volume", **more)

    return make


def remove(backend: TrashBackend, lib: Library, item: KnownFile, keeper: KnownFile) -> StoredFile:
    planned = backend.plan_destination(item.record.path, lib.root, "s1")
    return backend.remove(checked_request(planned, item, keeper, lib.root))


def restore_request(lib: Library, item: KnownFile, stored: StoredFile) -> RestoreRequest:
    return RestoreRequest(stored, item.record.path, lib.root, item.record.signature, item.full_hash)


def test_info_format() -> None:
    text = info_text("/a/b%20c.jpg", WHEN)
    assert text == "[Trash Info]\nPath=/a/b%20c.jpg\nDeletionDate=2026-11-26T08:30:05\n"
    assert recorded_value(text) == "/a/b%20c.jpg"
    assert recorded_value("Path=/x\n") is None
    assert recorded_value("[Trash Info]\nDeletionDate=x\n") is None


def test_home_trash_round_trip(home: Path, tlib: Library) -> None:
    backend = home_backend(home)
    item, keeper = tlib.add(f"2024/{ARABIC}", b"mine"), tlib.add("keep.jpg", b"mine")
    stored = remove(backend, tlib, item, keeper)
    assert stored.dst == home / "files" / ARABIC
    info = home / "info" / (ARABIC + ".trashinfo")
    expected_path = quote(os.fsencode(item.record.path), safe="/")
    assert info.read_text(encoding="utf-8") == info_text(expected_path, WHEN)
    assert "%D8%B5" in expected_path
    assert stat.S_IMODE(home.stat().st_mode) & 0o077 == 0
    restored = backend.restore(restore_request(tlib, item, stored))
    assert restored.path == item.record.path and not restored.renamed
    assert item.record.path.read_bytes() == b"mine"
    assert os.listdir(home / "files") == [] and os.listdir(home / "info") == []


def test_names_are_unique_in_files_and_info(home: Path, tlib: Library) -> None:
    backend = home_backend(home)
    write(home / "files" / "a.jpg", b"older")
    write(home / "info" / "a (2).jpg.trashinfo", b"orphan")
    item, keeper = tlib.add("a.jpg"), tlib.add("keep.jpg")
    assert remove(backend, tlib, item, keeper).dst.name == "a (3).jpg"
    assert (home / "files" / "a.jpg").read_bytes() == b"older"


def test_long_names_leave_room_for_the_suffix(home: Path, tlib: Library) -> None:
    name = "ص" * 125 + ".jpg"  # 254 bytes
    item, keeper = tlib.add(name), tlib.add("keep.jpg")
    stored = remove(home_backend(home), tlib, item, keeper)
    assert len(os.fsencode(stored.dst.name + ".trashinfo")) <= 255


def test_volume_trash_of_the_user(
    volume_trash: Callable[..., TrashBackend], tmp_path: Path, tlib: Library
) -> None:
    item, keeper = tlib.add("a b.jpg"), tlib.add("keep.jpg")
    stored = remove(volume_trash(), tlib, item, keeper)
    folder = tmp_path / "volume" / f".Trash-{os.getuid()}"
    assert stored.dst == folder / "files" / "a b.jpg"
    text = (folder / "info" / "a b.jpg.trashinfo").read_text(encoding="utf-8")
    assert recorded_value(text) == "photos/a%20b.jpg"


def test_shared_volume_trash(
    volume_trash: Callable[..., TrashBackend], tmp_path: Path, tlib: Library
) -> None:
    shared = tmp_path / "volume" / ".Trash"
    shared.mkdir()
    os.chmod(shared, 0o1777)
    item, keeper = tlib.add("a.jpg"), tlib.add("keep.jpg")
    stored = remove(volume_trash(), tlib, item, keeper)
    assert stored.dst == shared / str(os.getuid()) / "files" / "a.jpg"


@pytest.mark.parametrize("kind", ["not sticky", "symlink"])
def test_unsafe_shared_trash_is_skipped(
    volume_trash: Callable[..., TrashBackend], tmp_path: Path, tlib: Library, kind: str
) -> None:
    shared = tmp_path / "volume" / ".Trash"
    if kind == "symlink":
        (tmp_path / "elsewhere").mkdir()
        os.chmod(tmp_path / "elsewhere", 0o1777)
        shared.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    else:
        shared.mkdir()
    item, keeper = tlib.add("a.jpg"), tlib.add("keep.jpg")
    stored = remove(volume_trash(), tlib, item, keeper)
    assert stored.dst.parent.parent.name == f".Trash-{os.getuid()}"
    assert not (tmp_path / "elsewhere" / str(os.getuid())).exists()


def test_trash_of_someone_else(volume_trash: Callable[..., TrashBackend], tlib: Library) -> None:
    item, keeper = tlib.add("a.jpg"), tlib.add("keep.jpg")
    backend = volume_trash(uid=os.getuid() + 12345)
    assert code_of(lambda: remove(backend, tlib, item, keeper)) == "unsupported"
    assert item.record.path.exists()


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="Unix rights")
def test_unwritable_trash(home: Path, tlib: Library) -> None:
    (home / "files").mkdir(parents=True)
    (home / "info").mkdir()
    os.chmod(home / "info", 0o500)
    item, keeper = tlib.add("a.jpg"), tlib.add("keep.jpg")
    try:
        assert code_of(lambda: remove(home_backend(home), tlib, item, keeper)) == "unsupported"
    finally:
        os.chmod(home / "info", 0o700)
    assert item.record.path.exists()


def test_files_in_the_trash_are_refused(home: Path, tmp_path: Path) -> None:
    lib = make_library(tmp_path / "home")
    item = lib.add(".local/share/Trash/files/a.jpg")
    assert code_of(lambda: home_backend(home).check(item.record.path, lib.root)) == "in_quarantine"


def test_record_taken_after_planning(home: Path, tlib: Library) -> None:
    backend = home_backend(home)
    item, keeper = tlib.add("a.jpg"), tlib.add("keep.jpg")
    planned = backend.plan_destination(item.record.path, tlib.root, "s1")
    write(Path(planned.detail["info"]), b"someone else's")
    request = checked_request(planned, item, keeper, tlib.root)
    assert code_of(lambda: backend.remove(request)) == "target_exists"
    assert item.record.path.exists()
    assert Path(planned.detail["info"]).read_bytes() == b"someone else's"


def test_failed_move_takes_its_record_back(home: Path, tlib: Library) -> None:
    backend = home_backend(home)
    item, keeper = tlib.add("a.jpg"), tlib.add("keep.jpg")
    planned = backend.plan_destination(item.record.path, tlib.root, "s1")
    write(planned.dst, b"intruder")
    request = checked_request(planned, item, keeper, tlib.root)
    assert code_of(lambda: backend.remove(request)) == "target_exists"
    assert item.record.path.exists()
    assert planned.dst.read_bytes() == b"intruder"
    assert os.listdir(home / "info") == []


def test_keeper_lost_during_the_move(
    home: Path, tlib: Library, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = home_backend(home)
    item, keeper = tlib.add("a.jpg"), tlib.add("keep.jpg")
    request = checked_request(
        backend.plan_destination(item.record.path, tlib.root, "s1"), item, keeper, tlib.root
    )
    real = fsops.move_no_replace

    def move(src: Path, dst: Path, expected: object) -> object:
        moved = real(src, dst, expected)  # type: ignore[arg-type]
        keeper.record.path.unlink()
        return moved

    monkeypatch.setattr(fsops, "move_no_replace", move)
    assert code_of(lambda: backend.remove(request)) == "keeper_missing"
    assert item.record.path.exists()
    assert os.listdir(home / "files") == [] and os.listdir(home / "info") == []


def test_restore_beside_a_newcomer(home: Path, tlib: Library) -> None:
    backend = home_backend(home)
    item, keeper = tlib.add("a.jpg", b"mine"), tlib.add("keep.jpg", b"mine")
    stored = remove(backend, tlib, item, keeper)
    write(item.record.path, b"new")
    restored = backend.restore(restore_request(tlib, item, stored))
    assert restored.renamed and restored.path.name == "a (restored 1).jpg"
    assert item.record.path.read_bytes() == b"new"
    assert os.listdir(home / "info") == []


def test_restore_of_changed_or_missing(home: Path, tlib: Library) -> None:
    backend = home_backend(home)
    item, keeper = tlib.add("a.jpg", b"mine"), tlib.add("keep.jpg", b"mine")
    stored = remove(backend, tlib, item, keeper)
    stored.dst.write_bytes(b"MINE")
    assert code_of(lambda: backend.restore(restore_request(tlib, item, stored))) == "stored_changed"
    stored.dst.unlink()
    assert code_of(lambda: backend.restore(restore_request(tlib, item, stored))) == "stored_missing"
    assert os.listdir(home / "info") == ["a.jpg.trashinfo"]


def test_restore_refuses_other_folders(home: Path, tlib: Library, tmp_path: Path) -> None:
    item = tlib.add("a.jpg")
    stray = write(tmp_path / "files" / "a.jpg")
    request = RestoreRequest(StoredFile(stray), item.record.path, tlib.root, item.record.signature)
    assert code_of(lambda: home_backend(home).restore(request)) == "outside_root"


def test_locate_all_states(home: Path, tlib: Library) -> None:
    backend = home_backend(home)
    item, keeper = tlib.add("a.jpg"), tlib.add("keep.jpg")
    planned = backend.plan_destination(item.record.path, tlib.root, "s1")
    expected = item.record.signature
    assert backend.locate(planned, expected).location is Location.SOURCE
    (home / "files").mkdir(parents=True)
    (home / "info").mkdir()
    os.link(item.record.path, planned.dst)
    write(Path(planned.detail["info"]), info_text(planned.detail["trash_path"], WHEN).encode())
    assert backend.locate(planned, expected).location is Location.BOTH
    assert backend.tidy(planned, expected)
    assert backend.locate(planned, expected).location is Location.SOURCE
    assert os.listdir(home / "files") == [] and os.listdir(home / "info") == []
    backend.remove(checked_request(planned, item, keeper, tlib.root))
    assert backend.locate(planned, expected).location is Location.DESTINATION
    planned.dst.unlink()
    assert backend.locate(planned, expected).location is Location.NEITHER
    assert not backend.tidy(planned, expected)


def test_tidy_keeps_records_of_other_files(home: Path, tlib: Library) -> None:
    backend = home_backend(home)
    item = tlib.add("a.jpg")
    planned = backend.plan_destination(item.record.path, tlib.root, "s1")
    write(Path(planned.detail["info"]), info_text("/somewhere/else.jpg", WHEN).encode())
    assert backend.tidy(planned, item.record.signature)
    assert Path(planned.detail["info"]).exists()
