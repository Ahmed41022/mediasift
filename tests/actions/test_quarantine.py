from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from mediasift.actions import fsops
from mediasift.actions.backends import BACKENDS, for_setting
from mediasift.actions.backends.base import (
    Location,
    PlannedRemoval,
    RestoreRequest,
    StoredFile,
)
from mediasift.actions.backends.quarantine import QuarantineBackend
from mediasift.actions.checks import KnownFile
from mediasift.core import paths
from mediasift.core.errors import SafetyError
from mediasift.core.paths import QUARANTINE_FOLDER
from tests.actions.conftest import (
    ARABIC,
    WINDOWS,
    Library,
    checked_request,
    make_folder_link,
    on_other_device,
    write,
)

Q = QuarantineBackend()


def code_of(call: Callable[[], object]) -> str:
    with pytest.raises(SafetyError) as caught:
        call()
    return caught.value.code


def remove(lib: Library, item: KnownFile, keeper: KnownFile, session: str = "s1") -> StoredFile:
    planned = Q.plan_destination(item.record.path, lib.root, session)
    return Q.remove(checked_request(planned, item, keeper, lib.root))


def restore_request(
    lib: Library, item: KnownFile, stored: StoredFile, **more: object
) -> RestoreRequest:
    return RestoreRequest(
        stored,
        item.record.path,
        lib.root,
        item.record.signature,
        item.full_hash,
        **more,  # type: ignore[arg-type]
    )


def test_round_trip(lib: Library) -> None:
    item, keeper = lib.add(f"2024/{ARABIC}/{ARABIC}", b"family"), lib.add("keep.jpg", b"family")
    mtime = item.record.signature.mtime_ns
    stored = remove(lib, item, keeper)
    assert stored.dst == lib.folder / QUARANTINE_FOLDER / "s1" / "2024" / ARABIC / ARABIC
    assert not item.record.path.exists()
    assert stored.dst.read_bytes() == b"family"
    restored = Q.restore(restore_request(lib, item, stored))
    assert restored.path == item.record.path and not restored.renamed
    assert item.record.path.read_bytes() == b"family"
    assert item.record.path.stat().st_mtime_ns == mtime


@pytest.mark.skipif(not WINDOWS, reason="the hidden attribute is Windows'")
def test_quarantine_is_hidden_on_windows(lib: Library) -> None:
    remove(lib, lib.add("a.jpg"), lib.add("keep.jpg"))
    assert (lib.folder / QUARANTINE_FOLDER).stat().st_file_attributes & 0x2


def test_name_collisions(lib: Library) -> None:
    keeper = lib.add("keep.jpg")
    keeper = lib.add("keep2.jpg", b"one")
    first = remove(lib, lib.add("sub/a.jpg", b"one"), keeper)
    second = remove(lib, lib.add("sub/a.jpg", b"one"), keeper)
    assert first.dst.name == "a.jpg"
    assert second.dst.name == "a (2).jpg"
    assert first.dst.read_bytes() == second.dst.read_bytes() == b"one"


def test_case_collision(lib: Library) -> None:
    keeper = lib.add("keep.jpg")
    first = remove(lib, lib.add("A.JPG"), keeper)
    insensitive = (first.dst.parent / "a.jpg").exists()
    second = remove(lib, lib.add("a.jpg"), keeper)
    assert second.dst.name == ("a (2).jpg" if insensitive else "a.jpg")
    assert first.dst.exists() and second.dst.exists()


def test_folder_name_taken_by_a_file(lib: Library) -> None:
    keeper = lib.add("keep.jpg")
    write(lib.folder / QUARANTINE_FOLDER / "s1" / "sub", b"a file")
    stored = remove(lib, lib.add("sub/a.jpg"), keeper)
    assert stored.dst == lib.folder / QUARANTINE_FOLDER / "s1" / "sub (2)" / "a.jpg"


def test_long_names(lib: Library) -> None:
    keeper = lib.add("keep.jpg", b"long")
    rel = "/".join(["ص" * 60] * 4) + "/" + "ب" * 100 + ".jpg"
    folder = lib.folder
    os.makedirs(paths.fs_path(folder / os.path.dirname(rel)))
    item = lib.add(rel, b"long") if not WINDOWS else _add_long(lib, rel)
    stored = remove(lib, item, keeper)
    assert Path(paths.fs_path(stored.dst)).read_bytes() == b"long"
    Q.restore(restore_request(lib, item, stored))
    assert Path(paths.fs_path(item.record.path)).read_bytes() == b"long"


def _add_long(lib: Library, rel: str) -> KnownFile:
    Path(paths.fs_path(lib.folder / rel)).write_bytes(b"long")
    known = lib.add("placeholder.jpg", b"long")
    path = lib.folder / rel
    return KnownFile(
        replace(known.record, path=path, rel_path=Path(rel), signature=paths.signature(path)),
        known.full_hash,
    )


def test_target_taken_after_planning(lib: Library) -> None:
    item, keeper = lib.add("a.jpg"), lib.add("keep.jpg")
    planned = Q.plan_destination(item.record.path, lib.root, "s1")
    write(planned.dst, b"someone else")
    request = checked_request(planned, item, keeper, lib.root)
    assert code_of(lambda: Q.remove(request)) == "target_exists"
    assert item.record.path.exists() and planned.dst.read_bytes() == b"someone else"


def test_keeper_lost_during_the_move(lib: Library, monkeypatch: pytest.MonkeyPatch) -> None:
    item, keeper = lib.add("a.jpg"), lib.add("keep.jpg")
    request = checked_request(
        Q.plan_destination(item.record.path, lib.root, "s1"), item, keeper, lib.root
    )
    real = fsops.move_no_replace

    def move(src: Path, dst: Path, expected: object) -> object:
        moved = real(src, dst, expected)  # type: ignore[arg-type]
        keeper.record.path.unlink()
        return moved

    monkeypatch.setattr(fsops, "move_no_replace", move)
    assert code_of(lambda: Q.remove(request)) == "keeper_missing"
    assert item.record.path.read_bytes() == b"photo"
    assert not request.planned.dst.exists()


def test_quarantine_through_a_link_is_refused(lib: Library, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    make_folder_link(lib.folder / QUARANTINE_FOLDER, elsewhere)
    item, keeper = lib.add("a.jpg"), lib.add("keep.jpg")
    planned = Q.plan_destination(item.record.path, lib.root, "s1")
    assert code_of(lambda: Q.remove(checked_request(planned, item, keeper, lib.root))) == "is_link"
    assert item.record.path.exists() and not os.listdir(elsewhere)


def test_nested_volume_is_refused(lib: Library, monkeypatch: pytest.MonkeyPatch) -> None:
    item = lib.add("mnt/a.jpg")
    real = fsops.stat_or_none

    def stat(path: Path) -> os.stat_result | None:
        st = real(path)
        if st is not None and path == item.record.path:
            return on_other_device(st)
        return st

    monkeypatch.setattr(fsops, "stat_or_none", stat)
    assert code_of(lambda: Q.check(item.record.path, lib.root)) == "cross_volume"


def test_session_names(lib: Library) -> None:
    item = lib.add("a.jpg")
    for bad in ("", "..", ".bin-records", "a/b", "a\\b", "x" * 65):
        with pytest.raises(SafetyError, match="session"):
            Q.plan_destination(item.record.path, lib.root, bad)


# --- restoring ---


def test_restore_when_the_name_is_taken(lib: Library) -> None:
    item, keeper = lib.add("a.jpg", b"mine"), lib.add("keep.jpg", b"mine")
    stored = remove(lib, item, keeper)
    write(item.record.path, b"new")
    write(item.record.path.with_name("a (restored 1).jpg"), b"newer")
    restored = Q.restore(restore_request(lib, item, stored))
    assert restored.renamed
    assert restored.path.name == "a (restored 2).jpg"
    assert restored.path.read_bytes() == b"mine"
    assert item.record.path.read_bytes() == b"new"


def test_restore_when_the_name_differs_by_case(lib: Library) -> None:
    item, keeper = lib.add("a.jpg", b"mine"), lib.add("keep.jpg", b"mine")
    stored = remove(lib, item, keeper)
    write(lib.folder / "A.JPG", b"other")
    insensitive = (lib.folder / "a.jpg").exists()
    restored = Q.restore(restore_request(lib, item, stored))
    assert restored.renamed is insensitive
    assert restored.path.read_bytes() == b"mine"
    assert (lib.folder / "A.JPG").read_bytes() == b"other"


def test_restore_recreates_folders(lib: Library) -> None:
    item, keeper = lib.add("x/y/a.jpg"), lib.add("keep.jpg")
    stored = remove(lib, item, keeper)
    os.rmdir(lib.folder / "x" / "y")
    os.rmdir(lib.folder / "x")
    assert Q.restore(restore_request(lib, item, stored)).path == item.record.path
    assert item.record.path.read_bytes() == b"photo"


def test_restore_never_goes_through_links(lib: Library, tmp_path: Path) -> None:
    item, keeper = lib.add("x/a.jpg"), lib.add("keep.jpg")
    stored = remove(lib, item, keeper)
    os.rmdir(lib.folder / "x")
    (tmp_path / "elsewhere").mkdir()
    make_folder_link(lib.folder / "x", tmp_path / "elsewhere")
    assert code_of(lambda: Q.restore(restore_request(lib, item, stored))) in (
        "is_link",
        "outside_root",
    )
    assert stored.dst.exists() and not os.listdir(tmp_path / "elsewhere")


def test_restore_of_a_changed_file(lib: Library) -> None:
    item, keeper = lib.add("a.jpg", b"aaaa"), lib.add("keep.jpg", b"aaaa")
    stored = remove(lib, item, keeper)
    st = stored.dst.stat()
    stored.dst.write_bytes(b"bbbb")
    os.utime(stored.dst, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert code_of(lambda: Q.restore(restore_request(lib, item, stored))) == "stored_changed"
    no_hash = replace(restore_request(lib, item, stored), full_hash=None)
    assert Q.restore(no_hash).path == item.record.path  # same file, same size and time
    item2 = lib.add("b.jpg", b"aaaa")
    stored2 = remove(lib, item2, keeper)
    stored2.dst.write_bytes(b"longer content")
    no_hash2 = replace(restore_request(lib, item2, stored2), full_hash=None)
    assert code_of(lambda: Q.restore(no_hash2)) == "stored_changed"
    assert not item2.record.path.exists()


def test_restore_of_a_missing_file(lib: Library) -> None:
    item, keeper = lib.add("a.jpg"), lib.add("keep.jpg")
    stored = remove(lib, item, keeper)
    stored.dst.unlink()
    assert code_of(lambda: Q.restore(restore_request(lib, item, stored))) == "stored_missing"


def test_restore_refuses_places_outside(lib: Library, tmp_path: Path) -> None:
    item, keeper = lib.add("a.jpg"), lib.add("keep.jpg")
    stored = remove(lib, item, keeper)
    request = restore_request(lib, item, stored)
    outside = replace(request, original=tmp_path / "a.jpg")
    assert code_of(lambda: Q.restore(outside)) == "outside_root"
    stray = replace(request, stored=StoredFile(write(tmp_path / "b.jpg")))
    assert code_of(lambda: Q.restore(stray)) == "outside_root"
    read_only = replace(request, root=replace(lib.root, read_only=True))
    assert code_of(lambda: Q.restore(read_only)) == "root_read_only"
    assert stored.dst.exists()


# --- recovery ---


def test_locate_all_states(lib: Library) -> None:
    item, keeper = lib.add("a.jpg"), lib.add("keep.jpg")
    planned = Q.plan_destination(item.record.path, lib.root, "s1")
    expected = item.record.signature
    assert Q.locate(planned, expected).location is Location.SOURCE
    planned.dst.parent.mkdir(parents=True)
    os.link(item.record.path, planned.dst)
    assert Q.locate(planned, expected).location is Location.BOTH
    assert Q.tidy(planned, expected)
    assert Q.locate(planned, expected).location is Location.SOURCE
    assert item.record.path.exists() and not planned.dst.exists()
    Q.remove(checked_request(planned, item, keeper, lib.root))
    found = Q.locate(planned, expected)
    assert found.location is Location.DESTINATION
    assert found.stored is not None and found.stored.dst == planned.dst
    planned.dst.unlink()
    assert Q.locate(planned, expected).location is Location.NEITHER
    assert not Q.tidy(planned, expected)


def test_locate_ignores_a_stranger(lib: Library) -> None:
    item = lib.add("a.jpg")
    planned = PlannedRemoval(item.record.path, lib.folder / "q.jpg", lib.root, "s1")
    item.record.path.unlink()
    write(item.record.path, b"a stranger")  # file systems reuse the number of a deleted file
    assert Q.locate(planned, item.record.signature).location is Location.NEITHER


def test_tidy_keeps_two_different_files(lib: Library) -> None:
    item = lib.add("a.jpg")
    planned = Q.plan_destination(item.record.path, lib.root, "s1")
    write(planned.dst)
    assert Q.tidy(planned, item.record.signature)
    assert planned.dst.exists() and item.record.path.exists()


def test_registry() -> None:
    assert BACKENDS.get("quarantine").name == "quarantine"
    assert for_setting("quarantine").name == "quarantine"
    assert for_setting("trash").name == ("recycle_bin" if WINDOWS else "trash")
