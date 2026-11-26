from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from mediasift.actions import checks, fsops
from mediasift.actions.checks import KnownFile
from mediasift.core import paths
from mediasift.core.errors import SafetyError
from mediasift.core.paths import QUARANTINE_FOLDER
from mediasift.core.types import FileSignature, MatchKind
from tests.actions.conftest import (
    ARABIC,
    POSIX,
    Library,
    make_folder_link,
    make_library,
    make_symlink,
    set_read_only,
    verified,
    write,
)


def code_of(call: Callable[[], object]) -> str:
    with pytest.raises(SafetyError) as caught:
        call()
    return caught.value.code


def check(lib: Library, item: KnownFile, *keepers: KnownFile, **options: object) -> str:
    return code_of(lambda: checks.check_item(item, keepers, [lib.root], **options))  # type: ignore[arg-type]


def keep_mtime(path: Path, data: bytes) -> None:
    """Change the content behind the file's back: same size, same modified time."""
    st = path.stat()
    path.write_bytes(data)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def test_passes(lib: Library) -> None:
    item, keeper = lib.add("a/" + ARABIC), lib.add("b/keep.jpg")
    result = checks.check_item(item, [keeper], [lib.root])
    assert result.file.path == item.record.path
    assert result.keeper.path == keeper.record.path
    assert result.file.full_hash == item.full_hash
    assert result.root == lib.root


def test_one_intact_keeper_is_enough(lib: Library) -> None:
    item, gone, keeper = lib.add("a.jpg"), lib.add("gone.jpg"), lib.add("keep.jpg")
    gone.record.path.unlink()
    assert checks.check_item(item, [gone, keeper], [lib.root]).keeper.path == keeper.record.path


@pytest.mark.parametrize(
    ("change", "code"),
    [({"enabled": False}, "root_disabled"), ({"read_only": True}, "root_read_only")],
)
def test_root_state(lib: Library, change: dict[str, bool], code: str) -> None:
    item, keeper = lib.add("a.jpg"), lib.add("keep.jpg")
    other = lib.with_root(**change)
    assert check(other, item, keeper) == code


def test_outside_root(lib: Library, tmp_path: Path) -> None:
    item, keeper = lib.add("a.jpg"), lib.add("keep.jpg")
    assert code_of(lambda: checks.check_item(item, [keeper], [])) == "outside_root"
    outside = write(tmp_path / "outside.jpg")
    record = replace(item.record, path=outside, signature=paths.signature(outside))
    assert check(lib, KnownFile(record, item.full_hash), keeper) == "outside_root"
    sneaky = replace(item.record, path=lib.folder / ".." / "outside.jpg")
    assert check(lib, KnownFile(sneaky, item.full_hash), keeper) == "outside_root"


def test_folder_link_out_of_the_root(lib: Library, tmp_path: Path) -> None:
    keeper = lib.add("keep.jpg")
    outside = write(tmp_path / "outside" / "a.jpg")
    make_folder_link(lib.folder / "link", tmp_path / "outside")
    item = lib.add("x.jpg")
    record = replace(item.record, path=lib.folder / "link" / "a.jpg")
    record = replace(record, signature=paths.signature(outside))
    assert check(lib, KnownFile(record, item.full_hash), keeper) in ("outside_root", "is_link")
    assert outside.exists()


def test_folder_link_inside_the_root(lib: Library) -> None:
    keeper, item = lib.add("keep.jpg"), lib.add("real/a.jpg")
    make_folder_link(lib.folder / "link", lib.folder / "real")
    record = replace(item.record, path=lib.folder / "link" / "a.jpg")
    assert check(lib, KnownFile(record, item.full_hash), keeper) == "is_link"


def test_file_link(lib: Library) -> None:
    keeper, item = lib.add("keep.jpg"), lib.add("a.jpg")
    link = lib.folder / "link.jpg"
    make_symlink(link, item.record.path)
    record = replace(item.record, path=link, signature=paths.signature(link))
    assert check(lib, KnownFile(record, item.full_hash), keeper) == "is_link"


def test_read_only_file(lib: Library) -> None:
    item, keeper = lib.add("a.jpg"), lib.add("keep.jpg")
    set_read_only(item.record.path)
    try:
        item = KnownFile(
            replace(item.record, signature=paths.signature(item.record.path)), item.full_hash
        )
        assert check(lib, item, keeper) == "file_read_only"
    finally:
        set_read_only(item.record.path, False)


@pytest.mark.skipif(not POSIX or os.geteuid() == 0, reason="folder rights of Unix")
def test_read_only_folder(lib: Library) -> None:
    item, keeper = lib.add("ro/a.jpg"), lib.add("keep.jpg")
    os.chmod(item.record.path.parent, 0o555)
    try:
        assert check(lib, item, keeper) == "file_read_only"
    finally:
        os.chmod(item.record.path.parent, 0o755)


def test_placeholder(lib: Library, monkeypatch: pytest.MonkeyPatch) -> None:
    item, keeper = lib.add("a.jpg"), lib.add("keep.jpg")
    monkeypatch.setattr(paths, "is_placeholder", lambda st: True)
    assert check(lib, item, keeper) == "is_placeholder"


def test_missing_and_not_regular(lib: Library) -> None:
    item, keeper = lib.add("a.jpg"), lib.add("keep.jpg")
    item.record.path.unlink()
    assert check(lib, item, keeper) == "missing"
    item.record.path.mkdir()
    assert check(lib, item, keeper) == "not_regular"


def test_changed_size_and_time(lib: Library) -> None:
    item, keeper = lib.add("a.jpg", b"12345"), lib.add("keep.jpg", b"12345")
    st = item.record.path.stat()
    os.utime(item.record.path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert check(lib, item, keeper) == "changed"
    item2 = lib.add("b.jpg", b"12345")
    item2.record.path.write_bytes(b"123456")
    assert check(lib, item2, keeper) == "changed"


def test_replaced_by_an_identical_file(lib: Library) -> None:
    item, keeper = lib.add("a.jpg"), lib.add("keep.jpg")
    path = item.record.path
    st = path.stat()
    path.rename(path.with_name("old.jpg"))
    write(path, b"photo")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert check(lib, item, keeper) == "changed"


def test_content_changed_with_the_time_kept(lib: Library) -> None:
    item, keeper = lib.add("a.jpg", b"aaaa"), lib.add("keep.jpg", b"aaaa")
    keep_mtime(item.record.path, b"bbbb")
    assert check(lib, item, keeper) == "content_changed"


def test_in_quarantine(lib: Library) -> None:
    keeper = lib.add("keep.jpg")
    item = lib.add(f"{QUARANTINE_FOLDER}/s1/a.jpg")
    assert check(lib, item, keeper) == "in_quarantine"


def test_no_identity(lib: Library, monkeypatch: pytest.MonkeyPatch) -> None:
    item, keeper = lib.add("a.jpg"), lib.add("keep.jpg")
    real = paths.signature
    monkeypatch.setattr(paths, "signature", lambda source: replace(real(source), inode=0))
    assert check(lib, item, keeper) == "unsupported"


# --- keepers ---


def test_no_keeper(lib: Library) -> None:
    assert check(lib, lib.add("a.jpg")) == "keeper_missing"


def test_keeper_gone(lib: Library) -> None:
    item, keeper = lib.add("a.jpg"), lib.add("keep.jpg")
    keeper.record.path.unlink()
    assert check(lib, item, keeper) == "keeper_missing"


def test_keeper_in_disabled_or_unknown_root(lib: Library, tmp_path: Path) -> None:
    item = lib.add("a.jpg")
    other = make_library(tmp_path / "other", root_id=2)
    keeper = other.add("keep.jpg")
    assert check(lib, item, keeper) == "keeper_missing"
    roots = [lib.root, replace(other.root, enabled=False)]
    assert code_of(lambda: checks.check_item(item, [keeper], roots)) == "keeper_missing"
    roots = [lib.root, other.root]
    assert checks.check_item(item, [keeper], roots).keeper.path == keeper.record.path


def test_keeper_changed(lib: Library) -> None:
    item, keeper = lib.add("a.jpg", b"aaaa"), lib.add("keep.jpg", b"aaaa")
    keep_mtime(keeper.record.path, b"bbbb")
    assert check(lib, item, keeper) == "keeper_changed"


def test_exact_keeper_must_have_the_same_content(lib: Library) -> None:
    item, keeper = lib.add("a.jpg", b"aaaa"), lib.add("keep.jpg", b"other")
    assert check(lib, item, keeper) == "keeper_changed"
    near = checks.check_item(item, [keeper], [lib.root], kind=MatchKind.SIMILAR_IMAGE)
    assert near.keeper.path == keeper.record.path


def test_keeper_is_a_hard_link(lib: Library) -> None:
    item = lib.add("a.jpg")
    os.link(item.record.path, lib.folder / "keep.jpg")
    keeper = KnownFile(
        replace(
            item.record,
            id=99,
            path=lib.folder / "keep.jpg",
            rel_path=Path("keep.jpg"),
            signature=paths.signature(lib.folder / "keep.jpg"),
        ),
        item.full_hash,
    )
    item = KnownFile(
        replace(item.record, signature=paths.signature(item.record.path)), item.full_hash
    )
    good = lib.add("other.jpg")
    assert check(lib, item, good, keeper) == "keeper_same_file"


def test_keeper_is_the_same_path(lib: Library) -> None:
    item = lib.add("a.jpg")
    assert check(lib, item, item) == "keeper_same_file"


def test_keeper_is_a_link_to_the_file(lib: Library) -> None:
    item = lib.add("a.jpg")
    link = lib.folder / "keep.jpg"
    make_symlink(link, item.record.path)
    keeper = KnownFile(replace(item.record, path=link, rel_path=Path("keep.jpg")), item.full_hash)
    assert check(lib, item, keeper) == "keeper_same_file"


def test_keeper_changes_while_the_file_is_hashed(lib: Library) -> None:
    item, keeper = lib.add("a.jpg", b"aaaa"), lib.add("keep.jpg", b"aaaa")
    calls = [0]

    def cancel() -> bool:
        calls[0] += 1
        if calls[0] == 2:  # the keeper is hashed first, then the item
            write(keeper.record.path, b"changed!")
        return False

    assert check(lib, item, keeper, cancel=cancel) == "keeper_changed"


def test_still_there(lib: Library) -> None:
    keeper = lib.add("keep.jpg")
    state = verified(keeper)
    checks.expect_still_there(state)
    keeper.record.path.write_bytes(b"other content")
    assert code_of(lambda: checks.expect_still_there(state)) == "keeper_changed"
    keeper.record.path.unlink()
    assert code_of(lambda: checks.expect_still_there(state)) == "keeper_missing"


def test_unchanged_ignores_the_link_count() -> None:
    sig = FileSignature(size=1, mtime_ns=2, dev=3, inode=4, nlink=1)
    assert fsops.unchanged(replace(sig, nlink=2), sig)
    assert not fsops.unchanged(replace(sig, inode=5), sig)
    assert not fsops.unchanged(replace(sig, size=9), sig)
