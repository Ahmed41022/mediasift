from __future__ import annotations

import errno
import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from mediasift.actions import fsops
from mediasift.core import paths
from mediasift.core.errors import FileChanged, JobCancelled, SafetyError
from tests.actions.conftest import (
    ARABIC,
    POSIX,
    WINDOWS,
    digest,
    make_folder_link,
    make_symlink,
    on_other_device,
    write,
)

Rename = Callable[[Path, Path], None]


def code_of(call: Callable[[], object]) -> str:
    with pytest.raises(SafetyError) as caught:
        call()
    return caught.value.code


def hook_rename(monkeypatch: pytest.MonkeyPatch, before: Callable[[int, Path, Path], None]) -> None:
    """Run ``before(call number, src, dst)`` ahead of every real rename."""
    real: Rename = fsops._rename_no_replace
    calls = [0]

    def rename(src: Path, dst: Path) -> None:
        calls[0] += 1
        before(calls[0], src, dst)
        real(src, dst)

    monkeypatch.setattr(fsops, "_rename_no_replace", rename)


# --- moving ---


@pytest.mark.parametrize("name", ["a.jpg", ARABIC])
def test_move_keeps_the_file(tmp_path: Path, name: str) -> None:
    src = write(tmp_path / "a" / name, b"content")
    os.utime(src, ns=(1_000_000_000, 1_500_000_000_100))
    dst = tmp_path / "b" / name
    dst.parent.mkdir()
    before = paths.signature(src)
    after = fsops.move_no_replace(src, dst, before)
    assert not src.exists()
    assert dst.read_bytes() == b"content"
    assert fsops.unchanged(after, before)
    assert dst.stat().st_mtime_ns == 1_500_000_000_100


def test_long_paths(tmp_path: Path) -> None:
    folder = tmp_path
    for _ in range(6):
        folder = folder / ("ص" * 60)
    os.makedirs(paths.fs_path(folder))
    src = folder / "photo.jpg"
    Path(paths.fs_path(src)).write_bytes(b"x")
    dst = folder / ("ب" * 100 + ".jpg")
    fsops.move_no_replace(src, dst, paths.signature(src))
    assert len(str(dst)) > 400
    assert Path(paths.fs_path(dst)).read_bytes() == b"x"


def test_target_exists(tmp_path: Path) -> None:
    src, dst = write(tmp_path / "a.jpg", b"mine"), write(tmp_path / "b.jpg", b"other")
    assert code_of(lambda: fsops.move_no_replace(src, dst, paths.signature(src))) == (
        "target_exists"
    )
    assert (src.read_bytes(), dst.read_bytes()) == (b"mine", b"other")


def test_target_differs_only_by_case(tmp_path: Path) -> None:
    src = write(tmp_path / "a" / "x.jpg", b"mine")
    other = write(tmp_path / "b" / "Photo.jpg", b"other")
    dst = tmp_path / "b" / "PHOTO.JPG"
    if dst.exists():  # case-insensitive: the same name
        assert code_of(lambda: fsops.move_no_replace(src, dst, paths.signature(src))) == (
            "target_exists"
        )
        assert src.read_bytes() == b"mine"
    else:
        fsops.move_no_replace(src, dst, paths.signature(src))
        assert dst.read_bytes() == b"mine"
    assert other.read_bytes() == b"other"


def test_hard_link_target_is_refused(tmp_path: Path) -> None:
    src = write(tmp_path / "a.jpg")
    os.link(src, tmp_path / "b.jpg")
    sig = paths.signature(src)
    assert code_of(lambda: fsops.move_no_replace(src, tmp_path / "b.jpg", sig)) == ("target_exists")
    assert src.exists()


def test_target_created_after_the_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src, dst = write(tmp_path / "a.jpg", b"mine"), tmp_path / "b.jpg"
    hook_rename(monkeypatch, lambda n, s, d: write(dst, b"intruder") and None)
    assert code_of(lambda: fsops.move_no_replace(src, dst, paths.signature(src))) == (
        "target_exists"
    )
    assert (src.read_bytes(), dst.read_bytes()) == (b"mine", b"intruder")


def swap(src: Path, aside: Path) -> None:
    src.rename(aside)
    write(src, b"stranger")


def test_source_swapped_after_the_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src, dst, aside = write(tmp_path / "a.jpg", b"mine"), tmp_path / "b.jpg", tmp_path / "c"
    sig = paths.signature(src)
    hook_rename(monkeypatch, lambda n, s, d: swap(src, aside) if n == 1 else None)
    assert code_of(lambda: fsops.move_no_replace(src, dst, sig)) == "changed"
    assert src.read_bytes() == b"stranger"  # moved back
    assert aside.read_bytes() == b"mine"
    assert not dst.exists()


def test_stranger_cannot_be_moved_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src, dst, aside = write(tmp_path / "a.jpg", b"mine"), tmp_path / "b.jpg", tmp_path / "c"
    sig = paths.signature(src)

    def before(n: int, s: Path, d: Path) -> None:
        if n == 1:
            swap(src, aside)
        else:  # moving back: the name has been taken again
            write(src, b"third")

    hook_rename(monkeypatch, before)
    with pytest.raises(fsops.MisplacedFile) as caught:
        fsops.move_no_replace(src, dst, sig)
    assert caught.value.path == dst
    assert str(dst) in str(caught.value)
    assert dst.read_bytes() == b"stranger"
    assert (src.read_bytes(), aside.read_bytes()) == (b"third", b"mine")


def test_target_folder_replaced_during_the_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = write(tmp_path / "a.jpg", b"mine")
    folder, elsewhere = tmp_path / "q", tmp_path / "elsewhere"
    folder.mkdir()
    elsewhere.mkdir()

    def before(n: int, s: Path, d: Path) -> None:
        if n == 1:
            folder.rename(tmp_path / "q-old")
            make_folder_link(folder, elsewhere)

    hook_rename(monkeypatch, before)
    assert code_of(lambda: fsops.move_no_replace(src, folder / "a.jpg", paths.signature(src))) == (
        "changed"
    )
    assert src.read_bytes() == b"mine"
    assert not (elsewhere / "a.jpg").exists()


def test_cross_volume_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = write(tmp_path / "a.jpg")

    def cross(src: Path, dst: Path) -> None:
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(fsops, "_rename_no_replace", cross)
    sig = paths.signature(src)
    assert code_of(lambda: fsops.move_no_replace(src, tmp_path / "b.jpg", sig)) == ("cross_volume")
    assert src.exists()


def test_other_device_is_refused_before_moving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = write(tmp_path / "a.jpg")
    other = tmp_path / "mounted"
    other.mkdir()
    real = fsops.stat_or_none

    def stat(path: Path) -> os.stat_result | None:
        st = real(path)
        if st is None or path != other:
            return st
        return on_other_device(st)

    monkeypatch.setattr(fsops, "stat_or_none", stat)
    sig = paths.signature(src)
    assert code_of(lambda: fsops.move_no_replace(src, other / "a.jpg", sig)) == "cross_volume"
    assert src.exists()


def test_links_are_never_moved(tmp_path: Path) -> None:
    target = write(tmp_path / "real.jpg")
    link = tmp_path / "link.jpg"
    make_symlink(link, target)
    sig = paths.signature(link)
    assert code_of(lambda: fsops.move_no_replace(link, tmp_path / "x.jpg", sig)) == "is_link"
    folder = tmp_path / "folder"
    folder.mkdir()
    junction = tmp_path / "junction"
    make_folder_link(junction, folder)
    src_sig = paths.signature(target)
    code = code_of(lambda: fsops.move_no_replace(target, junction / "x.jpg", src_sig))
    assert code == "is_link"
    assert code_of(lambda: fsops.move_no_replace(junction, tmp_path / "j2", src_sig)) == "is_link"
    assert target.exists() and link.is_symlink() and not (folder / "x.jpg").exists()


def test_only_regular_files(tmp_path: Path) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    write(tmp_path / "x")
    sig = paths.signature(tmp_path / "x")
    assert code_of(lambda: fsops.move_no_replace(folder, tmp_path / "y", sig)) == "not_regular"
    assert code_of(lambda: fsops.move_no_replace(tmp_path / "gone", tmp_path / "y", sig)) == (
        "missing"
    )
    assert code_of(lambda: fsops.move_no_replace(tmp_path / "x", tmp_path / "no" / "y", sig)) == (
        "missing"
    )


def test_relative_paths_are_refused(tmp_path: Path) -> None:
    src = write(tmp_path / "a.jpg")
    sig = paths.signature(src)
    assert code_of(lambda: fsops.move_no_replace(Path("a.jpg"), tmp_path / "b", sig)) == (
        "unsupported"
    )
    assert code_of(lambda: fsops.move_no_replace(src, tmp_path / ".." / "b", sig)) == (
        "unsupported"
    )


def test_changed_file_is_refused(tmp_path: Path) -> None:
    src = write(tmp_path / "a.jpg", b"one")
    sig = paths.signature(src)
    write(src, b"three")
    assert code_of(lambda: fsops.move_no_replace(src, tmp_path / "b", sig)) == "changed"
    other = write(tmp_path / "c.jpg", b"one")
    os.utime(other, ns=(sig.mtime_ns, sig.mtime_ns))
    assert code_of(lambda: fsops.move_no_replace(other, tmp_path / "b", sig)) == "changed"


def test_placeholder_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = write(tmp_path / "a.jpg")
    monkeypatch.setattr(paths, "is_placeholder", lambda st: True)
    sig = paths.signature(src)
    assert code_of(lambda: fsops.move_no_replace(src, tmp_path / "b", sig)) == "is_placeholder"
    assert src.exists()


# --- the link fallback of Linux ---

linux_only = pytest.mark.skipif(not POSIX, reason="the fallback is for Linux file systems")


@linux_only
@pytest.mark.parametrize("error", [errno.ENOSYS, errno.EINVAL])
def test_fallback_moves_by_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: int
) -> None:
    monkeypatch.setattr(fsops, "_renameat2", lambda src, dst: error)
    src = write(tmp_path / "a.jpg", b"mine")
    dst = tmp_path / "q" / "a.jpg"
    dst.parent.mkdir()
    fsops.move_no_replace(src, dst, paths.signature(src))
    assert not src.exists()
    assert dst.read_bytes() == b"mine"
    assert dst.stat().st_nlink == 1
    assert os.listdir(dst.parent) == ["a.jpg"]


@linux_only
def test_fallback_never_replaces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fsops, "_renameat2", lambda src, dst: errno.ENOSYS)
    src, dst = write(tmp_path / "a.jpg", b"mine"), tmp_path / "b.jpg"
    hook_rename(monkeypatch, lambda n, s, d: write(dst, b"intruder") and None)
    assert code_of(lambda: fsops.move_no_replace(src, dst, paths.signature(src))) == (
        "target_exists"
    )
    assert (src.read_bytes(), dst.read_bytes()) == (b"mine", b"intruder")


@linux_only
def test_fallback_refuses_without_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fsops, "_renameat2", lambda src, dst: errno.ENOSYS)

    def no_links(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(fsops.os, "link", no_links)
    src = write(tmp_path / "a.jpg")
    assert code_of(lambda: fsops.move_no_replace(src, tmp_path / "b", paths.signature(src))) == (
        "unsupported"
    )
    assert src.exists() and not (tmp_path / "b").exists()


@linux_only
def test_fallback_with_a_swapped_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fsops, "_renameat2", lambda src, dst: errno.ENOSYS)
    src, dst = write(tmp_path / "a.jpg", b"mine"), tmp_path / "b.jpg"
    real_rename = os.rename

    def rename(a: object, b: object) -> None:
        if Path(str(b)).name == "file":  # the source goes aside: swap it just before
            os.unlink(src)
            write(src, b"stranger")
        real_rename(a, b)  # type: ignore[arg-type]

    monkeypatch.setattr(fsops.os, "rename", rename)
    with pytest.raises(fsops.MisplacedFile) as caught:
        fsops.move_no_replace(src, dst, paths.signature(src))
    assert caught.value.path == dst
    assert (src.read_bytes(), dst.read_bytes()) == (b"stranger", b"mine")
    assert sorted(os.listdir(tmp_path)) == ["a.jpg", "b.jpg"]


def test_no_way_to_move_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fsops.sys, "platform", "darwin")
    src = write(tmp_path / "a.jpg")
    assert code_of(lambda: fsops.move_no_replace(src, tmp_path / "b", paths.signature(src))) == (
        "unsupported"
    )
    assert src.exists()


# --- releasing a name ---


def test_release_extra_name(tmp_path: Path) -> None:
    keep = write(tmp_path / "a.jpg")
    extra = tmp_path / "b.jpg"
    os.link(keep, extra)
    fsops.release_extra_name(extra, keep)
    assert keep.exists() and not extra.exists()


def test_release_refuses_the_last_name(tmp_path: Path) -> None:
    keep, other = write(tmp_path / "a.jpg"), write(tmp_path / "b.jpg")
    assert code_of(lambda: fsops.release_extra_name(other, keep)) == "changed"
    assert code_of(lambda: fsops.release_extra_name(keep, keep)) == "changed"
    os.link(keep, tmp_path / "c.jpg")
    # Two spellings of one entry are not two names, even with another hard link elsewhere.
    upper = tmp_path / "A.JPG"
    if upper.exists():
        assert code_of(lambda: fsops.release_extra_name(upper, keep)) == "changed"
    assert code_of(lambda: fsops.release_extra_name(tmp_path / "gone", keep)) == "missing"
    assert keep.exists() and other.exists()


def test_release_checks_the_expected_file(tmp_path: Path) -> None:
    keep = write(tmp_path / "a.jpg")
    os.link(keep, tmp_path / "b.jpg")
    other = paths.signature(write(tmp_path / "c.jpg"))
    assert code_of(lambda: fsops.release_extra_name(tmp_path / "b.jpg", keep, other)) == ("changed")


def test_release_needs_a_second_link(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    keep = write(tmp_path / "a.jpg")
    os.link(keep, tmp_path / "b.jpg")
    real = paths.signature
    monkeypatch.setattr(paths, "signature", lambda source: replace(real(source), nlink=1))
    assert code_of(lambda: fsops.release_extra_name(tmp_path / "b.jpg", keep)) == "changed"
    assert (tmp_path / "b.jpg").exists()


# --- folders ---


def test_make_dirs_private(tmp_path: Path) -> None:
    target = tmp_path / "q" / "s" / ARABIC
    fsops.make_dirs_private(target, tmp_path)
    assert target.is_dir()
    if POSIX:
        assert (tmp_path / "q").stat().st_mode & 0o077 == 0
    fsops.make_dirs_private(target, tmp_path)  # again: nothing to do


def test_make_dirs_refuses_links_and_files(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    make_folder_link(root / "q", elsewhere)
    assert code_of(lambda: fsops.make_dirs_private(root / "q" / "s", root)) == "is_link"
    assert not (elsewhere / "s").exists()
    write(root / "file")
    assert code_of(lambda: fsops.make_dirs_private(root / "file" / "s", root)) == "not_regular"
    assert code_of(lambda: fsops.make_dirs_private(tmp_path / "x", root)) == "outside_root"
    assert code_of(lambda: fsops.make_dirs_private(root / ".." / "x", root)) == "outside_root"
    assert code_of(lambda: fsops.make_dirs_private(root / "q2", root / "gone")) == "missing"


# --- hashing ---


def test_hash_file(tmp_path: Path) -> None:
    data = os.urandom(3 * 1024 * 1024 + 5)
    path = write(tmp_path / "a.bin", data)
    assert fsops.hash_file(path) == digest(data)
    assert fsops.hash_file(write(tmp_path / "empty", b"")) == digest(b"")


def test_hash_notices_a_change(tmp_path: Path) -> None:
    path = write(tmp_path / "a.bin", b"x" * 100)

    def change() -> bool:  # called after each chunk
        os.utime(path, ns=(0, 1_000_000_000))
        return False

    with pytest.raises(FileChanged):
        fsops.hash_file(path, change)


def test_hash_can_be_cancelled(tmp_path: Path) -> None:
    path = write(tmp_path / "a.bin", b"x" * 100)
    with pytest.raises(JobCancelled):
        fsops.hash_file(path, lambda: True)


def test_hash_refuses_links(tmp_path: Path) -> None:
    link = tmp_path / "link"
    make_symlink(link, write(tmp_path / "a"))
    assert code_of(lambda: fsops.hash_file(link)) == "is_link"


# --- names ---


def test_numbered_names() -> None:
    assert fsops.numbered_name("photo.jpg", 2) == "photo (2).jpg"
    assert fsops.numbered_name("photo.jpg", 1, "restored") == "photo (restored 1).jpg"
    assert fsops.numbered_name(".hidden", 2) == ".hidden (2)"
    assert fsops.numbered_name("a." + "x" * 40, 2) == "a." + "x" * 40 + " (2)"


def test_numbered_names_stay_short() -> None:
    name = "ص" * (120 if not WINDOWS else 250) + ".jpg"
    longer = fsops.numbered_name(name, 12, "restored")
    assert longer.endswith(" (restored 12).jpg")
    assert fsops._name_length(longer) <= 255
    assert fsops._name_length(fsops.numbered_name(name, 3, limit=100)) <= 100
    assert list(fsops.numbered("a.jpg"))[:2] == ["a (2).jpg", "a (3).jpg"]
