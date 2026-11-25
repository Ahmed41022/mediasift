from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path

import pytest

from mediasift.core import paths
from mediasift.core.errors import ExitCode, LibraryError, SafetyError
from mediasift.db.connection import Database
from mediasift.db.roots import RootsRepo
from tests.core.test_paths import UNDECODABLE, make_folder_link


@pytest.fixture
def roots(db: Database) -> RootsRepo:
    return RootsRepo(db)


def folder(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir(parents=True)
    return path


def test_add_and_get(roots: RootsRepo, tmp_path: Path) -> None:
    path = folder(tmp_path, "صور العائلة")
    root = roots.add(path, read_only=True, prefer_rank=3)
    assert root.path == paths.resolve(path)
    assert (root.read_only, root.prefer_rank, root.enabled) == (True, 3, True)
    assert root.volume_id == paths.volume_id(path) != ""
    assert root.case_sensitive is paths.is_case_sensitive(path)
    assert roots.get(root.id) == root
    assert roots.list() == [root]
    assert roots.find(path) == root
    assert roots.find(tmp_path) is None


@pytest.mark.parametrize("name", [UNDECODABLE, unicodedata.normalize("NFD", "é")])
def test_odd_names_survive(roots: RootsRepo, tmp_path: Path, name: str) -> None:
    root = roots.add(folder(tmp_path, name))
    assert root.path.name == name
    assert roots.get(root.id).path.is_dir()


def test_long_path(roots: RootsRepo, tmp_path: Path) -> None:
    path = tmp_path.joinpath(*["x" * 60] * 8)
    os.makedirs(paths.fs_path(path))
    root = roots.add(path)
    assert os.path.isdir(paths.fs_path(root.path))
    assert len(str(root.path)) > 400


def test_order_and_enabled(roots: RootsRepo, tmp_path: Path) -> None:
    a, b, c = (roots.add(folder(tmp_path, name)) for name in "abc")
    roots.set_prefer_rank(c.id, 5)
    assert roots.set_enabled(a.id, False).enabled is False
    assert [root.id for root in roots.list()] == [c.id, a.id, b.id]
    assert [root.id for root in roots.list(enabled_only=True)] == [c.id, b.id]
    assert roots.set_read_only(b.id, True).read_only is True
    assert roots.set_volume_id(b.id, "other").volume_id == "other"
    assert roots.set_enabled(a.id, True) == a


def test_unknown_root(roots: RootsRepo) -> None:
    for call in (
        lambda: roots.get(9),
        lambda: roots.set_enabled(9, True),
        lambda: roots.set_prefer_rank(9, 1),
    ):
        with pytest.raises(LibraryError, match="number 9"):
            call()


def test_overlap_is_refused(roots: RootsRepo, tmp_path: Path) -> None:
    inner = folder(tmp_path, "photos/2024")
    photos = roots.add(tmp_path / "photos")
    for path in (inner, tmp_path, tmp_path / "photos" / ".." / "photos" / "2024"):
        with pytest.raises(SafetyError, match="overlaps") as caught:
            roots.add(path)
        assert caught.value.code == "root_overlap"
        assert caught.value.exit_code is ExitCode.REFUSED
    with pytest.raises(SafetyError) as caught:
        roots.add(tmp_path / "photos")
    assert caught.value.code == "root_exists"
    roots.set_enabled(photos.id, False)
    with pytest.raises(SafetyError, match="overlaps"):
        roots.add(inner)
    assert len(roots.list()) == 1


def test_similar_prefix_is_no_overlap(roots: RootsRepo, tmp_path: Path) -> None:
    roots.add(folder(tmp_path, "photos"))
    roots.add(folder(tmp_path, "photos2"))
    roots.add(folder(tmp_path, "photo"))
    assert len(roots.list()) == 3


def test_overlap_through_links_is_refused(roots: RootsRepo, tmp_path: Path) -> None:
    inner = folder(tmp_path, "photos/2024")
    make_folder_link(tmp_path / "link", tmp_path / "photos")
    make_folder_link(tmp_path / "deep", inner)
    root = roots.add(tmp_path / "link")
    assert root.path == paths.resolve(tmp_path / "photos")
    for path in (tmp_path / "deep", inner, tmp_path / "link" / "2024"):
        with pytest.raises(SafetyError, match="overlaps"):
            roots.add(path)
    with pytest.raises(SafetyError, match="already"):
        roots.add(tmp_path / "photos")


@pytest.mark.skipif(sys.platform != "win32", reason="case-insensitive names")
def test_other_spelling_is_the_same_root(roots: RootsRepo, tmp_path: Path) -> None:
    path = folder(tmp_path, "Photos")
    root = roots.add(str(path).upper())
    assert root.path.name == "Photos"  # as stored on disk
    assert roots.find(str(path).lower()) == root
    with pytest.raises(SafetyError, match="already"):
        roots.add(paths.fs_path(path))


def test_unreachable_root_still_counts(roots: RootsRepo, tmp_path: Path) -> None:
    path = folder(tmp_path, "usb/photos")
    roots.add(path)
    path.rmdir()
    with pytest.raises(SafetyError, match="overlaps"):
        roots.add(tmp_path / "usb")
    assert roots.find(path) is not None


def test_find_containing(roots: RootsRepo, tmp_path: Path) -> None:
    root = roots.add(folder(tmp_path, "photos"))
    assert roots.find_containing(tmp_path / "photos" / "a" / "b.jpg") == root
    assert roots.find_containing(tmp_path / "photos") == root
    assert roots.find_containing(tmp_path / "photos2" / "b.jpg") is None


def test_not_a_folder(roots: RootsRepo, tmp_path: Path) -> None:
    (tmp_path / "file").write_bytes(b"")
    for path in (tmp_path / "file", tmp_path / "gone", "bad\0name"):
        with pytest.raises(LibraryError):
            roots.add(path)
    assert roots.list() == []
