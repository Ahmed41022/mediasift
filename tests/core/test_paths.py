from __future__ import annotations

import os
import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from mediasift.core import paths
from mediasift.core.types import FileSignature

WINDOWS = sys.platform == "win32"
ARABIC = "صورة العائلة.jpg"
NFC = unicodedata.normalize("NFC", "café.jpg")
NFD = unicodedata.normalize("NFD", "café.jpg")
UNDECODABLE = "lone\ud83d.jpg" if WINDOWS else os.fsdecode(b"bad\xff.jpg")


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


# --- bytes, display, keys ---


@pytest.mark.parametrize("name", [ARABIC, NFC, NFD, UNDECODABLE, "a b.jpg"])
def test_bytes_round_trip_on_disk(tmp_path: Path, name: str) -> None:
    path = tmp_path / name
    path.write_bytes(b"x")
    raw = paths.path_to_bytes(path)
    assert isinstance(raw, bytes)
    again = paths.bytes_to_path(raw)
    assert str(again) == str(path)
    assert again.read_bytes() == b"x"
    assert paths.path_to_bytes(again) == raw


@pytest.mark.parametrize(
    "name", [UNDECODABLE, "a\nb", "a\x1b[31m", "evil\u202egpj.exe", "x\u2066y"]
)
def test_display_is_safe(name: str) -> None:
    shown = paths.display_path(name)
    shown.encode("utf-8")
    assert shown != name
    assert not set(shown) & {"\n", "\x1b", "\u202e", "\u2066"}


def test_display_keeps_ordinary_names() -> None:
    for name in (ARABIC, NFC, "👨\u200d👩\u200d👧.jpg"):
        assert paths.display_path(name) == name


def test_key_compares_by_nfc_and_case() -> None:
    def key(path: str, sensitive: bool) -> bytes:
        return paths.path_key(path, case_sensitive=sensitive)

    assert key(NFC, True) == key(NFD, True)
    assert key("IMG.JPG", False) == key("img.jpg", False)
    assert key("IMG.JPG", True) != key("img.jpg", True)
    assert key("É.jpg", False) == key(unicodedata.normalize("NFD", "é.jpg"), False)
    assert key(UNDECODABLE, False) != key("x.jpg", False)
    assert key(os.path.join("a", "b"), True) == b"a/b"


# --- long paths ---


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("C:\\a\\b.jpg", "\\\\?\\C:\\a\\b.jpg"),
        ("C:/a/../b.jpg", "\\\\?\\C:\\b.jpg"),
        ("C:\\a\\photo. ", "\\\\?\\C:\\a\\photo. "),
        ("C:\\a\\nul.jpg", "\\\\?\\C:\\a\\nul.jpg"),
        ("\\\\server\\share\\a.jpg", "\\\\?\\UNC\\server\\share\\a.jpg"),
        ("//server/share/a.jpg", "\\\\?\\UNC\\server\\share\\a.jpg"),
        ("\\\\?\\C:\\a", "\\\\?\\C:\\a"),
        ("\\\\?\\UNC\\s\\h\\a", "\\\\?\\UNC\\s\\h\\a"),
        ("\\\\.\\pipe\\x", "\\\\.\\pipe\\x"),
        ("a\\b.jpg", "a\\b.jpg"),
        ("C:a.jpg", "C:a.jpg"),
        ("\\a.jpg", "\\a.jpg"),
    ],
)
def test_windows_prefix(given: str, expected: str) -> None:
    assert paths._windows_fs_path(given) == expected


def test_plain_path_undoes_the_prefix() -> None:
    for path in ("C:\\a\\b.jpg", "\\\\server\\share\\a.jpg"):
        assert paths._windows_plain_path(paths._windows_fs_path(path)) == path
    volume = "\\\\?\\Volume{0}\\a"
    assert paths._windows_plain_path(volume) == volume


def test_nul_is_refused() -> None:
    with pytest.raises(ValueError, match="NUL"):
        paths.fs_path("a\0b")
    assert not paths.is_within("/a/\0", "/a")


def test_long_paths_work(tmp_path: Path) -> None:
    folder = tmp_path
    for _ in range(8):
        folder = folder / ("ص" * 60)
    os.makedirs(paths.fs_path(folder))
    file = folder / "photo.jpg"
    Path(paths.fs_path(file)).write_bytes(b"12345")
    assert len(str(file)) > 400
    assert paths.signature(file).size == 5
    assert paths.is_within(file, tmp_path)
    assert not paths.is_within(tmp_path, folder)
    assert not paths.is_link_like(file)
    assert paths.resolve(file).name == "photo.jpg"
    assert not str(paths.resolve(file)).startswith("\\\\?\\")


@pytest.mark.skipif(not WINDOWS, reason="Windows name rules")
def test_names_windows_would_rewrite(tmp_path: Path) -> None:
    for name, content in (("photo", b"1"), ("photo.", b"22"), ("photo ", b"333")):
        Path(paths.fs_path(tmp_path / name)).write_bytes(content)
    sizes = [paths.signature(f"{tmp_path}\\{name}").size for name in ("photo", "photo.", "photo ")]
    for name in ("photo.", "photo "):  # ordinary tools cannot remove these
        os.remove(paths.fs_path(f"{tmp_path}\\{name}"))
    assert sizes == [1, 2, 3]


# --- identity ---


def test_signature_has_identity(tmp_path: Path) -> None:
    first, second, other = (tmp_path / name for name in ("a.jpg", "b.jpg", "c.jpg"))
    first.write_bytes(b"same")
    os.link(first, second)
    other.write_bytes(b"same")
    sig = paths.signature(first)
    assert (sig.size, sig.nlink) == (4, 2)
    assert sig.mtime_ns == first.stat().st_mtime_ns
    assert sig.same_file(paths.signature(os.lstat(second)))
    assert not sig.same_file(paths.signature(other))


def test_signature_survives_rename(tmp_path: Path) -> None:
    before = tmp_path / "a.jpg"
    before.write_bytes(b"x")
    sig = paths.signature(before)
    assert paths.signature(before.rename(tmp_path / "b.jpg")) == sig


@pytest.mark.skipif(not WINDOWS, reason="only Windows leaves the inode out")
def test_dir_entry_stat_has_no_identity(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    entry = next(iter(os.scandir(tmp_path)))
    sig = paths.signature(entry.stat())
    assert (sig.dev, sig.inode) == (0, 0)
    assert not sig.same_file(sig)


def test_signature_does_not_follow_links(tmp_path: Path) -> None:
    target = tmp_path / "a.jpg"
    target.write_bytes(b"x" * 999)
    link = tmp_path / "b.jpg"
    make_symlink(link, target)
    assert not paths.signature(link).same_file(paths.signature(target))
    assert paths.is_link_like(link)


def test_extreme_stat_values_fit_the_index() -> None:
    row = (0o100644, 2**64 + 5, 7, 0, 0, 0, 10, 0, 0, 0)
    st = os.stat_result(row, {"st_mtime_ns": 2**70})
    sig = paths.signature(st)
    assert 0 < sig.inode < 2**64
    assert sig.mtime_ns == 2**63 - 1
    assert sig.nlink == 1
    assert paths._fold_inode(1 << 64 | 2) != paths._fold_inode(2 << 64 | 1)
    assert paths._fold_inode(12345) == 12345


def test_folder_links_are_links(tmp_path: Path) -> None:
    folder = tmp_path / "folder"
    folder.mkdir()
    link = tmp_path / "link"
    make_folder_link(link, folder)
    assert paths.is_link_like(link)
    assert paths.is_link_like(os.lstat(link))
    assert not paths.is_link_like(folder)
    assert not paths.is_link_like(tmp_path / "nothing")


@pytest.mark.skipif(not WINDOWS, reason="reparse points exist on Windows only")
def test_only_name_surrogates_are_links() -> None:
    def st(tag: int) -> os.stat_result:
        return os.stat_result((0o100644, 1, 1, 1, 0, 0, 1, 0, 0, 0), {"st_reparse_tag": tag})

    assert paths.is_link_like(st(0xA0000003))  # junction
    assert paths.is_link_like(st(0xA000001D))  # WSL symlink
    assert not paths.is_link_like(st(0x9000001A))  # OneDrive
    assert not paths.is_link_like(st(0x80000013))  # deduplicated


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [(0x20, False), (0x1000, True), (0x40000, True), (0x400000, True), (0x400, False)],
)
def test_placeholder(attributes: int, expected: bool) -> None:
    st = cast(Any, SimpleNamespace(st_file_attributes=attributes))
    assert paths.is_placeholder(st) is expected
    assert not paths.is_placeholder(os.lstat(__file__))


def test_volume_id(tmp_path: Path) -> None:
    file = tmp_path / "a"
    file.write_bytes(b"")
    assert paths.volume_id(tmp_path)
    assert paths.volume_id(tmp_path) == paths.volume_id(file)
    assert paths.volume_id(tmp_path / "gone") == ""


# --- containment ---


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for name in ("root/sub", "root2", "outside"):
        (tmp_path / name).mkdir(parents=True)
    for name in ("root/sub/a.jpg", "root2/a.jpg", "outside/a.jpg"):
        (tmp_path / name).write_bytes(b"x")
    return tmp_path


def test_within(tree: Path) -> None:
    root = tree / "root"
    assert paths.is_within(root / "sub" / "a.jpg", root)
    assert paths.is_within(root / "sub", root)
    assert paths.is_within(root / "not" / "yet.jpg", root)
    assert paths.is_within(root / "sub" / ".." / "sub" / "a.jpg", root)


def test_not_within(tree: Path) -> None:
    root = tree / "root"
    assert not paths.is_within(root, root)
    assert not paths.is_within(root / "sub" / "..", root)
    assert not paths.is_within(tree, root)
    assert not paths.is_within(tree / "root2" / "a.jpg", root)
    assert not paths.is_within(root / ".." / "outside" / "a.jpg", root)
    assert not paths.is_within(root / "sub" / ".." / ".." / "root2", root)


def test_relative_paths(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tree / "root")
    assert paths.is_within("sub/a.jpg", tree / "root")
    assert paths.is_within("sub", ".")
    assert not paths.is_within("../outside/a.jpg", ".")


def test_folder_link_out_of_the_root(tree: Path) -> None:
    root = tree / "root"
    make_folder_link(root / "link", tree / "outside")
    assert not paths.is_within(root / "link" / "a.jpg", root)
    assert not paths.is_within(root / "link", root)
    assert not paths.is_within(root / "link" / "missing.jpg", root)


def test_folder_link_into_the_root(tree: Path) -> None:
    root = tree / "root"
    make_folder_link(tree / "way-in", root)
    assert paths.is_within(tree / "way-in" / "sub" / "a.jpg", root)
    assert paths.is_within(root / "sub" / "a.jpg", tree / "way-in")
    assert paths.same_location(tree / "way-in", root)


def test_file_symlink_out_of_the_root(tree: Path) -> None:
    link = tree / "root" / "link.jpg"
    make_symlink(link, tree / "outside" / "a.jpg")
    assert not paths.is_within(link, tree / "root")


def test_dangling_symlink_out_of_the_root(tree: Path) -> None:
    link = tree / "root" / "link.jpg"
    make_symlink(link, tree / "outside" / "gone.jpg")
    assert not paths.is_within(link, tree / "root")


def test_link_loop(tree: Path) -> None:
    root = tree / "root"
    back = root / "sub" / "back"
    back.mkdir()  # a junction's target has to exist when it is made
    make_folder_link(root / "loop", back)
    back.rmdir()
    make_folder_link(back, root / "loop")
    assert not paths.is_within(root / "loop" / "a.jpg", root)


def test_case_twin_is_decided_by_the_file_system(tree: Path) -> None:
    other = tree / "ROOT" / "sub" / "a.jpg"
    assert paths.is_within(other, tree / "root") is other.exists()
    assert paths.same_location(tree / "ROOT", tree / "root") is (tree / "ROOT").exists()


def test_missing_root(tree: Path) -> None:
    gone = tree / "gone"
    assert paths.is_within(gone / "a.jpg", gone)
    assert not paths.is_within(tree / "gone2" / "a.jpg", gone)
    assert paths.same_location(gone, tree / "x" / ".." / "gone")
    assert not paths.same_location(gone, tree)


def test_unverifiable_is_outside(tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: object, **kwargs: object) -> os.stat_result:
        raise PermissionError(13, "denied")

    monkeypatch.setattr(paths.os, "lstat", denied)
    assert not paths.is_within(tree / "root" / "sub", tree / "root")


@pytest.mark.skipif(not WINDOWS, reason="Windows path forms")
def test_windows_forms_of_one_path(tree: Path) -> None:
    root = tree / "root"
    file = root / "sub" / "a.jpg"
    assert paths.is_within(paths.fs_path(file), root)
    assert paths.is_within(file, paths.fs_path(root))
    assert paths.is_within(str(file).replace("\\", "/"), root)
    assert paths.is_within(str(file).upper(), root)


# --- case ---


def test_case_probe_matches_reality_and_writes_nothing(tmp_path: Path) -> None:
    probe = tmp_path / "other"
    probe.mkdir()
    (probe / "Check").write_bytes(b"")
    truth = not (probe / "cHECK").exists()

    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "123").write_bytes(b"")
    (folder / "صورة").write_bytes(b"")
    (folder / "Photo.JPG").write_bytes(b"")
    before = (folder.stat().st_mtime_ns, sorted(os.listdir(folder)))
    assert paths.is_case_sensitive(folder) is truth
    assert (folder.stat().st_mtime_ns, sorted(os.listdir(folder))) == before


def test_case_probe_falls_back_to_the_platform(tmp_path: Path) -> None:
    default = sys.platform not in ("win32", "darwin")
    (tmp_path / "123").write_bytes(b"")
    assert paths.is_case_sensitive(tmp_path) is default
    assert paths.is_case_sensitive(tmp_path / "gone") is default


def fake_stat(inode: int, nlink: int = 1) -> os.stat_result:
    return os.stat_result((0o100644, inode, 1, nlink, 0, 0, 1, 0, 0, 0))


def test_case_probe_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    known: dict[str, os.stat_result] = {}

    def lstat(path: str) -> os.stat_result:
        try:
            return known[os.path.basename(path)]
        except KeyError:
            raise FileNotFoundError(path) from None

    monkeypatch.setattr(paths.os, "lstat", lstat)
    known.update({"a": fake_stat(5)})
    assert paths._probe_case(os.path.join("d", "a")) is True
    known.update({"A": fake_stat(5)})
    assert paths._probe_case(os.path.join("d", "a")) is False
    known.update({"A": fake_stat(6)})
    assert paths._probe_case(os.path.join("d", "a")) is True
    known.update({"a": fake_stat(5, 2), "A": fake_stat(5, 2)})  # perhaps two hard links
    assert paths._probe_case(os.path.join("d", "a")) is None
    known.update({"a": fake_stat(0), "A": fake_stat(0)})  # no identity
    assert paths._probe_case(os.path.join("d", "a")) is None
    assert paths._probe_case(os.path.join("d", "gone")) is None


def test_signature_type() -> None:
    assert isinstance(paths.signature(__file__), FileSignature)
