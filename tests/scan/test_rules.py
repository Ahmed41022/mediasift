from __future__ import annotations

import os
from pathlib import Path

import pytest

from mediasift.core.config import ScanSettings
from mediasift.core.errors import ConfigError
from mediasift.core.types import Root
from mediasift.scan.rules import QUARANTINE_FOLDER, Rules


def rules(
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    *,
    case_sensitive: bool = True,
    **other: object,
) -> Rules:
    root = Root(id=1, path=Path("/nowhere"), case_sensitive=case_sensitive)
    config = ScanSettings(include=include, exclude=exclude, **other)  # type: ignore[arg-type]
    return Rules.for_root(config, root)


def rel(text: str) -> str:
    return text.replace("/", os.sep)


@pytest.mark.parametrize(
    ("pattern", "path", "excluded"),
    [
        ("*.jpg", "a.jpg", True),
        ("*.jpg", "deep/er/a.jpg", True),
        ("*.jpg", "a.png", False),
        ("tmp", "x/tmp/a.jpg", True),
        ("tmp", "x/tmpx/a.jpg", False),
        ("/tmp", "tmp/a.jpg", True),
        ("/tmp", "x/tmp/a.jpg", False),
        ("x/tmp", "x/tmp/a.jpg", True),
        ("x/tmp", "y/x/tmp/a.jpg", False),
        ("**/tmp", "y/x/tmp/a.jpg", True),
        ("a/**/b.jpg", "a/b.jpg", True),
        ("a/**/b.jpg", "a/x/y/b.jpg", True),
        ("a/**", "a/x/y/b.jpg", True),
        ("a/**", "ab/c.jpg", False),
        ("IMG_????.jpg", "IMG_0001.jpg", True),
        ("IMG_????.jpg", "IMG_01.jpg", False),
        ("[ab]*.jpg", "b1.jpg", True),
        ("[!ab]*.jpg", "b1.jpg", False),
        ("[!ab]*.jpg", "c1.jpg", True),
        ("[0-9].jpg", "5.jpg", True),
        ("[.jpg", "[.jpg", True),
        ("*", "a/b.jpg", True),
        ("صور*", "صور العائلة/1.jpg", True),
        ("*.jpg/", "a.jpg", False),
    ],
)
def test_exclude_patterns(pattern: str, path: str, excluded: bool) -> None:
    assert rules(exclude=(pattern,)).skips_file(rel(path)) is excluded


def test_star_does_not_cross_folders() -> None:
    assert not rules(exclude=("/a*.jpg",)).skips_file(rel("ab/c.jpg"))


def test_trailing_slash_matches_folders_only() -> None:
    compiled = rules(exclude=("private/",))
    assert compiled.skips_folder(rel("x/private"))
    assert not compiled.skips_file("private")
    assert compiled.skips_file(rel("private/a.jpg"))


def test_folder_exclusion_prunes() -> None:
    assert rules(exclude=("cache",)).skips_folder(rel("a/cache"))
    assert not rules(exclude=("cache/*.jpg",)).skips_folder("cache")


def test_include_covers_folders_and_names() -> None:
    compiled = rules(include=("2023", "*.mp4"))
    assert not compiled.skips_file(rel("2023/x/a.jpg"))
    assert not compiled.skips_file(rel("other/a.mp4"))
    assert compiled.skips_file(rel("other/a.jpg"))
    # Folders are never pruned by includes: a file deep below may match.
    assert not compiled.skips_folder("other")


def test_exclude_wins_over_include() -> None:
    compiled = rules(include=("*.jpg",), exclude=("secret",))
    assert compiled.skips_file(rel("secret/a.jpg"))


def test_case_follows_the_root() -> None:
    assert rules(exclude=("*.JPG",), case_sensitive=False).skips_file("a.jpg")
    assert not rules(exclude=("*.JPG",), case_sensitive=True).skips_file("a.jpg")
    assert rules(exclude=("Straße",), case_sensitive=False).skips_folder("STRASSE")


def test_unicode_forms_are_equal() -> None:
    nfd = "éte"
    assert rules(exclude=("éte",)).skips_folder(nfd)


@pytest.mark.parametrize("bad", ["/", "..", "a/../b", "./"])
def test_bad_patterns_are_config_errors(bad: str) -> None:
    with pytest.raises(ConfigError, match=r"scan\.exclude"):
        rules(exclude=(bad,))


def test_hidden() -> None:
    compiled = rules()
    assert compiled.skips_file(".a.jpg")
    assert compiled.skips_folder(".thumbnails")
    assert compiled.skips_file("a.jpg", attributes=0x2)
    assert compiled.skips_folder("x", attributes=0x4)
    shown = rules(include_hidden=True)
    assert not shown.skips_file(".a.jpg")
    assert not shown.skips_folder(".thumbnails")


@pytest.mark.parametrize(
    "folder",
    [
        QUARANTINE_FOLDER,
        "$RECYCLE.BIN",
        "$Recycle.Bin",
        "System Volume Information",
        ".Trash",
        ".Trash-1000",
        "@eaDir",
        ".local/share/Trash",
    ],
)
def test_system_folders_are_always_skipped(folder: str) -> None:
    assert rules(include_hidden=True).skips_folder(rel(f"x/{folder}"))


def test_min_size() -> None:
    compiled = rules(min_size=10)
    assert compiled.skips_file("a.jpg", size=9)
    assert not compiled.skips_file("a.jpg", size=10)
    assert not compiled.skips_file("a.jpg")  # size unknown: decided later


def test_library_inside_a_root_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "root" / "app" / "lib").mkdir(parents=True)
    root = Root(id=1, path=tmp_path / "root", case_sensitive=True)
    compiled = Rules.for_root(ScanSettings(), root, tmp_path / "root" / "app" / "lib")
    assert compiled.skips_folder(rel("app/lib"))
    assert not compiled.skips_folder("app")
    outside = Rules.for_root(ScanSettings(), root, tmp_path / "elsewhere")
    assert outside.library is None
