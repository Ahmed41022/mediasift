"""Finding the reader for a file, and what is missing when a library is."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mediasift.core.plugins import Registry
from mediasift.core.types import IMAGE_EXTENSIONS, media_type_for
from mediasift.readers import (
    HeifReader,
    PillowReader,
    RawReader,
    default_readers,
    reader_for,
    unavailable_extensions,
)


def test_default_readers_are_pillow_heif_raw_in_that_order() -> None:
    readers = default_readers()
    assert [r.name for r in readers] == ["pillow", "heif", "raw"]
    assert [type(r) for r in readers] == [PillowReader, HeifReader, RawReader]
    assert len(readers) == 3
    assert readers.get("heif").extensions == {".heic", ".heif", ".hif"}


def test_each_call_gives_a_new_registry() -> None:
    assert default_readers() is not default_readers()
    assert isinstance(default_readers(), Registry)


def test_readers_cover_exactly_the_image_extensions() -> None:
    seen: set[str] = set()
    for reader in default_readers():
        assert not (seen & reader.extensions), f"{reader.name} overlaps an earlier reader"
        seen |= reader.extensions
    assert seen == IMAGE_EXTENSIONS


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.jpg", "pillow"),
        ("A.JPG", "pillow"),
        ("a.JpEg", "pillow"),
        ("x.tif", "pillow"),
        ("x.webp", "pillow"),
        ("IMG_1.HEIC", "heif"),
        ("x.hif", "heif"),
        ("x.dng", "raw"),
        ("X.CR3", "raw"),
        ("x.RAF", "raw"),
        ("x.srw", "raw"),
        ("archive.tar.jpg", "pillow"),
        (".jpg.png", "pillow"),
        ("x.mp4", None),
        ("x.mp3", None),
        ("x.txt", None),
        ("noextension", None),
        ("trailing.", None),
        (".jpg", None),  # a dotfile named jpg has no extension
        ("", None),
    ],
)
def test_reader_for_goes_by_extension(name: str, expected: str | None) -> None:
    reader = reader_for(Path("some") / name if name else Path(""))
    assert (reader.name if reader else None) == expected
    for ext in IMAGE_EXTENSIONS:
        assert reader_for(f"photo{ext}") is not None, ext
        assert reader_for(f"PHOTO{ext.upper()}") is not None, ext


def test_every_image_extension_agrees_with_the_core_types() -> None:
    for ext in IMAGE_EXTENSIONS:
        assert media_type_for(f"x{ext}") is not None
    assert reader_for("x.mp4") is None and media_type_for("x.mp4") is not None


def test_an_unavailable_reader_is_still_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pi_heif", None)
    monkeypatch.setitem(sys.modules, "pillow_heif", None)
    reader = reader_for("a.heic")
    assert reader is not None and reader.name == "heif" and reader.available() is False


def test_unavailable_extensions_follow_the_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ".jpg" not in unavailable_extensions()
    monkeypatch.setitem(sys.modules, "pi_heif", None)
    monkeypatch.setitem(sys.modules, "pillow_heif", None)
    assert unavailable_extensions() == HeifReader.extensions
    monkeypatch.setitem(sys.modules, "rawpy", None)
    assert unavailable_extensions() == HeifReader.extensions | RawReader.extensions


def test_explicit_readers_can_be_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rawpy", None)
    only_raw = [RawReader()]
    assert reader_for("a.jpg", only_raw) is None
    assert reader_for("a.DNG", only_raw) is only_raw[0]
    assert unavailable_extensions(only_raw) == RawReader.extensions
    assert unavailable_extensions([PillowReader()]) == frozenset()


def test_importing_the_package_does_not_import_the_optional_libraries(tmp_path: Path) -> None:
    import mediasift

    src = str(Path(mediasift.__file__).resolve().parents[1])
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import mediasift.readers as r; r.default_readers(); r.reader_for('a.heic');"
        "print(sorted(m for m in ('rawpy', 'pi_heif', 'pillow_heif') if m in sys.modules))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, src], capture_output=True, text=True, cwd=tmp_path, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"
