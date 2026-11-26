"""RawReader on the generated minimal DNG (and what happens without rawpy)."""

from __future__ import annotations

import contextlib
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from mediasift.core.errors import ReaderError
from mediasift.core.plugins import Reader
from mediasift.readers.image import ExifFields, apply_orientation
from mediasift.readers.raw import RawReader, TiffExif, _merge, _preview_orientation
from mediasift.readers.registry import unavailable_extensions
from tests.media import corrupt
from tests.media.dng import DngInfo, write_dng
from tests.readers.helpers import mean_diff

pytest.importorskip("rawpy")

READER = RawReader()
ORIENTATION_TAG = struct.pack("<HHIHH", 274, 3, 1, 1, 0)  # IFD0 entry: Orientation = 1


def _dng(path: Path, **kwargs: Any) -> DngInfo:
    return write_dng(path, **kwargs)


def _with_orientation(path: Path, orientation: int) -> Path:
    """Patch the generated DNG's IFD0 orientation entry in place."""
    data = bytearray(path.read_bytes())
    at = data.find(ORIENTATION_TAG)
    assert at > 0, "the DNG generator no longer writes an Orientation tag"
    data[at + 8 : at + 10] = struct.pack("<H", orientation)
    path.write_bytes(bytes(data))
    return path


# --------------------------------------------------------------------------- basics


def test_contract() -> None:
    reader: Reader = RawReader()
    assert reader.name == "raw" and reader.version >= 1 and reader.available()
    assert reader.extensions == {
        ".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf", ".pef", ".srw",
    }  # fmt: skip


def test_dng_metadata(tmp_path: Path) -> None:
    info = _dng(tmp_path / "a.dng", seed=3, make="MediaSift", model="Test Camera 9")
    meta = READER.read_metadata(info.path)
    assert (meta.width, meta.height) == (info.width, info.height) == (96, 64)
    assert (meta.camera_make, meta.camera_model) == ("MediaSift", "Test Camera 9")
    assert meta.taken_at == datetime(2021, 7, 4, 10, 30, 0) and meta.taken_source == "exif"
    assert meta.orientation == 1 and meta.codec == "RAW"
    assert meta.extra["exif_source"].startswith("tiff") and meta.extra["has_exif"] is True
    assert meta.extra["software"] == "mediasift tests"


def test_dng_pixels_come_from_the_preview_when_it_is_big_enough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _dng(tmp_path / "a.dng", seed=4)
    monkeypatch.setattr(
        RawReader, "_demosaic", staticmethod(lambda *a: pytest.fail("demosaic ran"))
    )
    img = READER.load_image(info.path, 48)
    assert img.mode == "RGB" and img.size == (48, 32)
    full = READER.load_image(info.path, 500)
    assert full.size == (96, 64)  # the preview is as big as the picture: never enlarged


def test_dng_without_preview_is_demosaiced(tmp_path: Path) -> None:
    with_preview = _dng(tmp_path / "p.dng", seed=5)
    without = _dng(tmp_path / "n.dng", seed=5, preview=False)
    big = READER.load_image(without.path, 500)
    assert big.mode == "RGB" and big.size == (96, 64)
    assert big.resize((1, 1)).getpixel((0, 0)) != (0, 0, 0)
    small = READER.load_image(without.path, 24)  # half size covers the request
    assert small.size == (24, 16)
    # Both routes show the same picture.
    ref = READER.load_image(with_preview.path, 48)
    assert mean_diff(READER.load_image(without.path, 48), ref) < 40
    meta = READER.read_metadata(without.path)
    assert (meta.width, meta.height) == (96, 64) and meta.camera_make == "MediaSift"


def test_half_size_decode_is_used_when_it_still_covers_the_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _dng(tmp_path / "n.dng", preview=False)
    seen: list[bool] = []
    real = RawReader._demosaic

    def spy(raw: Any, max_side: int, shown: tuple[int, int]) -> Image.Image:
        seen.append(max(shown) // 2 >= max_side)
        return real(raw, max_side, shown)

    monkeypatch.setattr(RawReader, "_demosaic", staticmethod(spy))
    READER.load_image(info.path, 40)
    READER.load_image(info.path, 80)
    assert seen == [True, False]


def test_arabic_path(tmp_path: Path) -> None:
    info = _dng(tmp_path / "صور" / "خام.dng")
    assert READER.read_metadata(info.path).width == 96
    assert READER.load_image(info.path, 48).size == (48, 32)
    assert READER.load_image(_dng(tmp_path / "ن" / "n.dng", preview=False).path, 48).size == (
        48,
        32,
    )


def test_the_extension_does_not_matter_to_libraw(tmp_path: Path) -> None:
    info = _dng(tmp_path / "a.dng")
    renamed = info.path.rename(tmp_path / "a.nef")
    assert READER.read_metadata(renamed).camera_model == "Test Camera 1"


# --------------------------------------------------------------------------- orientation


@pytest.mark.parametrize("with_preview", [True, False])
@pytest.mark.parametrize("orientation", [3, 6, 8])
def test_rotation_from_the_raw_applies_once(
    orientation: int, with_preview: bool, tmp_path: Path
) -> None:
    flat = _dng(tmp_path / "flat.dng", seed=2, preview=with_preview).path
    turned = _with_orientation(
        _dng(tmp_path / "t.dng", seed=2, preview=with_preview).path, orientation
    )
    meta = READER.read_metadata(turned)
    expected = (64, 96) if orientation in (6, 8) else (96, 64)
    assert (meta.width, meta.height) == expected and meta.orientation == orientation
    shown = READER.load_image(turned, 96)
    assert shown.size == expected
    reference = apply_orientation(READER.load_image(flat, 96), orientation)
    assert mean_diff(shown, reference) < (6 if with_preview else 12)


@pytest.mark.parametrize(
    ("own", "raw", "preview_size", "shown", "expected"),
    [
        (None, 6, (96, 64), (64, 96), 6),  # preview left unturned: use the raw's rotation
        (6, 6, (96, 64), (64, 96), 6),
        (1, 6, (96, 64), (64, 96), 6),  # claims upright but its shape says otherwise
        (None, 6, (64, 96), (64, 96), 1),  # the camera already turned the preview
        (None, 1, (96, 64), (96, 64), 1),
        (8, 6, (96, 64), (64, 96), 8),  # its own EXIF agrees with the shape: trusted
        (None, 6, (100, 100), (64, 96), 6),
    ],
)
def test_preview_orientation_choice(
    own: int | None,
    raw: int,
    preview_size: tuple[int, int],
    shown: tuple[int, int],
    expected: int,
) -> None:
    assert _preview_orientation(own, raw, preview_size, shown) == expected


# --------------------------------------------------------------------------- degrading


def test_non_tiff_raw_gives_the_size_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CR3 and RAF are not TIFF containers: simulate by hiding the TIFF structure."""
    info = _dng(tmp_path / "a.cr3", preview=False)
    monkeypatch.setattr(TiffExif, "read", classmethod(lambda cls, fp: None))
    meta = READER.read_metadata(info.path)
    assert (meta.width, meta.height) == (96, 64)
    assert meta.camera_make is None and meta.camera_model is None and meta.taken_at is None
    assert meta.taken_source is None and meta.extra["exif_source"] == "none"
    assert meta.extra["has_exif"] is False and meta.codec == "RAW"


def test_tiff_reader_declines_other_containers(tmp_path: Path) -> None:
    for name, data in {
        "short": b"II*",
        "orf": b"IIRO\x08\x00\x00\x00" + b"\x00" * 20,
        "cr3": b"\x00\x00\x00\x18ftypcrx " + b"\x00" * 20,
        "zero_offset": b"II*\x00\x00\x00\x00\x00" + b"\x00" * 20,
    }.items():
        path = tmp_path / name
        path.write_bytes(data)
        with path.open("rb") as fp:
            assert TiffExif.read(fp) is None, name


def test_missing_fields_are_filled_from_the_preview_exif() -> None:
    primary = ExifFields(make="A", orientation=6)
    fallback = ExifFields(
        make="B", model="M", taken_at=datetime(2020, 1, 1), software="S", has_exif=True
    )
    merged = _merge(primary, fallback)
    assert (merged.make, merged.model, merged.orientation) == ("A", "M", 6)
    assert merged.taken_at == datetime(2020, 1, 1) and merged.software == "S"
    assert merged.has_exif is True and merged.gps is None


def test_preview_exif_is_used_when_the_tiff_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _dng(tmp_path / "a.dng")
    monkeypatch.setattr(TiffExif, "read", classmethod(lambda cls, fp: None))
    monkeypatch.setattr(
        RawReader,
        "_preview_exif",
        staticmethod(lambda raw, lib: ExifFields(make="FromPreview", has_exif=True)),
    )
    meta = READER.read_metadata(info.path)
    assert meta.camera_make == "FromPreview" and meta.extra["exif_source"] == "preview"


# --------------------------------------------------------------------------- damage


def test_empty_garbage_and_unknown(tmp_path: Path) -> None:
    cases = {
        "empty.dng": (b"", "empty"),
        "garbage.dng": (corrupt.garbage(4000), "unsupported"),
        "text.cr2": (b"hello" * 500, "unsupported"),
    }
    for name, (data, kind) in cases.items():
        path = tmp_path / name
        path.write_bytes(data)
        for call in (READER.read_metadata, lambda p: READER.load_image(p, 64)):
            with pytest.raises(ReaderError) as info:
                call(path)
            assert info.value.kind == kind, name


def test_truncated_raw_without_preview_says_truncated(tmp_path: Path) -> None:
    info = _dng(tmp_path / "a.dng", preview=False)
    data = info.path.read_bytes()
    path = tmp_path / "t.dng"
    path.write_bytes(corrupt.truncated(data, 0.5))
    assert READER.read_metadata(path).width == 96  # the header is intact
    with pytest.raises(ReaderError) as info_:
        READER.load_image(path, 64)
    assert info_.value.kind == "truncated"


def test_truncated_raw_keeps_its_preview(tmp_path: Path) -> None:
    info = _dng(tmp_path / "a.dng")
    path = tmp_path / "t.dng"
    path.write_bytes(corrupt.truncated(info.path.read_bytes(), 0.6))
    assert READER.load_image(path, 48).size == (48, 32)  # the camera's own preview survives


def test_pixel_limit_comes_before_decoding(tmp_path: Path) -> None:
    path = tmp_path / "t.dng"
    path.write_bytes(
        corrupt.truncated(_dng(tmp_path / "a.dng", preview=False).path.read_bytes(), 0.5)
    )
    with pytest.raises(ReaderError) as info:
        RawReader(max_pixels=1000).load_image(path, 64)
    assert info.value.kind == "too_large"
    with pytest.raises(ValueError, match="max_pixels"):
        RawReader(max_pixels=0)
    with pytest.raises(ValueError, match="max_side"):
        READER.load_image(path, 0)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ReaderError) as info:
        READER.read_metadata(tmp_path / "nope.dng")
    assert info.value.kind == "io"


@pytest.mark.parametrize("case", ["good", "truncated", "garbage", "empty"])
def test_file_can_be_renamed_right_after_use(case: str, tmp_path: Path) -> None:
    good = _dng(tmp_path / "good.dng", preview=False).path.read_bytes()
    data = {
        "good": good,
        "truncated": corrupt.truncated(good, 0.5),
        "garbage": corrupt.garbage(500),
        "empty": b"",
    }[case]
    path = tmp_path / "x.dng"
    path.write_bytes(data)
    for call in (READER.read_metadata, lambda p: READER.load_image(p, 64)):
        with contextlib.suppress(ReaderError):  # only the release of the file matters here
            call(path)
        moved = path.rename(tmp_path / "y.dng")
        moved.rename(path)


# --------------------------------------------------------------------------- libraries


def test_missing_rawpy_makes_the_reader_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _dng(tmp_path / "a.dng")
    assert READER.available()
    monkeypatch.setitem(sys.modules, "rawpy", None)  # an import now raises ImportError
    assert READER.available() is False
    assert RawReader.extensions <= unavailable_extensions()
    for call in (READER.read_metadata, lambda p: READER.load_image(p, 64)):
        with pytest.raises(ReaderError) as error:
            call(info.path)
        assert error.value.kind == "unsupported" and "rawpy" in str(error.value)


def test_threads(tmp_path: Path) -> None:
    paths = [_dng(tmp_path / f"{i}.dng", seed=i, preview=bool(i % 2)).path for i in range(1, 5)]
    paths.append(_with_orientation(_dng(tmp_path / "r.dng", seed=9).path, 6))
    bad = tmp_path / "bad.dng"
    bad.write_bytes(corrupt.garbage(900))
    paths.append(bad)

    def work(path: Path) -> tuple[Any, ...]:
        try:
            meta = READER.read_metadata(path)
            img = READER.load_image(path, 48)
            return meta.width, meta.height, meta.camera_model, img.size, img.tobytes()
        except ReaderError as exc:
            return ("error", exc.kind)

    serial = [work(p) for p in paths]
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert list(pool.map(work, paths * 4)) == serial * 4
