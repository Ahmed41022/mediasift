"""HeifReader: generated HEIC files, orientation, damage, and a missing library."""

from __future__ import annotations

import contextlib
import io
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from mediasift.core.errors import ReaderError
from mediasift.core.plugins import Reader
from mediasift.readers import heif as heif_module
from mediasift.readers.heif import HeifReader, has_container_transform
from mediasift.readers.registry import unavailable_extensions
from tests.media import corrupt, images
from tests.media.images import requires_heif
from tests.readers.helpers import mean_diff

pytestmark = requires_heif

READER = HeifReader()
TAKEN = datetime(2021, 7, 4, 10, 30, 15)
SIZE = (160, 120)


@pytest.fixture(scope="module")
def picture() -> Image.Image:
    return images.scene(7, SIZE)


def _thumb(src: Image.Image, side: int) -> Image.Image:
    scale = side / max(src.size)
    return src.resize((round(src.width * scale), round(src.height * scale)), Image.Resampling.BOX)


def _exif_only(picture: Image.Image, orientation: int, path: Path) -> Path:
    """HEIC with the orientation in EXIF only (the kit's saver): nothing in the container."""
    stored = images.stored_for_orientation(picture, orientation)
    return images.save_heic(stored, path, exif=images.build_exif(orientation=orientation))


def _with_container_rotation(picture: Image.Image, orientation: int, path: Path) -> Path:
    """HEIC like a phone writes it: ``irot``/``imir`` in the container plus the EXIF value."""
    stored = images.stored_for_orientation(picture, orientation)
    exif = images.build_exif(orientation=orientation).tobytes()
    path.write_bytes(images.to_bytes(stored, "HEIF", quality=80, exif=exif))
    return path


# --------------------------------------------------------------------------- basics


def test_contract() -> None:
    reader: Reader = HeifReader()
    assert reader.name == "heif" and reader.version >= 1 and reader.available()
    assert reader.extensions == {".heic", ".heif", ".hif"}


def test_metadata_and_pixels(tmp_path: Path, picture: Image.Image) -> None:
    exif = images.build_exif(
        taken=TAKEN, offset="-04:00", make="MediaSift", model="Phone 1", gps=(48.85, 2.35)
    )
    path = images.save_heic(picture, tmp_path / "a.heic", exif=exif)
    meta = READER.read_metadata(path)
    assert (meta.width, meta.height) == SIZE and meta.codec == "HEIF"
    assert meta.taken_at == TAKEN and meta.taken_source == "exif"
    assert meta.extra["utc_offset"] == "-04:00"
    assert (meta.camera_make, meta.camera_model) == ("MediaSift", "Phone 1")
    assert meta.gps_lat == pytest.approx(48.85, abs=1e-4)
    assert meta.gps_lon == pytest.approx(2.35, abs=1e-4)
    assert meta.extra["has_exif"] is True and meta.extra["heif_transform"] is False

    full = READER.load_image(path, 500)
    assert full.mode == "RGB" and full.size == SIZE and mean_diff(full, picture) < 6
    small = READER.load_image(path, 40)
    assert small.size == (40, 30) and mean_diff(small, _thumb(picture, 40)) < 8


def test_without_exif(tmp_path: Path, picture: Image.Image) -> None:
    path = images.save_heic(picture, tmp_path / "a.heic")
    meta = READER.read_metadata(path)
    assert (meta.width, meta.height) == SIZE
    assert meta.taken_at is None and meta.orientation is None and meta.camera_make is None
    assert meta.extra["has_exif"] is False


def test_alpha_goes_on_white(tmp_path: Path) -> None:
    rgba = Image.new("RGBA", (64, 32), (0, 0, 255, 0))
    rgba.paste((255, 0, 0, 255), (32, 0, 64, 32))
    path = images.save_heic(rgba, tmp_path / "a.heic", quality=95)
    out = READER.load_image(path, 64)
    left, right = out.getpixel((8, 16)), out.getpixel((56, 16))
    assert isinstance(left, tuple) and isinstance(right, tuple)
    assert min(left) > 245 and right[0] > 235 and right[1] < 25


def test_arabic_path(tmp_path: Path, picture: Image.Image) -> None:
    path = images.save_heic(picture, tmp_path / "صور" / "صورة.heic")
    assert READER.read_metadata(path).width == 160
    assert READER.load_image(path, 40).size == (40, 30)


# --------------------------------------------------------------------------- orientation


@pytest.mark.parametrize("orientation", range(1, 9))
def test_orientation_only_in_exif_is_applied_once(
    orientation: int, tmp_path: Path, picture: Image.Image
) -> None:
    path = _exif_only(picture, orientation, tmp_path / "a.heic")
    meta = READER.read_metadata(path)
    assert meta.extra["heif_transform"] is False
    assert meta.orientation == (orientation if orientation != 1 else 1)
    assert (meta.width, meta.height) == SIZE
    shown = READER.load_image(path, 40)
    assert shown.size == (40, 30)
    assert mean_diff(shown, _thumb(picture, 40)) < 8


@pytest.mark.parametrize("orientation", range(2, 9))
def test_container_rotation_is_not_applied_twice(
    orientation: int, tmp_path: Path, picture: Image.Image
) -> None:
    path = _with_container_rotation(picture, orientation, tmp_path / "a.heic")
    data = path.read_bytes()
    assert b"irot" in data or b"imir" in data, "the generator no longer writes a transform"
    meta = READER.read_metadata(path)
    assert meta.extra["heif_transform"] is True
    assert meta.orientation == orientation  # the EXIF value is still reported as it is
    assert (meta.width, meta.height) == SIZE  # libheif's size is already the display size
    shown = READER.load_image(path, 40)
    assert shown.size == (40, 30)
    assert mean_diff(shown, _thumb(picture, 40)) < 8


def test_exif_orientation_is_ignored_when_the_container_has_a_transform(
    tmp_path: Path, picture: Image.Image, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _exif_only(picture, 6, tmp_path / "a.heic")  # stored 120x160, EXIF says 6
    assert READER.load_image(path, 200).size == (160, 120)
    monkeypatch.setattr(heif_module, "has_container_transform", lambda fp: True)
    stored_shape = READER.load_image(path, 200)
    assert stored_shape.size == (120, 160)  # untouched: the container is trusted
    meta = READER.read_metadata(path)
    assert (meta.width, meta.height) == (120, 160)


def _box(kind: bytes, payload: bytes = b"", large: bool = False) -> bytes:
    if large:
        return struct.pack(">I4sQ", 1, kind, 16 + len(payload)) + payload
    return struct.pack(">I4s", 8 + len(payload), kind) + payload


def _heic_like(*properties: bytes, mdat_first: bool = False, large: bool = False) -> bytes:
    ipco = _box(b"ipco", b"".join(properties))
    meta = _box(
        b"meta", b"\x00\x00\x00\x00" + _box(b"hdlr", b"\x00" * 24) + _box(b"iprp", ipco), large
    )
    mdat = _box(b"mdat", b"\x00" * 100)
    ftyp = _box(b"ftyp", b"heic\x00\x00\x00\x00mif1")
    return ftyp + (mdat + meta if mdat_first else meta + mdat)


@pytest.mark.parametrize("box", [b"irot", b"imir"])
@pytest.mark.parametrize("mdat_first", [False, True])
@pytest.mark.parametrize("large", [False, True])
def test_transform_boxes_are_found(box: bytes, mdat_first: bool, large: bool) -> None:
    ispe = _box(b"ispe", b"\x00" * 12)
    data = _heic_like(ispe, _box(box, b"\x01"), mdat_first=mdat_first, large=large)
    assert has_container_transform(io.BytesIO(data)) is True
    assert has_container_transform(io.BytesIO(_heic_like(ispe, mdat_first=mdat_first))) is False


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00\x00\x00",
        b"\x00" * 64,
        b"x" * 100,
        struct.pack(">I4s", 3, b"meta"),
        _box(b"meta", b"\x00\x00"),
        _box(b"ftyp") + struct.pack(">I4s", 0, b"mdat"),
    ],
)
def test_transform_probe_survives_junk(data: bytes) -> None:
    assert has_container_transform(io.BytesIO(data)) is False


# --------------------------------------------------------------------------- damage


def test_empty_garbage_and_header_only(tmp_path: Path, picture: Image.Image) -> None:
    good = images.save_heic(picture, tmp_path / "good.heic").read_bytes()
    cases = {
        "empty": (b"", "empty"),
        "garbage": (corrupt.garbage(3000), "unsupported"),
        "header": (corrupt.header_only(good, 24), "truncated"),
    }
    for name, (data, kind) in cases.items():
        path = tmp_path / f"{name}.heic"
        path.write_bytes(data)
        for call in (READER.read_metadata, lambda p: READER.load_image(p, 64)):
            with pytest.raises(ReaderError) as info:
                call(path)
            assert info.value.kind == kind, name


@pytest.mark.parametrize("fraction", [0.3, 0.7])
def test_truncated(fraction: float, tmp_path: Path, picture: Image.Image) -> None:
    good = images.save_heic(picture, tmp_path / "good.heic").read_bytes()
    path = tmp_path / "t.heic"
    path.write_bytes(corrupt.truncated(good, fraction))
    try:
        READER.read_metadata(path)
    except ReaderError as exc:
        assert exc.kind in {"truncated", "corrupt"}
    with pytest.raises(ReaderError) as info:
        READER.load_image(path, 64)
    assert info.value.kind == "truncated"


def test_pixel_limit_comes_before_decoding(tmp_path: Path, picture: Image.Image) -> None:
    good = images.save_heic(picture, tmp_path / "good.heic").read_bytes()
    path = tmp_path / "t.heic"
    path.write_bytes(corrupt.truncated(good, 0.7))  # decoding would say "truncated"
    with pytest.raises(ReaderError) as info:
        HeifReader(max_pixels=1000).load_image(path, 64)
    assert info.value.kind == "too_large"
    with pytest.raises(ValueError, match="max_pixels"):
        HeifReader(max_pixels=0)
    with pytest.raises(ValueError, match="max_side"):
        READER.load_image(path, 0)


def test_missing_file_is_an_io_error(tmp_path: Path) -> None:
    with pytest.raises(ReaderError) as info:
        READER.read_metadata(tmp_path / "nope.heic")
    assert info.value.kind == "io"


@pytest.mark.parametrize("case", ["good", "truncated", "garbage", "empty"])
def test_file_can_be_renamed_right_after_use(
    case: str, tmp_path: Path, picture: Image.Image
) -> None:
    good = images.save_heic(picture, tmp_path / "good.heic").read_bytes()
    data = {
        "good": good,
        "truncated": corrupt.truncated(good, 0.5),
        "garbage": corrupt.garbage(500),
        "empty": b"",
    }[case]
    path = tmp_path / "x.heic"
    path.write_bytes(data)
    for call in (READER.read_metadata, lambda p: READER.load_image(p, 64)):
        with contextlib.suppress(ReaderError):  # only the release of the file matters here
            call(path)
        moved = path.rename(tmp_path / "y.heic")
        moved.rename(path)


# --------------------------------------------------------------------------- libraries


def test_missing_library_makes_the_reader_unavailable(
    tmp_path: Path, picture: Image.Image, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = images.save_heic(picture, tmp_path / "a.heic")
    assert READER.available()
    monkeypatch.setitem(sys.modules, "pi_heif", None)  # an import now raises ImportError
    assert READER.available()  # pillow_heif can stand in
    monkeypatch.setitem(sys.modules, "pillow_heif", None)
    assert READER.available() is False
    assert {".heic", ".heif", ".hif"} <= unavailable_extensions()
    for call in (READER.read_metadata, lambda p: READER.load_image(p, 64)):
        with pytest.raises(ReaderError) as info:
            call(path)
        assert info.value.kind == "unsupported" and "pi-heif" in str(info.value)


def test_a_library_that_fails_to_import_counts_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    real = importlib.import_module

    def broken(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("pi_heif", "pillow_heif"):
            raise OSError("the native library is damaged")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", broken)
    assert READER.available() is False


def test_reading_registers_nothing_with_pillow(tmp_path: Path, picture: Image.Image) -> None:
    """In a fresh interpreter, HEIC reading must not install a global Pillow opener."""
    path = images.save_heic(picture, tmp_path / "a.heic")
    import mediasift

    src = str(Path(mediasift.__file__).resolve().parents[1])
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from pathlib import Path; from PIL import Image;"
        "from mediasift.readers.heif import HeifReader;"
        "r = HeifReader(); p = Path(sys.argv[2]);"
        "print(r.read_metadata(p).width, r.load_image(p, 32).size, 'HEIF' in Image.OPEN);"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, src, str(path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["160", "(32,", "24)", "False"]


def test_threads(tmp_path: Path, picture: Image.Image) -> None:
    paths = [_exif_only(picture, o, tmp_path / f"e{o}.heic") for o in (1, 6, 8)]
    paths += [_with_container_rotation(picture, o, tmp_path / f"c{o}.heic") for o in (3, 6)]
    bad = tmp_path / "bad.heic"
    bad.write_bytes(corrupt.garbage(300))
    paths.append(bad)

    def work(path: Path) -> tuple[Any, ...]:
        try:
            meta = READER.read_metadata(path)
            img = READER.load_image(path, 48)
            return meta.width, meta.height, img.size, img.tobytes()
        except ReaderError as exc:
            return ("error", exc.kind)

    serial = [work(p) for p in paths]
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert list(pool.map(work, paths * 4)) == serial * 4
