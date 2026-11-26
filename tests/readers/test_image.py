"""PillowReader: every format, EXIF, orientation, colour modes, damage, limits, threads."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageFile, TiffImagePlugin

from mediasift.core.errors import ReaderError
from mediasift.core.plugins import Reader
from mediasift.readers.image import PillowReader
from tests.media import corrupt, images
from tests.readers.helpers import jpeg_claiming, mean_diff, png_claiming

# pytest restores the warning filters per test, so the ones set at import are gone here.
pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning"),
]

READER = PillowReader()
TAKEN = datetime(2021, 7, 4, 10, 30, 15)


def _thumb(src: Image.Image, side: int) -> Image.Image:
    """What a correct reader returns for ``src``."""
    scale = side / max(src.size)
    return src.convert("RGB").resize(
        (round(src.width * scale), round(src.height * scale)), Image.Resampling.LANCZOS
    )


# --------------------------------------------------------------------------- contract


def test_reader_contract() -> None:
    reader: Reader = PillowReader()
    assert reader.name == "pillow" and reader.version >= 1 and reader.available()
    assert reader.extensions == {
        ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff",
    }  # fmt: skip


def test_bad_arguments_are_programming_errors(tmp_path: Path, small_scene: Image.Image) -> None:
    path = images.save_png(small_scene, tmp_path / "a.png")
    with pytest.raises(ValueError, match="max_side"):
        READER.load_image(path, 0)
    with pytest.raises(ValueError, match="max_pixels"):
        PillowReader(max_pixels=0)


# --------------------------------------------------------------------------- formats

SAVERS: dict[str, tuple[Callable[[Image.Image, Path], Path], str]] = {
    "jpeg": (lambda im, p: images.save_jpeg(im, p / "a.jpg", quality=95), "JPEG"),
    "png": (lambda im, p: images.save_png(im, p / "a.png"), "PNG"),
    "webp": (lambda im, p: images.save_webp(im, p / "a.webp", quality=95), "WEBP"),
    "gif": (lambda im, p: images.save_gif(im, p / "a.gif"), "GIF"),
    "bmp": (lambda im, p: images.save_bmp(im, p / "a.bmp"), "BMP"),
    "tiff": (lambda im, p: images.save_tiff(im, p / "a.tif"), "TIFF"),
}


@pytest.mark.parametrize("fmt", SAVERS)
def test_every_format_reads_metadata_and_pixels(
    fmt: str, tmp_path: Path, small_scene: Image.Image
) -> None:
    save, codec = SAVERS[fmt]
    path = save(small_scene, tmp_path)
    meta = READER.read_metadata(path)
    assert (meta.width, meta.height) == (320, 240)
    assert meta.codec == codec
    assert meta.taken_at is None and meta.camera_make is None and meta.gps_lat is None
    assert meta.extra["n_frames"] == 1 and meta.extra["has_exif"] is False
    assert meta.extra["has_icc"] is False and isinstance(meta.extra["mode"], str)

    full = READER.load_image(path, 1000)
    assert full.mode == "RGB" and full.size == (320, 240)
    assert mean_diff(full, small_scene) < (14 if fmt == "gif" else 4)
    small = READER.load_image(path, 80)
    assert small.mode == "RGB" and small.size == (80, 60)
    assert mean_diff(small, _thumb(small_scene, 80)) < (16 if fmt == "gif" else 5)


def test_a_small_image_is_never_enlarged(tmp_path: Path, small_scene: Image.Image) -> None:
    path = images.save_png(small_scene, tmp_path / "a.png")
    assert READER.load_image(path, 5000).size == (320, 240)


def test_reduction_is_exact_for_odd_ratios(tmp_path: Path) -> None:
    path = images.save_png(Image.new("RGB", (301, 199), (10, 20, 30)), tmp_path / "a.png")
    got = READER.load_image(path, 100)
    assert got.size == (100, 66)


def test_jpeg_uses_the_decoders_draft_mode(
    tmp_path: Path, small_scene: Image.Image, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import JpegImagePlugin

    calls: list[tuple[str | None, tuple[int, int] | None]] = []
    real = JpegImagePlugin.JpegImageFile.draft

    def spy(self: Any, mode: Any, size: Any) -> Any:
        calls.append((mode, size))
        return real(self, mode, size)

    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "draft", spy)
    path = images.save_jpeg(small_scene, tmp_path / "a.jpg")
    got = READER.load_image(path, 80)
    assert calls == [("RGB", (80, 60))]
    assert got.size == (80, 60)
    calls.clear()
    READER.load_image(path, 400)  # not smaller than the file: no draft needed
    assert calls == []


def test_jpeg_quality_estimate(tmp_path: Path, small_scene: Image.Image) -> None:
    for quality in (30, 60, 85, 95):
        path = images.save_jpeg(small_scene, tmp_path / f"q{quality}.jpg", quality=quality)
        got = READER.read_metadata(path).extra["jpeg_quality_estimate"]
        assert abs(got - quality) <= 3, (quality, got)
    png = images.save_png(small_scene, tmp_path / "a.png")
    assert "jpeg_quality_estimate" not in READER.read_metadata(png).extra


def test_wrong_extension_still_works(tmp_path: Path, small_scene: Image.Image) -> None:
    as_jpg = tmp_path / "really_png.jpg"
    as_jpg.write_bytes(images.to_bytes(small_scene, "PNG"))
    meta = READER.read_metadata(as_jpg)
    assert meta.codec == "PNG" and (meta.width, meta.height) == (320, 240)
    assert READER.load_image(as_jpg, 64).size == (64, 48)
    as_png = tmp_path / "really_jpeg.png"
    as_png.write_bytes(images.to_bytes(small_scene, "JPEG"))
    assert READER.read_metadata(as_png).codec == "JPEG"


def test_arabic_and_unusual_paths(tmp_path: Path, small_scene: Image.Image) -> None:
    folder = tmp_path / "صور العائلة" / "رحلة 2021"
    path = images.save_jpeg(
        small_scene, folder / "صورة جميلة.jpg", exif=images.build_exif(make="Canon")
    )
    meta = READER.read_metadata(path)
    assert meta.camera_make == "Canon" and meta.width == 320
    assert READER.load_image(path, 64).size == (64, 48)
    odd = images.save_png(small_scene, tmp_path / "a b  (1) #&%.png")
    assert READER.read_metadata(odd).width == 320


# --------------------------------------------------------------------------- EXIF


def test_exif_facts_survive_in_every_container(tmp_path: Path, small_scene: Image.Image) -> None:
    exif = images.build_exif(
        taken=TAKEN, offset="+02:00", make="MediaSift", model="Test 1", gps=(-33.85, 151.2)
    )
    for name, fmt in (("a.jpg", "JPEG"), ("a.png", "PNG"), ("a.webp", "WEBP")):
        path = tmp_path / name
        path.write_bytes(images.to_bytes(small_scene, fmt, exif=exif))
        meta = READER.read_metadata(path)
        assert meta.taken_at == TAKEN and meta.taken_at.tzinfo is None, name
        assert meta.taken_source == "exif", name
        assert meta.extra["utc_offset"] == "+02:00", name
        assert (meta.camera_make, meta.camera_model) == ("MediaSift", "Test 1"), name
        assert meta.gps_lat == pytest.approx(-33.85, abs=1e-4), name
        assert meta.gps_lon == pytest.approx(151.2, abs=1e-4), name
        assert meta.extra["has_exif"] is True, name


def test_tiff_keeps_make_and_model(tmp_path: Path, small_scene: Image.Image) -> None:
    # Pillow writes only first-IFD tags into a TIFF, so that is all there is to read back.
    path = tmp_path / "a.tif"
    path.write_bytes(
        images.to_bytes(small_scene, "TIFF", exif=images.build_exif(make="A", model="B"))
    )
    meta = READER.read_metadata(path)
    assert (meta.camera_make, meta.camera_model, meta.extra["has_exif"]) == ("A", "B", True)


def test_taken_without_offset_has_no_offset_entry(tmp_path: Path, small_scene: Image.Image) -> None:
    path = images.save_jpeg(small_scene, tmp_path / "a.jpg", exif=images.build_exif(taken=TAKEN))
    meta = READER.read_metadata(path)
    assert meta.taken_at == TAKEN and "utc_offset" not in meta.extra


def test_software_and_icc_are_reported(tmp_path: Path, small_scene: Image.Image) -> None:
    exif = Image.Exif()
    exif[0x0131] = "Editor 1.0"
    path = images.save_jpeg(small_scene, tmp_path / "a.jpg", exif=exif)
    assert READER.read_metadata(path).extra["software"] == "Editor 1.0"
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    withicc = tmp_path / "icc.png"
    withicc.write_bytes(images.to_bytes(small_scene, "PNG", icc_profile=icc))
    assert READER.read_metadata(withicc).extra["has_icc"] is True


def test_exif_date_fallbacks_in_a_real_file(tmp_path: Path, small_scene: Image.Image) -> None:
    exif = Image.Exif()
    exif[0x0132] = "2019:05:05 05:05:05"  # DateTime in the first IFD
    sub = exif.get_ifd(0x8769)
    sub[0x9003] = "0000:00:00 00:00:00"
    sub[0x9004] = "2020:06:06 06:06:06"
    path = images.save_jpeg(small_scene, tmp_path / "a.jpg", exif=exif)
    assert READER.read_metadata(path).taken_at == datetime(2020, 6, 6, 6, 6, 6)
    sub[0x9004] = "0000:00:00 00:00:00"
    path = images.save_jpeg(small_scene, tmp_path / "b.jpg", exif=exif)
    assert READER.read_metadata(path).taken_at == datetime(2019, 5, 5, 5, 5, 5)
    exif[0x0132] = "0000:00:00 00:00:00"
    path = images.save_jpeg(small_scene, tmp_path / "c.jpg", exif=exif)
    meta = READER.read_metadata(path)
    assert meta.taken_at is None and meta.taken_source is None and meta.width == 320


def test_gps_with_a_zero_denominator_is_ignored(tmp_path: Path, small_scene: Image.Image) -> None:
    rat = TiffImagePlugin.IFDRational
    exif = images.build_exif(make="Cam", taken=TAKEN)
    gps = exif.get_ifd(0x8825)
    gps[1], gps[2] = "N", (rat(40, 1), rat(1, 0), rat(0, 1))
    gps[3], gps[4] = "E", (rat(10, 1), rat(0, 1), rat(0, 1))
    path = images.save_jpeg(small_scene, tmp_path / "a.jpg", exif=exif)
    meta = READER.read_metadata(path)
    assert meta.gps_lat is None and meta.gps_lon is None
    assert meta.camera_make == "Cam" and meta.taken_at == TAKEN


def test_make_and_model_are_cleaned(tmp_path: Path, small_scene: Image.Image) -> None:
    exif = images.build_exif(make="  Canon\x00", model="EOS\x00\x00 ")
    path = images.save_jpeg(small_scene, tmp_path / "a.jpg", exif=exif)
    meta = READER.read_metadata(path)
    assert (meta.camera_make, meta.camera_model) == ("Canon", "EOS")


def _mutate_exif(data: bytes, mutate: Callable[[bytearray, int], None]) -> bytes:
    out = bytearray(data)
    tiff = out.index(b"Exif\x00\x00") + 6
    mutate(out, tiff)
    return bytes(out)


def _huge_ifd_offset(out: bytearray, tiff: int) -> None:
    out[tiff + 4 : tiff + 8] = b"\xff\xff\xff\x7f"


def _ifd_noise(out: bytearray, tiff: int) -> None:
    out[tiff + 8 : tiff + 8 + 60] = b"\xff" * 60


def _bad_byte_order(out: bytearray, tiff: int) -> None:
    out[tiff : tiff + 2] = b"ZZ"


def _wild_pointers(out: bytearray, tiff: int) -> None:
    for i in range(tiff + 10, tiff + 10 + 12 * 3, 12):
        out[i + 8 : i + 12] = b"\xff\xff\xff\x00"  # value offsets far outside the segment


@pytest.mark.parametrize("mutate", [_huge_ifd_offset, _ifd_noise, _bad_byte_order, _wild_pointers])
def test_broken_exif_never_raises(
    mutate: Callable[[bytearray, int], None], tmp_path: Path, small_scene: Image.Image
) -> None:
    exif = images.build_exif(taken=TAKEN, make="Cam", model="M" * 40, gps=(10.5, 20.5))
    good = images.to_bytes(small_scene, "JPEG", exif=exif)
    path = tmp_path / "a.jpg"
    path.write_bytes(_mutate_exif(good, mutate))
    meta = READER.read_metadata(path)
    assert (meta.width, meta.height) == (320, 240) and meta.codec == "JPEG"
    assert meta.taken_source in (None, "exif")
    assert READER.load_image(path, 64).size == (64, 48)


def test_exif_segment_shorter_than_its_tags(tmp_path: Path, small_scene: Image.Image) -> None:
    exif = images.build_exif(taken=TAKEN, make="Cam", gps=(10.5, 20.5))
    fake = bytearray(images.to_bytes(small_scene, "JPEG", exif=exif))
    start = fake.index(b"Exif\x00\x00")
    fake[start - 2 : start] = (20).to_bytes(2, "big")  # the APP1 segment ends inside the IFD
    path = tmp_path / "a.jpg"
    path.write_bytes(bytes(fake))
    try:
        meta = READER.read_metadata(path)
    except ReaderError as exc:  # Pillow may refuse the whole file; that is the only allowed way out
        assert exc.kind in {"corrupt", "truncated"}
    else:
        assert meta.codec == "JPEG" and (meta.width, meta.height) == (320, 240)


def test_missing_exif_leaves_fields_empty(tmp_path: Path, small_scene: Image.Image) -> None:
    path = images.save_jpeg(small_scene, tmp_path / "a.jpg")
    meta = READER.read_metadata(path)
    assert meta.orientation is None and meta.taken_at is None and meta.taken_source is None
    assert meta.camera_make is None and meta.camera_model is None
    assert meta.gps_lat is None and meta.gps_lon is None
    assert meta.extra["has_exif"] is False and "software" not in meta.extra


# --------------------------------------------------------------------------- orientation


def test_all_eight_orientations_display_identically(
    tmp_path: Path, small_scene: Image.Image
) -> None:
    reference = READER.load_image(
        images.save_jpeg(small_scene, tmp_path / "ref.jpg", quality=92), 64
    )
    assert reference.size == (64, 48)
    for orientation, data in images.all_orientations(small_scene).items():
        path = tmp_path / f"o{orientation}.jpg"
        path.write_bytes(data)
        meta = READER.read_metadata(path)
        assert meta.orientation == orientation
        assert (meta.width, meta.height) == (320, 240), orientation
        shown = READER.load_image(path, 64)
        assert shown.size == (64, 48), orientation
        assert mean_diff(shown, reference) < 4, orientation


def test_orientation_in_png_and_tiff(tmp_path: Path, small_scene: Image.Image) -> None:
    stored = images.stored_for_orientation(small_scene, 6)
    for fmt, name in (("PNG", "a.png"), ("TIFF", "a.tif"), ("WEBP", "a.webp")):
        path = tmp_path / name
        path.write_bytes(images.to_bytes(stored, fmt, exif=images.build_exif(orientation=6)))
        meta = READER.read_metadata(path)
        assert (meta.width, meta.height, meta.orientation) == (320, 240, 6), fmt
        assert mean_diff(READER.load_image(path, 64), _thumb(small_scene, 64)) < 12, fmt


@pytest.mark.parametrize("value", [0, 9, 99])
def test_invalid_orientation_is_ignored(
    value: int, tmp_path: Path, small_scene: Image.Image
) -> None:
    exif = Image.Exif()
    exif[0x0112] = value
    path = images.save_jpeg(small_scene, tmp_path / "a.jpg", exif=exif)
    meta = READER.read_metadata(path)
    assert meta.orientation is None and (meta.width, meta.height) == (320, 240)
    assert READER.load_image(path, 64).size == (64, 48)


# --------------------------------------------------------------------------- colour modes


def _save(img: Image.Image, path: Path, **params: Any) -> Path:
    path.write_bytes(images.to_bytes(img, "PNG", **params))
    return path


def _px(img: Image.Image, x: int, y: int) -> tuple[int, ...]:
    value = img.getpixel((x, y))
    assert isinstance(value, tuple)
    return value


@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "RGB", "RGBA", "I;16"])
def test_every_png_mode_loads_as_rgb(mode: str, tmp_path: Path) -> None:
    base = Image.linear_gradient("L").resize((60, 40))
    img = (
        base.convert(mode)
        if mode != "I;16"
        else Image.fromarray(np.asarray(base).astype(np.uint16) * 257)
    )
    if mode == "P":
        img = base.convert("P")
    path = _save(img, tmp_path / "a.png")
    meta = READER.read_metadata(path)
    assert (meta.width, meta.height) == (60, 40)
    out = READER.load_image(path, 30)
    assert out.mode == "RGB" and out.size == (30, 20)


def test_sixteen_bit_is_scaled_not_clipped(tmp_path: Path) -> None:
    ramp = np.tile(np.linspace(0, 65535, 256).astype(np.uint16), (32, 1))
    path = _save(Image.fromarray(ramp), tmp_path / "ramp.png")
    assert READER.read_metadata(path).extra["mode"].startswith("I;16")
    out = np.asarray(READER.load_image(path, 256), dtype=np.int32)[:, :, 0]
    row = out[0]
    assert row[0] <= 1 and row[-1] >= 254
    assert abs(int(row[128]) - 128) <= 2  # clipping would saturate at 255 early
    assert np.all(np.diff(row) >= 0)
    dark = Image.fromarray(np.full((8, 8), 6000, dtype=np.uint16))
    out = READER.load_image(_save(dark, tmp_path / "dark.png"), 8)
    assert _px(out, 0, 0) == (23, 23, 23)  # 6000 / 257, a dark picture stays dark


def test_cmyk_jpeg_has_the_right_colours(tmp_path: Path) -> None:
    patches = Image.new("RGB", (64, 64))
    for i, colour in enumerate([(200, 30, 30), (30, 160, 40), (40, 40, 200), (230, 230, 40)]):
        patches.paste(Image.new("RGB", (32, 32), colour), ((i % 2) * 32, (i // 2) * 32))
    path = tmp_path / "cmyk.jpg"
    path.write_bytes(images.to_bytes(patches.convert("CMYK"), "JPEG", quality=95))
    meta = READER.read_metadata(path)
    assert meta.extra["mode"] == "CMYK" and (meta.width, meta.height) == (64, 64)
    out = READER.load_image(path, 64)
    assert out.mode == "RGB" and mean_diff(out, patches) < 8  # inverted CMYK would be far off


def test_palette_with_transparency_goes_on_white(tmp_path: Path) -> None:
    img = Image.new("P", (40, 20), 1)
    img.putpalette([0, 0, 255, 255, 0, 0] + [0, 0, 0] * 254)  # 0 blue (transparent), 1 red
    img.paste(0, (0, 0, 20, 20))
    path = _save(img, tmp_path / "p.png", transparency=0)
    assert "transparency" in Image.open(path).info
    out = READER.load_image(path, 40)
    assert _px(out, 5, 10) == (255, 255, 255)  # not the blue behind it
    assert _px(out, 30, 10) == (255, 0, 0)
    out = READER.load_image(path, 20)
    assert _px(out, 2, 5) == (255, 255, 255) and _px(out, 17, 5) == (255, 0, 0)


def test_gif_transparency_goes_on_white(tmp_path: Path) -> None:
    img = Image.new("P", (40, 20), 1)
    img.putpalette([0, 0, 0, 0, 128, 0] + [0, 0, 0] * 254)
    img.paste(0, (0, 0, 20, 20))
    path = tmp_path / "t.gif"
    path.write_bytes(images.to_bytes(img, "GIF", transparency=0))
    out = READER.load_image(path, 40)
    assert _px(out, 5, 10) == (255, 255, 255) and _px(out, 30, 10) == (0, 128, 0)


def test_rgba_and_la_are_composited_on_white(tmp_path: Path) -> None:
    rgba = Image.new("RGBA", (40, 20), (0, 0, 0, 255))
    rgba.paste((0, 0, 255, 0), (0, 0, 20, 20))  # transparent blue, must not bleed
    rgba.paste((0, 0, 0, 128), (20, 0, 30, 20))  # half transparent black
    out = READER.load_image(_save(rgba, tmp_path / "a.png"), 40)
    assert _px(out, 5, 10) == (255, 255, 255) and _px(out, 35, 10) == (0, 0, 0)
    assert abs(_px(out, 25, 10)[0] - 127) <= 2
    small = READER.load_image(_save(rgba, tmp_path / "b.png"), 10)
    assert min(_px(small, 1, 2)) >= 250  # reducing must not pull the blue in
    la = Image.new("LA", (40, 20), (0, 255))
    la.paste((0, 0), (0, 0, 20, 20))
    out = READER.load_image(_save(la, tmp_path / "la.png"), 40)
    assert _px(out, 5, 10) == (255, 255, 255) and _px(out, 30, 10) == (0, 0, 0)


def test_animated_gif_gives_its_first_frame(tmp_path: Path) -> None:
    colours = [(250, 0, 0), (0, 250, 0), (0, 0, 250)]
    frames = [
        Image.new("RGB", (32, 24), c).convert("P", palette=Image.Palette.ADAPTIVE) for c in colours
    ]
    path = tmp_path / "anim.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    meta = READER.read_metadata(path)
    assert meta.extra["n_frames"] == 3 and (meta.width, meta.height) == (32, 24)
    out = READER.load_image(path, 32)
    assert _px(out, 10, 10)[0] > 200 and _px(out, 10, 10)[1] < 50


# --------------------------------------------------------------------------- damaged files


def _jpeg_bytes(img: Image.Image) -> bytes:
    return images.to_bytes(img, "JPEG", quality=90)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF"])
@pytest.mark.parametrize("fraction", [0.4, 0.8])
def test_truncated_files(
    fmt: str, fraction: float, tmp_path: Path, small_scene: Image.Image
) -> None:
    data = images.to_bytes(small_scene.convert("P") if fmt == "GIF" else small_scene, fmt)
    path = tmp_path / "t.img"
    path.write_bytes(corrupt.truncated(data, fraction))
    try:
        READER.read_metadata(path)  # the header is intact: the size may be known
    except ReaderError as exc:
        assert exc.kind in {"truncated", "corrupt"}
    with pytest.raises(ReaderError) as info:
        READER.load_image(path, 64)
    assert info.value.kind in {"truncated", "corrupt"}


def test_truncated_jpeg_says_truncated(tmp_path: Path, small_scene: Image.Image) -> None:
    path = tmp_path / "t.jpg"
    path.write_bytes(corrupt.truncated(_jpeg_bytes(small_scene), 0.5))
    assert READER.read_metadata(path).width == 320
    with pytest.raises(ReaderError) as info:
        READER.load_image(path, 64)
    assert info.value.kind == "truncated"


def test_truncated_files_do_not_change_pillows_global_switch(
    tmp_path: Path, small_scene: Image.Image
) -> None:
    path = tmp_path / "t.jpg"
    path.write_bytes(corrupt.truncated(_jpeg_bytes(small_scene), 0.5))
    before = (ImageFile.LOAD_TRUNCATED_IMAGES, Image.MAX_IMAGE_PIXELS)
    with pytest.raises(ReaderError):
        READER.load_image(path, 64)
    assert (ImageFile.LOAD_TRUNCATED_IMAGES, Image.MAX_IMAGE_PIXELS) == before == (False, before[1])


@pytest.mark.parametrize("name", ["a.jpg", "a.png", "a.webp", "a.gif", "a.bmp", "a.tif"])
def test_empty_file(name: str, tmp_path: Path) -> None:
    path = corrupt.write_zero_bytes(tmp_path / name)
    for call in (READER.read_metadata, lambda p: READER.load_image(p, 64)):
        with pytest.raises(ReaderError) as info:
            call(path)
        assert info.value.kind == "empty"


def test_garbage_is_unsupported(tmp_path: Path) -> None:
    path = corrupt.write_garbage(tmp_path / "a.jpg")
    for call in (READER.read_metadata, lambda p: READER.load_image(p, 64)):
        with pytest.raises(ReaderError) as info:
            call(path)
        assert info.value.kind == "unsupported"


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "GIF", "BMP", "TIFF", "WEBP"])
def test_header_only_is_corrupt(fmt: str, tmp_path: Path, small_scene: Image.Image) -> None:
    data = images.to_bytes(small_scene.convert("P") if fmt == "GIF" else small_scene, fmt)
    path = tmp_path / "h.img"
    path.write_bytes(corrupt.header_only(data, 24))
    for call in (READER.read_metadata, lambda p: READER.load_image(p, 64)):
        with pytest.raises(ReaderError) as info:
            call(path)
        assert info.value.kind in {"corrupt", "truncated"}


def test_flipped_bits_give_a_picture_or_a_reader_error(
    tmp_path: Path, small_scene: Image.Image
) -> None:
    clean = images.to_bytes(small_scene, "PNG")
    for seed in range(12):
        path = tmp_path / f"b{seed}.png"
        path.write_bytes(corrupt.bitflips(clean, count=6, seed=seed))
        try:
            READER.read_metadata(path)
            READER.load_image(path, 64)
        except ReaderError as exc:
            assert exc.kind in {"corrupt", "truncated", "unsupported"}


def test_missing_file_and_directory_are_io_errors(tmp_path: Path) -> None:
    for target in (tmp_path / "nope.jpg", tmp_path):
        for call in (READER.read_metadata, lambda p: READER.load_image(p, 64)):
            with pytest.raises(ReaderError) as info:
                call(target)
            assert info.value.kind == "io"


# --------------------------------------------------------------------------- limits


def test_hand_written_header_claiming_100k_by_100k(tmp_path: Path) -> None:
    path = tmp_path / "bomb.png"
    path.write_bytes(png_claiming(100_000, 100_000))
    assert READER.read_metadata(path).width == 100_000  # from the header alone
    with pytest.raises(ReaderError) as info:
        READER.load_image(path, 64)
    assert info.value.kind == "too_large"


def test_a_200_megapixel_phone_photo_is_not_a_bomb(tmp_path: Path) -> None:
    path = tmp_path / "phone.jpg"
    path.write_bytes(jpeg_claiming(16_320, 12_240))
    assert READER.read_metadata(path).width == 16_320
    with pytest.raises(ReaderError) as info:  # only because there are no pixels behind it
        READER.load_image(path, 64)
    assert info.value.kind != "too_large"


def test_png_metadata_never_decodes_the_pixels(
    tmp_path: Path, small_scene: Image.Image, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = images.save_png(small_scene, tmp_path / "a.png")

    def fail(*args: object) -> None:
        raise AssertionError("decoded")

    monkeypatch.setattr(ImageFile.ImageFile, "load", fail)
    assert READER.read_metadata(path).width == small_scene.width


def test_our_limit_is_checked_before_any_decoding(tmp_path: Path, small_scene: Image.Image) -> None:
    img = small_scene.crop((0, 0, 200, 100))
    cut = corrupt.truncated(images.to_bytes(img, "PNG"), 0.5)
    path = tmp_path / "a.png"
    path.write_bytes(cut)
    with pytest.raises(ReaderError) as info:  # decoding would say "truncated"
        PillowReader(max_pixels=10_000).load_image(path, 64)
    assert info.value.kind == "too_large"
    with pytest.raises(ReaderError) as info:
        PillowReader().load_image(path, 64)
    assert info.value.kind == "truncated"
    PillowReader(max_pixels=20_000)  # exactly the size is allowed
    ok = images.save_png(img, tmp_path / "ok.png")
    assert PillowReader(max_pixels=20_000).load_image(ok, 64).size == (64, 32)
    with pytest.raises(ReaderError):
        PillowReader(max_pixels=19_999).load_image(ok, 64)


def test_large_but_allowed_header_reads_without_decoding(tmp_path: Path) -> None:
    path = tmp_path / "wide.jpg"
    path.write_bytes(jpeg_claiming(12_000, 12_000))  # 144 MP: between Pillow's two limits
    meta = READER.read_metadata(path)
    assert (meta.width, meta.height) == (12_000, 12_000)
    with pytest.raises(ReaderError) as info:
        PillowReader(max_pixels=100_000_000).load_image(path, 64)
    assert info.value.kind == "too_large"


def test_pillow_global_state_is_untouched(tmp_path: Path, small_scene: Image.Image) -> None:
    before = (
        Image.MAX_IMAGE_PIXELS,
        ImageFile.LOAD_TRUNCATED_IMAGES,
        dict(Image.OPEN) and len(Image.OPEN),
        len(Image.ID),
    )
    good = images.save_jpeg(small_scene, tmp_path / "a.jpg")
    bomb = tmp_path / "b.png"
    bomb.write_bytes(png_claiming(100_000, 100_000))
    READER.read_metadata(good)
    READER.load_image(good, 64)
    for path in (bomb, corrupt.write_garbage(tmp_path / "c.jpg")):
        with pytest.raises(ReaderError):
            READER.load_image(path, 64)
    after = (
        Image.MAX_IMAGE_PIXELS,
        ImageFile.LOAD_TRUNCATED_IMAGES,
        dict(Image.OPEN) and len(Image.OPEN),
        len(Image.ID),
    )
    assert after == before


def test_no_decompression_bomb_warning_escapes(tmp_path: Path) -> None:
    """In a fresh interpreter with every warning shown, a 144 MP header stays silent."""
    path = tmp_path / "wide.jpg"
    path.write_bytes(jpeg_claiming(12_000, 12_000))
    import mediasift

    src = str(Path(mediasift.__file__).resolve().parents[1])
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from pathlib import Path;"
        "from mediasift.readers.image import PillowReader;"
        "m = PillowReader().read_metadata(Path(sys.argv[2]));"
        "print(m.width, m.height)"
    )
    done = subprocess.run(
        [sys.executable, "-W", "always", "-c", code, src, str(path)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONWARNINGS": "always"},
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["12000", "12000"]
    assert "DecompressionBomb" not in done.stderr and "Warning" not in done.stderr


# --------------------------------------------------------------------------- files are released


@pytest.mark.parametrize("case", ["good", "truncated", "garbage", "empty", "bomb", "header"])
def test_file_can_be_renamed_right_after_use(
    case: str, tmp_path: Path, small_scene: Image.Image
) -> None:
    data = {
        "good": images.to_bytes(small_scene, "JPEG"),
        "truncated": corrupt.truncated(images.to_bytes(small_scene, "JPEG"), 0.5),
        "garbage": corrupt.garbage(500),
        "empty": b"",
        "bomb": png_claiming(100_000, 100_000),
        "header": corrupt.header_only(images.to_bytes(small_scene, "JPEG"), 20),
    }[case]
    path = tmp_path / "x.jpg"
    path.write_bytes(data)
    for call in (READER.read_metadata, lambda p: READER.load_image(p, 64)):
        with contextlib.suppress(ReaderError):
            call(path)
        moved = path.rename(tmp_path / "y.jpg")  # a leaked handle makes this fail on Windows
        moved.rename(path)
        path.unlink()
        path.write_bytes(data)


def test_returned_image_stays_usable_after_the_file_is_gone(
    tmp_path: Path, small_scene: Image.Image
) -> None:
    path = images.save_jpeg(small_scene, tmp_path / "a.jpg")
    out = READER.load_image(path, 400)
    path.unlink()
    assert out.getpixel((0, 0)) is not None and out.size == (320, 240)
    out.load()


# --------------------------------------------------------------------------- threads


def test_threads_reading_concurrently(tmp_path: Path, small_scene: Image.Image) -> None:
    paths = []
    for i, (orientation, data) in enumerate(images.all_orientations(small_scene).items()):
        path = tmp_path / f"o{orientation}.jpg"
        path.write_bytes(data)
        paths.append(path)
        paths.append(images.save_png(small_scene.rotate(i * 7), tmp_path / f"p{i}.png"))
    paths.append(corrupt.write_garbage(tmp_path / "bad.jpg"))

    def work(path: Path) -> tuple[Any, ...]:
        try:
            meta = READER.read_metadata(path)
            img = READER.load_image(path, 64)
            return meta.width, meta.height, meta.orientation, img.size, img.tobytes()
        except ReaderError as exc:
            return ("error", exc.kind)

    serial = [work(p) for p in paths]
    with ThreadPoolExecutor(max_workers=8) as pool:
        parallel = list(pool.map(work, paths * 6))
    assert parallel == serial * 6
    assert serial[-1] == ("error", "unsupported")
