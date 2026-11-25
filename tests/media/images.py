"""Generated photo-like test images, savers, EXIF builder, variants and negatives.

Everything is a pure function of its seed (NumPy ``PCG64``), so a test can rebuild
the same pixels anywhere. Scenes have low-frequency structure (sky, ground, horizon
and large soft shapes) that changes with the seed, which is what makes two seeds look
different to a perceptual hash and not only to a byte comparison.
"""

from __future__ import annotations

import io
import random
import string
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, TiffImagePlugin

DEFAULT_SIZE = (1200, 900)
_F32 = NDArray[np.float32]


def _rng(*key: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(list(key)))


# --------------------------------------------------------------------------- scenes


def _fbm(rng: np.random.Generator, w: int, h: int, base: int, octaves: int) -> _F32:
    """Smooth multi-octave noise in 0..1, shape (h, w)."""
    total = np.zeros((h, w), dtype=np.float32)
    amp, norm = 1.0, 0.0
    for k in range(octaves):
        cells = base * 2**k
        grid = rng.random((max(2, cells * h // max(w, h)) + 1, cells + 1)).astype(np.float32)
        layer = Image.fromarray(grid, mode="F").resize((w, h), Image.Resampling.BICUBIC)
        total += amp * np.asarray(layer, dtype=np.float32)
        norm += amp
        amp *= 0.5
    total /= norm
    lo, hi = float(total.min()), float(total.max())
    return (total - lo) / max(hi - lo, 1e-6)


def _colour(rng: np.random.Generator) -> NDArray[np.float64]:
    return rng.uniform(15, 240, size=3)


def _render(
    seed: int,
    size: tuple[int, int],
    *,
    detail_seed: int | None = None,
    subject_move: float = 0.0,
    shape_range: tuple[int, int] = (8, 14),
    shape_scale: float = 1.0,
) -> Image.Image:
    w, h = size
    rng = _rng(seed)
    # Base layers come from `seed` alone: the composition.
    hy = rng.uniform(0.35, 0.65) * h
    sky_top, sky_bot = _colour(rng), _colour(rng)
    gnd_top, gnd_bot = _colour(rng), _colour(rng)
    clouds = _fbm(rng, w, h, 3, 4)
    ground_tex = _fbm(rng, w, h, 6, 5)
    ridge_pts = rng.random(7) * 0.12 * h
    ridge = np.interp(np.linspace(0, 6, w), np.arange(7), ridge_pts) - 0.06 * h
    ys = np.arange(h, dtype=np.float32)[:, None]
    edge = np.clip((ys - (hy + ridge[None, :])) / 6.0 + 0.5, 0.0, 1.0)[..., None]
    t_sky = np.clip(ys / hy, 0, 1)[..., None]
    t_gnd = np.clip((ys - hy) / max(h - hy, 1), 0, 1)[..., None]
    sky = sky_top * (1 - t_sky) + sky_bot * t_sky + (clouds[..., None] - 0.5) * 70
    ground = gnd_top * (1 - t_gnd) + gnd_bot * t_gnd + (ground_tex[..., None] - 0.5) * 90
    canvas = (sky * (1 - edge) + ground * edge).astype(np.float32)
    # Detail layers (shapes, texture) come from `detail_seed`.
    drng = _rng(seed if detail_seed is None else detail_seed, 99)
    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")
    n_shapes = int(drng.integers(shape_range[0], shape_range[1] + 1))
    short = min(w, h)
    for i in range(n_shapes):
        kind = int(drng.integers(0, 3))
        cx, cy = drng.uniform(0.05, 0.95) * w, drng.uniform(0.05, 0.95) * h
        rx, ry = drng.uniform(0.03, 0.22, size=2) * short * shape_scale
        colour = tuple(int(c) for c in _colour(drng))
        alpha = float(drng.uniform(0.55, 1.0))
        blur = float(drng.uniform(1.5, 7.0))
        angle = float(drng.uniform(0, 6.28))
        move = drng.uniform(-1, 1, size=2) * subject_move if i < 2 else (0.0, 0.0)
        cx, cy = cx + float(move[0]), cy + float(move[1])
        mask = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(mask)
        if kind == 0:
            d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=int(255 * alpha))
        elif kind == 1:
            d.rectangle((cx - rx, cy - ry, cx + rx, cy + ry), fill=int(255 * alpha))
        else:
            pts = [
                (
                    cx + rx * float(np.cos(angle + k * 2.09)),
                    cy + ry * float(np.sin(angle + k * 2.09)),
                )
                for k in range(3)
            ]
            d.polygon(pts, fill=int(255 * alpha))
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
        img.paste(Image.new("RGB", (w, h), colour), mask=mask)
    grain = drng.normal(0, 3.0, size=(h, w, 1)).astype(np.float32)
    fine = (_fbm(drng, w, h, 40, 2)[..., None] - 0.5) * 22
    out = np.asarray(img, dtype=np.float32) + grain + fine
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def scene(seed: int, size: tuple[int, int] = DEFAULT_SIZE) -> Image.Image:
    """A deterministic photo-like RGB image; different seeds differ in composition."""
    return _render(seed, size)


def similar_scene(seed: int, shift: int = 12, size: tuple[int, int] = DEFAULT_SIZE) -> Image.Image:
    """The same scene as a burst shot: camera shifted, subject moved, sensor noise added."""
    w, h = size
    base = _render(seed, size, subject_move=shift * 2.0)
    rng = _rng(seed, 7)
    dx, dy = (int(v) for v in rng.integers(-shift, shift + 1, size=2))
    m = abs(shift)
    box = (m + dx, m + dy, w - m + dx, h - m + dy)
    img = base.resize((w + 2 * m, h + 2 * m), Image.Resampling.BICUBIC).crop(box)
    img = img.resize((w, h), Image.Resampling.BICUBIC)
    noise = rng.normal(0, 2.5, size=(h, w, 3))
    arr = np.asarray(img, dtype=np.float64) + noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


# --------------------------------------------------------------------------- negatives


def distinct_scenes(
    count: int, start: int = 1000, size: tuple[int, int] = DEFAULT_SIZE
) -> list[Image.Image]:
    """Scenes with consecutive seeds: unrelated images."""
    return [scene(start + i, size) for i in range(count)]


def same_palette_shuffled(seed: int, size: tuple[int, int] = DEFAULT_SIZE) -> Image.Image:
    """Same pixels as ``scene(seed)`` cut into tiles and re-arranged: same colours, other layout."""
    img = scene(seed, size)
    w, h = size
    cols, rows = 6, 4
    tw, th = w // cols, h // rows
    order = list(range(cols * rows))
    random.Random(seed).shuffle(order)
    if order == sorted(order):
        order.reverse()
    out = img.copy()
    for dst, src in enumerate(order):
        sx, sy = (src % cols) * tw, (src // cols) * th
        tile = img.crop((sx, sy, sx + tw, sy + th))
        out.paste(tile, ((dst % cols) * tw, (dst // cols) * th))
    return out


def text_page(seed: int, size: tuple[int, int] = DEFAULT_SIZE) -> Image.Image:
    """A document-like page of black text lines on white; the words depend on the seed."""
    w, h = size
    rnd = random.Random(seed)
    img = Image.new("RGB", size, (250, 250, 247))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=max(10, h // 28))
    y = h // 12
    while y < h - h // 10:
        words = [
            "".join(rnd.choices(string.ascii_lowercase, k=rnd.randint(2, 9)))
            for _ in range(rnd.randint(4, 9))
        ]
        d.text((w // 10, y), " ".join(words), fill=(20, 20, 20), font=font)
        y += h // 18
    return img


def flat_image(
    colour: tuple[int, int, int] = (12, 12, 14), size: tuple[int, int] = DEFAULT_SIZE
) -> Image.Image:
    """A uniform image (plain dark by default)."""
    return Image.new("RGB", size, colour)


def horizon_scene(seed: int, detail_seed: int, size: tuple[int, int] = DEFAULT_SIZE) -> Image.Image:
    """Same sky, ground and horizon as ``seed`` but different small details."""
    return _render(seed, size, detail_seed=detail_seed, shape_range=(3, 5), shape_scale=0.35)


# --------------------------------------------------------------------------- EXIF


def _dms(value: float) -> tuple[Any, Any, Any]:
    value = abs(value)
    deg = int(value)
    minutes = int((value - deg) * 60)
    seconds = round((value - deg - minutes / 60) * 3600 * 10000)
    rat = TiffImagePlugin.IFDRational
    return rat(deg, 1), rat(minutes, 1), rat(seconds, 10000)


def build_exif(
    *,
    taken: datetime | None = None,
    offset: str | None = None,
    make: str | None = None,
    model: str | None = None,
    gps: tuple[float, float] | None = None,
    orientation: int | None = None,
) -> Image.Exif:
    """EXIF block with DateTimeOriginal (+ offset), make, model, GPS and orientation."""
    exif = Image.Exif()
    if make is not None:
        exif[0x010F] = make
    if model is not None:
        exif[0x0110] = model
    if orientation is not None:
        exif[0x0112] = orientation
    if taken is not None or offset is not None:
        sub = exif.get_ifd(0x8769)
        if taken is not None:
            sub[0x9003] = taken.strftime("%Y:%m:%d %H:%M:%S")
        if offset is not None:
            sub[0x9011] = offset
    if gps is not None:
        lat, lon = gps
        g = exif.get_ifd(0x8825)
        g[1] = "N" if lat >= 0 else "S"
        g[2] = _dms(lat)
        g[3] = "E" if lon >= 0 else "W"
        g[4] = _dms(lon)
    return exif


def read_exif(img: Image.Image) -> dict[str, Any]:
    """Decode the fields written by :func:`build_exif` (missing ones are ``None``)."""
    exif = img.getexif()
    sub = exif.get_ifd(0x8769)
    g = exif.get_ifd(0x8825)
    taken = sub.get(0x9003)
    gps = None
    if g.get(2) and g.get(4):
        lat = sum(float(v) / 60**i for i, v in enumerate(g[2]))
        lon = sum(float(v) / 60**i for i, v in enumerate(g[4]))
        gps = (lat if g.get(1) == "N" else -lat, lon if g.get(3) == "E" else -lon)
    return {
        "taken": datetime.strptime(taken, "%Y:%m:%d %H:%M:%S") if taken else None,
        "offset": sub.get(0x9011),
        "make": exif.get(0x010F),
        "model": exif.get(0x0110),
        "orientation": exif.get(0x0112),
        "gps": gps,
    }


# --------------------------------------------------------------------------- savers


def to_bytes(img: Image.Image, fmt: str, **params: Any) -> bytes:
    """Encode ``img`` in memory."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, **params)
    return buf.getvalue()


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def save_jpeg(
    img: Image.Image, path: Path, quality: int = 90, exif: Image.Exif | None = None
) -> Path:
    params: dict[str, Any] = {"quality": quality}
    if exif is not None:
        params["exif"] = exif
    return _write(
        path,
        to_bytes(img.convert("RGB") if img.mode not in ("L", "RGB") else img, "JPEG", **params),
    )


def save_png(img: Image.Image, path: Path) -> Path:
    return _write(path, to_bytes(img, "PNG"))


def save_webp(img: Image.Image, path: Path, quality: int = 85) -> Path:
    return _write(path, to_bytes(img, "WEBP", quality=quality))


def save_gif(img: Image.Image, path: Path) -> Path:
    return _write(path, to_bytes(img.convert("P", palette=Image.Palette.ADAPTIVE), "GIF"))


def save_bmp(img: Image.Image, path: Path) -> Path:
    return _write(path, to_bytes(img, "BMP"))


def save_tiff(img: Image.Image, path: Path) -> Path:
    return _write(path, to_bytes(img, "TIFF"))


def heif_available() -> bool:
    """True when pillow_heif can be imported and registered."""
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except Exception:  # noqa: BLE001 - any import or plugin failure means "unavailable"
        return False
    return True


requires_heif = pytest.mark.skipif(not heif_available(), reason="pillow_heif not available")


def save_heic(
    img: Image.Image, path: Path, quality: int = 80, exif: Image.Exif | None = None
) -> Path:
    """Write a HEIC file; callers should be guarded by :data:`requires_heif`."""
    import pillow_heif

    pillow_heif.register_heif_opener()
    params: dict[str, Any] = {"quality": quality}
    if exif is not None:
        params["exif"] = exif
    return _write(path, to_bytes(img, "HEIF", **params))


# --------------------------------------------------------------------------- orientation

_UNDO = {
    1: None,
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_90,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_270,
}


def stored_for_orientation(img: Image.Image, orientation: int) -> Image.Image:
    """Pixels to store so that a viewer honouring ``orientation`` shows ``img`` upright."""
    op = _UNDO[orientation]
    return img if op is None else img.transpose(op)


def all_orientations(img: Image.Image, quality: int = 92) -> dict[int, bytes]:
    """JPEG bytes for orientations 1..8; each displays like ``img`` after ``exif_transpose``."""
    return {
        o: to_bytes(
            stored_for_orientation(img, o), "JPEG", quality=quality, exif=build_exif(orientation=o)
        )
        for o in range(1, 9)
    }


# --------------------------------------------------------------------------- variants


@dataclass(frozen=True)
class EncodedImage:
    """Encoded variant bytes and the file extension that matches them."""

    data: bytes
    ext: str

    def image(self) -> Image.Image:
        im = Image.open(io.BytesIO(self.data))
        im.load()
        return im


Variant = Callable[[Image.Image], EncodedImage]
_JPEG = {"quality": 90}


def _jpeg(img: Image.Image, **params: Any) -> EncodedImage:
    params = {**_JPEG, **params}
    return EncodedImage(to_bytes(img, "JPEG", **params), ".jpg")


def _resized(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    return img.resize(
        (max(1, round(w * factor)), max(1, round(h * factor))), Image.Resampling.LANCZOS
    )


def whatsapp_size(size: tuple[int, int]) -> tuple[int, int]:
    """Size a chat app produces: longest side 1600, or 0.6x for smaller pictures."""
    w, h = size
    factor = 1600 / max(w, h) if max(w, h) > 1600 else 0.6
    return max(1, round(w * factor)), max(1, round(h * factor))


def _whatsapp(img: Image.Image) -> EncodedImage:
    return _jpeg(img.resize(whatsapp_size(img.size), Image.Resampling.LANCZOS), quality=65)


def _crop_3pct(img: Image.Image) -> Image.Image:
    w, h = img.size
    mx, my = int(w * 0.015), int(h * 0.015)
    return img.crop((mx, my, w - mx, h - my))


def _watermark(img: Image.Image) -> Image.Image:
    out = img.copy().convert("RGBA")
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = ImageFont.load_default(size=max(10, out.height // 40))
    d.text((out.width * 0.78, out.height * 0.93), "mediasift", fill=(255, 255, 255, 170), font=font)
    return Image.alpha_composite(out, layer).convert("RGB")


SCREENSHOT_BAR = (10, 10, 12)


def screenshot_size(size: tuple[int, int]) -> tuple[int, int]:
    """Canvas of the ``screenshot`` variant: same width, 1.5x height."""
    return size[0], round(size[1] * 1.5)


def _screenshot(img: Image.Image) -> Image.Image:
    h = img.height
    canvas = Image.new("RGB", screenshot_size(img.size), SCREENSHOT_BAR)
    canvas.paste(img, (0, (canvas.height - h) // 2))
    return canvas


def _rotated_with_exif(img: Image.Image) -> EncodedImage:
    return _jpeg(stored_for_orientation(img, 6), exif=build_exif(orientation=6))


def _crop_size(size: tuple[int, int]) -> tuple[int, int]:
    w, h = size
    return w - 2 * int(w * 0.015), h - 2 * int(h * 0.015)


VARIANTS: dict[str, Variant] = {
    "resize_50": lambda im: _jpeg(_resized(im, 0.5)),
    "resize_25": lambda im: _jpeg(_resized(im, 0.25)),
    "recompress_q80": lambda im: _jpeg(im, quality=80),
    "recompress_q40": lambda im: _jpeg(im, quality=40),
    "whatsapp": _whatsapp,
    "brightness_up": lambda im: _jpeg(ImageEnhance.Brightness(im).enhance(1.10)),
    "brightness_down": lambda im: _jpeg(ImageEnhance.Brightness(im).enhance(0.90)),
    "contrast_up": lambda im: _jpeg(ImageEnhance.Contrast(im).enhance(1.25)),
    "saturation_up": lambda im: _jpeg(ImageEnhance.Color(im).enhance(1.30)),
    "slight_crop_3pct": lambda im: _jpeg(_crop_3pct(im)),
    "slight_blur": lambda im: _jpeg(im.filter(ImageFilter.GaussianBlur(1.2))),
    "sharpen": lambda im: _jpeg(
        im.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=2))
    ),
    "small_watermark": lambda im: _jpeg(_watermark(im)),
    "screenshot": lambda im: EncodedImage(to_bytes(_screenshot(im), "PNG"), ".png"),
    "png_conversion": lambda im: EncodedImage(to_bytes(im, "PNG"), ".png"),
    "webp_conversion": lambda im: EncodedImage(to_bytes(im, "WEBP", quality=85), ".webp"),
    "grayscale": lambda im: _jpeg(ImageOps.grayscale(im)),
    "exif_rotated": _rotated_with_exif,
}


def variant_size(name: str, size: tuple[int, int]) -> tuple[int, int]:
    """Expected pixel size of a variant as *stored* (before any EXIF orientation is applied)."""
    w, h = size
    table: dict[str, tuple[int, int]] = {
        "resize_50": (round(w * 0.5), round(h * 0.5)),
        "resize_25": (round(w * 0.25), round(h * 0.25)),
        "whatsapp": whatsapp_size(size),
        "slight_crop_3pct": _crop_size(size),
        "screenshot": screenshot_size(size),
        "exif_rotated": (h, w),
    }
    return table.get(name, size)
