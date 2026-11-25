"""Pillow-based image reader, and the helpers the HEIC and RAW readers share.

Why the shared helpers live here: every image reader has to turn EXIF into
:class:`MediaMetadata` and turn whatever it decoded into an upright RGB picture of
about the requested size. Doing that once keeps the three readers consistent.

Decisions worth knowing:

* Nothing here changes Pillow's global state (``MAX_IMAGE_PIXELS``,
  ``LOAD_TRUNCATED_IMAGES``). The pixel limit is our own and is checked on the
  header, before any decoding. Pillow's own hard limit (twice its
  ``MAX_IMAGE_PIXELS``, about 179 megapixels) still fires first and is reported as
  ``too_large`` as well.
* Pillow reports a *warning* between its soft and hard limit. ``warnings.catch_warnings``
  is not thread-safe, so the warning category is silenced once, at import.
* Files are opened through our own file object and closed by a ``with`` block, so a
  failed read never leaves the file locked (Windows refuses to rename a locked file).
"""

from __future__ import annotations

import dataclasses
import io
import math
import re
import struct
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import numpy as np
from PIL import Image, ImageCms, UnidentifiedImageError

from mediasift.core.errors import ReaderError
from mediasift.core.types import MediaMetadata

# Process-wide, set once and never changed again: safe to read from any thread.
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
# Pillow reports damaged EXIF and TIFF tags as UserWarning; a bulk scan needs no noise.
warnings.filterwarnings("ignore", category=UserWarning, module=r"PIL\.")

DEFAULT_MAX_PIXELS = 300_000_000
"""Default decompression-bomb limit: width times height of one frame."""

EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
)

# Errors a decoder may raise on bad input. Programming errors (AttributeError, ...)
# are deliberately not in the list: they must not be reported as "corrupt files".
DECODE_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    ValueError,
    SyntaxError,
    EOFError,
    struct.error,
    TypeError,
    IndexError,
    KeyError,
    OverflowError,
    MemoryError,
    ZeroDivisionError,
    AssertionError,
    Image.DecompressionBombError,
)

_TAG_ORIENTATION = 0x0112
_TAG_MAKE = 0x010F
_TAG_MODEL = 0x0110
_TAG_SOFTWARE = 0x0131
_TAG_DATETIME = 0x0132
_IFD_EXIF = 0x8769
_IFD_GPS = 0x8825
_EXIF_DATETIME_ORIGINAL = 0x9003
_EXIF_DATETIME_DIGITIZED = 0x9004
_EXIF_OFFSET_TIME = 0x9010
_EXIF_OFFSET_TIME_ORIGINAL = 0x9011
_EXIF_OFFSET_TIME_DIGITIZED = 0x9012

_ORIENTATION_OPS: dict[int, Image.Transpose] = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}

_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"BM", b"II*\x00", b"MM\x00*", b"RIFF")

_T = TypeVar("_T")


# --------------------------------------------------------------------------- errors


_TRUNCATED = re.compile(
    r"truncated|unexpected end|premature end|not enough image data|image data too short"
    r"|file size too small",
    re.IGNORECASE,
)
_TOO_LARGE = re.compile(r"security limit|too large|exceeds limit", re.IGNORECASE)


def translate_decode_error(exc: BaseException) -> ReaderError:
    """Map an exception raised by a decoder to a :class:`ReaderError` with a stable kind."""
    if isinstance(exc, (Image.DecompressionBombError, MemoryError)) or _TOO_LARGE.search(str(exc)):
        return ReaderError(f"image is too large to decode: {exc}", kind="too_large")
    if isinstance(exc, UnidentifiedImageError):
        return ReaderError("not a recognised image", kind="unsupported")
    if isinstance(exc, (FileNotFoundError, PermissionError, IsADirectoryError)):
        return ReaderError(f"cannot read the file: {exc}", kind="io")
    if _TRUNCATED.search(str(exc)):
        return ReaderError(f"the file is cut short: {exc}", kind="truncated")
    return ReaderError(f"the image data is damaged: {exc}", kind="corrupt")


def check_pixels(size: tuple[int, int], max_pixels: int) -> None:
    """Raise ``too_large`` when ``size`` (read from a header) exceeds ``max_pixels``."""
    pixels = size[0] * size[1]
    if pixels > max_pixels:
        raise ReaderError(f"image has {pixels} pixels, the limit is {max_pixels}", kind="too_large")


@contextmanager
def open_image(path: Path) -> Iterator[Image.Image]:
    """Open ``path`` with Pillow and always release the file, even after a failure.

    Raises :class:`ReaderError` for a missing, empty or unidentifiable file; decoder
    errors that happen later propagate unchanged for :func:`translate_decode_error`.
    """
    try:
        fp = path.open("rb")
    except OSError as exc:
        raise ReaderError(f"cannot open the file: {exc}", kind="io") from exc
    with fp:
        head = fp.read(16)
        if not head:
            raise ReaderError("the file is empty", kind="empty")
        fp.seek(0)
        try:
            image = Image.open(fp)
        except UnidentifiedImageError:
            known = head.startswith(_MAGIC)
            raise ReaderError(
                "the image header cannot be read" if known else "not a recognised image",
                kind="corrupt" if known else "unsupported",
            ) from None
        with image:
            yield image


def _attempt(work: Callable[[], _T], default: _T) -> _T:
    """Run ``work``; EXIF comes from untrusted bytes, so any failure just means unknown."""
    try:
        return work()
    except Exception:  # noqa: BLE001 - parsers of damaged metadata can raise anything
        return default


# --------------------------------------------------------------------------- EXIF


class ExifSource(Protocol):
    """The part of ``PIL.Image.Exif`` the parser needs; a RAW reader supplies its own."""

    def __contains__(self, tag: object, /) -> bool: ...

    def get(self, tag: int, default: Any = None, /) -> Any: ...

    def get_ifd(self, tag: int, /) -> Mapping[int, Any]: ...


@dataclass(frozen=True, slots=True)
class ExifFields:
    """The EXIF facts MediaSift uses. Unknown values are None."""

    orientation: int | None = None
    taken_at: datetime | None = None
    utc_offset: str | None = None
    make: str | None = None
    model: str | None = None
    software: str | None = None
    gps: tuple[float, float] | None = None
    has_exif: bool = False


_DATE = re.compile(r"\s*(\d{4})\D(\d{2})\D(\d{2})[\sT]+(\d{2})\D(\d{2})\D(\d{2})")
_OFFSET = re.compile(r"\s*([+-])(\d{2}):?(\d{2})\s*")


def _text(value: Any) -> str | None:
    """Text of an ASCII tag with NULs and blanks removed; None when nothing is left."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return None
    cleaned = value.replace("\x00", "").strip()
    return cleaned or None


def parse_exif_datetime(value: Any) -> datetime | None:
    """Parse ``YYYY:MM:DD HH:MM:SS``; impossible dates such as ``0000:00:00`` give None."""
    text = _text(value)
    match = _DATE.match(text) if text else None
    if match is None:
        return None
    try:
        year, month, day, hour, minute, second = (int(part) for part in match.groups())
        parsed = datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None
    return parsed if parsed.year >= 1900 else None


def _utc_offset(value: Any) -> str | None:
    text = _text(value)
    match = _OFFSET.fullmatch(text) if text else None
    if match is None:
        return None
    sign, hours, minutes = match.groups()
    if int(hours) > 14 or int(minutes) > 59:
        return None
    return f"{sign}{hours}:{minutes}"


def _taken(exif: ExifSource, exif_ifd: Mapping[int, Any]) -> tuple[datetime, str | None] | None:
    sources = (
        (exif_ifd.get(_EXIF_DATETIME_ORIGINAL), _EXIF_OFFSET_TIME_ORIGINAL),
        (exif_ifd.get(_EXIF_DATETIME_DIGITIZED), _EXIF_OFFSET_TIME_DIGITIZED),
        (exif.get(_TAG_DATETIME), _EXIF_OFFSET_TIME),
    )
    for raw, offset_tag in sources:
        taken = parse_exif_datetime(raw)
        if taken is not None:
            return taken, _utc_offset(exif_ifd.get(offset_tag))
    return None


def _number(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):  # a zero denominator reads as NaN
        raise ValueError("not a finite number")
    return number


def _coordinate(value: Any, ref: Any, negative: str, positive: str, limit: float) -> float:
    """Degrees, minutes, seconds plus a hemisphere letter as signed decimal degrees."""
    parts = [_number(p) for p in value] if isinstance(value, Sequence) else [_number(value)]
    if not 1 <= len(parts) <= 3 or any(p < 0 for p in parts):
        raise ValueError("bad coordinate")
    degrees, minutes, seconds = [*parts, 0.0, 0.0][:3]
    if minutes > 60 or seconds > 60:  # writers that round up produce exactly 60
        raise ValueError("bad minutes or seconds")
    total = degrees + minutes / 60 + seconds / 3600
    letter = (_text(ref) or "").upper()
    if total > limit or letter not in (negative, positive):
        raise ValueError("coordinate out of range")
    return -total if letter == negative else total


def _gps(gps_ifd: Mapping[int, Any]) -> tuple[float, float] | None:
    """Decimal latitude and longitude; (0, 0) is what cameras write without a fix."""
    lat = _coordinate(gps_ifd[2], gps_ifd[1], "S", "N", 90.0)
    lon = _coordinate(gps_ifd[4], gps_ifd[3], "W", "E", 180.0)
    return None if lat == 0.0 and lon == 0.0 else (lat, lon)


def _orientation(exif: ExifSource) -> int | None:
    value = exif.get(_TAG_ORIENTATION)
    return value if isinstance(value, int) and 1 <= value <= 8 else None


def _has_exif(exif: ExifSource, exif_ifd: Mapping[int, Any]) -> bool:
    """True when the file carries EXIF proper: a TIFF's structural tags alone do not count."""
    own_tags = (_TAG_ORIENTATION, _TAG_MAKE, _TAG_MODEL, _TAG_SOFTWARE, _TAG_DATETIME)
    return bool(exif_ifd) or bool(exif.get_ifd(_IFD_GPS)) or any(t in exif for t in own_tags)


def parse_exif(exif: ExifSource | None) -> ExifFields:
    """Read the fields MediaSift uses. Never raises: a bad field is simply unknown."""
    if exif is None:
        return ExifFields()
    exif_ifd: Mapping[int, Any] = _attempt(lambda: exif.get_ifd(_IFD_EXIF), {})
    taken = _attempt(lambda: _taken(exif, exif_ifd), None)
    return ExifFields(
        orientation=_attempt(lambda: _orientation(exif), None),
        taken_at=taken[0] if taken else None,
        utc_offset=taken[1] if taken else None,
        make=_attempt(lambda: _text(exif.get(_TAG_MAKE)), None),
        model=_attempt(lambda: _text(exif.get(_TAG_MODEL)), None),
        software=_attempt(lambda: _text(exif.get(_TAG_SOFTWARE)), None),
        gps=_attempt(lambda: _gps(exif.get_ifd(_IFD_GPS)), None),
        has_exif=_attempt(lambda: _has_exif(exif, exif_ifd), False),
    )


def pillow_exif(im: Image.Image) -> ExifSource | None:
    """The image's EXIF block, or None when Pillow cannot parse it."""
    return _attempt(lambda: cast(ExifSource, im.getexif()), None)


def exif_from_bytes(data: bytes) -> ExifSource | None:
    """Parse a raw EXIF block (with or without the ``Exif`` prefix); None when unreadable."""

    def parse() -> ExifSource:
        exif = Image.Exif()
        exif.load(data)
        return cast(ExifSource, exif)

    return _attempt(parse, None)


def exif_extra(fields: ExifFields) -> dict[str, Any]:
    """The EXIF-derived entries of ``MediaMetadata.extra``."""
    extra: dict[str, Any] = {"has_exif": fields.has_exif}
    if fields.utc_offset is not None:
        extra["utc_offset"] = fields.utc_offset
    if fields.software is not None:
        extra["software"] = fields.software
    return extra


def display_size(size: tuple[int, int], orientation: int | None) -> tuple[int, int]:
    """Width and height as shown: orientations 5 to 8 turn the picture by a quarter."""
    return (size[1], size[0]) if orientation is not None and orientation >= 5 else size


def metadata_from_exif(
    fields: ExifFields, size: tuple[int, int], codec: str | None, extra: Mapping[str, Any]
) -> MediaMetadata:
    """Assemble :class:`MediaMetadata` from EXIF facts and the stored pixel size."""
    width, height = display_size(size, fields.orientation)
    return MediaMetadata(
        width=width,
        height=height,
        orientation=fields.orientation,
        taken_at=fields.taken_at,
        taken_source="exif" if fields.taken_at else None,
        camera_make=fields.make,
        camera_model=fields.model,
        gps_lat=fields.gps[0] if fields.gps else None,
        gps_lon=fields.gps[1] if fields.gps else None,
        codec=codec,
        extra={**exif_extra(fields), **extra},
    )


# --------------------------------------------------------------------------- JPEG quality

# The JPEG standard's example luminance table: quality 50 in libjpeg's scale.
_STD_LUMA = (
    (16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55),
    (14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62),
    (18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92),
    (49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99),
)
_STD_LUMA_SUM = sum(sum(row) for row in _STD_LUMA)


def estimate_jpeg_quality(tables: Mapping[int, Sequence[int]] | None) -> int | None:
    """Quality 1 to 100 as libjpeg would have scaled the standard luminance table.

    An estimate: cameras with their own tables map to the nearest libjpeg quality.
    """
    table = tables.get(0) if tables else None
    if not table or len(table) != 64:
        return None
    scale = 100.0 * sum(table) / _STD_LUMA_SUM
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return max(1, min(100, round(quality)))


# --------------------------------------------------------------------------- pixels


def apply_orientation(im: Image.Image, orientation: int | None) -> Image.Image:
    """Return ``im`` turned so that a viewer honouring ``orientation`` shows it upright."""
    op = _ORIENTATION_OPS.get(orientation or 1)
    return im if op is None else im.transpose(op)


def _high_depth_to_l(im: Image.Image) -> Image.Image:
    """Scale 16-bit, 32-bit integer and float pictures to 8 bits without clipping."""
    if im.mode == "F":
        data = np.asarray(im, dtype=np.float64)
    else:
        data = np.asarray(im.convert("I"), dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return Image.new("L", im.size, 0)
    low, high = float(finite.min()), float(finite.max())
    if im.mode == "F" and low >= 0.0 and high <= 1.0:
        scaled = data * 255.0
    elif low >= 0.0 and high <= 255.0 and im.mode != "F":
        scaled = data
    elif low >= 0.0 and high <= 65535.0 and im.mode != "F":
        scaled = data / 257.0
    else:
        scaled = (data - low) * (255.0 / max(high - low, 1e-12))
    return Image.fromarray(np.clip(np.rint(np.nan_to_num(scaled)), 0, 255).astype(np.uint8), "L")


def _cmyk_to_rgb(im: Image.Image) -> Image.Image:
    """CMYK to sRGB through the embedded profile when there is one."""
    profile = im.info.get("icc_profile")
    if profile:
        try:
            source = ImageCms.ImageCmsProfile(io.BytesIO(profile))
            target = ImageCms.createProfile("sRGB")
            converted = ImageCms.profileToProfile(im, source, target, outputMode="RGB")
        except (ImageCms.PyCMSError, OSError, ValueError, ImportError):
            converted = None
        if converted is not None:
            return converted
    return im.convert("RGB")


def _to_rgb_or_rgba(im: Image.Image) -> Image.Image:
    """Bring any decoded mode to 8-bit RGB, or RGBA when there is transparency."""
    mode = im.mode
    if mode in ("RGB", "RGBA"):
        return im
    if mode in ("1", "L"):
        return im.convert("RGB")
    if mode == "P":
        return im.convert("RGBA" if "transparency" in im.info else "RGB")
    if mode in ("PA", "LA", "La", "RGBa"):
        return im.convert("RGBA")
    if mode == "CMYK":
        return _cmyk_to_rgb(im)
    if mode == "F" or mode == "I" or mode.startswith("I;16"):
        return _high_depth_to_l(im).convert("RGB")
    try:
        return im.convert("RGB")
    except ValueError as exc:
        raise ReaderError(f"unsupported pixel format {mode}: {exc}", kind="unsupported") from exc


def _on_white(im: Image.Image) -> Image.Image:
    background = Image.new("RGB", im.size, (255, 255, 255))
    background.paste(im, mask=im.getchannel("A"))
    return background


def shrink(im: Image.Image, max_side: int) -> Image.Image:
    """Reduce so the longer side is ``max_side``; never enlarge."""
    longest = max(im.size)
    if longest <= max_side:
        return im
    factor = longest // max_side
    if factor >= 2:
        im = im.reduce(factor)  # integer box filter: fast, and exact for alpha
        longest = max(im.size)
    if longest > max_side:
        scale = max_side / longest
        size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
        im = im.resize(size, Image.Resampling.LANCZOS)
    return im


def finish_image(im: Image.Image, max_side: int, orientation: int | None) -> Image.Image:
    """Turn a decoded picture into upright RGB of about ``max_side``.

    Always returns a new image, so the caller may close ``im`` afterwards. Transparency
    is composited on white; the order (reduce, then composite) keeps colours of
    transparent pixels from bleeding into the result.
    """
    prepared = shrink(_to_rgb_or_rgba(im), max_side)
    if prepared.mode == "RGBA":
        prepared = _on_white(prepared)
    upright = apply_orientation(prepared, orientation)
    return upright.copy() if upright is im else upright


@cache
def pillow_orients_tiff() -> bool:
    """True when this Pillow turns TIFF pixels by the orientation tag while decoding.

    Recent versions do (and swap ``size`` to match); older ones leave it to the caller.
    Probed once with a tiny file so no version table has to be kept.
    """
    exif = Image.Exif()
    exif[_TAG_ORIENTATION] = 6
    buffer = io.BytesIO()
    Image.new("L", (2, 1)).save(buffer, format="TIFF", exif=exif)
    buffer.seek(0)
    with Image.open(buffer) as probe:
        return probe.size == (1, 2)


def orientation_to_apply(im: Image.Image, orientation: int | None) -> int | None:
    """The orientation left for us to apply: none when Pillow already did it (TIFF)."""
    return None if im.format == "TIFF" and pillow_orients_tiff() else orientation


def request_draft(im: Image.Image, max_side: int) -> None:
    """Let the JPEG decoder scale by 1/2, 1/4 or 1/8 while decoding: far less work."""
    if im.format not in ("JPEG", "MPO"):
        return
    longest = max(im.size)
    if longest <= max_side:
        return
    scale = max_side / longest
    target = (max(1, math.ceil(im.width * scale)), max(1, math.ceil(im.height * scale)))
    im.draft("RGB", target)


# --------------------------------------------------------------------------- reader


class PillowReader:
    """Reads JPEG, PNG, WebP, GIF, BMP and TIFF with Pillow."""

    name = "pillow"
    version = 1
    extensions = EXTENSIONS

    def __init__(self, max_pixels: int = DEFAULT_MAX_PIXELS) -> None:
        if max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        self.max_pixels = max_pixels

    def available(self) -> bool:
        """Pillow is a hard dependency."""
        return True

    def read_metadata(self, path: Path) -> MediaMetadata:
        """Facts from the header and EXIF. Damaged EXIF leaves fields empty, never raises."""
        try:
            with open_image(path) as im:
                return self._metadata(im)
        except ReaderError:
            raise
        except DECODE_ERRORS as exc:
            raise translate_decode_error(exc) from exc

    def load_image(self, path: Path, max_side: int) -> Image.Image:
        """Upright RGB image whose longer side is at most ``max_side`` (first frame only)."""
        if max_side < 1:
            raise ValueError("max_side must be at least 1")
        try:
            with open_image(path) as im:
                check_pixels(im.size, self.max_pixels)
                exif = pillow_exif(im)
                orientation = orientation_to_apply(im, parse_exif(exif).orientation)
                request_draft(im, max_side)
                im.load()
                return finish_image(im, max_side, orientation)
        except ReaderError:
            raise
        except DECODE_ERRORS as exc:
            raise translate_decode_error(exc) from exc

    @staticmethod
    def _metadata(im: Image.Image) -> MediaMetadata:
        fields = parse_exif(pillow_exif(im))
        extra: dict[str, Any] = {
            "mode": im.mode,
            "has_icc": bool(im.info.get("icc_profile")),
        }
        if im.format == "JPEG":
            quantization = _attempt(lambda: getattr(im, "quantization", None), None)
            extra["jpeg_quality_estimate"] = estimate_jpeg_quality(quantization)
        # Last: counting GIF or TIFF frames walks the file.
        extra["n_frames"] = _attempt(lambda: int(getattr(im, "n_frames", 1)), 1)
        meta = metadata_from_exif(fields, im.size, im.format, extra)
        if orientation_to_apply(im, fields.orientation) is None and fields.orientation:
            # Pillow already turned the pixels: its size is the display size.
            meta = dataclasses.replace(meta, width=im.size[0], height=im.size[1])
        return meta
