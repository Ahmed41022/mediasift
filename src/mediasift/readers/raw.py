"""Camera RAW reader on top of ``rawpy`` (LibRaw), which is optional.

Pixels: the embedded JPEG preview when it is big enough for the request (fast and
what the camera showed), otherwise LibRaw's demosaic, at half size when that still
covers the request.

Metadata: LibRaw gives the size and the rotation. Most RAW formats are TIFF
containers, so make, model, date and GPS are read from the first IFD and its Exif
sub-IFD with Pillow's TIFF directory parser; failing that, from the EXIF of the
embedded preview. Formats that are not TIFF (CR3, RAF, and ORF or RW2 with their own
header) give the size only unless the preview carries EXIF.
"""

from __future__ import annotations

import dataclasses
import importlib
import io
import struct
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import IO, Any

import numpy as np
from PIL import Image, TiffImagePlugin

from mediasift.core.errors import ReaderError
from mediasift.core.paths import fs_path
from mediasift.core.types import MediaMetadata
from mediasift.readers.image import (
    DECODE_ERRORS,
    DEFAULT_MAX_PIXELS,
    ExifFields,
    check_pixels,
    display_size,
    finish_image,
    metadata_from_exif,
    parse_exif,
    pillow_exif,
    request_draft,
    translate_decode_error,
)

EXTENSIONS = frozenset(
    {".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf", ".pef", ".srw"}
)

# LibRaw's "flip" (0 none, 3 turn 180, 5 turn 90 counter-clockwise, 6 turn 90 clockwise)
# written as the EXIF orientation that means the same.
_FLIP_TO_ORIENTATION = {0: 1, 3: 3, 5: 8, 6: 6}
_TIFF_HEADS = (b"II*\x00", b"MM\x00*")
_IFD_EXIF = 0x8769
_IFD_GPS = 0x8825


def _import_rawpy() -> ModuleType | None:
    """Import rawpy, or None. Never raises."""
    try:
        return importlib.import_module("rawpy")
    except Exception:  # noqa: BLE001 - a broken native library must read as "missing"
        return None


# --------------------------------------------------------------------------- TIFF EXIF


class TiffExif:
    """EXIF-like view of the first IFD of a TIFF-based RAW file (reads lazily from ``fp``)."""

    def __init__(self, fp: IO[bytes], head: bytes, ifd0: TiffImagePlugin.ImageFileDirectory_v2):
        self._fp = fp
        self._head = head
        self._ifd0 = ifd0
        self._sub: dict[int, Mapping[int, Any]] = {}

    @classmethod
    def read(cls, fp: IO[bytes]) -> TiffExif | None:
        """Parse the first IFD; None when the file is not a plain TIFF container."""
        fp.seek(0)
        head = fp.read(8)
        if len(head) < 8 or head[:4] not in _TIFF_HEADS:
            return None
        ifd = cls._load(fp, head, struct.unpack(cls._order(head) + "I", head[4:8])[0])
        return cls(fp, head, ifd) if ifd is not None else None

    @staticmethod
    def _order(head: bytes) -> str:
        return "<" if head[:2] == b"II" else ">"

    @staticmethod
    def _load(
        fp: IO[bytes], head: bytes, offset: int
    ) -> TiffImagePlugin.ImageFileDirectory_v2 | None:
        if offset < 8:
            return None
        ifd = TiffImagePlugin.ImageFileDirectory_v2(ifh=head)
        fp.seek(offset)
        ifd.load(fp)
        return ifd

    def __contains__(self, tag: object, /) -> bool:
        return tag in self._ifd0

    def get(self, tag: int, default: Any = None, /) -> Any:
        return self._ifd0.get(tag, default)

    def get_ifd(self, tag: int, /) -> Mapping[int, Any]:
        if tag not in self._sub:
            offset = self._ifd0.get(tag)
            ifd = self._load(self._fp, self._head, offset) if isinstance(offset, int) else None
            self._sub[tag] = dict(ifd) if ifd is not None else {}
        return self._sub[tag]


# --------------------------------------------------------------------------- helpers


@contextmanager
def _open_file(path: Path) -> Iterator[IO[bytes]]:
    """Open ``path`` for LibRaw. Handing LibRaw a file object keeps non-ASCII paths safe."""
    try:
        fp = Path(fs_path(path)).open("rb")  # noqa: SIM115 - closed below
    except OSError as exc:
        raise ReaderError(f"cannot open the file: {exc}", kind="io") from exc
    with fp:
        if not fp.read(1):
            raise ReaderError("the file is empty", kind="empty")
        fp.seek(0)
        yield fp


def _same_shape(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True when both are landscape or both portrait; a nearly square one fits either."""
    if min(a) >= 0.98 * max(a) or min(b) >= 0.98 * max(b):
        return True
    return (a[0] > a[1]) == (b[0] > b[1])


def _preview_orientation(
    own: int | None, raw_orientation: int, size: tuple[int, int], target: tuple[int, int]
) -> int:
    """Orientation to apply to a preview.

    Cameras differ: some write the preview already turned, some leave it for the raw's
    rotation, some give it its own EXIF. The first candidate that makes the preview's
    shape agree with the RAW's displayed shape wins.
    """
    for candidate in (own, raw_orientation, 1):
        if candidate is not None and _same_shape(display_size(size, candidate), target):
            return candidate
    return raw_orientation


def _merge(primary: ExifFields, fallback: ExifFields) -> ExifFields:
    """Fill the unknown fields of ``primary`` from ``fallback``."""
    values = {
        f.name: getattr(primary, f.name)
        if getattr(primary, f.name) not in (None, False)
        else getattr(fallback, f.name)
        for f in dataclasses.fields(ExifFields)
    }
    return ExifFields(**values)


class RawReader:
    """Reads camera RAW files when ``rawpy`` is installed."""

    name = "raw"
    version = 1
    extensions = EXTENSIONS

    def __init__(self, max_pixels: int = DEFAULT_MAX_PIXELS) -> None:
        if max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        self.max_pixels = max_pixels

    def available(self) -> bool:
        """True when rawpy can be imported. Checked every time, never cached."""
        return _import_rawpy() is not None

    def read_metadata(self, path: Path) -> MediaMetadata:
        """Size with rotation applied, plus whatever EXIF the file or its preview holds."""
        lib = self._library()
        try:
            with _open_file(path) as fp, lib.imread(fp) as raw:
                sizes = raw.sizes
                flip = int(sizes.flip)
                size = (int(sizes.width), int(sizes.height))
                fields, source = self._exif(fp, raw, lib)
                orientation = fields.orientation or _FLIP_TO_ORIENTATION.get(flip, 1)
                fields = dataclasses.replace(fields, orientation=orientation)
        except ReaderError:
            raise
        except (*DECODE_ERRORS, lib.LibRawError) as exc:
            raise self._translate(exc, lib) from exc
        meta = metadata_from_exif(fields, size, "RAW", {"exif_source": source})
        # LibRaw's flip is the authority for the displayed size, not the EXIF value.
        width, height = display_size(size, _FLIP_TO_ORIENTATION.get(flip, 1))
        return dataclasses.replace(meta, width=width, height=height)

    def load_image(self, path: Path, max_side: int) -> Image.Image:
        """Upright RGB image of about ``max_side``: preview when big enough, else demosaic."""
        if max_side < 1:
            raise ValueError("max_side must be at least 1")
        lib = self._library()
        try:
            with _open_file(path) as fp, lib.imread(fp) as raw:
                sizes = raw.sizes
                size = (int(sizes.width), int(sizes.height))
                check_pixels(size, self.max_pixels)
                raw_orientation = _FLIP_TO_ORIENTATION.get(int(sizes.flip), 1)
                shown = display_size(size, raw_orientation)
                preview = self._preview(raw, lib, max_side, shown, raw_orientation)
                if preview is not None:
                    return preview
                return self._demosaic(raw, max_side, shown)
        except ReaderError:
            raise
        except (*DECODE_ERRORS, lib.LibRawError) as exc:
            raise self._translate(exc, lib) from exc

    # ---- pieces

    @staticmethod
    def _exif(fp: IO[bytes], raw: Any, lib: Any) -> tuple[ExifFields, str]:
        tiff = TiffExif.read(fp)
        fields = parse_exif(tiff)
        sources = ["tiff"] if tiff is not None else []
        if fields.taken_at is None or fields.make is None:
            preview = RawReader._preview_exif(raw, lib)
            if preview is not None:
                fields = _merge(fields, preview)
                sources.append("preview")
        return fields, "+".join(sources) or "none"

    @staticmethod
    def _preview_exif(raw: Any, lib: Any) -> ExifFields | None:
        try:
            thumb = raw.extract_thumb()
            if thumb.format != lib.ThumbFormat.JPEG:
                return None
            with Image.open(io.BytesIO(thumb.data)) as image:
                return parse_exif(pillow_exif(image))
        except (*DECODE_ERRORS, lib.LibRawError):
            return None

    def _preview(
        self,
        raw: Any,
        lib: Any,
        max_side: int,
        shown: tuple[int, int],
        raw_orientation: int,
    ) -> Image.Image | None:
        """The embedded preview, upright, or None when absent or too small for the request."""
        try:
            thumb = raw.extract_thumb()
            if thumb.format == lib.ThumbFormat.JPEG:
                with Image.open(io.BytesIO(thumb.data)) as image:
                    return self._finish_preview(image, max_side, shown, raw_orientation)
            if thumb.format == lib.ThumbFormat.BITMAP:
                with Image.fromarray(np.asarray(thumb.data)) as image:
                    return self._finish_preview(image, max_side, shown, raw_orientation)
        except (*DECODE_ERRORS, lib.LibRawError):
            return None  # an unusable preview is not an error: demosaic instead
        return None

    @staticmethod
    def _finish_preview(
        image: Image.Image, max_side: int, shown: tuple[int, int], raw_orientation: int
    ) -> Image.Image | None:
        check_pixels(image.size, DEFAULT_MAX_PIXELS)
        longest = max(image.size)
        if longest < min(max_side, max(shown)):
            return None  # too small to stand in for the picture
        own = parse_exif(pillow_exif(image)).orientation
        orientation = _preview_orientation(own, raw_orientation, image.size, shown)
        request_draft(image, max_side)
        image.load()
        return finish_image(image, max_side, orientation)

    @staticmethod
    def _demosaic(raw: Any, max_side: int, shown: tuple[int, int]) -> Image.Image:
        # Half size is four times cheaper; use it whenever it still covers the request.
        half = max(shown) // 2 >= max_side
        pixels = raw.postprocess(half_size=half, use_camera_wb=True, output_bps=8)
        with Image.fromarray(pixels) as image:  # LibRaw has already applied the rotation
            return finish_image(image, max_side, None)

    def _library(self) -> Any:
        lib = _import_rawpy()
        if lib is None:
            raise ReaderError(
                "RAW support needs the optional package rawpy (pip install mediasift[raw])",
                kind="unsupported",
            )
        return lib

    @staticmethod
    def _translate(exc: BaseException, lib: Any) -> ReaderError:
        if isinstance(exc, lib.LibRawFileUnsupportedError):
            return ReaderError("not a supported RAW file", kind="unsupported")
        if isinstance(exc, lib.LibRawIOError):
            return ReaderError("the RAW file is cut short", kind="truncated")
        if isinstance(exc, (lib.LibRawTooBigError, lib.LibRawUnsufficientMemoryError)):
            return ReaderError("the RAW image is too large to decode", kind="too_large")
        if isinstance(exc, lib.LibRawError):
            return ReaderError(f"the RAW data is damaged: {exc}", kind="corrupt")
        return translate_decode_error(exc)
