"""HEIC and HEIF reader on top of ``pi_heif`` (or ``pillow_heif``), both optional.

The library is imported lazily and used directly through ``open_heif``: it is never
registered as a Pillow plugin, so importing MediaSift changes nothing global.

Orientation, the one subtle part. libheif applies the container's own rotation and
mirroring boxes (``irot`` and ``imir``) while decoding, so sizes and pixels are
already upright for such files and the EXIF orientation inside them must be ignored,
or the picture would be turned twice. A file that has no such box is turned by its
EXIF orientation, like a JPEG. The check looks for the boxes anywhere in the file's
property container, so a thumbnail's rotation can in theory be mistaken for the
main picture's; phones write both the same way.
"""

from __future__ import annotations

import dataclasses
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import IO, Any

from PIL import Image

from mediasift.core.errors import ReaderError
from mediasift.core.types import MediaMetadata
from mediasift.readers.image import (
    DECODE_ERRORS,
    DEFAULT_MAX_PIXELS,
    ExifSource,
    check_pixels,
    exif_from_bytes,
    finish_image,
    metadata_from_exif,
    parse_exif,
    translate_decode_error,
)

EXTENSIONS = frozenset({".heic", ".heif", ".hif"})
_LIBRARIES = ("pi_heif", "pillow_heif")
_MAX_META_BYTES = 16 * 1024 * 1024
_TRANSFORM_BOXES = (b"irot", b"imir")


def _import_library() -> ModuleType | None:
    """Import the first available HEIF library, or None. Never raises."""
    import importlib

    for name in _LIBRARIES:
        try:
            return importlib.import_module(name)
        except Exception:  # noqa: BLE001 - a broken native library must read as "missing"
            continue
    return None


def _boxes(data: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """Yield ``(type, payload_start, payload_end)`` of the ISO boxes in ``data[start:end]``."""
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", data, pos)
        header = 8
        if size == 1 and pos + 16 <= end:
            (size,) = struct.unpack_from(">Q", data, pos + 8)
            header = 16
        elif size == 0:
            size = end - pos
        if size < header:
            return
        yield kind, pos + header, min(pos + size, end)
        pos += size


def _find(data: bytes, start: int, end: int, kind: bytes) -> tuple[int, int] | None:
    for found, begin, stop in _boxes(data, start, end):
        if found == kind:
            return begin, stop
    return None


def has_container_transform(fp: IO[bytes]) -> bool:
    """True when the file declares a rotation or mirror (``irot``, ``imir``) itself."""
    fp.seek(0)
    while True:
        header = fp.read(16)
        if len(header) < 8:
            return False
        size, kind = struct.unpack_from(">I4s", header)
        header_len = 8
        if size == 1 and len(header) == 16:
            (size,) = struct.unpack_from(">Q", header, 8)
            header_len = 16
        if kind == b"meta":
            fp.seek(-(len(header) - header_len), 1)
            payload = fp.read(min(max(size - header_len, 0), _MAX_META_BYTES))
            return _meta_has_transform(payload)
        if size < header_len:
            return False
        fp.seek(size - len(header), 1)


def _meta_has_transform(payload: bytes) -> bool:
    """Look for ``irot`` or ``imir`` in the ``iprp/ipco`` boxes of a ``meta`` payload."""
    iprp = _find(payload, 4, len(payload), b"iprp")  # skip the full-box version and flags
    ipco = _find(payload, *iprp, b"ipco") if iprp else None
    if ipco is None:
        return False
    return any(kind in _TRANSFORM_BOXES for kind, _a, _b in _boxes(payload, *ipco))


@contextmanager
def _open_file(path: Path) -> Iterator[IO[bytes]]:
    try:
        fp = path.open("rb")
    except OSError as exc:
        raise ReaderError(f"cannot open the file: {exc}", kind="io") from exc
    with fp:
        if not fp.read(1):
            raise ReaderError("the file is empty", kind="empty")
        fp.seek(0)
        yield fp


def _exif_of(heif: Any) -> ExifSource | None:
    data = heif.info.get("exif")
    return exif_from_bytes(data) if data else None


class HeifReader:
    """Reads HEIC and HEIF files when ``pi_heif`` or ``pillow_heif`` is installed."""

    name = "heif"
    version = 1
    extensions = EXTENSIONS

    def __init__(self, max_pixels: int = DEFAULT_MAX_PIXELS) -> None:
        if max_pixels < 1:
            raise ValueError("max_pixels must be positive")
        self.max_pixels = max_pixels

    def available(self) -> bool:
        """True when a HEIF library can be imported. Checked every time, never cached."""
        return _import_library() is not None

    def read_metadata(self, path: Path) -> MediaMetadata:
        """Size as displayed, EXIF facts and the codec name."""
        lib = self._library()
        try:
            with _open_file(path) as fp:
                transformed = has_container_transform(fp)
                fp.seek(0)
                heif = lib.open_heif(fp)
                fields = parse_exif(_exif_of(heif))
                size = (int(heif.size[0]), int(heif.size[1]))
                meta = metadata_from_exif(
                    fields, size, "HEIF", {"mode": heif.mode, "heif_transform": transformed}
                )
        except ReaderError:
            raise
        except DECODE_ERRORS as exc:
            raise self._translate(exc) from exc
        # libheif already turned the pixels: the size is the display size.
        return dataclasses.replace(meta, width=size[0], height=size[1]) if transformed else meta

    def load_image(self, path: Path, max_side: int) -> Image.Image:
        """Upright RGB image of about ``max_side``; HDR is reduced to 8 bits."""
        if max_side < 1:
            raise ValueError("max_side must be at least 1")
        lib = self._library()
        try:
            with _open_file(path) as fp:
                transformed = has_container_transform(fp)
                fp.seek(0)
                heif = lib.open_heif(fp)
                check_pixels((int(heif.size[0]), int(heif.size[1])), self.max_pixels)
                orientation = None if transformed else parse_exif(_exif_of(heif)).orientation
                decoded = heif.to_pillow()
                decoded.load()
                return finish_image(decoded, max_side, orientation)
        except ReaderError:
            raise
        except DECODE_ERRORS as exc:
            raise self._translate(exc) from exc

    def _library(self) -> Any:
        lib = _import_library()
        if lib is None:
            raise ReaderError(
                "HEIC support needs the optional package pi-heif (pip install mediasift[heic])",
                kind="unsupported",
            )
        return lib

    @staticmethod
    def _translate(exc: BaseException) -> ReaderError:
        if isinstance(exc, ValueError) and "ftyp" in str(exc):
            return ReaderError("not a HEIF file", kind="unsupported")
        return translate_decode_error(exc)
