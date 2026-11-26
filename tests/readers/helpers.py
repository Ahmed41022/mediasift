"""Small builders and comparisons shared by the reader tests."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image


def mean_diff(a: Image.Image, b: Image.Image) -> float:
    """Mean absolute difference per channel of two RGB images of the same size."""
    assert a.size == b.size, f"sizes differ: {a.size} vs {b.size}"
    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB"), dtype=np.float64)
    return float(np.abs(x - y).mean())


def png_claiming(width: int, height: int) -> bytes:
    """A PNG whose header says ``width`` x ``height`` and whose data is a few bytes.

    Opening it must be safe, and decoding must never be attempted.
    """

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def jpeg_claiming(width: int, height: int) -> bytes:
    """A JPEG header (SOI, SOF0, SOS) that claims ``width`` x ``height`` and has no image data."""
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, height, width, 3)
    sof += b"\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    sos = b"\xff\xda" + struct.pack(">HB", 12, 3) + b"\x01\x00\x02\x11\x03\x11\x00\x3f\x00"
    return b"\xff\xd8" + sof + sos + b"\x00" * 16 + b"\xff\xd9"


class FakeExif:
    """A stand-in for ``PIL.Image.Exif`` built from plain dictionaries."""

    def __init__(
        self,
        ifd0: Mapping[int, Any] | None = None,
        exif: Mapping[int, Any] | None = None,
        gps: Mapping[int, Any] | None = None,
    ) -> None:
        self._ifd0 = dict(ifd0 or {})
        self._ifds = {0x8769: dict(exif or {}), 0x8825: dict(gps or {})}

    def __contains__(self, tag: object, /) -> bool:
        return tag in self._ifd0

    def get(self, tag: int, default: Any = None, /) -> Any:
        return self._ifd0.get(tag, default)

    def get_ifd(self, tag: int, /) -> Mapping[int, Any]:
        return self._ifds.get(tag, {})
