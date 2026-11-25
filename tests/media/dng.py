"""A tiny valid DNG written by hand (TIFF structure with struct).

Uncompressed 16-bit RGGB CFA in IFD0, an Exif sub-IFD holding DateTimeOriginal, and
a small JPEG preview in a SubIFD. libraw (rawpy) opens it and can demosaic it.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tests.media.images import scene

BYTE, ASCII, SHORT, LONG, RATIONAL, UNDEFINED, SRATIONAL = 1, 2, 3, 4, 5, 7, 10
_SIZES = {BYTE: 1, ASCII: 1, SHORT: 2, LONG: 4, RATIONAL: 8, UNDEFINED: 1, SRATIONAL: 8}
_FMT = {BYTE: "B", ASCII: "B", SHORT: "H", LONG: "I", UNDEFINED: "B"}

# A tag value: (tag, type, values). Values for RATIONAL types are flat (num, den, ...).
Entry = tuple[int, int, list[int] | bytes]


def _pack(kind: int, values: list[int] | bytes) -> bytes:
    if isinstance(values, bytes):
        return values
    if kind in (RATIONAL, SRATIONAL):
        return struct.pack("<" + ("I" if kind == RATIONAL else "i") * len(values), *values)
    return struct.pack("<" + _FMT[kind] * len(values), *values)


def _count(kind: int, values: list[int] | bytes) -> int:
    n = len(values)
    return n // 2 if kind in (RATIONAL, SRATIONAL) and not isinstance(values, bytes) else n


def _ifd(entries: list[Entry], offset: int) -> bytes:
    """Serialise one IFD placed at ``offset``; oversized values follow the entry table."""
    entries = sorted(entries, key=lambda e: e[0])
    table = struct.pack("<H", len(entries))
    extra = b""
    extra_base = offset + 2 + 12 * len(entries) + 4
    for tag, kind, values in entries:
        raw = _pack(kind, values)
        count = _count(kind, values)
        if len(raw) <= 4:
            field = raw.ljust(4, b"\x00")
        else:
            field = struct.pack("<I", extra_base + len(extra))
            extra += raw + (b"\x00" if len(raw) % 2 else b"")
        table += struct.pack("<HHI", tag, kind, count) + field
    return table + struct.pack("<I", 0) + extra


def _ascii(text: str) -> bytes:
    return text.encode("ascii") + b"\x00"


def _mosaic(width: int, height: int, seed: int) -> np.ndarray:
    """RGGB Bayer samples (12-bit range, stored as uint16) taken from a generated scene."""
    rgb = np.asarray(scene(seed, (width, height)).convert("RGB"), dtype=np.float32) / 255.0
    lin = rgb**2.2
    mosaic = np.empty((height, width), dtype=np.float32)
    mosaic[0::2, 0::2] = lin[0::2, 0::2, 0]
    mosaic[0::2, 1::2] = lin[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = lin[1::2, 0::2, 1]
    mosaic[1::2, 1::2] = lin[1::2, 1::2, 2]
    return np.clip(mosaic * 4000 + 64, 0, 4095).astype("<u2")


@dataclass(frozen=True)
class DngInfo:
    path: Path
    width: int
    height: int
    make: str
    model: str
    taken: str


def write_dng(
    path: Path,
    *,
    seed: int = 1,
    width: int = 96,
    height: int = 64,
    make: str = "MediaSift",
    model: str = "Test Camera 1",
    taken: str = "2021:07:04 10:30:00",
    preview: bool = True,
) -> DngInfo:
    """Write a valid uncompressed CFA DNG and return what it contains."""
    if width % 2 or height % 2:
        raise ValueError("width and height must be even for an RGGB mosaic")
    pixels = _mosaic(width, height, seed).tobytes()
    jpeg = b""
    if preview:
        buf = io.BytesIO()
        scene(seed, (width, height)).save(buf, format="JPEG", quality=85)
        jpeg = buf.getvalue()

    def build(off: dict[str, int]) -> list[tuple[int, list[Entry]]]:
        ifd0: list[Entry] = [
            (254, LONG, [0]),
            (256, LONG, [width]),
            (257, LONG, [height]),
            (258, SHORT, [16]),
            (259, SHORT, [1]),
            (262, SHORT, [32803]),
            (271, ASCII, _ascii(make)),
            (272, ASCII, _ascii(model)),
            (273, LONG, [off["pixels"]]),
            (274, SHORT, [1]),
            (277, SHORT, [1]),
            (278, LONG, [height]),
            (279, LONG, [len(pixels)]),
            (284, SHORT, [1]),
            (305, ASCII, _ascii("mediasift tests")),
            (306, ASCII, _ascii(taken)),
            (34665, LONG, [off["exif"]]),
            (33421, SHORT, [2, 2]),
            (33422, BYTE, [0, 1, 1, 2]),
            (50706, BYTE, [1, 4, 0, 0]),
            (50707, BYTE, [1, 1, 0, 0]),
            (50708, ASCII, _ascii(f"{make} {model}")),
            (50711, SHORT, [1]),
            (50714, SHORT, [64]),
            (50717, SHORT, [4095]),
            (
                50721,
                SRATIONAL,
                [
                    10000,
                    10000,
                    0,
                    10000,
                    0,
                    10000,
                    0,
                    10000,
                    10000,
                    10000,
                    0,
                    10000,
                    0,
                    10000,
                    0,
                    10000,
                    10000,
                    10000,
                ],
            ),
            (50728, RATIONAL, [1, 1, 1, 1, 1, 1]),
            (50778, SHORT, [21]),
        ]
        if preview:
            ifd0.append((330, LONG, [off["preview"]]))
        exif: list[Entry] = [(36867, ASCII, _ascii(taken)), (36868, ASCII, _ascii(taken))]
        out = [(off["ifd0"], ifd0), (off["exif"], exif)]
        if preview:
            thumb: list[Entry] = [
                (254, LONG, [1]),
                (256, LONG, [width]),
                (257, LONG, [height]),
                (258, SHORT, [8, 8, 8]),
                (259, SHORT, [6]),
                (262, SHORT, [6]),
                (277, SHORT, [3]),
                (513, LONG, [off["jpeg"]]),
                (514, LONG, [len(jpeg)]),
            ]
            out.append((off["preview"], thumb))
        return out

    offsets = {"ifd0": 8, "exif": 0, "preview": 0, "jpeg": 0, "pixels": 0}
    for _ in range(2):  # sizes do not depend on offset values: two passes settle them
        pos = 8
        for name in ("ifd0", "exif", "preview"):
            if name == "preview" and not preview:
                continue
            offsets[name] = pos
            entries = dict(build(offsets))[offsets[name]]
            pos += len(_ifd(entries, pos))
            pos += pos % 2
        offsets["jpeg"], pos = pos, pos + len(jpeg) + len(jpeg) % 2
        offsets["pixels"] = pos
    body = bytearray(b"II*\x00" + struct.pack("<I", 8))
    for at, entries in build(offsets):
        body.extend(b"\x00" * (at - len(body)))
        body.extend(_ifd(entries, at))
    body.extend(b"\x00" * (offsets["jpeg"] - len(body)))
    body.extend(jpeg)
    body.extend(b"\x00" * (offsets["pixels"] - len(body)))
    body.extend(pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(body))
    return DngInfo(path, width, height, make, model, taken)
