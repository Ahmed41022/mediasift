"""Broken-file generators: every function is deterministic."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def truncated(data: bytes, fraction: float) -> bytes:
    """The first ``fraction`` of the bytes (at least one byte, never all of them)."""
    n = min(max(1, int(len(data) * fraction)), max(1, len(data) - 1))
    return data[:n]


def bitflips(data: bytes, count: int = 8, seed: int = 0, protect_header: int = 32) -> bytes:
    """Flip ``count`` random bits after the first ``protect_header`` bytes (keeps the magic)."""
    if len(data) <= protect_header:
        raise ValueError("data too short to flip bits outside the header")
    rng = np.random.Generator(np.random.PCG64(seed))
    out = bytearray(data)
    positions = rng.choice(len(data) - protect_header, size=count, replace=False)
    for pos in positions:
        out[protect_header + int(pos)] ^= 1 << int(rng.integers(0, 8))
    return bytes(out)


def header_only(data: bytes, length: int = 24) -> bytes:
    """Just the first ``length`` bytes: a valid magic number and nothing else."""
    return data[:length]


def garbage(size: int = 4096, seed: int = 0) -> bytes:
    """Random bytes that start with no known magic number."""
    rng = np.random.Generator(np.random.PCG64(seed))
    body = rng.bytes(size)
    return b"\x00NOT" + body[4:] if body[:2] in (b"\xff\xd8", b"\x89P") else body


def write_zero_bytes(path: Path) -> Path:
    """An empty file with an image-looking name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def write_garbage(path: Path, size: int = 4096, seed: int = 0) -> Path:
    """Random bytes under ``path`` (use an image extension to imitate a broken photo)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(garbage(size, seed))
    return path
