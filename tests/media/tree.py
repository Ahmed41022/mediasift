"""Small libraries on disk with planted duplicates, plus a fast bulk writer for benchmarks."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tests.media.images import scene, to_bytes

KIB = 1024
ARABIC_DIR = "صور العائلة"
TRIP_DIR = "رحلة/صيف ٢٠٢٣"


@dataclass(frozen=True)
class ExactGroup:
    """Files with identical content. ``hardlinked`` groups are one inode with several names."""

    sha256: str
    size: int
    paths: tuple[Path, ...]
    hardlinked: bool = False


@dataclass
class Expected:
    """What :func:`build_tree` planted, for a test to compare a scan against."""

    root: Path
    seed: int
    exact_groups: list[ExactGroup] = field(default_factory=list)
    media_paths: list[Path] = field(default_factory=list)
    non_media_paths: list[Path] = field(default_factory=list)
    empty_files: list[Path] = field(default_factory=list)
    # Different content, identical size: must never be reported as duplicates.
    same_size_different: list[tuple[Path, ...]] = field(default_factory=list)
    # Identical first and last blocks, different middle, larger than 256 KiB.
    same_head_tail_different_middle: list[tuple[Path, ...]] = field(default_factory=list)

    @property
    def duplicate_groups(self) -> list[ExactGroup]:
        """Groups of separate files (hard-linked names excluded)."""
        return [g for g in self.exact_groups if not g.hardlinked]

    @property
    def hardlink_groups(self) -> list[ExactGroup]:
        return [g for g in self.exact_groups if g.hardlinked]

    @property
    def reclaimable_bytes(self) -> int:
        """Bytes freed by keeping one file of each duplicate group."""
        return sum(g.size * (len(g.paths) - 1) for g in self.duplicate_groups)

    @property
    def all_paths(self) -> list[Path]:
        return [*self.media_paths, *self.non_media_paths, *self.empty_files]


def _bytes(seed: int, tag: int, size: int) -> bytes:
    """Deterministic pseudo-random content for (seed, tag)."""
    return np.random.Generator(np.random.PCG64([seed, tag])).bytes(size)


def _jpeg(seed: int, tag: int) -> bytes:
    return to_bytes(scene(seed * 1000 + tag, (96, 72)), "JPEG", quality=85)


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_tree(root: Path, seed: int = 0, *, hardlinks: bool = True) -> Expected:
    """Write a small library under ``root`` and describe it.

    Planted: duplicate groups spread over folders (Arabic names, spaces, emoji, upper-case
    extensions), a hard-linked pair, empty files, same-size files with different content,
    large files that share their head and tail, and non-media files. Names are fixed;
    contents depend on ``seed``.
    """
    exp = Expected(root=root, seed=seed)
    root.mkdir(parents=True, exist_ok=True)
    groups: list[tuple[bytes, list[str]]] = [
        (
            _jpeg(seed, 1),
            ["2023/IMG_0001.jpg", f"{ARABIC_DIR}/صورة ١.jpg", "backup/IMG_0001 copy.JPG"],
        ),
        (_jpeg(seed, 2), [f"{TRIP_DIR}/شاطئ.jpg", "My Holiday ✈ 2020 😀.jpg"]),
        (
            _bytes(seed, 3, 40 * KIB),
            ["clips/clip.MP4", "clips/old/clip final.mp4", f"{ARABIC_DIR}/فيديو.MOV"],
        ),
        (_bytes(seed, 4, 9 * KIB), ["music/Song.Mp3", "music/copy of song.mp3"]),
        (_bytes(seed, 5, 300 * KIB), ["raw/DSC_0100.NEF", "raw/archive/DSC_0100.nef"]),
    ]
    for data, names in groups:
        paths = tuple(_write(root / name, data) for name in names)
        exp.exact_groups.append(ExactGroup(_sha(data), len(data), paths))
        exp.media_paths.extend(paths)

    unique = [
        ("unique/sunset.jpg", _jpeg(seed, 10)),
        ("unique/space name/beach day.jpeg", _jpeg(seed, 11)),
        (f"{ARABIC_DIR}/فريدة.jpg", _jpeg(seed, 12)),
        ("unique/talk.m4a", _bytes(seed, 13, 6 * KIB)),
    ]
    for name, data in unique:
        exp.media_paths.append(_write(root / name, data))

    # Same size, different content: an exact matcher must keep them apart.
    twins = tuple(
        _write(root / f"samesize/file{i}.jpg", _bytes(seed, 20 + i, 5000)) for i in range(3)
    )
    exp.same_size_different.append(twins)
    exp.media_paths.extend(twins)

    # Same head and tail, different middle (sampling hashes must not call these equal).
    head, tail = _bytes(seed, 30, 200 * KIB), _bytes(seed, 31, 200 * KIB)
    big = tuple(
        _write(root / f"bigfiles/video{i}.mp4", head + _bytes(seed, 32 + i, 300 * KIB) + tail)
        for i in range(2)
    )
    exp.same_head_tail_different_middle.append(big)
    exp.media_paths.extend(big)

    for name in ("empty/zero.jpg", "empty/zero2.JPG", f"{ARABIC_DIR}/فارغ.png"):
        exp.empty_files.append(_write(root / name, b""))

    for name, text in (
        ("docs/notes.txt", f"notes {seed}\n"),
        ("docs/ملاحظات.txt", "ملاحظات\n"),
        ("docs/data.json", "{}\n"),
        ("unique/Thumbs.db", f"db {seed}"),
        ("README", "no extension\n"),
    ):
        exp.non_media_paths.append(_write(root / name, text.encode("utf-8")))

    if hardlinks:
        _plant_hardlink(exp, root, _jpeg(seed, 40))
    return exp


def _plant_hardlink(exp: Expected, root: Path, data: bytes) -> None:
    first = _write(root / "links/original.jpg", data)
    second = root / "links/hard link.jpg"
    try:
        os.link(first, second)
    except (OSError, NotImplementedError):
        exp.media_paths.append(first)
        return
    exp.exact_groups.append(ExactGroup(_sha(data), len(data), (first, second), hardlinked=True))
    exp.media_paths.extend([first, second])


# ----------------------------------------------------------------------- bulk benchmark tree


@dataclass(frozen=True)
class CheapTree:
    root: Path
    count: int
    duplicate_groups: int  # pairs of identical files
    duplicate_files: int  # files that are an extra copy


_EXTS = (".jpg", ".png", ".mp4", ".mp3", ".heic", ".mov")


def cheap_tree(root: Path, count: int, dup_every: int = 10, per_dir: int = 500) -> CheapTree:
    """Write ``count`` tiny fake media files quickly; every ``dup_every``-th copies its predecessor.

    Content is a few dozen bytes derived from the index (not decodable media), so only
    scanning and hashing speed is measured.
    """
    if dup_every < 2:
        raise ValueError("dup_every must be at least 2")
    root.mkdir(parents=True, exist_ok=True)
    made_dirs: set[int] = set()
    dups = 0
    for i in range(count):
        d = i // per_dir
        if d not in made_dirs:
            (root / f"d{d:05d}").mkdir(exist_ok=True)
            made_dirs.add(d)
        is_dup = i > 0 and i % dup_every == 0
        dups += is_dup
        content = i - 1 if is_dup else i
        payload = b"MEDIASIFT-CHEAP:%010d:" % content + b"x" * (content % 37)
        with open(root / f"d{d:05d}" / f"f{i:07d}{_EXTS[i % len(_EXTS)]}", "wb") as f:
            f.write(payload)
    return CheapTree(root, count, dups, dups)
