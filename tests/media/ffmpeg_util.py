"""Shared ffmpeg helpers for the generated video and audio kit."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

# Applied to whole test modules or single tests that need the encoder.
requires_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None, reason="ffmpeg / ffprobe not installed"
)

# Reproducible output: no encoder version strings, no timestamps.
_DETERMINISTIC = [
    "-map_metadata",
    "-1",
    "-fflags",
    "+bitexact",
    "-flags:v",
    "+bitexact",
    "-flags:a",
    "+bitexact",
]


def run_ffmpeg(args: list[str], out: Path) -> Path:
    """Run ffmpeg with ``args`` (inputs and filters) writing ``out``; raises on failure."""
    if FFMPEG is None:
        raise RuntimeError("ffmpeg is not installed")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-threads", "1", *args]
    proc = subprocess.run(
        [*cmd, *_DETERMINISTIC, str(out)], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")
    return out


def run_ffmpeg_with_tags(args: list[str], out: Path, tags: dict[str, str]) -> Path:
    """Like :func:`run_ffmpeg` but writes container metadata ``tags`` (title, artist ...)."""
    meta: list[str] = []
    for key, value in tags.items():
        meta += ["-metadata", f"{key}={value}"]
    if FFMPEG is None:
        raise RuntimeError("ffmpeg is not installed")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-threads",
        "1",
        *args,
        *meta,
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")
    return out


def probe(path: Path) -> dict[str, Any]:
    """ffprobe output (format and streams) as a dict; raises when the file is unreadable."""
    if FFPROBE is None:
        raise RuntimeError("ffprobe is not installed")
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()}")
    data: dict[str, Any] = json.loads(proc.stdout)
    return data


def streams(path: Path, kind: str) -> list[dict[str, Any]]:
    """The probed streams of ``kind`` ("video" or "audio")."""
    return [s for s in probe(path)["streams"] if s.get("codec_type") == kind]


def tags_of(path: Path) -> dict[str, str]:
    """Lower-cased container and stream tags merged (Ogg and Opus keep them on the stream)."""
    data = probe(path)
    found: dict[str, str] = {}
    for holder in [data["format"], *data["streams"]]:
        found.update({k.lower(): v for k, v in holder.get("tags", {}).items()})
    return found
