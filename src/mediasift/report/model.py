"""Plain rows for reports.

Reports sit below the facade, so they define their own input: the facade converts what it
knows into these and the writers never see a database type.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO


@dataclass(frozen=True, slots=True)
class ReportMember:
    """One file of a group. ``path`` is text fit for people (always encodable)."""

    path: str
    size: int
    modified: str
    media_type: str
    hardlink_of: str = ""


@dataclass(frozen=True, slots=True)
class ReportGroup:
    """A set of identical files. ``size`` is the size of one file."""

    number: int
    kind: str
    size: int
    reclaimable_bytes: int
    members: tuple[ReportMember, ...]


def human_size(size: int) -> str:
    """``1536`` becomes ``1.5 KiB``."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


@contextmanager
def atomic_text(path: Path, *, encoding: str, newline: str | None = None) -> Iterator[IO[str]]:
    """Write to a temporary file next to ``path`` and move it into place when done.

    A report that stops half way (full disk, Ctrl+C) must not replace a good older one.
    """
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding=encoding, newline=newline) as stream:
            yield stream
        Path(temp_name).replace(path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
