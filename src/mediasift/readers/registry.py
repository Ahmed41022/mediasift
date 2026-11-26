"""The built-in image readers, found by file extension."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cache
from pathlib import Path

from mediasift.core.plugins import Reader, Registry
from mediasift.readers.heif import HeifReader
from mediasift.readers.image import PillowReader
from mediasift.readers.raw import RawReader


def default_readers() -> Registry[Reader]:
    """A new registry holding the built-in readers, in the order they are tried."""
    registry: Registry[Reader] = Registry("reader")
    registry.register(PillowReader())
    registry.register(HeifReader())
    registry.register(RawReader())
    return registry


@cache
def _shared() -> Registry[Reader]:
    return default_readers()


def reader_for(path: Path | str, readers: Iterable[Reader] | None = None) -> Reader | None:
    """The reader that handles ``path`` by its extension (any case), or None.

    A reader whose optional library is missing is still returned: callers check
    ``available()`` and report the skipped file, which is better than treating a
    HEIC photo as "not an image".
    """
    extension = Path(path).suffix.lower()
    for reader in _shared() if readers is None else readers:
        if extension in reader.extensions:
            return reader
    return None


def unavailable_extensions(readers: Iterable[Reader] | None = None) -> frozenset[str]:
    """Extensions that cannot be read right now because an optional library is missing."""
    missing: set[str] = set()
    for reader in _shared() if readers is None else readers:
        if not reader.available():
            missing |= reader.extensions
    return frozenset(missing)
