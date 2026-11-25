"""Read files in chunks that a cancel can interrupt."""

from __future__ import annotations

import os
from collections.abc import Iterator
from io import FileIO
from typing import Protocol

from mediasift.jobs.context import CancelToken

DEFAULT_CHUNK_SIZE = 1024 * 1024


class SupportsRead(Protocol):
    def read(self, size: int, /) -> bytes | None: ...


AnyPath = str | bytes | os.PathLike[str] | os.PathLike[bytes]


def read_chunks(
    source: AnyPath | SupportsRead,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    cancel: CancelToken | None = None,
) -> Iterator[bytes]:
    """Yield ``source`` in chunks of at most ``chunk_size`` bytes.

    The token is checked before every read, the one that finds the end included, so a loop that
    ends normally has seen the whole file. A path is opened unbuffered at the first chunk and
    closed at the end; an open file is read from its position and left open. Each chunk is its
    own object: a shared buffer would be faster, but corrupts data for a caller that keeps one.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if isinstance(source, str | bytes | os.PathLike):
        return _read_path(source, chunk_size, cancel)
    return _read_file(source, chunk_size, cancel)


def _read_path(path: AnyPath, chunk_size: int, cancel: CancelToken | None) -> Iterator[bytes]:
    if cancel is not None:
        cancel.raise_if_cancelled()
    # FileIO takes the path as it is: no normalising that would alter long or odd names.
    with FileIO(os.fspath(path)) as file:
        yield from _read_file(file, chunk_size, cancel)


def _read_file(file: SupportsRead, chunk_size: int, cancel: CancelToken | None) -> Iterator[bytes]:
    while True:
        if cancel is not None:
            cancel.raise_if_cancelled()
        chunk = file.read(chunk_size)
        if not chunk:
            return
        yield chunk
