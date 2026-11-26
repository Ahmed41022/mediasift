"""Content hashes: BLAKE2b-256 of a sample of a file, and of all of it.

A hash is only returned when the open file was the expected one before and after it was read.
Links are never followed and files that live in the cloud are never opened.
"""

from __future__ import annotations

import errno
import hashlib
import os
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from mediasift.core import paths
from mediasift.core.errors import FileChanged, ReaderError
from mediasift.core.paths import StrPath
from mediasift.core.types import FileSignature
from mediasift.jobs.context import CancelToken

FEATURE_PARTIAL = "partial"
PARTIAL_VERSION = 1
FEATURE_FULL = "full"
FULL_VERSION = 1

DIGEST_SIZE = 32
BLOCK = 64 * 1024
# Up to this size the three blocks of a sample would cover the whole file, so it is read whole.
WHOLE_LIMIT = 3 * BLOCK
BUFFER = 1024 * 1024

# ReaderError kind of a file whose content is not on this disk. It passes: try again later.
PLACEHOLDER = "placeholder"

OnBytes = Callable[[int], None]

# O_NONBLOCK: a pipe put in the file's place must not hang the open.
_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_BINARY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_SEQUENTIAL = getattr(os, "O_SEQUENTIAL", 0)
# What O_NOFOLLOW answers for a link: ELOOP on Linux and macOS, EMLINK on FreeBSD.
_LINK_ERRORS = {errno.ELOOP, errno.EMLINK}


@dataclass(frozen=True, slots=True)
class PartialHash:
    """Hash of a sample. ``full`` is set when the sample was the whole file.

    ``full`` is then exactly what ``full_hash`` returns, so it can be stored as the full hash.
    """

    partial: bytes
    full: bytes | None = None


def sample_size(size: int) -> int:
    """How many bytes ``partial_hash`` reads of a file of this size."""
    return min(size, WHOLE_LIMIT)


def partial_hash(
    path: StrPath,
    expected: FileSignature,
    cancel: CancelToken | None = None,
    on_bytes: OnBytes | None = None,
) -> PartialHash:
    """Hash the size and the first, middle and last 64 KiB.

    A small file is read whole and gets its full hash from the same read. Different partial
    hashes prove different content; equal ones prove nothing unless ``full`` is set.
    Raises FileChanged, ReaderError (kind "placeholder"), OSError or JobCancelled.
    """
    size = expected.size
    whole = size <= WHOLE_LIMIT
    if whole:
        ranges = [(0, size)]
    else:
        ranges = [(0, BLOCK), (size // 2 - BLOCK // 2, BLOCK), (size - BLOCK, BLOCK)]
    partial = hashlib.blake2b(size.to_bytes(8, "little"), digest_size=DIGEST_SIZE)
    full = hashlib.blake2b(digest_size=DIGEST_SIZE)
    with _open_unchanged(path, expected) as fd:
        for offset, length in ranges:
            for chunk in _read(fd, path, offset, length, cancel, on_bytes):
                partial.update(chunk)
                if whole:
                    full.update(chunk)
    return PartialHash(partial.digest(), full.digest() if whole else None)


def full_hash(
    path: StrPath,
    expected: FileSignature,
    cancel: CancelToken | None = None,
    on_bytes: OnBytes | None = None,
) -> bytes:
    """Plain BLAKE2b-256 of the whole content. Raises like ``partial_hash``."""
    digest = hashlib.blake2b(digest_size=DIGEST_SIZE)
    with _open_unchanged(path, expected, sequential=True) as fd:
        for chunk in _read(fd, path, 0, expected.size, cancel, on_bytes):
            digest.update(chunk)
    return digest.digest()


def check_unchanged(path: StrPath, expected: FileSignature) -> None:
    """Raise FileChanged unless ``path`` still is the expected regular file.

    Raises ReaderError (kind "placeholder") when the content has left this disk, and OSError
    when the path cannot be looked at.
    """
    _check(os.lstat(paths.fs_path(path)), path, expected)


def _check(st: os.stat_result, path: StrPath, expected: FileSignature) -> None:
    name = paths.display_path(path)
    if paths.is_link_like(st) or not stat.S_ISREG(st.st_mode):
        raise FileChanged(f"{name} is no longer a regular file")
    if paths.is_placeholder(st):
        raise ReaderError(f"{name} is not on this disk", kind=PLACEHOLDER)
    now = paths.signature(st)
    # Identity is compared only when both sides know it; unknown is not evidence of a swap.
    known = expected.inode != 0 and now.inode != 0
    if not now.same_content_state(expected) or (known and not now.same_file(expected)):
        raise FileChanged(f"{name} has changed since it was scanned")


@contextmanager
def _open_unchanged(
    path: StrPath, expected: FileSignature, *, sequential: bool = False
) -> Iterator[int]:
    """Open the expected file, and check on leaving that it is still the one at ``path``.

    The look before opening keeps links and placeholders from being opened at all. The look
    at the open handle is the one that counts, as the path may be swapped in between; the
    last look catches a file renamed over the path while it was read.
    """
    name = paths.fs_path(path)
    _check(os.lstat(name), path, expected)
    try:
        fd = os.open(name, _OPEN_FLAGS | (_SEQUENTIAL if sequential else 0))
    except OSError as exc:
        if exc.errno in _LINK_ERRORS:
            raise FileChanged(f"{paths.display_path(path)} has become a link") from exc
        raise
    try:
        _check(os.fstat(fd), path, expected)
        if sequential:
            _advise_sequential(fd)
        yield fd
        _check(os.fstat(fd), path, expected)
    finally:
        os.close(fd)
    _check(os.lstat(name), path, expected)


def _advise_sequential(fd: int) -> None:
    if sys.platform == "linux":
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        except OSError:
            return  # only a hint, which some file systems do not take


def _read(
    fd: int,
    path: StrPath,
    offset: int,
    length: int,
    cancel: CancelToken | None,
    on_bytes: OnBytes | None,
) -> Iterator[bytes]:
    """Yield exactly ``length`` bytes. No more are read of a file that keeps growing."""
    if offset:
        os.lseek(fd, offset, os.SEEK_SET)
    while length:
        if cancel is not None:
            cancel.raise_if_cancelled()
        chunk = os.read(fd, min(BUFFER, length))
        if not chunk:
            raise FileChanged(f"{paths.display_path(path)} became shorter while being read")
        length -= len(chunk)
        yield chunk
        if on_bytes is not None:
            on_bytes(len(chunk))
