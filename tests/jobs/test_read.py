from __future__ import annotations

import io
import os
from pathlib import Path

import pytest

from mediasift.core.errors import JobCancelled
from mediasift.jobs.context import CancelToken
from mediasift.jobs.read import read_chunks


@pytest.mark.parametrize("size", [0, 1, 9, 10, 11, 25])
@pytest.mark.parametrize("kind", ["path", "str", "bytes"])
def test_path(tmp_path: Path, size: int, kind: str) -> None:
    path = tmp_path / "صورة.bin"
    data = bytes(range(size))
    path.write_bytes(data)
    source = {"path": path, "str": str(path), "bytes": os.fsencode(path)}[kind]
    chunks = list(read_chunks(source, 10))
    assert b"".join(chunks) == data
    assert [len(c) for c in chunks[:-1]] == [10] * (len(chunks) - 1)
    assert all(chunks)


def test_open_file_is_read_from_its_position_and_left_open() -> None:
    file = io.BytesIO(b"0123456789")
    file.seek(3)
    assert list(read_chunks(file, 4)) == [b"3456", b"789"]
    assert not file.closed


def test_path_is_closed_at_the_end_and_on_cancel(tmp_path: Path) -> None:
    path = tmp_path / "a"
    path.write_bytes(b"abc")
    list(read_chunks(path))
    path.rename(tmp_path / "b")  # Windows refuses while the file is open

    token = CancelToken()
    chunks = read_chunks(tmp_path / "b", 1, token)
    next(chunks)
    token.cancel()
    with pytest.raises(JobCancelled):
        next(chunks)
    (tmp_path / "b").rename(tmp_path / "c")


def test_path_is_opened_at_the_first_chunk(tmp_path: Path) -> None:
    chunks = read_chunks(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        next(chunks)


def test_a_cancelled_token_prevents_opening(tmp_path: Path) -> None:
    token = CancelToken()
    token.cancel()
    with pytest.raises(JobCancelled):
        next(read_chunks(tmp_path / "missing", cancel=token))


class CountingFile(io.BytesIO):
    reads = 0

    def read(self, size: int | None = -1, /) -> bytes:
        self.reads += 1
        return super().read(size)


def test_cancel_between_chunks() -> None:
    token = CancelToken()
    file = CountingFile(b"abcdef")
    got = []
    with pytest.raises(JobCancelled):
        for chunk in read_chunks(file, 2, token):
            got.append(chunk)
            if len(got) == 2:
                token.cancel()
    assert got == [b"ab", b"cd"]
    assert file.reads == 2


def test_cancel_after_the_last_chunk_does_not_look_like_the_end() -> None:
    token = CancelToken()
    chunks = read_chunks(io.BytesIO(b"ab"), 2, token)
    assert next(chunks) == b"ab"
    token.cancel()
    with pytest.raises(JobCancelled):
        next(chunks)


def test_chunks_can_be_kept() -> None:
    assert list(read_chunks(io.BytesIO(b"abcd"), 2)) == [b"ab", b"cd"]


def test_chunk_size_is_checked_at_the_call() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        read_chunks(io.BytesIO(), 0)
