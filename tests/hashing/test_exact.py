from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from mediasift.core import paths
from mediasift.core.errors import FileChanged, JobCancelled, ReaderError
from mediasift.hashing import exact
from mediasift.hashing.exact import (
    BLOCK,
    BUFFER,
    WHOLE_LIMIT,
    PartialHash,
    check_unchanged,
    full_hash,
    partial_hash,
    sample_size,
)
from mediasift.jobs.context import CancelToken

KIB = 1024


def write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def blake(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=32).digest()


def noise(size: int, seed: int = 0) -> bytes:
    return (hashlib.sha256(str(seed).encode()).digest() * (size // 32 + 1))[:size]


def test_full_hash_is_blake2b_256_of_the_content(tmp_path: Path) -> None:
    data = os.urandom(3 * BUFFER + 17)
    path = write(tmp_path / "a.bin", data)
    seen: list[int] = []
    assert full_hash(path, paths.signature(path), on_bytes=seen.append) == blake(data)
    assert sum(seen) == len(data)
    assert max(seen) <= BUFFER


@pytest.mark.parametrize("size", [0, 1, BLOCK, WHOLE_LIMIT])
def test_small_files_get_both_hashes_from_one_read(tmp_path: Path, size: int) -> None:
    data = os.urandom(size)
    path = write(tmp_path / "a.bin", data)
    seen: list[int] = []
    result = partial_hash(path, paths.signature(path), on_bytes=seen.append)
    assert result.full == blake(data) == full_hash(path, paths.signature(path))
    assert sum(seen) == size == sample_size(size)


def test_larger_files_are_sampled(tmp_path: Path) -> None:
    size = WHOLE_LIMIT + 1
    path = write(tmp_path / "a.bin", os.urandom(size))
    seen: list[int] = []
    result = partial_hash(path, paths.signature(path), on_bytes=seen.append)
    assert result.full is None
    assert sum(seen) == 3 * BLOCK == sample_size(size)


def test_the_sample_covers_head_middle_tail_and_size(tmp_path: Path) -> None:
    size = 700 * KIB
    base = noise(size)

    def partial_of(data: bytes, name: str) -> bytes:
        path = write(tmp_path / name, data)
        return partial_hash(path, paths.signature(path)).partial

    reference = partial_of(base, "base")
    for n, offset in enumerate([0, size - 1, size // 2, size // 2 - BLOCK // 2]):
        changed = bytearray(base)
        changed[offset] ^= 1
        assert partial_of(bytes(changed), f"c{n}") != reference
    # Outside the three blocks a change goes unnoticed: that is what full hashes are for.
    changed = bytearray(base)
    changed[BLOCK + 10] ^= 1
    assert partial_of(bytes(changed), "outside") == reference
    # The size is part of the sample, so a longer file with the same blocks differs.
    assert partial_of(base[: size // 2] + b"x" * 2 + base[size // 2 :], "longer") != reference


def test_a_partial_hash_never_equals_a_full_hash(tmp_path: Path) -> None:
    path = write(tmp_path / "a.bin", b"abc")
    result = partial_hash(path, paths.signature(path))
    assert result == PartialHash(result.partial, blake(b"abc"))
    assert result.partial != result.full


def test_a_stale_signature_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "a.bin", b"abc")
    sig = paths.signature(path)
    write(path, b"abcd")
    for hasher in (partial_hash, full_hash):
        with pytest.raises(FileChanged):
            hasher(path, sig)


def test_a_file_replaced_with_same_size_and_time_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "a.bin", b"abc")
    sig = paths.signature(path)
    if not sig.inode:
        pytest.skip("this file system has no file identity")
    other = write(tmp_path / "b.bin", b"xyz")
    os.utime(other, ns=(sig.mtime_ns, sig.mtime_ns))
    os.replace(other, path)
    with pytest.raises(FileChanged):
        full_hash(path, sig)


def test_growth_while_reading(tmp_path: Path) -> None:
    path = write(tmp_path / "a.bin", os.urandom(2 * BUFFER))
    sig = paths.signature(path)

    def grow(_: int) -> None:
        with path.open("ab") as f:
            f.write(b"more")

    with pytest.raises(FileChanged):
        full_hash(path, sig, on_bytes=grow)
    with pytest.raises(FileChanged):
        partial_hash(path, sig, on_bytes=grow)


def test_truncation_while_reading(tmp_path: Path) -> None:
    path = write(tmp_path / "a.bin", os.urandom(3 * BUFFER))
    sig = paths.signature(path)

    def cut(_: int) -> None:
        os.truncate(path, BUFFER)

    with pytest.raises(FileChanged, match="shorter"):
        full_hash(path, sig, on_bytes=cut)


def test_replacement_while_reading(tmp_path: Path) -> None:
    data = os.urandom(2 * BUFFER)
    path = write(tmp_path / "a.bin", data)
    sig = paths.signature(path)
    other = write(tmp_path / "b.bin", os.urandom(2 * BUFFER))
    os.utime(other, ns=(sig.mtime_ns, sig.mtime_ns))
    replaced: list[bool] = []

    def swap(_: int) -> None:
        if replaced:
            return
        try:
            os.replace(other, path)
            replaced.append(True)
        except PermissionError:
            replaced.append(False)  # Windows: the open file cannot be replaced

    if not sig.inode:
        pytest.skip("this file system has no file identity")
    if os.name == "nt":
        assert full_hash(path, sig, on_bytes=swap) == blake(data)
        assert replaced == [False]
    else:
        with pytest.raises(FileChanged):
            full_hash(path, sig, on_bytes=swap)
        assert replaced == [True]


def test_a_symlink_in_place_of_the_file_is_not_followed(tmp_path: Path) -> None:
    path = write(tmp_path / "a.bin", b"abc")
    sig = paths.signature(path)
    target = write(tmp_path / "elsewhere.bin", b"abc")
    os.utime(target, ns=(sig.mtime_ns, sig.mtime_ns))
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("cannot create symlinks here")
    with pytest.raises(FileChanged):
        check_unchanged(path, sig)
    for hasher in (partial_hash, full_hash):
        with pytest.raises(FileChanged):
            hasher(path, sig)


def test_a_link_swapped_in_after_the_first_look_is_not_followed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(tmp_path / "a.bin", b"abc")
    sig = paths.signature(path)
    target = write(tmp_path / "elsewhere.bin", b"abc")
    os.utime(target, ns=(sig.mtime_ns, sig.mtime_ns))
    real_open = os.open

    def swap_then_open(name: str, flags: int, *args: int) -> int:
        os.unlink(name)
        os.symlink(target, name)
        return real_open(name, flags, *args)

    try:
        os.symlink(target, tmp_path / "probe")
    except OSError:
        pytest.skip("cannot create symlinks here")
    monkeypatch.setattr(exact.os, "open", swap_then_open)
    with pytest.raises(FileChanged):
        full_hash(path, sig)


def test_placeholders_are_never_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write(tmp_path / "cloud.jpg", b"abc")
    sig = paths.signature(path)
    monkeypatch.setattr(paths, "is_placeholder", lambda st: True)

    def no_open(*args: object) -> int:
        raise AssertionError("a placeholder was opened")

    monkeypatch.setattr(exact.os, "open", no_open)
    for hasher in (partial_hash, full_hash):
        with pytest.raises(ReaderError) as info:
            hasher(path, sig)
        assert info.value.kind == exact.PLACEHOLDER


def test_a_folder_in_place_of_the_file(tmp_path: Path) -> None:
    path = write(tmp_path / "a.bin", b"abc")
    sig = paths.signature(path)
    path.unlink()
    path.mkdir()
    with pytest.raises(FileChanged):
        full_hash(path, sig)


def test_cancel_between_chunks(tmp_path: Path) -> None:
    path = write(tmp_path / "a.bin", os.urandom(3 * BUFFER))
    token = CancelToken()
    seen: list[int] = []

    def count_and_cancel(n: int) -> None:
        seen.append(n)
        token.cancel()

    with pytest.raises(JobCancelled):
        full_hash(path, paths.signature(path), token, count_and_cancel)
    assert seen == [BUFFER]


def test_long_and_arabic_paths(tmp_path: Path) -> None:
    folder = tmp_path
    for n in range(6):
        folder = folder / f"مجلد طويل جدا {n} {'x' * 40}"
    os.makedirs(paths.fs_path(folder), exist_ok=True)
    data = os.urandom(300 * KIB)
    with open(paths.fs_path(folder / "صورة.jpg"), "wb") as f:
        f.write(data)
    plain = folder / "صورة.jpg"
    assert len(str(plain)) > 300
    sig = paths.signature(plain)
    assert full_hash(plain, sig) == blake(data)
    assert partial_hash(plain, sig).full is None
