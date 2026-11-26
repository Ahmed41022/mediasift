from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from mediasift.db.connection import Database
from mediasift.hashing.exact import PartialHash
from mediasift.jobs.runner import THREAD_PREFIX
from mediasift.match import exact
from tests.match.library import Library


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    with Database.open(tmp_path / "library") as database:
        yield database


@pytest.fixture
def lib(db: Database) -> Library:
    return Library(db)


@pytest.fixture(autouse=True)
def no_leaked_workers() -> Iterator[None]:
    yield
    assert [t.name for t in threading.enumerate() if t.name.startswith(THREAD_PREFIX)] == []


class HashCalls:
    """Records which files the matcher hashes; ``before`` runs ahead of each hash."""

    def __init__(self) -> None:
        self.partial: list[Path] = []
        self.full: list[Path] = []
        self.before: Callable[[str, Path], None] | None = None
        self._lock = threading.Lock()

    def note(self, kind: str, path: Path) -> None:
        with self._lock:
            (self.partial if kind == "partial" else self.full).append(path)
        if self.before is not None:
            self.before(kind, path)

    def clear(self) -> None:
        self.partial.clear()
        self.full.clear()


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> HashCalls:
    record = HashCalls()
    real_partial, real_full = exact.partial_hash, exact.full_hash

    def partial(path: Path, *args: object, **kwargs: object) -> PartialHash:
        record.note("partial", path)
        return real_partial(path, *args, **kwargs)

    def full(path: Path, *args: object, **kwargs: object) -> bytes:
        record.note("full", path)
        return real_full(path, *args, **kwargs)

    monkeypatch.setattr(exact, "partial_hash", partial)
    monkeypatch.setattr(exact, "full_hash", full)
    return record
