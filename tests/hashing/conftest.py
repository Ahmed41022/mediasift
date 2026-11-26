from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from mediasift.db.connection import Database
from tests.match.library import Library


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    with Database.open(tmp_path / "library") as database:
        yield database


@pytest.fixture
def lib(db: Database) -> Library:
    return Library(db)
