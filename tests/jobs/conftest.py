from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest

from mediasift.jobs.runner import THREAD_PREFIX


@pytest.fixture(autouse=True)
def no_leaked_workers() -> Iterator[None]:
    """Every test in this folder must leave no worker thread behind."""
    yield
    assert [t.name for t in threading.enumerate() if t.name.startswith(THREAD_PREFIX)] == []
