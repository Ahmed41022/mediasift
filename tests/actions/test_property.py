"""Remove then restore is the identity: same bytes, name and modified time."""

from __future__ import annotations

import itertools
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mediasift.actions.backends.base import RestoreRequest
from mediasift.actions.backends.quarantine import QuarantineBackend
from tests.actions.conftest import checked_request, make_library

Q = QuarantineBackend()
_COUNTER = itertools.count()

# Letters that every file system of both systems accepts, Arabic and accents included.
_LETTERS = st.sampled_from("abcXYZ019 _-صورةالعائلةéñ")
names = st.text(_LETTERS, min_size=1, max_size=20).filter(
    lambda name: name.strip() == name and not name.startswith("-")
)


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    folders=st.lists(names, max_size=3),
    name=names,
    data=st.binary(max_size=4096),
    mtime=st.integers(min_value=0, max_value=4_000_000_000).map(lambda s: s * 1_000_000_000),
    taken=st.booleans(),
)
def test_remove_then_restore(
    tmp_path: Path, folders: list[str], name: str, data: bytes, mtime: int, taken: bool
) -> None:
    lib = make_library(tmp_path / f"case{next(_COUNTER)}")
    rel = "/".join([*folders, name + ".jpg"])
    path = lib.folder / rel
    item = lib.add(rel, data, mtime)
    keeper = lib.add("keeper.bin", data)
    planned = Q.plan_destination(path, lib.root, "s1")
    stored = Q.remove(checked_request(planned, item, keeper, lib.root))
    assert not path.exists()
    if taken:
        path.write_bytes(b"newcomer")
    request = RestoreRequest(stored, path, lib.root, item.record.signature, item.full_hash)
    restored = Q.restore(request)
    assert restored.renamed is taken
    assert restored.path.parent == path.parent
    if not taken:
        assert restored.path == path
    assert restored.path.read_bytes() == data
    assert restored.path.stat().st_mtime_ns == mtime
