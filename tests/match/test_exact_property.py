"""Exact groups equal grouping by content, whatever the contents."""

from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mediasift.db.connection import Database
from mediasift.hashing.exact import BLOCK, WHOLE_LIMIT
from mediasift.jobs.context import JobContext
from mediasift.match.exact import ExactMatcher
from tests.match.library import Library

BIG = WHOLE_LIMIT + 3 * BLOCK
_BASE = bytes(range(256)) * (BIG // 256)

# Small contents from a tiny alphabet collide often; large ones share almost everything, so
# sizes and samples agree and only the full hash tells them apart.
small = st.text(alphabet="ab", max_size=3).map(str.encode)
large = st.tuples(
    st.sampled_from([0, BLOCK + 7, BIG // 2, BIG - 1, BIG - BLOCK - 5]), st.integers(0, 2)
).map(lambda change: _BASE[: change[0]] + bytes([change[1]]) + _BASE[change[0] + 1 :])
contents = st.lists(st.one_of(small, large), min_size=2, max_size=8)

_counter = itertools.count()


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(contents)
def test_groups_are_the_partition_by_content(tmp_path: Path, files: list[bytes]) -> None:
    case = tmp_path / f"case{next(_counter)}"
    folder = case / "photos"
    folder.mkdir(parents=True)
    by_content: dict[bytes, set[str]] = defaultdict(set)
    for n, data in enumerate(files):
        name = f"f{n}.jpg"
        (folder / name).write_bytes(data)
        if data:
            by_content[data].add(name)
    want = {
        (frozenset(names), len(data) * (len(names) - 1))
        for data, names in by_content.items()
        if len(names) > 1
    }
    with Database.open(case / "library") as db:
        lib = Library(db)
        lib.index(lib.add_root(folder))
        result = ExactMatcher(db, workers=2).run(JobContext())
        assert lib.groups() == want
        assert result.empty_files == sum(1 for data in files if not data)
