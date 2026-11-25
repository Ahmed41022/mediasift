from __future__ import annotations

from pathlib import Path

import pytest

from mediasift.core.errors import LibraryError
from mediasift.core.types import Root
from mediasift.db.connection import Database
from mediasift.db.scans import ScanIssue, ScansRepo, ScanState
from tests.core.test_paths import UNDECODABLE


@pytest.fixture
def scans(db: Database) -> ScansRepo:
    return ScansRepo(db)


def test_life_of_a_scan(scans: ScansRepo) -> None:
    assert scans.latest() is None
    scan = scans.start({"roots": [1, 2]})
    record = scans.get(scan)
    assert (record.state, record.finished_at) == (ScanState.RUNNING, None)
    assert record.options == {"roots": [1, 2]}
    scans.set_counters(scan, {"seen": 5})
    scans.finish(scan, ScanState.DONE, {"seen": 9, "hashed": 2})
    record = scans.get(scan)
    assert record.state is ScanState.DONE
    assert record.finished_at is not None and record.finished_at >= record.started_at
    assert record.counters == {"seen": 9, "hashed": 2}
    assert scans.latest() == record


def test_finish_keeps_counters_unless_given(scans: ScansRepo) -> None:
    scan = scans.start()
    scans.set_counters(scan, {"seen": 5})
    scans.finish(scan, ScanState.CANCELLED)
    assert scans.get(scan).counters == {"seen": 5}


def test_scan_of_a_dead_process_is_interrupted(scans: ScansRepo) -> None:
    first = scans.start()
    second = scans.start()
    assert scans.get(first).state is ScanState.INTERRUPTED
    assert scans.get(first).finished_at is not None
    assert scans.get(second).state is ScanState.RUNNING


def test_unknown_scan(scans: ScansRepo) -> None:
    with pytest.raises(LibraryError, match="number 9"):
        scans.get(9)
    with pytest.raises(LibraryError, match="number 9"):
        scans.finish(9, ScanState.DONE)


def test_issues(scans: ScansRepo, root: Root) -> None:
    scan, other = scans.start(), scans.start()
    issues = [
        ScanIssue(Path("sub") / UNDECODABLE, "permission", "denied: \udcff", root.id),
        ScanIssue(Path("صورة.jpg"), "corrupt", "مشكلة"),
        ScanIssue(Path("b.jpg"), "corrupt"),
    ]
    assert scans.add_issues(scan, iter(issues)) == 3
    read = scans.issues(scan)
    assert [issue.path for issue in read] == [issue.path for issue in issues]
    assert read[0].message == "denied: \\udcff"
    assert (read[0].root_id, read[1].root_id) == (root.id, None)
    assert scans.issues(scan, limit=1, offset=2) == [issues[2]]
    assert scans.issue_counts(scan) == {"corrupt": 2, "permission": 1}
    assert scans.issues(other) == []
