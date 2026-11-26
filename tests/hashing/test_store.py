from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from mediasift.core.errors import FileChanged, JobCancelled, ReaderError
from mediasift.core.types import FileRecord
from mediasift.db.features import FeaturesRepo
from mediasift.hashing.store import FeatureSummary, FileIssue, ensure_features
from mediasift.jobs.context import JobContext, Progress
from tests.match.library import Library

VERSIONS = {"a": 1, "b": 2}


def make_files(tmp_path: Path, lib: Library, count: int = 4) -> list[FileRecord]:
    folder = tmp_path / "photos"
    folder.mkdir()
    for n in range(count):
        (folder / f"f{n}.jpg").write_bytes(b"x" * (n + 1))
    return sorted(lib.index(lib.add_root(folder)).values(), key=lambda r: r.id)


def run(
    lib: Library,
    records: list[FileRecord],
    compute: object,
    *,
    job: JobContext | None = None,
    **kwargs: object,
) -> FeatureSummary:
    job = job or JobContext()
    with job.stage("test", len(records)) as stage:
        return ensure_features(
            lib.db,
            records,
            compute,  # type: ignore[arg-type]
            VERSIONS,
            job=job,
            stage=stage,
            workers=2,
            **kwargs,  # type: ignore[arg-type]
        )


def test_several_features_from_one_computation(tmp_path: Path, lib: Library) -> None:
    records = make_files(tmp_path, lib)

    def compute(record: FileRecord) -> Mapping[str, bytes]:
        return {"a": b"A%d" % record.id, "b": b"B%d" % record.id}

    summary = run(lib, records, compute)
    assert summary.computed == 4 and summary.issue_count == 0
    repo = FeaturesRepo(lib.db)
    ids = [r.id for r in records]
    assert repo.get_many(ids, "a", 1) == {i: b"A%d" % i for i in ids}
    assert repo.get_many(ids, "b", 2) == {i: b"B%d" % i for i in ids}
    assert repo.get_many(ids, "b", 1) == {}
    assert list(repo.iter_lacking("a", 1)) == []


def test_errors_about_the_content_are_stored(tmp_path: Path, lib: Library) -> None:
    records = make_files(tmp_path, lib)
    bad = records[1]

    def compute(record: FileRecord) -> Mapping[str, bytes]:
        if record.id == bad.id:
            raise ReaderError("truncated image", kind="corrupt")
        return {"a": b"ok"}

    summary = run(lib, records, compute)
    assert summary.computed == 3
    assert summary.issues == [FileIssue(bad.id, bad.path, "corrupt", "truncated image", True)]
    repo = FeaturesRepo(lib.db)
    assert repo.errors("a", 1) == [(bad.id, "truncated image")]
    assert repo.errors("b", 2) == [(bad.id, "truncated image")]
    assert bad.id not in {r.id for r in repo.iter_lacking("a", 1)}


@pytest.mark.parametrize(
    "error",
    [
        ReaderError("needs a library", kind="unavailable"),
        PermissionError(13, "denied"),
        OSError(5, "input/output error"),
        ValueError("a bug"),
    ],
)
def test_passing_errors_are_issues_only(tmp_path: Path, lib: Library, error: Exception) -> None:
    records = make_files(tmp_path, lib, 2)

    def compute(record: FileRecord) -> Mapping[str, bytes]:
        if record.id == records[0].id:
            raise error
        return {"a": b"ok"}

    summary = run(lib, records, compute)
    assert summary.computed == 1
    [issue] = summary.issues
    assert issue.file_id == records[0].id and not issue.permanent
    assert summary.changed == set()
    repo = FeaturesRepo(lib.db)
    assert repo.errors("a", 1) == []
    assert [r.id for r in repo.iter_lacking("a", 1)] == [records[0].id]


def test_a_locked_file_on_windows_is_its_own_kind(tmp_path: Path, lib: Library) -> None:
    records = make_files(tmp_path, lib, 1)
    locked = PermissionError(13, "in use")
    locked.winerror = 32  # type: ignore[attr-defined]

    def compute(record: FileRecord) -> Mapping[str, bytes]:
        raise locked

    assert [i.kind for i in run(lib, records, compute).issues] == ["locked"]


def test_changed_files_are_reported_and_not_stored(tmp_path: Path, lib: Library) -> None:
    records = make_files(tmp_path, lib)

    def compute(record: FileRecord) -> Mapping[str, bytes]:
        if record.id == records[0].id:
            raise FileChanged("moved on")
        if record.id == records[1].id:
            # Changed by someone else while it was computed: the check afterwards notices.
            record.path.write_bytes(b"something else entirely")
        return {"a": b"ok"}

    summary = run(lib, records, compute)
    assert summary.changed == {records[0].id, records[1].id}
    assert {i.kind for i in summary.issues} == {"changed"}
    assert set(FeaturesRepo(lib.db).get_many([r.id for r in records], "a", 1)) == {
        records[2].id,
        records[3].id,
    }


def test_without_verify_the_computation_is_trusted(tmp_path: Path, lib: Library) -> None:
    records = make_files(tmp_path, lib, 1)
    os.remove(records[0].path)

    def compute(record: FileRecord) -> Mapping[str, bytes]:
        return {"a": b"ok"}

    assert run(lib, records, compute, verify=False).computed == 1
    assert [i.kind for i in run(lib, records, compute).issues] == ["missing"]


def test_a_feature_without_a_version_is_a_bug(tmp_path: Path, lib: Library) -> None:
    records = make_files(tmp_path, lib, 1)
    with pytest.raises(ValueError, match="without a version: c"):
        run(lib, records, lambda record: {"c": b""})


def test_issues_are_capped_but_counted(tmp_path: Path, lib: Library) -> None:
    records = make_files(tmp_path, lib, 5)

    def compute(record: FileRecord) -> Mapping[str, bytes]:
        raise FileChanged("changed")

    summary = run(lib, records, compute, issue_limit=2)
    assert len(summary.issues) == 2
    assert summary.issue_count == 5
    assert len(summary.changed) == 5


def test_cancel_keeps_finished_work_and_progress_counts(tmp_path: Path, lib: Library) -> None:
    records = make_files(tmp_path, lib, 8)
    seen: list[Progress] = []
    job = JobContext(on_progress=seen.append, max_rate=1e9)

    def compute(record: FileRecord) -> Mapping[str, bytes]:
        if record.id == records[3].id:
            job.cancel.cancel()
        return {"a": b"ok"}

    with job.stage("test", len(records)) as stage, pytest.raises(JobCancelled):
        ensure_features(lib.db, records, compute, VERSIONS, job=job, stage=stage, workers=1)
    stored = FeaturesRepo(lib.db).get_many([r.id for r in records], "a", 1)
    assert set(stored) == {r.id for r in records[:4]}
    assert seen[-1].done == 4
