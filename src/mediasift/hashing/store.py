"""Compute features of many files in worker threads and store them in batches."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from mediasift.core import paths
from mediasift.core.errors import FileChanged, ReaderError
from mediasift.core.types import FileRecord
from mediasift.db.connection import Database
from mediasift.db.features import FeaturesRepo, FeatureValue
from mediasift.hashing.exact import PLACEHOLDER, check_unchanged
from mediasift.jobs.batch import batched_commit
from mediasift.jobs.context import JobContext, Stage
from mediasift.jobs.runner import Outcome, run_bounded

log = logging.getLogger(__name__)

MAX_ISSUES = 1000
CHANGED = "changed"

# Kinds of ReaderError that say nothing about the content, so they are tried again next time.
TRANSIENT_KINDS = frozenset({PLACEHOLDER, "unavailable", "missing", "permission", "locked", "io"})

# ERROR_SHARING_VIOLATION and ERROR_LOCK_VIOLATION: the file is open in another program.
_LOCKED_WINERRORS = {32, 33}

Compute = Callable[[FileRecord], Mapping[str, bytes]]


@dataclass(frozen=True, slots=True)
class FileIssue:
    """Why a file has no feature. Permanent ones are stored and not tried again."""

    file_id: int
    path: Path
    kind: str
    message: str
    permanent: bool = False


@dataclass(slots=True)
class FeatureSummary:
    """What became of the files. ``issues`` is capped; ``issue_count`` and ``changed`` are not.

    ``changed`` holds the ids of files that are no longer what the index says.
    """

    computed: int = 0
    changed: set[int] = field(default_factory=set)
    issue_count: int = 0
    issues: list[FileIssue] = field(default_factory=list)

    def add_issue(self, issue: FileIssue, limit: int = MAX_ISSUES) -> None:
        """Count an issue, and keep it while there is room."""
        self.issue_count += 1
        if issue.kind == CHANGED:
            self.changed.add(issue.file_id)
        if len(self.issues) < limit:
            self.issues.append(issue)


def ensure_features(
    db: Database,
    records: Iterable[FileRecord],
    compute: Compute,
    versions: Mapping[str, int],
    *,
    job: JobContext,
    stage: Stage,
    workers: int = 0,
    verify: bool = True,
    issue_limit: int = MAX_ISSUES,
) -> FeatureSummary:
    """Run ``compute`` on every record and store the features it returns.

    ``compute`` runs in worker threads and must not touch the database. It returns some or all
    of the features named in ``versions``, which gives each one's version. It may count the
    bytes it reads with ``stage.advance(0, bytes=n)``; files and errors are counted here.
    ``verify`` checks the file before and after ``compute``; turn it off only when ``compute``
    checks the file it reads itself, as the exact hashes do.

    One bad file never stops the job:

    - ReaderError, except for the kinds in TRANSIENT_KINDS, is about the content. It is stored
      as the error of every feature in ``versions`` and stands until the file changes.
    - FileChanged means the index is out of date. Nothing is stored and the file is reported
      as changed: it has to be scanned again first.
    - Anything else, such as a locked file, is reported and tried again next time.

    Everything finished before a cancel is stored, then JobCancelled is raised.
    """
    repo = FeaturesRepo(db)
    summary = FeatureSummary()

    def work(record: FileRecord) -> Mapping[str, bytes]:
        if not verify:
            return compute(record)
        check_unchanged(record.path, record.signature)
        values = compute(record)
        check_unchanged(record.path, record.signature)
        return values

    def rows() -> Iterator[list[FeatureValue]]:
        with closing(run_bounded(records, work, job=job, workers=workers)) as outcomes:
            for outcome in outcomes:
                if outcome.ok:
                    found = _values(outcome.item, outcome.unwrap(), versions)
                    summary.computed += 1
                else:
                    found = _failure(outcome, versions, summary, issue_limit)
                name = paths.display_path(outcome.item.rel_path)
                stage.advance(current=name, errors=0 if outcome.ok else 1)
                yield found

    def flush(batch: list[list[FeatureValue]]) -> None:
        repo.put_many(value for values in batch for value in values)

    batched_commit(rows(), flush)
    return summary


def issue_for(record: FileRecord, error: Exception) -> FileIssue:
    """Describe the error of one file."""
    permanent = isinstance(error, ReaderError) and error.kind not in TRANSIENT_KINDS
    message = str(error) or type(error).__name__
    return FileIssue(record.id, record.path, _kind(error), message, permanent)


def _kind(error: Exception) -> str:
    if isinstance(error, FileChanged):
        return CHANGED
    if isinstance(error, ReaderError):
        return error.kind
    if isinstance(error, FileNotFoundError | NotADirectoryError):
        return "missing"
    if getattr(error, "winerror", None) in _LOCKED_WINERRORS:
        return "locked"
    if isinstance(error, PermissionError):
        return "permission"
    return "io" if isinstance(error, OSError) else "error"


def _values(
    record: FileRecord, values: Mapping[str, bytes], versions: Mapping[str, int]
) -> list[FeatureValue]:
    if unknown := values.keys() - versions.keys():
        raise ValueError(f"features without a version: {', '.join(sorted(unknown))}")
    sig = record.signature
    return [
        FeatureValue(record.id, name, versions[name], value, sig.size, sig.mtime_ns)
        for name, value in values.items()
    ]


def _failure(
    outcome: Outcome[FileRecord, Mapping[str, bytes]],
    versions: Mapping[str, int],
    summary: FeatureSummary,
    issue_limit: int,
) -> list[FeatureValue]:
    """Record a failed file. Return the error rows to store: none unless it is permanent."""
    record, error = outcome.item, outcome.error
    assert error is not None
    issue = issue_for(record, error)
    if issue.kind == "error":
        # Not something a file should be able to cause: leave a trace to look into.
        log.warning("cannot process %s", paths.display_path(record.path), exc_info=error)
    summary.add_issue(issue, issue_limit)
    if not issue.permanent:
        return []
    sig = record.signature
    return [
        FeatureValue(record.id, name, version, None, sig.size, sig.mtime_ns, issue.message)
        for name, version in versions.items()
    ]
