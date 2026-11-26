"""Exact duplicates: files with the same content, whatever their names and types.

Files are told apart by size, then by a hash of a sample, and only what is left is read whole.
Equal 256-bit hashes are taken as equal content: an accidental collision is far less likely
than a disk returning wrong data, so there is no byte-for-byte comparison. Actions hash a file
again right before they touch it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import closing
from dataclasses import dataclass, field

from mediasift.core.errors import FileChanged, ReaderError
from mediasift.core.types import FileRecord, FileStatus, MatchGroup, MatchKind, MatchMember
from mediasift.db.connection import Database
from mediasift.db.features import FeatureGroup, FeaturesRepo
from mediasift.db.files import ENABLED, FilesRepo
from mediasift.db.matches import MatchesRepo, MatchRun
from mediasift.db.roots import RootsRepo
from mediasift.hashing.exact import (
    FEATURE_FULL,
    FEATURE_PARTIAL,
    FULL_VERSION,
    PARTIAL_VERSION,
    OnBytes,
    check_unchanged,
    full_hash,
    partial_hash,
    sample_size,
)
from mediasift.hashing.store import (
    MAX_ISSUES,
    FeatureSummary,
    FileIssue,
    ensure_features,
    issue_for,
)
from mediasift.jobs.context import JobContext, Stage
from mediasift.jobs.runner import run_bounded

log = logging.getLogger(__name__)

STAGE_PARTIAL = "partial hashes"
STAGE_FULL = "full hashes"
STAGE_CHECK = "checking"
STAGE_GROUPS = "grouping"

_FULL_ONLY = {FEATURE_FULL: FULL_VERSION}
_BOTH = {FEATURE_PARTIAL: PARTIAL_VERSION, **_FULL_ONLY}

# Files looked up together while groups stream past.
_LOOKUP = 500


@dataclass(frozen=True, slots=True)
class ExactResult:
    """Counts of a finished run. The groups themselves are in the library.

    ``hardlink_sets`` are files with several names and no copy: not duplicates.
    """

    run: MatchRun
    groups: int
    reclaimable_bytes: int
    hardlink_sets: int
    empty_files: int
    partial: FeatureSummary
    full: FeatureSummary
    check: FeatureSummary

    @property
    def changed(self) -> set[int]:
        """Files left out because they are no longer what the index says: scan again."""
        return self.partial.changed | self.full.changed | self.check.changed

    @property
    def issue_count(self) -> int:
        return self.partial.issue_count + self.full.issue_count + self.check.issue_count

    @property
    def issues(self) -> list[FileIssue]:
        """Files that could not be hashed or have changed; capped per stage."""
        return [*self.partial.issues, *self.full.issues, *self.check.issues]


@dataclass(slots=True)
class _Work:
    """What a stage is about to do."""

    files: int = 0
    bytes: int = 0
    hardlink_sets: int = 0


@dataclass(slots=True)
class _Tally:
    groups: int = 0
    reclaimable: int = 0
    hardlink_sets: int = 0


class ExactMatcher:
    """Finds exact duplicates among the present files of enabled roots.

    Memory stays flat with the size of the library: files and groups stream from the index,
    and only the ids of changed files are kept.
    """

    name = "exact"
    version = 1

    def __init__(self, db: Database, *, workers: int = 0) -> None:
        self._db = db
        self._workers = workers
        self._files = FilesRepo(db)
        self._features = FeaturesRepo(db)
        self._matches = MatchesRepo(db)

    def run(self, job: JobContext) -> ExactResult:
        """Hash what is needed and replace the stored groups.

        A cancelled run leaves the previous groups as they were and keeps its hashes.
        """
        partial = self._hash_samples(job)
        full, linked = self._hash_whole(job)
        check = self._check_members(job, partial.changed | full.changed)
        changed = partial.changed | full.changed | check.changed
        tally = _Tally(hardlink_sets=linked)
        with job.stage(STAGE_GROUPS) as stage:
            job.raise_if_cancelled()
            run = self._matches.replace_run(
                MatchKind.EXACT,
                self._groups(job, stage, changed, tally),
                matcher=self.name,
                matcher_version=self.version,
            )
        log.info(
            "exact: %d groups; %d sample and %d full hashes computed; %d issues;"
            " equal hashes are not compared byte for byte",
            tally.groups,
            partial.computed,
            full.computed,
            partial.issue_count + full.issue_count + check.issue_count,
        )
        return ExactResult(
            run=run,
            groups=tally.groups,
            reclaimable_bytes=tally.reclaimable,
            hardlink_sets=tally.hardlink_sets,
            empty_files=self._count_empty(),
            partial=partial,
            full=full,
            check=check,
        )

    # --- hashing ---

    def _hash_samples(self, job: JobContext) -> FeatureSummary:
        work = _Work()
        for record in self._lacking_partial(job):
            work.files += 1
            work.bytes += sample_size(record.signature.size)
        with job.stage(STAGE_PARTIAL, work.files, work.bytes) as stage:
            if not work.files:
                return FeatureSummary()

            def compute(record: FileRecord) -> dict[str, bytes]:
                found = partial_hash(record.path, record.signature, job.cancel, _counter(stage))
                values = {FEATURE_PARTIAL: found.partial}
                if found.full is not None:
                    values[FEATURE_FULL] = found.full
                return values

            return ensure_features(
                self._db,
                self._lacking_partial(job),
                compute,
                _BOTH,
                job=job,
                stage=stage,
                workers=self._workers,
                verify=False,
            )

    def _lacking_partial(self, job: JobContext) -> Iterator[FileRecord]:
        lacking = self._features.iter_lacking(
            FEATURE_PARTIAL, PARTIAL_VERSION, repeated_sizes_only=True
        )
        for record in lacking:
            job.raise_if_cancelled()
            if record.signature.size:
                yield record

    def _hash_whole(self, job: JobContext) -> tuple[FeatureSummary, int]:
        """Full hashes of files whose samples agree, and the hard link sets seen on the way."""
        work = _Work()
        for records, known in self._partial_groups(job):
            if _identities(records) < 2:
                # With two or more known full hashes it is counted with the groups instead.
                work.hardlink_sets += len(known) < 2
                continue
            sizes = [r.signature.size for r in records if r.id not in known]
            work.files += len(sizes)
            work.bytes += sum(sizes)
        with job.stage(STAGE_FULL, work.files, work.bytes) as stage:
            if not work.files:
                return FeatureSummary(), work.hardlink_sets
            summary = _store_full(self._db, self._lacking_full(job), job, stage, self._workers)
        return summary, work.hardlink_sets

    def _lacking_full(self, job: JobContext) -> Iterator[FileRecord]:
        for records, known in self._partial_groups(job):
            if _identities(records) > 1:
                yield from (record for record in records if record.id not in known)

    def _partial_groups(
        self, job: JobContext
    ) -> Iterator[tuple[list[FileRecord], dict[int, bytes]]]:
        """Files with equal size and sample, and the full hashes already known of them."""
        groups = self._features.iter_groups(FEATURE_PARTIAL, PARTIAL_VERSION, same_size=True)
        for records in self._with_records(groups, job):
            ids = [record.id for record in records]
            yield records, self._features.get_many(ids, FEATURE_FULL, FULL_VERSION)

    # --- grouping ---

    def _full_groups(self, job: JobContext, changed: set[int]) -> Iterator[list[FileRecord]]:
        """Files with equal content, without the changed ones. Empty files are no duplicates."""
        groups = self._features.iter_groups(FEATURE_FULL, FULL_VERSION, same_size=True)
        for records in self._with_records((g for g in groups if g.size), job):
            unchanged = [record for record in records if record.id not in changed]
            if len(unchanged) > 1:
                yield unchanged

    def _check_members(self, job: JobContext, changed: set[int]) -> FeatureSummary:
        """Find members that changed after they were hashed, in this run or an earlier one.

        A file that cannot be looked at stays: its drive may be unplugged, which must not make
        the user's decisions about it disappear.
        """
        summary = FeatureSummary()
        members = (
            record
            for records in self._full_groups(job, changed)
            if _identities(records) > 1
            for record in records
        )

        def look(record: FileRecord) -> None:
            check_unchanged(record.path, record.signature)

        with (
            job.stage(STAGE_CHECK) as stage,
            closing(run_bounded(members, look, job=job, workers=self._workers)) as outcomes,
        ):
            for outcome in outcomes:
                error = outcome.error
                stage.advance(errors=0 if error is None else 1)
                if error is None:
                    continue
                issue = issue_for(outcome.item, error)
                summary.add_issue(issue)
                if isinstance(error, FileChanged | ReaderError):
                    summary.changed.add(issue.file_id)
        return summary

    def _groups(
        self, job: JobContext, stage: Stage, changed: set[int], tally: _Tally
    ) -> Iterator[MatchGroup]:
        for records in self._full_groups(job, changed):
            group = _match_group(records)
            if group is None:
                tally.hardlink_sets += 1
                continue
            tally.groups += 1
            tally.reclaimable += group.reclaimable_bytes
            stage.advance()
            yield group
        job.raise_if_cancelled()

    def _with_records(
        self, groups: Iterable[FeatureGroup], job: JobContext
    ) -> Iterator[list[FileRecord]]:
        """Look the files of groups up, several groups at a time."""
        pending: list[FeatureGroup] = []
        count = 0
        for group in groups:
            job.raise_if_cancelled()
            pending.append(group)
            count += len(group.file_ids)
            if count >= _LOOKUP:
                yield from self._lookup(pending)
                pending, count = [], 0
        yield from self._lookup(pending)

    def _lookup(self, groups: list[FeatureGroup]) -> Iterator[list[FileRecord]]:
        found = self._files.get_many(file_id for group in groups for file_id in group.file_ids)
        for group in groups:
            yield [found[file_id] for file_id in group.file_ids]

    def _count_empty(self) -> int:
        where = f"f.size = 0 AND f.status = 'present' AND {ENABLED}"
        return sum(1 for _ in self._files.iter_where(where, {}))


# --- full hashes of given files -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class FullHashes:
    """Full hashes by file id. Files without one are explained in ``issues``."""

    values: dict[int, bytes]
    computed: int = 0
    issues: list[FileIssue] = field(default_factory=list)


def ensure_full_hashes(
    db: Database, file_ids: Iterable[int], job: JobContext, *, workers: int = 0
) -> FullHashes:
    """Full hashes of these files, computing the ones that are lacking.

    Values are valid for the size and modified time in the index; whoever is about to act on
    a file still hashes it again. Files that are not present, or whose root is disabled, are
    never read and get no value. Unknown ids are left out.
    """
    features = FeaturesRepo(db)
    enabled = {root.id for root in RootsRepo(db).list(enabled_only=True)}
    usable: dict[int, FileRecord] = {}
    issues: list[FileIssue] = []
    for file_id, record in FilesRepo(db).get_many(dict.fromkeys(file_ids)).items():
        if record.root_id not in enabled:
            issues.append(FileIssue(file_id, record.path, "disabled", "its folder is disabled"))
        elif record.status is not FileStatus.PRESENT:
            message = f"the file is marked as {record.status.value}"
            issues.append(FileIssue(file_id, record.path, record.status.value, message))
        else:
            usable[file_id] = record
    values = features.get_many(usable, FEATURE_FULL, FULL_VERSION)
    lacking = [record for file_id, record in usable.items() if file_id not in values]
    if not lacking:
        return FullHashes(values, 0, issues)
    total = sum(record.signature.size for record in lacking)
    with job.stage(STAGE_FULL, len(lacking), total) as stage:
        summary = _store_full(db, lacking, job, stage, workers, issue_limit=len(lacking))
    values |= features.get_many((r.id for r in lacking), FEATURE_FULL, FULL_VERSION)
    return FullHashes(values, summary.computed, issues + summary.issues)


def _store_full(
    db: Database,
    records: Iterable[FileRecord],
    job: JobContext,
    stage: Stage,
    workers: int,
    issue_limit: int = MAX_ISSUES,
) -> FeatureSummary:
    def compute(record: FileRecord) -> dict[str, bytes]:
        digest = full_hash(record.path, record.signature, job.cancel, _counter(stage))
        return {FEATURE_FULL: digest}

    return ensure_features(
        db,
        records,
        compute,
        _FULL_ONLY,
        job=job,
        stage=stage,
        workers=workers,
        verify=False,
        issue_limit=issue_limit,
    )


# --- helpers ------------------------------------------------------------------------


def _counter(stage: Stage) -> OnBytes:
    """Count bytes read on a stage, from any thread."""
    return lambda count: stage.advance(0, bytes=count)


def _identity(record: FileRecord) -> tuple[int, int]:
    """Equal only for names of one physical file. An unknown identity is its own."""
    sig = record.signature
    return (sig.dev, sig.inode) if sig.inode else (-1, record.id)


def _identities(records: Iterable[FileRecord]) -> int:
    return len({_identity(record) for record in records})


def _match_group(records: list[FileRecord]) -> MatchGroup | None:
    """The group, or None when all members are names of one file.

    Extra names of a file are members, so that the user sees them, but they free no space.
    """
    first: dict[tuple[int, int], int] = {}
    members = []
    for record in records:
        owner = first.setdefault(_identity(record), record.id)
        evidence = {} if owner == record.id else {"hardlink_of": owner}
        members.append(MatchMember(record.id, evidence=evidence))
    if len(first) < 2:
        return None
    return MatchGroup(
        kind=MatchKind.EXACT,
        members=tuple(members),
        reclaimable_bytes=records[0].signature.size * (len(first) - 1),
    )
