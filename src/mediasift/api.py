"""The facade every interface uses: command line, window, web.

Interfaces import this module and ``mediasift.core``, nothing else (an architecture test
enforces it). It returns plain frozen dataclasses, so no database type leaks out and the
layers below can change without breaking an interface.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Self

from mediasift.core import paths
from mediasift.core.config import Settings, config_path, library_dir, load_settings
from mediasift.core.errors import (
    ActionError,
    ConfigError,
    ExitCode,
    JobCancelled,
    LibraryBusy,
    LibraryError,
    MediaSiftError,
    SafetyError,
)
from mediasift.core.log import close_logging, set_up_logging
from mediasift.core.types import FileRecord, FileStatus, MatchGroup, MatchKind, Root
from mediasift.db.connection import Database
from mediasift.db.files import FilesRepo
from mediasift.db.matches import GroupCursor, MatchesRepo
from mediasift.db.roots import RootsRepo
from mediasift.db.scans import ScanRecord, ScansRepo, ScanState
from mediasift.jobs import CancelToken, JobContext, Progress, cancel_on_sigint
from mediasift.match.exact import ExactMatcher, ExactResult
from mediasift.report import ReportGroup, ReportMember, human_size, write_csv, write_html
from mediasift.scan.scanner import Scanner, ScanResult

__all__ = [
    "ActionError",
    "CancelToken",
    "ConfigError",
    "ExactSummary",
    "ExitCode",
    "GroupInfo",
    "GroupPage",
    "IssueInfo",
    "JobCancelled",
    "JobContext",
    "Library",
    "LibraryBusy",
    "LibraryError",
    "LibraryStats",
    "MediaSiftError",
    "MemberInfo",
    "PageCursor",
    "Progress",
    "ReportSummary",
    "RootInfo",
    "SafetyError",
    "ScanInfo",
    "ScanSummary",
    "TypeStat",
    "cancel_on_sigint",
    "human_size",
]

log = logging.getLogger(__name__)

EXACT = MatchKind.EXACT.value
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


# --- plain results ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RootInfo:
    """A folder in the library."""

    id: int
    path: Path
    enabled: bool
    read_only: bool
    prefer_rank: int
    volume_id: str


@dataclass(frozen=True, slots=True)
class IssueInfo:
    """A problem with one path. ``kind`` is a short stable code."""

    path: Path
    kind: str
    message: str


@dataclass(frozen=True, slots=True)
class ScanSummary:
    """What a scan did. ``state`` is done, cancelled or failed."""

    scan_id: int
    state: str
    seen: int
    new: int
    changed: int
    unchanged: int
    moved: int
    missing: int
    restored: int
    placeholders: int
    skipped_links: int
    vanished: int
    errors: int
    by_type: Mapping[str, int]
    unreachable_roots: tuple[int, ...]

    @property
    def cancelled(self) -> bool:
        return self.state == ScanState.CANCELLED.value

    @property
    def has_issues(self) -> bool:
        """True when some files or folders could not be looked at."""
        return self.errors > 0 or bool(self.unreachable_roots)


@dataclass(frozen=True, slots=True)
class ExactSummary:
    """What exact matching did. Hash counts say how much work was new."""

    groups: int
    reclaimable_bytes: int
    hardlink_sets: int
    empty_files: int
    hashed_samples: int
    hashed_full: int
    issue_count: int
    changed_files: int
    issues: tuple[IssueInfo, ...]

    @property
    def has_issues(self) -> bool:
        return self.issue_count > 0 or self.changed_files > 0


@dataclass(frozen=True, slots=True)
class MemberInfo:
    """A file of a group. ``hardlink_of`` is the id of the file this is another name of."""

    file_id: int
    path: Path
    size: int
    mtime: datetime
    media_type: str
    root_id: int
    hardlink_of: int | None = None


@dataclass(frozen=True, slots=True)
class GroupInfo:
    """A set of matching files. ``size`` is the size of one file."""

    id: int
    kind: str
    size: int
    reclaimable_bytes: int
    members: tuple[MemberInfo, ...]


@dataclass(frozen=True, slots=True)
class PageCursor:
    """Where a page of groups ended. Pass it as ``after`` to get the next page."""

    reclaimable_bytes: int
    group_id: int


@dataclass(frozen=True, slots=True)
class GroupPage:
    """Groups that free most first. ``next`` is None on the last page."""

    groups: tuple[GroupInfo, ...]
    next: PageCursor | None


@dataclass(frozen=True, slots=True)
class TypeStat:
    """Files of one media type and status."""

    media_type: str
    status: str
    count: int
    bytes: int


@dataclass(frozen=True, slots=True)
class ScanInfo:
    """A scan as recorded in the library."""

    id: int
    state: str
    started_at: str
    finished_at: str | None
    counters: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LibraryStats:
    """The state of the library. ``exact_run_at`` is None until duplicates were searched."""

    directory: Path
    files: tuple[TypeStat, ...]
    roots: tuple[RootInfo, ...]
    last_scan: ScanInfo | None
    issue_count: int
    groups: int
    reclaimable_bytes: int
    exact_run_at: str | None

    @property
    def present_files(self) -> int:
        return sum(s.count for s in self.files if s.status == FileStatus.PRESENT.value)

    @property
    def present_bytes(self) -> int:
        return sum(s.bytes for s in self.files if s.status == FileStatus.PRESENT.value)


@dataclass(frozen=True, slots=True)
class ReportSummary:
    """What ``export_report`` wrote."""

    groups: int
    files: int
    reclaimable_bytes: int
    csv: Path | None
    html: Path | None


# --- the library ------------------------------------------------------------------------


class Library:
    """An open library. Use it as a context manager; it holds the writer lock until closed."""

    def __init__(self, db: Database, settings: Settings) -> None:
        self._db = db
        self.settings = settings
        self._logging = db.writable

    @classmethod
    def open(
        cls,
        path: str | Path | None = None,
        *,
        write: bool = True,
        settings: Settings | None = None,
    ) -> Self:
        """Open the library in ``path``, or in the default place (see ``library_dir``).

        Settings come from the library's ``config.toml`` unless given. Writing creates the
        library when it is new and raises ``LibraryBusy`` when another process writes;
        reading needs an existing library and never waits for a writer.
        """
        folder = library_dir(path)
        chosen = settings if settings is not None else load_settings(config_path(folder))
        db = Database.open(folder, write=write)
        try:
            if write:
                set_up_logging(folder)
            return cls(db, chosen)
        except BaseException:
            db.close()
            raise

    @property
    def directory(self) -> Path:
        return self._db.directory

    @property
    def writable(self) -> bool:
        return self._db.writable

    def close(self) -> None:
        """Release the library. Safe to call twice."""
        try:
            self._db.close()
        finally:
            if self._logging:
                self._logging = False
                close_logging()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _need_write(self) -> None:
        if not self._db.writable:
            raise LibraryError("the library is open for reading only")

    # --- roots ---

    def add_root(self, path: str | Path, read_only: bool = False, prefer_rank: int = 0) -> RootInfo:
        """Add a folder. A folder that contains, lies in, or is a known one is refused."""
        self._need_write()
        return _root_info(
            RootsRepo(self._db).add(path, read_only=read_only, prefer_rank=prefer_rank)
        )

    def find_root(self, path: str | Path) -> RootInfo | None:
        """The root that is exactly this folder, if there is one."""
        found = RootsRepo(self._db).find(path)
        return None if found is None else _root_info(found)

    def list_roots(self) -> tuple[RootInfo, ...]:
        """Every root, preferred ones first."""
        return tuple(_root_info(root) for root in RootsRepo(self._db).list())

    def disable_root(self, root_id: int) -> RootInfo:
        """Leave a root out of scans and matching. Its files stay in the index."""
        self._need_write()
        return _root_info(RootsRepo(self._db).set_enabled(root_id, False))

    def enable_root(self, root_id: int) -> RootInfo:
        self._need_write()
        return _root_info(RootsRepo(self._db).set_enabled(root_id, True))

    def set_root_volume(self, root_id: int) -> RootInfo:
        """Accept the volume now mounted at a root as the right one.

        A scan refuses a root whose volume changed, because an unplugged drive would look
        like every file was deleted. After a real change (a new disk, a reformat) the user
        says so here.
        """
        self._need_write()
        repo = RootsRepo(self._db)
        root = repo.get(root_id)
        volume = paths.volume_id(root.path)
        if not volume:
            raise LibraryError(
                f"the volume of {paths.display_path(root.path)} cannot be identified right now"
            )
        return _root_info(repo.set_volume_id(root_id, volume))

    # --- work ---

    def scan(
        self, roots: Iterable[int] | None = None, job: JobContext | None = None
    ) -> ScanSummary:
        """Scan the given roots (ids), or every enabled one.

        A cancelled scan returns normally with state ``cancelled``; what it indexed is kept.
        """
        self._need_write()
        repo = RootsRepo(self._db)
        chosen = None if roots is None else [repo.get(root_id) for root_id in roots]
        result = Scanner(self._db, self.settings).scan(chosen, job)
        return _scan_summary(result)

    def find_exact(self, job: JobContext | None = None) -> ExactSummary:
        """Hash what is needed and store the groups of identical files.

        Raises ``JobCancelled`` when stopped; the hashes made so far are kept.
        """
        self._need_write()
        matcher = ExactMatcher(self._db, workers=self.settings.jobs.workers)
        return _exact_summary(matcher.run(job or JobContext()))

    # --- reading ---

    def list_groups(
        self, kind: str = EXACT, limit: int = 100, after: PageCursor | None = None
    ) -> GroupPage:
        """A page of groups, the ones that free most first."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        cursor = None if after is None else GroupCursor(after.reclaimable_bytes, after.group_id)
        rows = MatchesRepo(self._db).list_groups(MatchKind(kind), limit=limit + 1, after=cursor)
        page = rows[:limit]
        more = len(rows) > limit
        groups = self._group_infos(page)
        last = groups[-1] if groups else None
        nxt = PageCursor(last.reclaimable_bytes, last.id) if more and last else None
        return GroupPage(tuple(groups), nxt)

    def iter_groups(self, kind: str = EXACT) -> Iterator[GroupInfo]:
        """Every group in the order of ``list_groups``, a page at a time."""
        after: PageCursor | None = None
        while True:
            page = self.list_groups(kind, limit=500, after=after)
            yield from page.groups
            if page.next is None:
                return
            after = page.next

    def stats(self) -> LibraryStats:
        """Counts for the enabled roots, the last scan and the stored groups."""
        scans = ScansRepo(self._db)
        latest = scans.latest()
        matches = MatchesRepo(self._db)
        groups, reclaimable = matches.count_groups(MatchKind.EXACT)
        run = matches.latest_run(MatchKind.EXACT)
        issues = sum(scans.issue_counts(latest.id).values()) if latest else 0
        return LibraryStats(
            directory=self.directory,
            files=tuple(
                TypeStat(s.media_type.value, s.status.value, s.count, s.bytes)
                for s in FilesRepo(self._db).stats()
            ),
            roots=self.list_roots(),
            last_scan=None if latest is None else _scan_info(latest),
            issue_count=issues,
            groups=groups,
            reclaimable_bytes=reclaimable,
            exact_run_at=None if run is None else run.created_at,
        )

    def issues(self, scan_id: int | None = None, limit: int = 1000) -> tuple[IssueInfo, ...]:
        """Problems of a scan, the latest by default, in the order they were met."""
        scans = ScansRepo(self._db)
        if scan_id is None:
            latest = scans.latest()
            if latest is None:
                return ()
            scan_id = latest.id
        found = scans.issues(scan_id, limit=limit)
        return tuple(IssueInfo(Path(i.path), i.kind, i.message) for i in found)

    # --- reports ---

    def export_report(
        self, *, csv: Path | str | None = None, html: Path | str | None = None
    ) -> ReportSummary:
        """Write the stored exact groups to a CSV file, an HTML page, or both.

        Raises ``LibraryError`` when duplicates were never searched: an empty report would
        wrongly say that there are none.
        """
        if csv is None and html is None:
            raise ValueError("give a CSV file, an HTML file or both")
        stats = self.stats()
        if stats.exact_run_at is None:
            raise LibraryError("duplicates were not searched yet; run `mediasift dupes` first")
        csv_path = None if csv is None else Path(csv)
        html_path = None if html is None else Path(html)
        files = 0
        try:
            if csv_path is not None:
                files = write_csv(csv_path, self._report_groups())
            if html_path is not None:
                write_html(html_path, self._report_groups())
        except OSError as exc:
            raise ActionError(f"cannot write the report: {exc}") from exc
        return ReportSummary(stats.groups, files, stats.reclaimable_bytes, csv_path, html_path)

    def _report_groups(self) -> Iterator[ReportGroup]:
        for number, group in enumerate(self.iter_groups(), start=1):
            by_id = {m.file_id: m for m in group.members}
            members = tuple(
                ReportMember(
                    path=paths.display_path(m.path),
                    size=m.size,
                    modified=m.mtime.isoformat(timespec="seconds"),
                    media_type=m.media_type,
                    hardlink_of=_owner_path(by_id, m),
                )
                for m in group.members
            )
            yield ReportGroup(number, group.kind, group.size, group.reclaimable_bytes, members)

    # --- conversion ---

    def _group_infos(self, groups: list[MatchGroup]) -> list[GroupInfo]:
        ids = {member.file_id for group in groups for member in group.members}
        records = FilesRepo(self._db).get_many(ids)
        return [_group_info(group, records) for group in groups]


def _owner_path(by_id: Mapping[int, MemberInfo], member: MemberInfo) -> str:
    if member.hardlink_of is None or member.hardlink_of not in by_id:
        return ""
    return paths.display_path(by_id[member.hardlink_of].path)


def _root_info(root: Root) -> RootInfo:
    return RootInfo(
        root.id, root.path, root.enabled, root.read_only, root.prefer_rank, root.volume_id
    )


def _mtime(mtime_ns: int) -> datetime:
    """UTC time of a file; odd values (before 1970, far future) are clamped, never an error."""
    try:
        return _EPOCH + timedelta(microseconds=mtime_ns // 1000)
    except OverflowError:
        return datetime.max.replace(tzinfo=UTC) if mtime_ns > 0 else _EPOCH


def _member_info(file_id: int, record: FileRecord, owner: object) -> MemberInfo:
    return MemberInfo(
        file_id=file_id,
        path=record.path,
        size=record.signature.size,
        mtime=_mtime(record.signature.mtime_ns),
        media_type=record.media_type.value,
        root_id=record.root_id,
        hardlink_of=owner if isinstance(owner, int) else None,
    )


def _group_info(group: MatchGroup, records: Mapping[int, FileRecord]) -> GroupInfo:
    members = tuple(
        _member_info(m.file_id, records[m.file_id], m.evidence.get("hardlink_of"))
        for m in group.members
        if m.file_id in records
    )
    size = members[0].size if members else 0
    return GroupInfo(group.id, group.kind.value, size, group.reclaimable_bytes, members)


def _scan_summary(result: ScanResult) -> ScanSummary:
    return ScanSummary(
        scan_id=result.scan_id,
        state=result.state.value,
        seen=result.seen,
        new=result.new,
        changed=result.changed,
        unchanged=result.unchanged,
        moved=result.moved,
        missing=result.missing,
        restored=result.restored,
        placeholders=result.placeholders,
        skipped_links=result.skipped_links,
        vanished=result.vanished,
        errors=result.errors,
        by_type=dict(result.by_type),
        unreachable_roots=tuple(result.unreachable_roots),
    )


def _exact_summary(result: ExactResult) -> ExactSummary:
    return ExactSummary(
        groups=result.groups,
        reclaimable_bytes=result.reclaimable_bytes,
        hardlink_sets=result.hardlink_sets,
        empty_files=result.empty_files,
        hashed_samples=result.partial.computed,
        hashed_full=result.full.computed,
        issue_count=result.issue_count,
        changed_files=len(result.changed),
        issues=tuple(IssueInfo(i.path, i.kind, i.message) for i in result.issues),
    )


def _scan_info(record: ScanRecord) -> ScanInfo:
    return ScanInfo(
        record.id, record.state.value, record.started_at, record.finished_at, dict(record.counters)
    )
