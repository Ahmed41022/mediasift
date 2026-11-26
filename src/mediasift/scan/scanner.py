"""Keeping the index in step with the disk.

A scan walks each root, compares what it lists with the index in batches, and commits every
batch, so a cancelled scan keeps its work and the next one only re-reads folders.

Identity of a file:

* Same place, same size and time: unchanged. A changed file at the same place keeps its id;
  its cached features stop being valid because they carry the size and time they came from.
* A file at a new place whose (device, inode, size, time) equal those of a known file whose
  old place no longer holds it was moved: it keeps its id, features and decisions. The old
  place is checked with ``lstat`` when the decision is made, so a hard link (old place still
  there) is a new file, and so is a file that reuses the inode of a deleted one with another
  size or time.
* Across volumes there is no identity. A file there takes over a gone file with the same
  name, size and time only when exactly one such file is gone, nothing else could be it, and
  no other new file claims it too. These, and files that replaced a known one at its place,
  are settled after every root was walked, so the result does not depend on the order of the
  walk (renames in a cycle, such as a swap of two names, work too).
* Files are marked missing only after a complete walk of a root that is still the folder it
  was, and never below a folder that could not be read.

On Windows a folder listing carries no inode, so finding one costs an ``lstat`` (an open and
a close of the file). A rescan skips it when the listing shows the size and time the index
has, the index knows the identity and the file has one link: the listing of a file with
several hard links may be out of date. A new or changed file is always asked. The cost is
that a replacement with the very same size and time at the same place goes unseen on
Windows; the full hash is re-checked before any action, so this can never remove a file.
"""

from __future__ import annotations

import contextlib
import enum
import os
import stat as stat_module
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mediasift.core import paths
from mediasift.core.config import Settings
from mediasift.core.errors import JobCancelled, LibraryError
from mediasift.core.types import FileRecord, FileSignature, FileStatus, MediaType, Root
from mediasift.db.connection import Database
from mediasift.db.files import FileMove, FilesRepo, FoundFile
from mediasift.db.roots import RootsRepo
from mediasift.db.scans import ScanIssue, ScansRepo, ScanState
from mediasift.jobs.batch import batched_commit
from mediasift.jobs.context import JobContext, Stage
from mediasift.scan.rules import Rules
from mediasift.scan.walker import Entry, Walker, reason, root_problem

STAGE = "scan"


@dataclass
class ScanResult:
    """What a scan did. ``seen`` counts the media files it listed."""

    scan_id: int
    state: ScanState = ScanState.RUNNING
    seen: int = 0
    new: int = 0
    changed: int = 0
    unchanged: int = 0
    moved: int = 0
    missing: int = 0
    restored: int = 0
    placeholders: int = 0
    skipped_links: int = 0
    vanished: int = 0
    errors: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    unreachable_roots: list[int] = field(default_factory=list)

    def counters(self) -> dict[str, Any]:
        """The counters, for the scans table."""
        return {key: value for key, value in asdict(self).items() if key not in ("scan_id",)}


class Scanner:
    """Scans roots into the index. One scan at a time per library."""

    def __init__(self, db: Database, settings: Settings | None = None) -> None:
        self._db = db
        self._settings = settings or Settings()

    def scan(
        self, roots: Iterable[Root] | None = None, job: JobContext | None = None
    ) -> ScanResult:
        """Scan the given roots, or every enabled one.

        A cancelled scan returns with state ``cancelled``: its batches are kept, nothing is
        marked missing, and running it again picks up cheaply. Other errors mark the scan
        failed and propagate.
        """
        chosen = self._chosen(roots)
        scans = ScansRepo(self._db)
        scan_id = scans.start({"roots": [root.id for root in chosen]})
        run = _Run(self._db, self._settings, scan_id, job or JobContext())
        try:
            run.scan(chosen)
        except JobCancelled:
            run.result.state = ScanState.CANCELLED
        except BaseException:
            run.result.state = ScanState.FAILED
            with contextlib.suppress(LibraryError):  # the original error says more
                scans.finish(scan_id, ScanState.FAILED, run.result.counters())
            raise
        else:
            run.result.state = ScanState.DONE
        scans.finish(scan_id, run.result.state, run.result.counters())
        return run.result

    def _chosen(self, roots: Iterable[Root] | None) -> list[Root]:
        repo = RootsRepo(self._db)
        if roots is None:
            return repo.list(enabled_only=True)
        chosen: dict[int, Root] = {}
        for given in roots:
            root = repo.get(given.id)  # as stored now, not as the caller remembers it
            if not root.enabled:
                raise LibraryError(f"{paths.display_path(root.path)} is disabled")
            chosen[root.id] = root
        return list(chosen.values())


class _Place(enum.Enum):
    HERE = "here"
    GONE = "gone"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class _Later:
    """A file whose fate is settled after the walk, with what was found at its place."""

    root_id: int
    rel: str
    media_type: MediaType
    signature: FileSignature
    status: FileStatus

    def found(self) -> FoundFile:
        return FoundFile(self.root_id, self.rel, self.media_type, self.signature, self.status)

    @property
    def name(self) -> str:
        return os.path.basename(self.rel)


@dataclass
class _Writes:
    """Changes of one batch, written together."""

    moves: list[FileMove] = field(default_factory=list)
    upserts: list[FoundFile] = field(default_factory=list)
    seen: dict[FileStatus, list[int]] = field(default_factory=lambda: defaultdict(list))


class _Places:
    """Answers whether a known file is still at its place, looking at the disk now."""

    def __init__(self, roots: dict[int, Root]) -> None:
        self._roots = roots
        self._listings: dict[str, frozenset[str]] = {}

    def forget(self) -> None:
        """Drop cached folder listings; called between batches."""
        self._listings.clear()

    def look(self, record: FileRecord) -> _Place:
        root = self._roots[record.root_id]
        try:
            place = self._probe(root, record)
        except (FileNotFoundError, NotADirectoryError):
            place = _Place.GONE
        except (OSError, ValueError):
            return _Place.UNKNOWN
        # Gone only counts on the volume the root lives on: an unplugged drive proves nothing.
        if place is _Place.GONE and root_problem(root) is not None:
            return _Place.UNKNOWN
        return place

    def _probe(self, root: Root, record: FileRecord) -> _Place:
        folder = root.path
        for part in record.rel_path.parts[:-1]:
            folder = folder / part
            st = os.lstat(paths.fs_path(folder))
            # A folder moved away and replaced by a link: its files are elsewhere now.
            if paths.is_link_like(st) or not stat_module.S_ISDIR(st.st_mode):
                return _Place.GONE
        st = os.lstat(paths.fs_path(record.path))
        if paths.is_link_like(st) or not stat_module.S_ISREG(st.st_mode):
            return _Place.GONE
        here, known = paths.signature(st), record.signature
        if here.inode and known.inode:
            same = here.inode == known.inode
        else:
            same = here.same_content_state(known)
        if not same:
            return _Place.GONE
        if not root.case_sensitive and not self._spelled(root, record.rel_path):
            return _Place.GONE  # renamed in case only: the old spelling is gone
        return _Place.HERE

    def _spelled(self, root: Root, rel: Path) -> bool:
        folder = paths.fs_path(root.path)
        for part in rel.parts:
            names = self._listings.get(folder)
            if names is None:
                try:
                    names = self._listings[folder] = frozenset(os.listdir(folder))
                except OSError:
                    return True  # cannot tell: not a move
            if part not in names:
                return False
            folder = os.path.join(folder, part)
        return True


class _Run:
    """The state of one scan."""

    def __init__(self, db: Database, settings: Settings, scan_id: int, job: JobContext) -> None:
        self._db = db
        self._settings = settings
        self._files = FilesRepo(db)
        self._scans = ScansRepo(db)
        self._job = job
        self.result = ScanResult(scan_id)
        self._roots = {root.id: root for root in RootsRepo(db).list()}
        self._places = _Places(self._roots)
        self._claimed: set[int] = set()
        self._replaced: list[_Later] = []
        self._arrivals: list[_Later] = []
        self._complete: list[Walker] = []

    @property
    def _scan_id(self) -> int:
        return self.result.scan_id

    # --- the whole scan ---

    def scan(self, roots: list[Root]) -> None:
        with self._job.stage(STAGE) as stage:
            for root in roots:
                self._scan_root(root, stage)
            self._job.raise_if_cancelled()
            self._settle_later()
            self._mark_missing()
            stage.set_total(self.result.seen)

    def _scan_root(self, root: Root, stage: Stage) -> None:
        problem = root_problem(root)
        if problem is not None:
            self._unreachable(root, problem)
            return
        if not root.volume_id:
            root = self._roots[root.id] = RootsRepo(self._db).set_volume_id(
                root.id, paths.volume_id(root.path)
            )
        rules = Rules.for_root(self._settings.scan, root, self._db.directory)
        walker = Walker(root, rules, self._job)
        try:
            batched_commit(
                walker.files(),
                lambda batch: self._flush(walker, batch, stage),
                size=self._settings.jobs.batch_size,
            )
        finally:
            self.result.skipped_links += walker.skipped_links
            self.result.vanished += walker.vanished
            self._record_issues(walker.take_issues())
        problem = root_problem(root)
        if problem is not None:
            self._unreachable(root, problem)
        elif "." not in walker.unread_folders:
            self._complete.append(walker)

    def _unreachable(self, root: Root, problem: str) -> None:
        self.result.unreachable_roots.append(root.id)
        self._record_issues([ScanIssue(root.path, "root_unreachable", problem, root.id)])

    def _record_issues(self, issues: list[ScanIssue]) -> None:
        if issues:
            self.result.errors += self._scans.add_issues(self._scan_id, issues)

    # --- one batch ---

    def _flush(self, walker: Walker, batch: list[Entry], stage: Stage) -> None:
        root = walker.root
        writes = _Writes()
        with self._db.transaction():
            known = self._files.find_by_paths(root.id, [entry.rel for entry in batch])
            for entry in batch:
                self._count_seen(entry)
                self._settle(walker, entry, known.get(entry.rel), writes)
            self._write(writes)
            self._record_issues(walker.take_issues())
        self._claimed.clear()
        self._places.forget()
        size = sum(entry.stat.st_size for entry in batch if entry.stat is not None)
        last = os.path.join(root.path, batch[-1].rel)
        stage.advance(len(batch), bytes=size, current=paths.display_path(last))

    def _count_seen(self, entry: Entry) -> None:
        self.result.seen += 1
        kind = entry.media_type.value
        self.result.by_type[kind] = self.result.by_type.get(kind, 0) + 1

    def _write(self, writes: _Writes) -> None:
        if writes.moves:
            self._files.move_many(writes.moves, self._scan_id)
        if writes.upserts:
            self._files.upsert_many(writes.upserts, self._scan_id)
        for status, ids in writes.seen.items():
            self._files.mark_seen(ids, self._scan_id, status)

    def _settle(self, walker: Walker, entry: Entry, row: FileRecord | None, out: _Writes) -> None:
        if entry.stat is None:  # exists but cannot be asked: keep what is known
            if row is not None:
                out.seen[FileStatus.ERROR].append(row.id)
            return
        if paths.is_placeholder(entry.stat):
            self.result.placeholders += 1
        if row is not None and self._unchanged_by_listing(entry.stat, row):
            self._count_known(row, row.signature)
            out.seen[_status_of(entry.stat)].append(row.id)
            return
        asked = self._ask(walker, entry, row, out)
        if asked is None:
            return
        later = _Later(walker.root.id, entry.rel, entry.media_type, *asked)
        if row is None:
            self._settle_new(later, out)
        else:
            self._settle_known(later, row, out)

    def _unchanged_by_listing(self, st: os.stat_result, row: FileRecord) -> bool:
        if row.status is FileStatus.MISSING:
            return False
        listed = paths.signature(st)
        placeholder = paths.is_placeholder(st)
        if placeholder != (row.status is FileStatus.PLACEHOLDER):
            return False
        if os.name != "nt":
            return listed == row.signature
        known = row.signature
        trusted = placeholder or (known.inode != 0 and known.nlink == 1)
        return trusted and listed.same_content_state(known)

    def _ask(
        self, walker: Walker, entry: Entry, row: FileRecord | None, out: _Writes
    ) -> tuple[FileSignature, FileStatus] | None:
        """Full signature and status of a listed file; None when it is not there to index."""
        st = entry.stat
        assert st is not None
        # A placeholder is never opened, not even for its identity.
        if os.name == "nt" and not paths.is_placeholder(st):
            try:
                st = os.lstat(paths.fs_path(os.path.join(walker.root.path, entry.rel)))
            except FileNotFoundError:
                walker.vanished += 1
                return None
            except OSError as exc:
                walker.add_issue(entry.rel, "unreadable_file", reason(exc))
                if row is not None:
                    out.seen[FileStatus.ERROR].append(row.id)
                return None
            if paths.is_link_like(st) or not stat_module.S_ISREG(st.st_mode):
                walker.skipped_links += paths.is_link_like(st)
                return None
        return paths.signature(st), _status_of(st)

    def _settle_known(self, later: _Later, row: FileRecord, out: _Writes) -> None:
        sig, known = later.signature, row.signature
        other_file = bool(sig.inode and known.inode and sig.inode != known.inode)
        if other_file and (
            not sig.same_content_state(known) or self._heir(later, exclude=row.id) is not None
        ):
            # Another file took this place; where it came from is settled after the walk.
            self._replaced.append(later)
            return
        self._count_known(row, sig)
        if sig == known and later.status is row.status:
            out.seen[later.status].append(row.id)
        else:
            out.upserts.append(later.found())

    def _settle_new(self, later: _Later, out: _Writes) -> None:
        heir = self._heir(later)
        if heir is not None:
            self._claimed.add(heir.id)
            self.result.moved += 1
            out.moves.append(FileMove(heir.id, later.root_id, later.rel, later.signature))
            out.upserts.append(later.found())
        elif self._relics(later):
            self._arrivals.append(later)
        else:
            self.result.new += 1
            out.upserts.append(later.found())

    def _count_known(self, row: FileRecord, sig: FileSignature) -> None:
        if row.status is FileStatus.MISSING:
            self.result.restored += 1
        elif sig.same_content_state(row.signature):
            self.result.unchanged += 1
        else:
            self.result.changed += 1

    # --- where did it come from ---

    def _heir(self, later: _Later, exclude: int | None = None) -> FileRecord | None:
        """The known file that ``later`` is, by identity, when its old place lost it."""
        sig = later.signature
        gone = []
        for record in self._files.find_by_identity(sig.dev, sig.inode):
            if record.id in self._claimed or record.id == exclude:
                continue
            if not record.signature.same_content_state(sig):
                continue  # an inode used again by another file
            place = self._places.look(record)
            if place is _Place.GONE:
                gone.append(record)
            elif place is _Place.HERE and record.signature.nlink != sig.nlink:
                self._relink(record.id, sig.nlink)
        # Several gone names of one inode are all the same file: prefer the same name.
        return min(gone, key=lambda r: (r.path.name != later.name, r.id), default=None)

    def _relink(self, file_id: int, nlink: int) -> None:
        """A hard link was added: the other names must stop trusting their listing."""
        self._db.execute("UPDATE files SET nlink = ? WHERE id = ?", (nlink, file_id))

    def _relics(self, later: _Later) -> list[FileRecord]:
        """Known files that a file from another volume may be: same name, size and time."""
        sig = later.signature
        if sig.size == 0:
            return []
        return [
            record
            for record in self._files.find_by_size_mtime(sig.size, sig.mtime_ns)
            if record.id not in self._claimed
            and record.path.name == later.name
            and (record.signature.dev != sig.dev or not (sig.inode and record.signature.inode))
            and not (record.root_id == later.root_id and str(record.rel_path) == later.rel)
        ]

    # --- after the walk ---

    def _settle_later(self) -> None:
        if not (self._replaced or self._arrivals):
            return
        with self._db.transaction():
            self._settle_replaced()
            self._settle_arrivals()

    def _settle_replaced(self) -> None:
        moves: list[tuple[_Later, FileRecord]] = []
        targets: dict[int, FileRecord] = {}
        for later in self._replaced:
            self._job.raise_if_cancelled()
            row = self._row_at(later)
            heir = self._heir(later, exclude=row.id if row else None)
            if heir is not None:
                self._claimed.add(heir.id)
                moves.append((later, heir))
                if row is not None:
                    targets[row.id] = row
        movers = {heir.id for _, heir in moves}
        # A file whose place was taken and which went nowhere known is forgotten.
        displaced = [row for row in targets.values() if row.id not in movers]
        self._files.set_status((row.id for row in displaced), FileStatus.MISSING)
        self.result.missing += sum(row.status is not FileStatus.MISSING for row in displaced)
        self._apply_moves(moves)
        moved = {(later.root_id, later.rel) for later, _ in moves}
        rest = [later for later in self._replaced if (later.root_id, later.rel) not in moved]
        for later in rest:
            row = self._row_at(later)
            if row is None:
                self.result.new += 1
            else:
                self._count_known(row, later.signature)
        self._files.upsert_many((later.found() for later in rest), self._scan_id)

    def _settle_arrivals(self) -> None:
        wanted: dict[int, list[tuple[_Later, FileRecord]]] = defaultdict(list)
        alone: list[_Later] = []
        for later in self._arrivals:
            self._job.raise_if_cancelled()
            gone, unsure = [], False
            for record in self._relics(later):
                place = self._places.look(record)
                gone += [record] if place is _Place.GONE else []
                unsure |= place is _Place.UNKNOWN
            if len(gone) == 1 and not unsure and self._row_at(later) is None:
                wanted[gone[0].id].append((later, gone[0]))
            else:
                alone.append(later)
        moves = []
        for claims in wanted.values():
            if len(claims) == 1:
                moves.extend(claims)
            else:  # never guess between several
                alone.extend(later for later, _ in claims)
        self._apply_moves(moves)
        for later in alone:
            row = self._row_at(later)
            if row is None:
                self.result.new += 1
            else:
                self._count_known(row, later.signature)
        self._files.upsert_many((later.found() for later in alone), self._scan_id)

    def _apply_moves(self, moves: list[tuple[_Later, FileRecord]]) -> None:
        if not moves:
            return
        self._files.move_many(
            [FileMove(heir.id, later.root_id, later.rel, later.signature) for later, heir in moves],
            self._scan_id,
        )
        self._files.upsert_many((later.found() for later, _ in moves), self._scan_id)
        self.result.moved += len(moves)

    def _row_at(self, later: _Later) -> FileRecord | None:
        return self._files.find_by_paths(later.root_id, [later.rel]).get(later.rel)

    def _mark_missing(self) -> None:
        """Only roots walked completely, and checked again afterwards, lose files."""
        self._job.raise_if_cancelled()
        with self._db.transaction():
            for walker in self._complete:
                self.result.missing += self._files.mark_missing(
                    walker.root.id, self._scan_id, unread_folders=walker.unread_folders
                )


def _status_of(st: os.stat_result) -> FileStatus:
    return FileStatus.PLACEHOLDER if paths.is_placeholder(st) else FileStatus.PRESENT
