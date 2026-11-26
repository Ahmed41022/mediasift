"""Walking one root.

The walk is iterative, so neither depth nor path length is limited, and it yields files one
by one, so memory holds little more than the names of folders still to be visited. It reads
folder listings and file attributes only: no file is opened.
"""

from __future__ import annotations

import os
import stat as stat_module
from collections.abc import Iterator
from dataclasses import dataclass

from mediasift.core import paths
from mediasift.core.types import MediaType, Root, media_type_for
from mediasift.db.scans import ScanIssue
from mediasift.jobs.context import JobContext
from mediasift.scan.rules import Rules


@dataclass(frozen=True, slots=True)
class Entry:
    """A media file as its folder lists it.

    ``stat`` is complete on POSIX. On Windows it comes from the listing: no identity, no link
    count, and size and time may be out of date for files with several hard links. None means
    that the file exists but could not be asked.
    """

    rel: str
    media_type: MediaType
    stat: os.stat_result | None


def root_problem(root: Root) -> str | None:
    """Why a root cannot be trusted right now, or None when it is the folder it used to be.

    An unplugged drive leaves either nothing or the empty folder it was mounted on; the
    latter lives on another volume.
    """
    try:
        st = os.lstat(paths.fs_path(root.path))
    except (OSError, ValueError) as exc:
        return f"the folder cannot be reached: {reason(exc)}"
    if paths.is_link_like(st):
        return "the folder was replaced by a link"
    if not stat_module.S_ISDIR(st.st_mode):
        return "there is a file in place of the folder"
    volume = paths.volume_id(root.path)
    if not volume:
        return "the volume of the folder cannot be identified"
    if root.volume_id and volume != root.volume_id:
        return "another volume is mounted at the folder"
    return None


class Walker:
    """One walk over one root. Its attributes tell what was not looked at."""

    def __init__(self, root: Root, rules: Rules, job: JobContext) -> None:
        self.root = root
        self.skipped_links = 0
        self.vanished = 0
        self.unread_folders: list[str] = []
        self._issues: list[ScanIssue] = []
        self._rules = rules
        self._job = job
        self._base = paths.fs_path(root.path)
        self._device: int | None = None

    def add_issue(self, rel: str, kind: str, message: str) -> None:
        """Note a problem with a path below the root."""
        path = os.path.join(self.root.path, rel) if rel else self.root.path
        self._issues.append(ScanIssue(path, kind, message, self.root.id))

    def take_issues(self) -> list[ScanIssue]:
        """Problems noted since the last call."""
        issues, self._issues = self._issues, []
        return issues

    def files(self) -> Iterator[Entry]:
        """Media files that the rules let through, in no particular order."""
        if os.name != "nt":
            try:
                self._device = os.lstat(self._base).st_dev
            except OSError as exc:
                self._unread("", exc)
                return
        pending = [""]
        while pending:
            yield from self._folder(pending.pop(), pending)

    def _folder(self, rel: str, pending: list[str]) -> Iterator[Entry]:
        self._job.raise_if_cancelled()
        try:
            entries = os.scandir(os.path.join(self._base, rel) if rel else self._base)
        except OSError as exc:
            self._unread(rel, exc)
            return
        with entries:
            while True:
                self._job.raise_if_cancelled()
                try:
                    entry = next(entries)
                except StopIteration:
                    return
                except OSError as exc:
                    self._unread(rel, exc)
                    return
                child = os.path.join(rel, entry.name) if rel else entry.name
                found = self._look(entry, child, pending)
                if found is not None:
                    yield found

    def _look(self, entry: os.DirEntry[str], rel: str, pending: list[str]) -> Entry | None:
        media: MediaType | None = None
        try:
            # The type comes with the listing, so files of other kinds cost no system call.
            if entry.is_symlink():
                self.skipped_links += 1
                return None
            is_folder = entry.is_dir(follow_symlinks=False)
            media = None if is_folder else media_type_for(entry.name)
            if not is_folder and media is None:
                return None
            st = entry.stat(follow_symlinks=False)
        except FileNotFoundError:
            self.vanished += 1
            return None
        except OSError as exc:
            return self._unknown(rel, media, exc)
        if paths.is_link_like(st):
            self.skipped_links += 1
            return None
        attributes = getattr(st, "st_file_attributes", 0)
        if media is None:
            if not self._rules.skips_folder(rel, attributes) and not self._mounted(rel, st):
                pending.append(rel)
            return None
        if not stat_module.S_ISREG(st.st_mode):
            return None
        if self._rules.skips_file(rel, st.st_size, attributes):
            return None
        return Entry(rel, media, st)

    def _unknown(self, rel: str, media: MediaType | None, exc: OSError) -> Entry | None:
        """An entry that cannot be asked. It exists, so nothing known may become missing."""
        if media is None:
            if not self._rules.skips_folder(rel):
                self._unread(rel, exc)
            return None
        if self._rules.skips_file(rel):
            return None
        self.add_issue(rel, "unreadable_file", reason(exc))
        return Entry(rel, media, None)

    def _mounted(self, rel: str, st: os.stat_result) -> bool:
        """Another volume mounted inside the root (Windows mounts are links, seen above)."""
        if self._device is None or st.st_dev == self._device:
            return False
        self.skipped_links += 1
        self.add_issue(rel, "mount_point", "another volume is mounted here; it is not scanned")
        return True

    def _unread(self, rel: str, exc: OSError) -> None:
        # A folder that vanished is unread too: where its files went is not known yet.
        self.unread_folders.append(rel or ".")
        if rel and isinstance(exc, FileNotFoundError | NotADirectoryError):
            self.vanished += 1
        else:
            self.add_issue(rel, "unreadable_folder", reason(exc))


def reason(exc: Exception) -> str:
    """Short text for an issue, without the path that the issue carries anyway."""
    return (exc.strerror if isinstance(exc, OSError) else None) or str(exc) or type(exc).__name__
