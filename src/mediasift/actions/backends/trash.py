"""The Trash of Linux desktops, after the freedesktop.org Trash specification 1.0.

A file goes to the home Trash when it is on the same volume, otherwise to the Trash of its own
volume: ``$topdir/.Trash/$uid`` when the administrator provided a shared, sticky ``.Trash``,
else ``$topdir/.Trash-$uid``. Its ``.trashinfo`` is written first, as the specification asks,
so a file manager never sees a trashed file it cannot restore.
"""

from __future__ import annotations

import os
import stat as stat_module
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from mediasift.actions import fsops
from mediasift.actions.backends import base
from mediasift.actions.backends.base import (
    Located,
    Location,
    PlannedRemoval,
    RemovalRequest,
    RestoredFile,
    RestoreRequest,
    StoredFile,
)
from mediasift.core import paths
from mediasift.core.errors import ActionError, MediaSiftError, SafetyError
from mediasift.core.paths import display_path
from mediasift.core.types import FileSignature, Root

INFO_SUFFIX = ".trashinfo"
_NAME_LIMIT = 255 - len(INFO_SUFFIX)
_HEADER = "[Trash Info]"
_MAX_INFO = 64 * 1024


@dataclass(frozen=True, slots=True)
class TrashFolder:
    """A Trash: ``folder`` holds ``files`` and ``info``.

    ``base`` is the existing folder it is created below; ``topdir`` is None for the home
    Trash, whose records hold absolute paths.
    """

    folder: Path
    base: Path
    topdir: Path | None


def mount_point(path: Path) -> Path:
    """The top folder of the volume that holds ``path``."""
    current = path.parent
    while not os.path.ismount(current) and current.parent != current:
        current = current.parent
    return current


class TrashBackend:
    """Moves files into a Trash that desktop file managers can restore from."""

    name = "trash"

    def __init__(
        self,
        *,
        home_trash: Path | None = None,
        topdir: Callable[[Path], Path] = mount_point,
        uid: int | None = None,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._home_trash = home_trash
        self._topdir = topdir
        self._uid_value = uid
        self._now = now

    def check(self, path: Path, root: Root) -> None:
        base.check_removable(path, root)
        trash = self.trash_for(path)
        if _is_below(path, trash.folder):
            raise SafetyError(f"{display_path(path)} is in the Trash", code="in_quarantine")
        st = fsops.stat_or_none(trash.folder)
        if st is None:
            writable = os.access(_existing_ancestor(trash.folder), os.W_OK | os.X_OK)
        else:
            writable = self._usable(st) and all(
                os.access(trash.folder / part, os.W_OK | os.X_OK)
                for part in ("files", "info")
                if fsops.stat_or_none(trash.folder / part) is not None
            )
        if not writable:
            raise SafetyError(
                f"the Trash {display_path(trash.folder)} cannot be written to", code="unsupported"
            )

    def plan_destination(self, path: Path, root: Root, session: str) -> PlannedRemoval:
        trash = self.trash_for(path)
        files, info = trash.folder / "files", trash.folder / "info"
        name = path.name if _fits(path.name) else ""
        for candidate in (name, *fsops.numbered(path.name, limit=_NAME_LIMIT)):
            record = info / (candidate + INFO_SUFFIX)
            if candidate and not _taken(files / candidate) and not _taken(record):
                detail = {"info": os.fspath(record), "trash_path": _recorded_path(path, trash)}
                return PlannedRemoval(path, files / candidate, root, session, detail)
        raise ActionError(f"no free name for {display_path(path)} in the Trash")

    def remove(self, request: RemovalRequest) -> StoredFile:
        planned = request.planned
        src, dst = planned.src, planned.dst
        self.check(src, planned.root)
        trash = self.trash_for(src)
        info = trash.folder / "info" / (dst.name + INFO_SUFFIX)
        value = planned.detail.get("trash_path", "")
        if dst.parent != trash.folder / "files" or planned.detail.get("info") != os.fspath(info):
            raise SafetyError(
                f"the Trash for {display_path(src)} is not the planned one; plan again",
                code="target_exists",
            )
        self._prepare(trash)
        _write_info(info, value, self._now())
        try:
            fsops.move_no_replace(src, dst, request.expected)
            base.confirm_keeper(request, lambda: fsops.put_back(dst, src))
        except BaseException:
            # The record goes only when it describes no file: ours is not in the Trash.
            if not base.is_at(dst, request.expected):
                with suppress(MediaSiftError):
                    _drop_info(info, value)
            raise
        return StoredFile(dst, planned.detail)

    def restore(self, request: RestoreRequest) -> RestoredFile:
        stored = request.stored
        folder = stored.dst.parent.parent
        allowed = self._trash_folders(request.root.path)
        if stored.dst.parent.name != "files" or folder not in allowed:
            raise SafetyError(
                f"{display_path(stored.dst)} is not in a Trash of this volume",
                code="outside_root",
            )
        fsops.refuse_links(stored.dst, folder)
        restored = base.restore_file(stored.dst, request)
        value = stored.detail.get("trash_path")
        if value:
            with suppress(MediaSiftError):
                _drop_info(folder / "info" / (stored.dst.name + INFO_SUFFIX), value)
        return restored

    def locate(self, planned: PlannedRemoval, expected: FileSignature) -> Located:
        return base.locate_by_name(planned, expected)

    def tidy(self, planned: PlannedRemoval, expected: FileSignature) -> bool:
        if self.locate(planned, expected).location is Location.BOTH:
            fsops.release_extra_name(planned.dst, planned.src)
        location = self.locate(planned, expected).location
        info, value = planned.detail.get("info"), planned.detail.get("trash_path")
        if location is Location.SOURCE and info and value and not _taken(planned.dst):
            with suppress(MediaSiftError):
                _drop_info(Path(info), value)
        return location in (Location.SOURCE, Location.DESTINATION)

    # --- which Trash ---

    def trash_for(self, path: Path) -> TrashFolder:
        """The Trash the specification picks for a file."""
        file = fsops.stat_or_none(path)
        if file is None:
            raise SafetyError(f"{display_path(path)} does not exist", code="missing")
        home = self._home()
        home_base = _existing_ancestor(home)
        if os.stat(home_base).st_dev == file.st_dev:
            return TrashFolder(home, home_base, None)
        top = self._topdir(path)
        shared = top / ".Trash"
        st = fsops.stat_or_none(shared)
        if st is not None and _is_folder(st) and st.st_mode & stat_module.S_ISVTX:
            own = shared / str(self._uid())
            own_st = fsops.stat_or_none(own)
            if own_st is None and os.access(shared, os.W_OK | os.X_OK):
                return TrashFolder(own, shared, top)
            if own_st is not None and self._usable(own_st):
                return TrashFolder(own, shared, top)
        return TrashFolder(top / f".Trash-{self._uid()}", top, top)

    def _trash_folders(self, path: Path) -> set[Path]:
        top = self._topdir(path / "x")
        uid = str(self._uid())
        return {self._home(), top / ".Trash" / uid, top / f".Trash-{uid}"}

    def _home(self) -> Path:
        if self._home_trash is not None:
            return self._home_trash
        data = os.environ.get("XDG_DATA_HOME", "")
        if not os.path.isabs(data):
            data = os.path.join(os.path.expanduser("~"), ".local", "share")
        return paths.resolve(data) / "Trash"

    def _uid(self) -> int:
        if self._uid_value is not None:
            return self._uid_value
        if sys.platform == "win32":
            raise SafetyError("there is no Trash on this system", code="unsupported")
        return os.getuid()

    def _usable(self, st: os.stat_result) -> bool:
        return _is_folder(st) and st.st_uid == self._uid()

    def _prepare(self, trash: TrashFolder) -> None:
        for part in ("files", "info"):
            if fsops.stat_or_none(trash.folder / part) is None:
                fsops.make_dirs_private(trash.folder / part, trash.base)
            fsops.refuse_links(trash.folder / part / "x", trash.base)
        st = fsops.stat_or_none(trash.folder)
        if st is None or not self._usable(st):
            raise SafetyError(
                f"the Trash {display_path(trash.folder)} belongs to someone else",
                code="unsupported",
            )


# --- the .trashinfo record ----------------------------------------------------------


def _recorded_path(path: Path, trash: TrashFolder) -> str:
    """Percent-encoded bytes: absolute for the home Trash, relative to the volume otherwise."""
    shown = path if trash.topdir is None else path.relative_to(trash.topdir)
    return quote(os.fsencode(shown), safe="/")


def info_text(value: str, when: datetime) -> str:
    """The content of a ``.trashinfo`` file."""
    return f"{_HEADER}\nPath={value}\nDeletionDate={when.strftime('%Y-%m-%dT%H:%M:%S')}\n"


def recorded_value(text: str) -> str | None:
    """The ``Path`` of a ``.trashinfo`` file, or None when it is not one."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != _HEADER:
        return None
    for line in lines[1:]:
        if line.startswith("Path="):
            return line[len("Path=") :]
    return None


def _write_info(info: Path, value: str, when: datetime) -> None:
    """Create the record; it exists only if this call made it."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        handle = os.open(info, flags, 0o600)
    except FileExistsError as exc:
        raise SafetyError(f"{display_path(info)} already exists", code="target_exists") from exc
    except OSError as exc:
        raise SafetyError(
            f"cannot write to the Trash ({exc.strerror or exc})", code="unsupported"
        ) from exc
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
            file.write(info_text(value, when))
            file.flush()
            os.fsync(file.fileno())
    except OSError as exc:
        with suppress(MediaSiftError):
            _drop_info(info, value)
        raise ActionError(f"cannot write to the Trash ({exc.strerror or exc})") from exc


def _drop_info(info: Path, value: str) -> None:
    """Remove a record MediaSift wrote, after making sure that it is that record."""
    st = fsops.stat_or_none(info)
    if st is None or not stat_module.S_ISREG(st.st_mode) or st.st_size > _MAX_INFO:
        raise SafetyError(f"{display_path(info)} is not a Trash record", code="changed")
    try:
        with os.fdopen(fsops.open_for_reading(info), "rb") as file:
            text = file.read(_MAX_INFO).decode("utf-8", "replace")
    except OSError as exc:
        raise ActionError(f"cannot read {display_path(info)}: {exc.strerror or exc}") from exc
    if recorded_value(text) != value:
        raise SafetyError(f"{display_path(info)} describes another file", code="changed")
    try:
        os.unlink(info)
    except OSError as exc:
        raise ActionError(f"cannot remove {display_path(info)}: {exc.strerror or exc}") from exc


# --- helpers ------------------------------------------------------------------------


def _existing_ancestor(path: Path) -> Path:
    current = path
    while fsops.stat_or_none(current) is None and current.parent != current:
        current = current.parent
    return current


def _is_folder(st: os.stat_result) -> bool:
    return stat_module.S_ISDIR(st.st_mode) and not paths.is_link_like(st)


def _is_below(path: Path, folder: Path) -> bool:
    try:
        path.relative_to(folder)
    except ValueError:
        return False
    return True


def _taken(path: Path) -> bool:
    return fsops.stat_or_none(path) is not None


def _fits(name: str) -> bool:
    return len(os.fsencode(name)) <= _NAME_LIMIT
