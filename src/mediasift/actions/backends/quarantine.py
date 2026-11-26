"""Quarantine: files wait in a hidden folder of their own root, one rename away."""

from __future__ import annotations

import ctypes
import stat as stat_module
import sys
from pathlib import Path

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
from mediasift.core.errors import ActionError, SafetyError
from mediasift.core.paths import QUARANTINE_FOLDER, display_path, fs_path
from mediasift.core.types import FileSignature, Root

_HIDDEN_ATTRIBUTE = 0x2


class QuarantineBackend:
    """``<root>/.mediasift-quarantine/<session>/<path within the root>``."""

    name = "quarantine"

    def check(self, path: Path, root: Root) -> None:
        base.check_removable(path, root)
        base.check_same_volume(path, root)
        folder = root.path / QUARANTINE_FOLDER
        if fsops.stat_or_none(folder) is not None:
            fsops.refuse_links(folder / "any", root.path)
            base.check_same_volume(folder, root)

    def plan_destination(self, path: Path, root: Root, session: str) -> PlannedRemoval:
        return PlannedRemoval(path, quarantine_place(path, root, session), root, session)

    def remove(self, request: RemovalRequest) -> StoredFile:
        planned = request.planned
        src, dst = planned.src, planned.dst
        self.check(src, planned.root)
        make_quarantine_folder(planned)
        fsops.move_no_replace(src, dst, request.expected)
        base.confirm_keeper(request, lambda: fsops.put_back(dst, src))
        return StoredFile(dst, planned.detail)

    def restore(self, request: RestoreRequest) -> RestoredFile:
        return restore_from_quarantine(request.stored.dst, request)

    def locate(self, planned: PlannedRemoval, expected: FileSignature) -> Located:
        return base.locate_by_name(planned, expected)

    def tidy(self, planned: PlannedRemoval, expected: FileSignature) -> bool:
        # Two names can be left by a crash on file systems where a move is link plus unlink.
        if self.locate(planned, expected).location is Location.BOTH:
            fsops.release_extra_name(planned.dst, planned.src)
        return self.locate(planned, expected).location in (Location.SOURCE, Location.DESTINATION)


def quarantine_place(path: Path, root: Root, session: str) -> Path:
    """A free place that mirrors where the file was.

    Names already there, also under another spelling the file system treats as the same, get
    a number.
    """
    current = base.session_folder(root, session)
    *folders, name = fsops.parts_below(path, root.path)
    for folder in folders:
        current = current / _pick(current, folder, folder_allowed=True)
    return current / _pick(current, name, folder_allowed=False)


def _pick(folder: Path, name: str, *, folder_allowed: bool) -> str:
    for candidate in (name, *fsops.numbered(name)):
        st = fsops.stat_or_none(folder / candidate)
        if st is None:
            return candidate
        if folder_allowed and stat_module.S_ISDIR(st.st_mode) and not paths.is_link_like(st):
            return candidate
    raise ActionError(f"no free name for {display_path(name)} in {display_path(folder)}")


def make_quarantine_folder(planned: PlannedRemoval) -> None:
    """Create the folder of the planned destination, which must be in the session's."""
    root = planned.root
    fsops.parts_below(planned.dst, base.session_folder(root, planned.session))
    top = root.path / QUARANTINE_FOLDER
    existed = fsops.stat_or_none(top) is not None
    fsops.make_dirs_private(planned.dst.parent, root.path)
    if not existed:
        _hide(top)


def _hide(folder: Path) -> None:
    """Keep photo apps and people out of it on Windows. Cosmetic, so failure is ignored."""
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetFileAttributesW(fs_path(folder), _HIDDEN_ATTRIBUTE)


def restore_from_quarantine(source: Path, request: RestoreRequest) -> RestoredFile:
    """Restore a file that lies in the quarantine of its root."""
    quarantine = request.root.path / QUARANTINE_FOLDER
    fsops.parts_below(source, quarantine)
    try:
        fsops.refuse_links(source, request.root.path)
    except SafetyError as exc:
        code = "stored_missing" if exc.code == "missing" else exc.code
        raise SafetyError(str(exc), code=code) from exc
    return base.restore_file(source, request)
