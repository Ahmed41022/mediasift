"""File-system primitives: the only code in MediaSift that may move a user's file.

Rules every function here keeps:

* Nothing is overwritten. A target that exists, under whatever spelling the file system
  treats as the same name, is an error.
* Nothing crosses a volume, because that would be a copy followed by a delete.
* Only regular files move. Links, folders, placeholders and devices are refused.
* A name is unlinked only while the same file provably has another name
  (``release_extra_name``), so the content stays reachable.

What stays open between a check and the rename (see ``checks`` for the checks): operating
systems offer no "rename if this is still the file I looked at". So after every move the file
at the target is compared with the expected one, and a stranger is moved back.
"""

from __future__ import annotations

import ctypes
import errno
import functools
import hashlib
import os
import stat as stat_module
import sys
import uuid
from collections.abc import Callable, Iterator
from contextlib import suppress
from pathlib import Path

from mediasift.core import paths
from mediasift.core.errors import (
    ActionError,
    FileChanged,
    JobCancelled,
    MediaSiftError,
    SafetyError,
)
from mediasift.core.paths import display_path, fs_path
from mediasift.core.types import FileSignature

DIGEST_SIZE = 32
_CHUNK = 1024 * 1024
_NAME_LIMIT = 255
_LONGEST_KEPT_EXTENSION = 32
_MAX_NUMBER = 10_000

_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_NO_SUCH_RENAME = {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}


class MisplacedFile(ActionError):
    """A file ended up somewhere it should not be, and could not be moved back.

    Nothing is lost: ``path`` is where the file is.
    """

    def __init__(self, message: str, path: Path) -> None:
        super().__init__(message)
        self.path = path


# --- looking ------------------------------------------------------------------------


def stat_or_none(path: Path) -> os.stat_result | None:
    """``lstat``, or None when nothing has that name. Other problems are not hidden."""
    try:
        return os.lstat(fs_path(path))
    except (FileNotFoundError, NotADirectoryError):
        return None
    except ValueError as exc:
        raise SafetyError(str(exc), code="unsupported") from exc
    except OSError as exc:
        raise ActionError(f"cannot look at {display_path(path)}: {exc.strerror or exc}") from exc


def unchanged(current: FileSignature, expected: FileSignature) -> bool:
    """True for the same file on disk with the same size and modified time.

    The link count is left out: a guard hard link changes it.
    """
    return current.same_file(expected) and current.same_content_state(expected)


def regular_signature(path: Path) -> FileSignature:
    """Signature of a regular file whose content is on this disk. Refuses anything else."""
    st = stat_or_none(path)
    shown = display_path(path)
    if st is None:
        raise SafetyError(f"{shown} does not exist", code="missing")
    if paths.is_link_like(st):
        raise SafetyError(f"{shown} is a link", code="is_link")
    if not stat_module.S_ISREG(st.st_mode):
        raise SafetyError(f"{shown} is not a regular file", code="not_regular")
    if paths.is_placeholder(st):
        raise SafetyError(f"the content of {shown} is not on this disk", code="is_placeholder")
    current = paths.signature(st)
    if not current.inode:
        raise SafetyError(
            f"{shown} is on a file system that cannot tell files apart", code="unsupported"
        )
    return current


def expect_unchanged(path: Path, expected: FileSignature) -> FileSignature:
    """``regular_signature``, refusing a file that is not the expected one."""
    current = regular_signature(path)
    if not unchanged(current, expected):
        raise SafetyError(f"{display_path(path)} has changed", code="changed")
    return current


def refuse_links(path: Path, within: Path) -> None:
    """Refuse when ``within``, or a folder between it and ``path``, is a link or is gone.

    ``path`` itself is not looked at.
    """
    for folder in _folders_between(path, within):
        st = stat_or_none(folder)
        if st is None:
            raise SafetyError(f"{display_path(folder)} does not exist", code="missing")
        _expect_folder(folder, st)


def _folders_between(path: Path, within: Path) -> Iterator[Path]:
    yield within
    current = within
    for part in parts_below(path, within)[:-1]:
        current = current / part
        yield current


def parts_below(path: Path, within: Path) -> tuple[str, ...]:
    """Names leading from ``within`` to ``path``, which must lie below it as written."""
    try:
        parts = path.relative_to(within).parts
    except ValueError:
        parts = ()
    if not parts or ".." in parts:
        raise SafetyError(
            f"{display_path(path)} is not inside {display_path(within)}", code="outside_root"
        )
    return parts


def _expect_folder(folder: Path, st: os.stat_result) -> None:
    if paths.is_link_like(st):
        raise SafetyError(f"{display_path(folder)} is a link", code="is_link")
    if not stat_module.S_ISDIR(st.st_mode):
        raise SafetyError(f"{display_path(folder)} is not a folder", code="not_regular")


# --- hashing ------------------------------------------------------------------------


def hash_file(path: Path, cancel: Callable[[], bool] | None = None) -> bytes:
    """BLAKE2b-256 of the content, read from disk now.

    Deliberately separate from the cached hashes of the index: this is the last look before a
    file is moved, and a cached value would only repeat what was true at the last scan.
    Raises ``FileChanged`` when the file changes meanwhile and ``JobCancelled`` on cancel.
    """
    return hash_with_signature(path, cancel)[0]


def hash_with_signature(
    path: Path, cancel: Callable[[], bool] | None = None
) -> tuple[bytes, FileSignature]:
    """The hash together with the signature the file had throughout."""
    before = regular_signature(path)
    digest = hashlib.blake2b(digest_size=DIGEST_SIZE)
    try:
        with os.fdopen(open_for_reading(path), "rb", buffering=0) as file:
            if not unchanged(paths.signature(os.fstat(file.fileno())), before):
                raise FileChanged(f"{display_path(path)} was replaced while it was being opened")
            while chunk := file.read(_CHUNK):
                if cancel is not None and cancel():
                    raise JobCancelled("cancelled")
                digest.update(chunk)
    except OSError as exc:
        raise ActionError(f"cannot read {display_path(path)}: {exc.strerror or exc}") from exc
    if not unchanged(regular_signature(path), before):
        raise FileChanged(f"{display_path(path)} changed while it was being read")
    return digest.digest(), before


def open_for_reading(path: Path) -> int:
    """Open a file without following a link in its place. Returns a descriptor."""
    if sys.platform == "win32":
        return os.open(fs_path(path), os.O_RDONLY | os.O_BINARY)
    # A link or a pipe that replaced the file a moment ago must not be followed or waited for.
    return os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)


# --- folders ------------------------------------------------------------------------


def make_dirs_private(folder: Path, within: Path, *, mode: int = 0o700) -> None:
    """Create ``folder`` below the existing folder ``within``.

    Refuses to pass through anything but real folders of the same volume. New folders get
    ``mode``, limited by the umask.
    """
    top = stat_or_none(within)
    if top is None:
        raise SafetyError(f"{display_path(within)} does not exist", code="missing")
    _expect_folder(within, top)
    current = within
    for part in parts_below(folder, within):
        current = current / part
        st = stat_or_none(current) or _make_dir(current, mode)
        _expect_folder(current, st)
        if st.st_dev != top.st_dev:
            raise SafetyError(
                f"{display_path(current)} is on another volume than {display_path(within)}",
                code="cross_volume",
            )


def _make_dir(folder: Path, mode: int) -> os.stat_result:
    try:
        # Someone else creating it at the same time is fine; what is there gets checked.
        with suppress(FileExistsError):
            os.mkdir(fs_path(folder), mode)
        return os.lstat(fs_path(folder))
    except OSError as exc:
        raise ActionError(f"cannot create {display_path(folder)}: {exc.strerror or exc}") from exc


def sync_folders(*folders: Path) -> None:
    """Make renames durable on Linux. Best effort: not every file system supports it."""
    if sys.platform == "win32":
        return
    for folder in set(folders):
        with suppress(OSError):
            handle = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(handle)
            finally:
                os.close(handle)


# --- names --------------------------------------------------------------------------


def numbered_name(name: str, number: int, label: str = "", *, limit: int = _NAME_LIMIT) -> str:
    """``photo.jpg`` becomes ``photo (restored 2).jpg``, or ``photo (2).jpg`` without a label.

    The stem is shortened when the result would be longer than a name may be.
    """
    stem, extension = os.path.splitext(name)
    if _name_length(extension) > _LONGEST_KEPT_EXTENSION:
        stem, extension = name, ""
    ending = f" ({label} {number}){extension}" if label else f" ({number}){extension}"
    while stem and _name_length(stem + ending) > limit:
        stem = stem[:-1]
    return stem + ending


def _name_length(name: str) -> int:
    """Length in the unit the file system limits: UTF-16 units on Windows, bytes elsewhere."""
    if os.name == "nt":
        return len(name.encode("utf-16-le", "surrogatepass")) // 2
    return len(os.fsencode(name))


def numbered(name: str, label: str = "", *, limit: int = _NAME_LIMIT) -> Iterator[str]:
    """The name with numbers, for as long as anyone could need."""
    for number in range(1 if label else 2, _MAX_NUMBER):
        yield numbered_name(name, number, label, limit=limit)


# --- moving -------------------------------------------------------------------------


def move_no_replace(src: Path, dst: Path, expected: FileSignature) -> FileSignature:
    """Rename ``src`` to ``dst`` on the same volume, never overwriting.

    ``expected`` is the file the caller checked. It is compared before the move and again
    after it, when nothing can slip in between any more: a different file that took its place
    is moved back. Raises ``SafetyError`` when nothing has moved (codes ``missing``,
    ``is_link``, ``not_regular``, ``is_placeholder``, ``changed``, ``target_exists``,
    ``cross_volume``, ``unsupported``), ``MisplacedFile`` when a file could not be moved back,
    and ``ActionError`` when the system refused.
    """
    _expect_absolute(src)
    _expect_absolute(dst)
    expect_unchanged(src, expected)
    folder = _expect_free_target(src, dst)
    try:
        _rename_no_replace(src, dst)
    except OSError as exc:
        raise _move_error(exc, src, dst) from exc
    moved = _verify_moved(src, dst, expected, folder)
    sync_folders(src.parent, dst.parent)
    return moved


def _expect_absolute(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise SafetyError(f"{display_path(path)} is not a full path", code="unsupported")


def _expect_free_target(src: Path, dst: Path) -> os.stat_result:
    """Check the target; return the stat of its folder."""
    folder = stat_or_none(dst.parent)
    if folder is None:
        raise SafetyError(f"{display_path(dst.parent)} does not exist", code="missing")
    _expect_folder(dst.parent, folder)
    if stat_or_none(dst) is not None:
        raise SafetyError(f"{display_path(dst)} already exists", code="target_exists")
    source = stat_or_none(src)
    if source is None or source.st_dev != folder.st_dev:
        raise SafetyError(
            f"{display_path(src)} and {display_path(dst.parent)} are on different volumes",
            code="cross_volume",
        )
    return folder


def _move_error(exc: OSError, src: Path, dst: Path) -> MediaSiftError:
    what = f"cannot move {display_path(src)} to {display_path(dst)}: {exc.strerror or exc}"
    codes = {
        errno.EEXIST: "target_exists",
        errno.EXDEV: "cross_volume",
        errno.ENOENT: "missing",
        errno.ENOSYS: "unsupported",
    }
    code = codes.get(exc.errno or 0)
    return SafetyError(what, code=code) if code else ActionError(what)


def _verify_moved(
    src: Path, dst: Path, expected: FileSignature, folder: os.stat_result
) -> FileSignature:
    try:
        st = os.lstat(fs_path(dst))
        folder_now = os.lstat(fs_path(dst.parent))
    except OSError as exc:
        raise MisplacedFile(
            f"{display_path(src)} was moved to {display_path(dst)}, which something else has "
            f"changed since: {exc.strerror or exc}",
            dst,
        ) from exc
    same_folder = (folder.st_dev, folder.st_ino) == (folder_now.st_dev, folder_now.st_ino)
    regular = stat_module.S_ISREG(st.st_mode) and not paths.is_link_like(st)
    moved = paths.signature(st)
    if same_folder and regular and unchanged(moved, expected):
        return moved
    put_back(dst, src)
    raise SafetyError(
        f"{display_path(src)} changed while it was being moved; it was moved back",
        code="changed",
    )


def put_back(current: Path, original: Path) -> None:
    """Undo a rename that just happened, never overwriting."""
    try:
        _rename_no_replace(current, original)
    except OSError as exc:
        raise MisplacedFile(
            f"{display_path(original)} could not be moved back ({exc.strerror or exc}); "
            f"the file is at {display_path(current)}",
            current,
        ) from exc
    sync_folders(current.parent, original.parent)


def _rename_no_replace(src: Path, dst: Path) -> None:
    if sys.platform == "win32":
        # MoveFileExW without flags: fails on an existing target and across volumes.
        os.rename(fs_path(src), fs_path(dst))
    elif sys.platform == "linux":
        _rename_linux(src, dst)
    else:
        raise OSError(errno.ENOSYS, "this system has no supported way to move safely")


def _rename_linux(src: Path, dst: Path) -> None:
    code = _renameat2(os.fsencode(src), os.fsencode(dst))
    if code in _NO_SUCH_RENAME:
        _move_by_link(src, dst)
    elif code:
        raise OSError(code, os.strerror(code), os.fspath(src), None, os.fspath(dst))


def _renameat2(src: bytes, dst: bytes) -> int:
    """Return 0 or the error number."""
    function = _libc_renameat2()
    if function is None:
        return errno.ENOSYS
    if function(_AT_FDCWD, src, _AT_FDCWD, dst, _RENAME_NOREPLACE) == 0:
        return 0
    return ctypes.get_errno() or errno.EIO


@functools.cache
def _libc_renameat2() -> Callable[[int, bytes, int, bytes, int], int] | None:
    function = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if function is None:
        return None
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    typed: Callable[[int, bytes, int, bytes, int], int] = function
    return typed


def _move_by_link(src: Path, dst: Path) -> None:
    """Move on a Linux file system without ``RENAME_NOREPLACE`` (NFS, some FUSE ones).

    A hard link creates the target and can never replace anything. Removing the source name
    directly would leave a moment in which another file saved under it is removed instead. So
    the name is first moved into a new private folder, where nothing can exist or change, and
    dropped only when it turns out to be the file that was linked. At every moment the file
    has a name at ``src`` or ``dst``.
    """
    try:
        os.link(src, dst, follow_symlinks=False)
    except OSError as exc:
        if exc.errno in (errno.EEXIST, errno.ENOENT, errno.EXDEV):
            raise
        raise OSError(errno.ENOSYS, "this file system has no safe way to move") from exc
    private = dst.parent / f".mediasift-move-{uuid.uuid4().hex}"
    aside = private / "file"
    try:
        os.mkdir(private, 0o700)
        os.rename(src, aside)
    except OSError:
        with suppress(OSError):
            os.rmdir(private)
        release_extra_name(dst, src)
        raise
    if not paths.signature(aside).same_file(paths.signature(dst)):
        _return_stranger(aside, src, dst)
    release_extra_name(aside, dst)
    with suppress(OSError):
        os.rmdir(private)


def _return_stranger(aside: Path, src: Path, dst: Path) -> None:
    """Another file took the name ``src`` after the link was made. Give it its name back."""
    try:
        os.link(aside, src, follow_symlinks=False)
    except OSError as exc:
        raise MisplacedFile(
            f"another file appeared at {display_path(src)}; it is now at {display_path(aside)}, "
            f"and the file that was being moved is at {display_path(dst)}",
            aside,
        ) from exc
    release_extra_name(aside, src)
    with suppress(OSError):
        os.rmdir(aside.parent)
    raise MisplacedFile(
        f"{display_path(src)} was replaced by another file while it was being moved; "
        f"the earlier one is at {display_path(dst)}",
        dst,
    )


def release_extra_name(extra: Path, keep: Path, expected: FileSignature | None = None) -> None:
    """Remove one of two names of the same file. The only unlink of a file that may be a user's.

    Refuses unless ``keep`` is, right now, a different name of the very same regular file. The
    caller makes sure that nothing else works on ``extra``.
    """
    first, second = stat_or_none(extra), stat_or_none(keep)
    if first is None or second is None:
        raise SafetyError(
            f"{display_path(extra)} and {display_path(keep)} are not both there", code="missing"
        )
    current, kept = paths.signature(first), paths.signature(second)
    same = current.same_file(kept) and (expected is None or current.same_file(expected))
    if (
        not same
        or not stat_module.S_ISREG(first.st_mode)
        or paths.is_link_like(first)
        or current.nlink < 2
        or _same_entry(extra, keep)
    ):
        raise SafetyError(
            f"{display_path(extra)} is not just another name of {display_path(keep)}",
            code="changed",
        )
    try:
        os.unlink(fs_path(extra))
    except OSError as exc:
        raise ActionError(f"cannot release {display_path(extra)}: {exc.strerror or exc}") from exc


def _same_entry(first: Path, second: Path) -> bool:
    """True when two paths may be one directory entry rather than two hard links."""
    names = {paths.path_key(path.name, case_sensitive=False) for path in (first, second)}
    return len(names) == 1 and paths.same_location(first.parent, second.parent)
