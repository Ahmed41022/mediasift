"""Paths: exact bytes, comparison keys, containment, identity, long paths.

Everything here reads file-system metadata at most. Nothing is opened for reading and
nothing is ever written, so it is safe to use on the user's folders.
"""

from __future__ import annotations

import hashlib
import itertools
import ntpath
import os
import stat as stat_module
import sys
import unicodedata
from pathlib import Path

from mediasift.core.types import FileSignature

StrPath = str | os.PathLike[str]

_INT64_MAX = 2**63 - 1
_UINT64 = 2**64

# Windows file attributes of files whose content is not on this disk. Opening one may
# start a download. FILE_ATTRIBUTE_OFFLINE | RECALL_ON_OPEN | RECALL_ON_DATA_ACCESS.
_PLACEHOLDER_ATTRIBUTES = 0x1000 | 0x40000 | 0x400000

# Set in reparse tags that stand for another path: symlinks, junctions, mounted folders,
# WSL links. Other tags (OneDrive, deduplication, compression) are ordinary files.
_NAME_SURROGATE = 0x20000000

# Embeddings, overrides and isolates can make a name read as a different one.
_BIDI_CONTROLS = frozenset(map(chr, (*range(0x202A, 0x202F), *range(0x2066, 0x206A))))

_PROBE_LIMIT = 1000


# --- bytes, display, keys -----------------------------------------------------------


def path_to_bytes(path: StrPath) -> bytes:
    """Exact bytes of a path. Undecodable Linux names and lone surrogates survive."""
    return os.fsencode(path)


def bytes_to_path(raw: bytes) -> Path:
    """Inverse of ``path_to_bytes``."""
    return Path(os.fsdecode(raw))


def display_path(path: StrPath) -> str:
    """Text for people: always encodable, one line, no control or bidi-override characters."""
    text = os.fspath(path)
    if text.isprintable():
        return text
    return "".join(_display_char(char) for char in text)


def _display_char(char: str) -> str:
    code = ord(char)
    category = unicodedata.category(char)
    if category == "Cs":
        undecodable_byte = os.name != "nt" and 0xDC80 <= code <= 0xDCFF
        return f"\\x{code - 0xDC00:02x}" if undecodable_byte else f"\\u{code:04x}"
    if category in ("Cc", "Zl", "Zp") or char in _BIDI_CONTROLS:
        return f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}"
    return char


def bidi_isolate(text: str) -> str:
    """Wrap text so that Arabic names do not reorder the line they are printed in."""
    return f"⁨{text}⁩"


def path_key(path: StrPath, *, case_sensitive: bool) -> bytes:
    """Key for comparing paths: NFC, and casefolded on case-insensitive roots.

    Bytes rather than text because lone surrogates cannot be stored as text. Two different
    files may share a key (NFC and NFD twins on Linux): a key finds candidates, it never
    proves that two paths are the same file.
    """
    text = unicodedata.normalize("NFC", os.fspath(path))
    if os.sep == "\\":
        text = text.replace("\\", "/")
    if not case_sensitive:
        text = unicodedata.normalize("NFC", text.casefold())
    return text.encode("utf-8", "surrogatepass")


# --- long paths ---------------------------------------------------------------------


def fs_path(path: StrPath) -> str:
    r"""Form to hand to the operating system.

    On Windows absolute paths get the ``\\?\`` prefix: no length limit, and names such as
    ``photo.`` or ``nul.jpg`` mean exactly themselves. Elsewhere, and for relative paths,
    the path is returned unchanged.
    """
    text = os.fspath(path)
    if "\0" in text:
        raise ValueError("path contains a NUL character")
    return _windows_fs_path(text) if os.name == "nt" else text


def _windows_fs_path(text: str) -> str:
    if text.startswith(("\\\\?\\", "\\\\.\\")):
        return text
    text = text.replace("/", "\\")
    drive, rest = ntpath.splitdrive(text)
    if not drive or (drive.endswith(":") and not rest.startswith("\\")):
        return text  # relative, or relative to a drive's current folder
    # normpath, not abspath: abspath drops trailing dots and spaces, which names another file.
    text = ntpath.normpath(text)
    if text.startswith("\\\\"):
        return "\\\\?\\UNC" + text[1:]
    return "\\\\?\\" + text


def plain_path(path: StrPath) -> str:
    """Remove the prefix that ``fs_path`` adds."""
    text = os.fspath(path)
    return _windows_plain_path(text) if os.name == "nt" else text


def _windows_plain_path(text: str) -> str:
    if not text.startswith("\\\\?\\"):
        return text
    if text[:8].upper() == "\\\\?\\UNC\\":
        return "\\\\" + text[8:]
    return text[4:] if text[5:6] == ":" else text


# --- identity -----------------------------------------------------------------------


def signature(source: StrPath | os.stat_result) -> FileSignature:
    """Signature of a file, not following links.

    Pass a path or the result of ``os.lstat``. On Windows do not pass
    ``DirEntry.stat()``: it carries no inode, device or link count, so the identity would
    come out as unknown.
    """
    st = source if isinstance(source, os.stat_result) else os.lstat(fs_path(source))
    return FileSignature(
        size=st.st_size,
        mtime_ns=max(-_INT64_MAX, min(_INT64_MAX, st.st_mtime_ns)),
        dev=_device(st) if st.st_ino else 0,
        inode=_fold_inode(st.st_ino),
        nlink=max(1, st.st_nlink),
    )


def _device(st: os.stat_result) -> int:
    # Python 3.12 widened st_dev on Windows to 64 bits. The low half is the serial that
    # older versions report, so an index stays valid across Python upgrades.
    return st.st_dev & 0xFFFFFFFF if os.name == "nt" else st.st_dev % _UINT64


def _fold_inode(inode: int) -> int:
    """ReFS identifiers have 128 bits; hash them down to what the index can store."""
    if inode < _UINT64:
        return inode
    digest = hashlib.blake2b(inode.to_bytes(16, "little"), digest_size=8).digest()
    return int.from_bytes(digest, "little") or 1


def is_link_like(source: StrPath | os.stat_result) -> bool:
    """True for symlinks, junctions and mounted folders, which are never followed.

    A stat result must come from ``lstat``. A missing path is not a link.
    """
    if isinstance(source, os.stat_result):
        st = source
    else:
        try:
            st = os.lstat(fs_path(source))
        except FileNotFoundError:
            return False
    if stat_module.S_ISLNK(st.st_mode):
        return True
    return bool(getattr(st, "st_reparse_tag", 0) & _NAME_SURROGATE)


def is_placeholder(st: os.stat_result) -> bool:
    """True for cloud files whose content is not on this disk. Never open one."""
    return bool(getattr(st, "st_file_attributes", 0) & _PLACEHOLDER_ATTRIBUTES)


def volume_id(path: StrPath) -> str:
    """Identity of the volume holding ``path``; empty when the path cannot be reached.

    A different value for a root means another volume is mounted there, for example the
    empty mount point of an unplugged drive. Best effort: some Linux file systems derive
    it from the device number, which may change between boots.
    """
    try:
        if sys.platform != "win32":
            fsid = os.statvfs(path).f_fsid
            if fsid:
                return f"fs:{fsid % _UINT64:x}"
        return f"dev:{_device(os.stat(fs_path(path))):x}"
    except (OSError, ValueError):
        return ""


# --- containment --------------------------------------------------------------------


def resolve(path: StrPath) -> Path:
    """Absolute path with links resolved. Missing parts are kept as they are."""
    return Path(plain_path(_resolve(path)))


def _resolve(path: StrPath) -> str:
    text = os.fspath(path)
    if os.name == "nt":
        text = plain_path(text)
        if not os.path.isabs(text) or not ntpath.splitdrive(text)[0]:
            text = os.path.abspath(text)
    return os.path.realpath(fs_path(text))


def same_location(first: StrPath, second: StrPath) -> bool:
    """True when both paths lead to the same file or folder."""
    try:
        a, b = _resolve(first), _resolve(second)
        if a == b:
            return True
        identity, other = _identity(a), _identity(b)
    except (OSError, ValueError):
        return False
    if identity is None and other is None:
        return os.path.normcase(a) == os.path.normcase(b)
    return identity == other


def is_within(child: StrPath, root: StrPath) -> bool:
    """True when ``child`` really lies inside ``root`` (the root itself does not count).

    Both are resolved first, so ``..``, symlinks and junctions cannot lead outside, and
    folders are compared as whole names or by identity, so ``/a/bc`` is not inside ``/a/b``.
    Anything that cannot be verified counts as outside.
    """
    try:
        current, top = _resolve(child), _resolve(root)
        if current == top:
            return False
        top_identity = _identity(top)
        while True:
            parent = os.path.dirname(current)
            # A link left after resolving is a loop or unreadable: do not trust it.
            if parent == current or is_link_like(current):
                return False
            if parent == top or (top_identity is not None and _identity(parent) == top_identity):
                return True
            if top_identity is None and os.path.normcase(parent) == os.path.normcase(top):
                return True
            current = parent
    except (OSError, ValueError):
        return False


def _identity(path: str) -> tuple[int, int] | None:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    return (st.st_dev, st.st_ino) if st.st_ino else None


# --- case ---------------------------------------------------------------------------


def is_case_sensitive(folder: StrPath) -> bool:
    """Find out whether names in ``folder`` differ by case, without writing to it.

    Looks an existing entry up under its name with the case swapped. Falls back to the
    platform's usual behaviour when no entry can tell.
    """
    try:
        with os.scandir(fs_path(folder)) as entries:
            for entry in itertools.islice(entries, _PROBE_LIMIT):
                answer = _probe_case(entry.path)
                if answer is not None:
                    return answer
    except OSError:
        pass
    return sys.platform not in ("win32", "darwin")


def _probe_case(path: str) -> bool | None:
    head, name = os.path.split(path)
    # Only ASCII: file systems disagree about the case of other letters.
    swapped = "".join(c.swapcase() if c.isascii() else c for c in name)
    if swapped == name:
        return None
    try:
        original = os.lstat(path)
    except OSError:
        return None
    try:
        other = os.lstat(os.path.join(head, swapped))
    except FileNotFoundError:
        return True
    except OSError:
        return None
    if not original.st_ino:
        return None
    if (original.st_dev, original.st_ino) != (other.st_dev, other.st_ino):
        return True
    # Two hard links that differ only by case would look like one file.
    hard_linked = not stat_module.S_ISDIR(original.st_mode) and original.st_nlink > 1
    return None if hard_linked else False
