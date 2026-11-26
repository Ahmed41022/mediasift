"""Which files and folders a scan looks at.

Patterns are globs on the path relative to the root, written with ``/``:

* ``*`` and ``?`` stay inside one name, ``**`` spans folders, ``[abc]`` and ``[!abc]`` work.
* A pattern without ``/`` matches a name at any depth; with one, or with a leading ``/``, it
  is anchored at the root. A trailing ``/`` matches folders only.
* A pattern that matches a folder covers everything below it.

Excluding wins over including. Files that the index knows and that the rules no longer cover
are treated like files that are gone: the scanner marks them missing, and they come back with
their id when the rule is lifted.
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from mediasift.core import paths
from mediasift.core.config import ScanSettings
from mediasift.core.errors import ConfigError
from mediasift.core.paths import StrPath
from mediasift.core.types import Root

# The folder the quarantine backend keeps inside a root.
QUARANTINE_FOLDER = ".mediasift-quarantine"

# Compared without case on every root: nothing in these is ever part of a library.
_NEVER = frozenset(
    name.casefold()
    for name in (
        QUARANTINE_FOLDER,
        "$RECYCLE.BIN",
        "System Volume Information",
        ".Trash",
        "@eaDir",
    )
)
_NEVER_PREFIX = ".trash-"
_HOME_TRASH = (".local", "share", "trash")

_HIDDEN_ATTRIBUTES = 0x2 | 0x4  # FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM


@dataclass(frozen=True, slots=True)
class Rules:
    """Compiled rules for one root. Paths are relative to it."""

    case_sensitive: bool
    include_hidden: bool = False
    min_size: int = 0
    include: re.Pattern[str] | None = None
    exclude: re.Pattern[str] | None = None
    library: bytes | None = None

    @classmethod
    def for_root(cls, settings: ScanSettings, root: Root, library: StrPath | None = None) -> Rules:
        """Rules of a root. ``library`` is the index folder, skipped when it lies inside."""
        sensitive = root.case_sensitive
        return cls(
            case_sensitive=sensitive,
            include_hidden=settings.include_hidden,
            min_size=settings.min_size,
            include=_compile("scan.include", settings.include, sensitive),
            exclude=_compile("scan.exclude", settings.exclude, sensitive),
            library=_library_key(library, root),
        )

    def skips_folder(self, rel: str, attributes: int = 0) -> bool:
        """True when nothing below this folder is to be looked at."""
        name = os.path.basename(rel)
        if self._hidden(name, attributes) or _is_system_folder(rel, name):
            return True
        if self.library is not None and self.library == self._key(rel):
            return True
        return self.exclude is not None and self._matches(self.exclude, rel + "/")

    def skips_file(self, rel: str, size: int | None = None, attributes: int = 0) -> bool:
        """True when a media file is to be ignored. Its folders have been checked before."""
        if self._hidden(os.path.basename(rel), attributes):
            return True
        if size is not None and size < self.min_size:
            return True
        if self.exclude is not None and self._matches(self.exclude, rel):
            return True
        return self.include is not None and not self._matches(self.include, rel)

    def _hidden(self, name: str, attributes: int) -> bool:
        return not self.include_hidden and (
            name.startswith(".") or bool(attributes & _HIDDEN_ATTRIBUTES)
        )

    def _matches(self, pattern: re.Pattern[str], rel: str) -> bool:
        return pattern.fullmatch(_normal(rel, self.case_sensitive)) is not None

    def _key(self, rel: str) -> bytes:
        return paths.path_key(rel, case_sensitive=self.case_sensitive)


def _is_system_folder(rel: str, name: str) -> bool:
    folded = name.casefold()
    if folded in _NEVER or folded.startswith(_NEVER_PREFIX):
        return True
    return folded == _HOME_TRASH[-1] and _tail(rel) == _HOME_TRASH


def _tail(rel: str) -> tuple[str, ...]:
    return tuple(part.casefold() for part in rel.split(os.sep)[-len(_HOME_TRASH) :])


def _library_key(library: StrPath | None, root: Root) -> bytes | None:
    if library is None or not paths.is_within(library, root.path):
        return None
    rel = os.path.relpath(paths.resolve(library), paths.resolve(root.path))
    return paths.path_key(rel, case_sensitive=root.case_sensitive)


# --- patterns -----------------------------------------------------------------------


def _normal(text: str, case_sensitive: bool) -> str:
    """The form patterns and paths are compared in; the same steps as ``path_key``."""
    if os.sep == "\\":
        text = text.replace("\\", "/")
    text = unicodedata.normalize("NFC", text)
    return text if case_sensitive else unicodedata.normalize("NFC", text.casefold())


def _compile(key: str, patterns: Iterable[str], case_sensitive: bool) -> re.Pattern[str] | None:
    parts = [_translate(key, _normal(pattern, case_sensitive)) for pattern in patterns]
    return re.compile("|".join(f"(?:{part})" for part in parts), re.DOTALL) if parts else None


def _translate(key: str, pattern: str) -> str:
    """Regular expression for one pattern. Folders are matched with a ``/`` at their end."""
    text = pattern
    while text.startswith("./"):
        text = text[2:]
    segments = [segment for segment in text.split("/") if segment]
    if not segments or ".." in segments or "." in segments:
        raise ConfigError(f"{key}: {pattern!r} is not a pattern inside a folder")
    anchored = "/" in text.rstrip("/")
    body = "" if anchored else "(?:[^/]+/)*"
    for segment in segments[:-1]:
        body += "(?:[^/]+/)*" if segment == "**" else _translate_name(segment) + "/"
    folders_only = text.endswith("/")
    if segments[-1] == "**":
        return body + (".+/.*" if folders_only else ".+")
    return body + _translate_name(segments[-1]) + ("/.*" if folders_only else "(?:/.*)?")


def _translate_name(segment: str) -> str:
    out, i = "", 0
    while i < len(segment):
        char = segment[i]
        i += 1
        if char == "*":
            out += "" if out.endswith("[^/]*") else "[^/]*"
        elif char == "?":
            out += "[^/]"
        elif char == "[" and (end := _class_end(segment, i)) > 0:
            out += _translate_class(segment[i:end])
            i = end + 1
        else:
            out += re.escape(char)
    return out


def _class_end(segment: str, start: int) -> int:
    """Index of the ``]`` that closes a class, or -1: an unclosed ``[`` is an ordinary one."""
    first = start + 1 if segment[start : start + 1] == "!" else start
    return segment.find("]", first + 1)


def _translate_class(body: str) -> str:
    negated = body.startswith("!")
    chars = body[1:] if negated else body
    ranges, i = "", 0
    while i < len(chars):
        low = chars[i]
        if chars[i + 1 : i + 2] == "-" and i + 2 < len(chars):
            high = chars[i + 2]
            # A reversed range matches nothing; as a regular expression it is an error.
            ranges += f"{re.escape(low)}-{re.escape(high)}" if low <= high else ""
            i += 3
        else:
            ranges += re.escape(low)
            i += 1
    if not ranges:
        return "[^/]" if negated else "(?!)"
    return f"(?!/)[{'^' if negated else ''}{ranges}]"
