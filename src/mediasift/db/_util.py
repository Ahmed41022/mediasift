"""Conversions shared by the repositories."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime
from itertools import islice
from typing import Any, TypeVar

T = TypeVar("T")

# Below SQLite's oldest default limit of 999 parameters per statement.
MAX_PARAMETERS = 500
PAGE_SIZE = 1000


def now() -> str:
    """Current time, UTC, as text that sorts."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def chunked(items: Iterable[T], size: int = MAX_PARAMETERS) -> Iterator[list[T]]:
    """Lists of at most ``size`` items, pulled lazily."""
    iterator = iter(items)
    while chunk := list(islice(iterator, size)):
        yield chunk


def marks(count: int) -> str:
    """Placeholders for an IN list."""
    return ",".join("?" * count)


def to_signed(value: int) -> int:
    """Store an unsigned 64-bit number in SQLite's signed one."""
    return value - 2**64 if value >= 2**63 else value


def to_unsigned(value: int) -> int:
    """Inverse of ``to_signed``."""
    return value + 2**64 if value < 0 else value


def clean_text(text: str) -> str:
    """Text SQLite can store: no lone surrogates, and no NULs (EXIF pads with them)."""
    return text.replace("\0", "").encode("utf-8", "backslashreplace").decode("utf-8")


def clean_optional(text: str | None) -> str | None:
    """``clean_text`` that lets None through."""
    return None if text is None else clean_text(text)


def dump_json(value: object) -> str:
    """JSON for a column. Values JSON does not know become text instead of failing a scan."""
    return json.dumps(_jsonable(value), allow_nan=False)


def load_json(text: str) -> dict[str, Any]:
    """A JSON object from a column."""
    value = json.loads(text)
    return value if isinstance(value, dict) else {}


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, bool | int | str):
        return str(value) if isinstance(value, str) else value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    return str(value)
