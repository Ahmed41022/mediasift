"""Plugin interfaces and the registry that holds them."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable

from mediasift.core.errors import ConfigError
from mediasift.core.types import FileRecord, MediaMetadata, MediaType, RuleScore

if TYPE_CHECKING:
    from PIL.Image import Image


@runtime_checkable
class Named(Protocol):
    """Anything with a stable name."""

    @property
    def name(self) -> str: ...


class Reader(Named, Protocol):
    """Reads metadata and, for images, pixels."""

    @property
    def version(self) -> int: ...

    @property
    def extensions(self) -> frozenset[str]: ...

    def available(self) -> bool:
        """False when an optional library is missing. Must not raise."""
        ...

    def read_metadata(self, path: Path) -> MediaMetadata:
        """Raise ReaderError when the file cannot be read."""
        ...

    def load_image(self, path: Path, max_side: int) -> Image:
        """Return an upright RGB image, reduced to about ``max_side``. Raise ReaderError."""
        ...


class Hasher(Named, Protocol):
    """Computes named features of a decoded image."""

    @property
    def version(self) -> int: ...

    @property
    def media_types(self) -> frozenset[MediaType]: ...

    def compute(self, image: Image) -> Mapping[str, bytes]:
        """Return feature name to value."""
        ...


class KeeperRule(Named, Protocol):
    """Scores a candidate against the other members of its group."""

    def score(
        self,
        candidate: FileRecord,
        metadata: MediaMetadata | None,
        group: tuple[FileRecord, ...],
    ) -> RuleScore:
        """Higher is better; only the order within a group matters."""
        ...


T = TypeVar("T", bound=Named)


class Registry(Generic[T]):
    """Plugins by name, in registration order."""

    def __init__(self, what: str) -> None:
        self._what = what
        self._items: dict[str, T] = {}

    def register(self, item: T) -> T:
        """Add a plugin. Two plugins may not share a name."""
        if item.name in self._items:
            raise ConfigError(f"{self._what} {item.name!r} is registered twice")
        self._items[item.name] = item
        return item

    def get(self, name: str) -> T:
        """Return a plugin by name."""
        try:
            return self._items[name]
        except KeyError:
            raise ConfigError(f"unknown {self._what}: {name!r}") from None

    def __iter__(self) -> Iterator[T]:
        return iter(tuple(self._items.values()))

    def __len__(self) -> int:
        return len(self._items)
