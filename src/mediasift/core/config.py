"""Settings from a TOML file, and where the library folder is."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import platformdirs

from mediasift.core.errors import ConfigError

CONFIG_FILE = "config.toml"
HOME_VARIABLE = "MEDIASIFT_HOME"


@dataclass(frozen=True, slots=True)
class ScanSettings:
    """What a scan looks at. Links are never followed, so there is no setting for that.

    An empty ``include`` means everything.
    """

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    include_hidden: bool = False
    min_size: int = 0


@dataclass(frozen=True, slots=True)
class JobSettings:
    """How work is spread. ``workers`` 0 means automatic."""

    workers: int = 0
    batch_size: int = 200


@dataclass(frozen=True, slots=True)
class SimilarSettings:
    """Near-duplicate matching."""

    threshold: int = 10
    burst_seconds: int = 5


@dataclass(frozen=True, slots=True)
class KeeperSettings:
    """Folders whose files are kept, or given up, first."""

    prefer_folders: tuple[str, ...] = ()
    avoid_folders: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ActionSettings:
    """Where removed files go."""

    backend: str = "quarantine"


@dataclass(frozen=True, slots=True)
class Settings:
    """All settings."""

    scan: ScanSettings = field(default_factory=ScanSettings)
    jobs: JobSettings = field(default_factory=JobSettings)
    similar: SimilarSettings = field(default_factory=SimilarSettings)
    keeper: KeeperSettings = field(default_factory=KeeperSettings)
    actions: ActionSettings = field(default_factory=ActionSettings)


Check = Callable[[str, object], Any]


def _integer(low: int, high: int) -> Check:
    def check(key: str, value: object) -> int:
        # bool is an int in Python, but `workers = true` is a mistake.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{key}: expected a whole number")
        if not low <= value <= high:
            raise ConfigError(f"{key}: must be between {low} and {high}, got {value}")
        return value

    return check


def _boolean(key: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{key}: expected true or false")
    return value


def _strings(key: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{key}: expected a list of texts")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{key}: expected a list of non-empty texts")
    return tuple(value)


def _one_of(*allowed: str) -> Check:
    def check(key: str, value: object) -> str:
        if not isinstance(value, str) or value not in allowed:
            raise ConfigError(f"{key}: must be one of {', '.join(allowed)}")
        return value

    return check


_SECTIONS: dict[str, tuple[Callable[..., Any], dict[str, Check]]] = {
    "scan": (
        ScanSettings,
        {
            "include": _strings,
            "exclude": _strings,
            "include_hidden": _boolean,
            "min_size": _integer(0, 2**62),
        },
    ),
    "jobs": (
        JobSettings,
        {"workers": _integer(0, 256), "batch_size": _integer(1, 100_000)},
    ),
    "similar": (
        SimilarSettings,
        {"threshold": _integer(0, 32), "burst_seconds": _integer(0, 3600)},
    ),
    "keeper": (KeeperSettings, {"prefer_folders": _strings, "avoid_folders": _strings}),
    "actions": (ActionSettings, {"backend": _one_of("quarantine", "trash")}),
}


def parse_settings(data: Mapping[str, object]) -> Settings:
    """Build settings from parsed TOML. Unknown keys are errors, so typos do not go unseen."""
    sections: dict[str, Any] = {}
    for name, values in data.items():
        if name not in _SECTIONS:
            raise ConfigError(f"unknown setting: {name}")
        if not isinstance(values, dict):
            raise ConfigError(f"{name}: expected a section, [{name}]")
        build, checks = _SECTIONS[name]
        for key in values:
            if key not in checks:
                raise ConfigError(f"unknown setting: {name}.{key}")
        parsed = {key: checks[key](f"{name}.{key}", value) for key, value in values.items()}
        sections[name] = build(**parsed)
    return Settings(**sections)


def load_settings(path: Path) -> Settings:
    """Read settings from a TOML file. A missing file means the defaults."""
    try:
        # utf-8-sig: Windows editors like to add a byte order mark.
        text = path.read_bytes().decode("utf-8-sig")
    except FileNotFoundError:
        return Settings()
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    try:
        return parse_settings(tomllib.loads(text))
    except (tomllib.TOMLDecodeError, ConfigError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def library_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Folder of the library: argument, then MEDIASIFT_HOME, then the user's data folder."""
    if explicit is not None:
        chosen = os.fspath(explicit)
        if not chosen:
            raise ConfigError("the library folder must not be empty")
    else:
        chosen = os.environ.get(HOME_VARIABLE, "")
    if chosen:
        return Path(os.path.abspath(Path(chosen).expanduser()))  # noqa: PTH100
    return Path(platformdirs.user_data_path("mediasift", appauthor=False))


def config_path(library: Path) -> Path:
    """The settings file of a library."""
    return library / CONFIG_FILE
