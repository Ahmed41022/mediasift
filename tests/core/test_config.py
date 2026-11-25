from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from mediasift.core.config import Settings, config_path, library_dir, load_settings
from mediasift.core.errors import ConfigError, ExitCode


def write(tmp_path: Path, text: str) -> Path:
    path = config_path(tmp_path)
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_without_a_file(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "none.toml")
    assert settings == Settings()
    assert settings.similar.threshold == 10
    assert settings.similar.burst_seconds == 5
    assert settings.jobs.workers == 0
    assert settings.actions.backend == "quarantine"
    with pytest.raises(FrozenInstanceError):
        settings.jobs.workers = 3  # type: ignore[misc]


def test_every_setting(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [scan]
        include = ["*.jpg"]
        exclude = ["صور/*", "*.tmp"]
        include_hidden = true
        min_size = 1024
        [jobs]
        workers = 4
        batch_size = 50
        [similar]
        threshold = 32
        burst_seconds = 0
        [keeper]
        prefer_folders = ["D:/Photos"]
        avoid_folders = ["D:/Downloads"]
        [actions]
        backend = "trash"
        """,
    )
    settings = load_settings(path)
    assert settings.scan.include == ("*.jpg",)
    assert settings.scan.exclude == ("صور/*", "*.tmp")
    assert settings.scan.include_hidden
    assert settings.scan.min_size == 1024
    assert (settings.jobs.workers, settings.jobs.batch_size) == (4, 50)
    assert (settings.similar.threshold, settings.similar.burst_seconds) == (32, 0)
    assert settings.keeper.prefer_folders == ("D:/Photos",)
    assert settings.keeper.avoid_folders == ("D:/Downloads",)
    assert settings.actions.backend == "trash"


def test_partial_file_keeps_other_defaults(tmp_path: Path) -> None:
    settings = load_settings(write(tmp_path, "[jobs]\nworkers = 2\n"))
    assert settings.jobs.workers == 2
    assert settings.jobs.batch_size == Settings().jobs.batch_size
    assert settings.scan == Settings().scan


def test_byte_order_mark(tmp_path: Path) -> None:
    path = tmp_path / "bom.toml"
    path.write_bytes(b"\xef\xbb\xbf[jobs]\r\nworkers = 2\r\n")
    assert load_settings(path).jobs.workers == 2


@pytest.mark.parametrize(
    ("text", "named"),
    [
        ("[scan]\nfollow_links = true", "scan.follow_links"),
        ("[speed]\nx = 1", "speed"),
        ("workers = 3", "workers"),
        ("scan = 3", "scan"),
        ("[jobs]\nworkers = -1", "jobs.workers"),
        ("[jobs]\nworkers = true", "jobs.workers"),
        ("[jobs]\nworkers = 2.0", "jobs.workers"),
        ("[jobs]\nbatch_size = 0", "jobs.batch_size"),
        ("[similar]\nthreshold = 33", "similar.threshold"),
        ("[similar]\nburst_seconds = '5'", "similar.burst_seconds"),
        ("[scan]\nmin_size = -5", "scan.min_size"),
        ("[scan]\ninclude = '*.jpg'", "scan.include"),
        ("[scan]\nexclude = ['ok', 3]", "scan.exclude"),
        ("[scan]\nexclude = ['']", "scan.exclude"),
        ("[scan]\ninclude_hidden = 1", "scan.include_hidden"),
        ("[keeper]\nprefer_folders = [[]]", "keeper.prefer_folders"),
        ("[actions]\nbackend = 'delete'", "actions.backend"),
        ("[actions]\nbackend = ['trash']", "actions.backend"),
    ],
)
def test_errors_name_the_key(tmp_path: Path, text: str, named: str) -> None:
    path = write(tmp_path, text)
    with pytest.raises(ConfigError) as caught:
        load_settings(path)
    assert named in str(caught.value)
    assert str(path) in str(caught.value)
    assert caught.value.exit_code is ExitCode.USAGE


@pytest.mark.parametrize("raw", [b"[scan", b"\xff\xfe\x00", b"[a]\n[a]\n"])
def test_unreadable_file(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "bad.toml"
    path.write_bytes(raw)
    with pytest.raises(ConfigError, match=r"bad\.toml"):
        load_settings(path)


def test_config_that_is_a_folder(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings(tmp_path)


def test_library_dir_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIASIFT_HOME", str(tmp_path / "env"))
    assert library_dir(tmp_path / "given") == tmp_path / "given"
    assert library_dir() == tmp_path / "env"
    for value in ("", None):
        if value is None:
            monkeypatch.delenv("MEDIASIFT_HOME")
        else:
            monkeypatch.setenv("MEDIASIFT_HOME", value)
        default = library_dir()
        assert default.is_absolute()
        assert default.name == "mediasift"


def test_library_dir_is_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert library_dir("here") == Path.cwd() / "here"
    assert library_dir("a/../here") == Path.cwd() / "here"
    assert library_dir("~/x") == Path.home() / "x"
    with pytest.raises(ConfigError):
        library_dir("")
