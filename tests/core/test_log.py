from __future__ import annotations

import logging
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from mediasift.core import log

LOGGER = logging.getLogger("mediasift.test")


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    yield
    log.close_logging()
    logging.getLogger("mediasift").setLevel(logging.NOTSET)


def handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger("mediasift").handlers if isinstance(h, log._LogHandler)]


def test_set_up_twice_logs_once(tmp_path: Path) -> None:
    path = log.set_up_logging(tmp_path / "lib")
    assert log.set_up_logging(tmp_path / "lib") == path
    assert len(handlers()) == 1
    LOGGER.info("ملف %s", "صورة.jpg")
    LOGGER.debug("not wanted")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0].endswith("mediasift.test: ملف صورة.jpg")


def test_another_library_replaces_the_file(tmp_path: Path) -> None:
    first = log.set_up_logging(tmp_path / "one")
    LOGGER.info("to one")
    second = log.set_up_logging(tmp_path / "two")
    LOGGER.info("to two")
    assert "to two" not in first.read_text(encoding="utf-8")
    assert "to two" in second.read_text(encoding="utf-8")
    assert len(handlers()) == 1


def test_lone_surrogates_do_not_break_logging(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = log.set_up_logging(tmp_path)
    LOGGER.info("bad \udcff name")
    assert "\\udcff name" in path.read_text(encoding="utf-8")
    assert capsys.readouterr().err == ""


def test_rotates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(log, "MAX_BYTES", 2000)
    log.set_up_logging(tmp_path)
    for number in range(200):
        LOGGER.info("line %d", number)
    names = sorted(item.name for item in tmp_path.iterdir())
    assert names == [log.LOG_FILE] + [f"{log.LOG_FILE}.{n}" for n in range(1, log.BACKUPS + 1)]
    assert "line 199" in (tmp_path / log.LOG_FILE).read_text(encoding="utf-8")


def test_goes_on_when_rotation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(self: RotatingFileHandler, source: str, dest: str) -> None:
        raise PermissionError(13, "in use")

    monkeypatch.setattr(log, "MAX_BYTES", 500)
    monkeypatch.setattr(RotatingFileHandler, "rotate", refuse)
    path = log.set_up_logging(tmp_path)
    for number in range(50):
        LOGGER.info("line %d", number)
    text = path.read_text(encoding="utf-8")
    assert all(f"line {number}\n" in text for number in range(50))
    assert "Logging error" not in capsys.readouterr().err
