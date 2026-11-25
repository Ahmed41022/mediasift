"""The log file in the library folder. Logs never leave the computer."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "mediasift"
LOG_FILE = "mediasift.log"
MAX_BYTES = 2 * 1024 * 1024
BACKUPS = 3


class _LogHandler(RotatingFileHandler):
    """Keeps writing to the same file when it cannot be rotated.

    Windows refuses to rename a file another process has open, and a reader and a writer
    of one library may both be logging.
    """

    def doRollover(self) -> None:
        try:
            super().doRollover()
        except OSError:
            self.stream = self._open()


def set_up_logging(library: Path, level: int = logging.INFO) -> Path:
    """Log to a rotating file in the library folder. Safe to call again."""
    path = (library / LOG_FILE).absolute()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    handlers = [h for h in logger.handlers if isinstance(h, _LogHandler)]
    if any(Path(handler.baseFilename) == path for handler in handlers):
        return path
    close_logging()
    library.mkdir(parents=True, exist_ok=True)
    handler = _LogHandler(
        path,
        maxBytes=MAX_BYTES,
        backupCount=BACKUPS,
        encoding="utf-8",
        errors="backslashreplace",  # file names may hold lone surrogates
        delay=True,
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(process)d %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    return path


def close_logging() -> None:
    """Detach and close the log file."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if isinstance(handler, _LogHandler):
            logger.removeHandler(handler)
            handler.close()
