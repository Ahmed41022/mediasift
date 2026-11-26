"""A progress bar on stderr, and the job context that feeds it."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType

from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.progress import Progress as RichProgress

from mediasift.api import CancelToken, JobContext, Progress, cancel_on_sigint
from mediasift.cli.common import env_of


class ProgressBar:
    """One bar per stage, drawn on stderr. Does nothing when ``enabled`` is false."""

    def __init__(self, enabled: bool, console: Console | None = None) -> None:
        self._bar: RichProgress | None = None
        self._tasks: dict[str, TaskID] = {}
        if enabled:
            self._bar = RichProgress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console or Console(file=sys.stderr),
                transient=True,
            )

    def __enter__(self) -> ProgressBar:
        if self._bar is not None:
            self._bar.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._bar is not None:
            self._bar.stop()

    def update(self, progress: Progress) -> None:
        """Progress callback of a job."""
        if self._bar is None:
            return
        task = self._tasks.get(progress.stage)
        if task is None:
            task = self._tasks[progress.stage] = self._bar.add_task(
                escape(progress.stage), total=progress.total
            )
        self._bar.update(task, completed=progress.done, total=progress.total)


@contextmanager
def job_context(*, show_bar: bool | None = None) -> Iterator[JobContext]:
    """A job that Ctrl+C cancels and that reports to a bar.

    The bar shows only when stderr is a terminal and the output is not JSON.
    """
    if show_bar is None:
        show_bar = sys.stderr.isatty() and not env_of().json
    token = CancelToken()
    with ProgressBar(show_bar) as bar, cancel_on_sigint(token):
        yield JobContext(token, bar.update)
