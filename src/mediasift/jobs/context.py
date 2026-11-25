"""Cancellation and progress, shared by every long-running operation."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Self

from mediasift.core.errors import JobCancelled

log = logging.getLogger(__name__)

DEFAULT_MAX_RATE = 10.0


class CancelToken:
    """A one-way switch that asks a job to stop. Any thread may flip it.

    It is a plain attribute rather than a ``threading.Event`` on purpose: ``cancel()`` is called
    from signal handlers, which run between two bytecodes of the main thread. If the main thread
    were inside an ``Event`` method at that moment, taking the event's lock again would deadlock.
    """

    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise JobCancelled("cancelled")


@dataclass(frozen=True, slots=True)
class Progress:
    """A snapshot of one stage. Totals are None while they are unknown."""

    stage: str
    done: int = 0
    total: int | None = None
    bytes_done: int = 0
    bytes_total: int | None = None
    current: str | None = None
    errors: int = 0


ProgressCallback = Callable[[Progress], None]


class JobContext:
    """What a running job needs from its surroundings: a way to be stopped and a way to report.

    ``JobContext()`` reports nowhere and is only cancelled by its own token, which makes it the
    no-op context for tests and scripts. It belongs to one job, so make a new one each time.

    The thread that opens the first stage owns the job. Progress callbacks run on that thread
    only, so an interface never has to guard against calls from worker threads. A callback that
    raises is logged and otherwise ignored; a callback stops the job through the token.
    """

    def __init__(
        self,
        cancel: CancelToken | None = None,
        on_progress: ProgressCallback | None = None,
        *,
        max_rate: float = DEFAULT_MAX_RATE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_rate <= 0:
            raise ValueError("max_rate must be positive")
        self.cancel = cancel if cancel is not None else CancelToken()
        self._on_progress = on_progress
        self._interval = 1.0 / max_rate
        self._clock = clock
        self._owner: int | None = None
        self._open: list[Stage] = []
        self._callback_failed = False

    @property
    def cancelled(self) -> bool:
        return self.cancel.cancelled

    def raise_if_cancelled(self) -> None:
        self.cancel.raise_if_cancelled()

    def stage(self, name: str, total: int | None = None, bytes_total: int | None = None) -> Stage:
        """Open a stage and report its start. Use it as a context manager, or call ``finish()``.

        Stages may overlap: a lazy pipeline walks and hashes at the same time. The rate limit
        applies to each stage on its own.
        """
        here = threading.get_ident()
        if self._owner is None:
            self._owner = here
        elif self._owner != here:
            raise RuntimeError("stages are opened on the thread that owns the job")
        stage = Stage(self, name, total, bytes_total)
        self._open.append(stage)
        stage._deliver(force=True)
        return stage

    def deliver_progress(self) -> None:
        """Report what worker threads have added since the last callback.

        Workers may advance a stage but never run callbacks, so whoever waits for them on the
        owning thread calls this. ``run_bounded`` does. Elsewhere it does nothing.
        """
        if self._open:
            for stage in tuple(self._open):
                stage._deliver()

    def _owned_here(self) -> bool:
        return self._owner == threading.get_ident()

    def _call(self, progress: Progress) -> None:
        if self._on_progress is None:
            return
        try:
            self._on_progress(progress)
        except Exception:
            # A broken display must not stop the job, nor fill the log ten times a second.
            level = logging.DEBUG if self._callback_failed else logging.ERROR
            log.log(level, "the progress callback failed", exc_info=True)
            self._callback_failed = True


class Stage:
    """Counts the work of one stage. Any thread may advance it; only the owner reports."""

    def __init__(
        self, job: JobContext, name: str, total: int | None, bytes_total: int | None
    ) -> None:
        self._job = job
        self._name = name
        self._total = total
        self._bytes_total = bytes_total
        self._done = 0
        self._bytes_done = 0
        self._current: str | None = None
        self._errors = 0
        self._lock = threading.Lock()
        # The fields below are only used on the owning thread.
        self._finished = False
        self._sent: Progress | None = None
        self._sent_at = 0.0

    def advance(
        self, n: int = 1, bytes: int = 0, current: str | None = None, errors: int = 0
    ) -> None:
        """Add to the counters. ``current=None`` keeps the current item."""
        with self._lock:
            self._done += n
            self._bytes_done += bytes
            self._errors += errors
            if current is not None:
                self._current = current
        self._deliver()

    def set_total(self, total: int | None, bytes_total: int | None = None) -> None:
        """Replace both totals, for work that is discovered while it runs."""
        with self._lock:
            self._total = total
            self._bytes_total = bytes_total
        self._deliver()

    def snapshot(self) -> Progress:
        with self._lock:
            return Progress(
                stage=self._name,
                done=self._done,
                total=self._total,
                bytes_done=self._bytes_done,
                bytes_total=self._bytes_total,
                current=self._current,
                errors=self._errors,
            )

    def finish(self) -> None:
        """Report the final state. Nothing is reported for this stage afterwards."""
        if not self._job._owned_here():
            raise RuntimeError("stages are finished on the thread that owns the job")
        if self._finished:
            return
        self._deliver(force=True)
        self._finished = True
        self._job._open.remove(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.finish()

    def _deliver(self, force: bool = False) -> None:
        job = self._job
        if job._on_progress is None or not job._owned_here() or self._finished:
            return
        if not force and job._clock() - self._sent_at < job._interval:
            return
        progress = self.snapshot()
        if progress == self._sent:
            return
        self._sent = progress
        # Set before the call because a callback may advance the stage, and after it so that a
        # slow callback cannot take up all of the job's time.
        self._sent_at = job._clock()
        job._call(progress)
        self._sent_at = job._clock()
