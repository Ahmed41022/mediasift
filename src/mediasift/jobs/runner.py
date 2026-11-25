"""Run work on a bounded pool of threads."""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections.abc import Callable, Generator, Iterable, Iterator
from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Generic, TypeVar, cast

from mediasift.core.errors import JobCancelled
from mediasift.jobs.context import CancelToken, JobContext

T = TypeVar("T")
R = TypeVar("R")
E = TypeVar("E", bound=Exception)

MAX_AUTO_WORKERS = 8
AUTO_WINDOW_PER_WORKER = 4
THREAD_PREFIX = "mediasift-worker"

# A wait on a lock cannot be interrupted by Ctrl+C on Windows, so the caller's thread never waits
# longer than this. It also sets how soon progress made inside one long item is reported.
_POLL_SECONDS = 0.1

# How long an interrupted run waits for workers before leaving them behind.
_GRACE_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class Outcome(Generic[T, R]):
    """One finished item: ``error`` is None when the work returned ``value``."""

    item: T
    value: R | None = None
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> R:
        """Return the value, or raise the error."""
        if self.error is not None:
            raise self.error
        return cast(R, self.value)


def run_bounded(
    items: Iterable[T],
    work: Callable[[T], R],
    *,
    job: JobContext,
    workers: int = 0,
    window: int = 0,
) -> Generator[Outcome[T, R], None, None]:
    """Run ``work`` on every item in worker threads and yield outcomes in completion order.

    Items are pulled lazily on the caller's thread. At most ``window`` items are pulled but not
    yet yielded, finished ones included, so a slow consumer slows the workers down instead of
    filling memory. ``workers=0`` and ``window=0`` pick sensible values.

    - An exception from ``work`` becomes an outcome with ``error`` set.
    - When the job is cancelled, or ``work`` or ``items`` raises ``JobCancelled``, nothing more is
      started. Outcomes that are complete, and those of items already running, are still yielded
      so their work is not lost; then ``JobCancelled`` is raised. This holds even if the cancel
      came too late to skip anything: a cancelled job never looks like a complete one. A consumer
      that wants those last outcomes must not check the token inside its own loop.
    - Another exception from ``items`` is raised after everything already pulled has been
      processed and yielded. It wins over a cancel that arrives meanwhile.
    - Closing the generator early skips queued items and waits for running ones. Without a
      cancel they run to their end, as nothing else can tell them to stop.
    - ``KeyboardInterrupt`` and ``SystemExit``, on either side, cancel the job's token so running
      work stops at its next check, and are raised without yielding anything more. A worker that
      is stuck, say on a dead network share, is left behind after a short wait.

    Use ``contextlib.closing`` when the generator is kept in a variable and the loop may be left
    early.
    """
    if workers < 0 or window < 0:
        raise ValueError("workers and window must not be negative")
    workers = workers or min(MAX_AUTO_WORKERS, os.cpu_count() or 1)
    window = window or workers * AUTO_WINDOW_PER_WORKER
    return _outcomes(_Pool(work, job.cancel, workers), iter(items), job, window)


def _outcomes(
    pool: _Pool[T, R], source: Iterator[T], job: JobContext, window: int
) -> Generator[Outcome[T, R], None, None]:
    failure: Exception | None = None
    grace: float | None = None
    pulling = True
    try:
        while True:
            job.deliver_progress()
            if job.cancelled:
                pulling = False
            # One pull and at most one yield per turn: workers stay fed while the consumer is
            # busy, and finished outcomes do not wait for a slow source to fill the window.
            if pulling and pool.in_flight < window:
                try:
                    pool.submit(next(source))
                except StopIteration:
                    pulling = False
                except Exception as error:  # noqa: BLE001 - raised below, after the drain
                    failure, pulling = error, False
                    if isinstance(error, JobCancelled):
                        pool.skip_queued()
            if not pool.in_flight:
                if pulling:
                    continue
                break
            has_room = pulling and pool.in_flight < window
            result = pool.take(0 if has_room else _POLL_SECONDS)
            if isinstance(result, Outcome):
                yield result
            elif isinstance(result, _Stopped):
                if isinstance(result.error, JobCancelled):
                    failure, pulling = failure or result.error, False
                elif result.error is not None:
                    raise result.error
        if failure is not None:
            raise failure
        job.raise_if_cancelled()
    except (KeyboardInterrupt, SystemExit):
        job.cancel.cancel()
        grace = _GRACE_SECONDS
        raise
    finally:
        failure = None  # its traceback points to this frame: do not leave a cycle behind
        pool.close(grace)


@dataclass(frozen=True, slots=True)
class _Stopped:
    """An item without an outcome: skipped (no error), cancelled, or ended by a BaseException."""

    error: BaseException | None = None


class _Pool(Generic[T, R]):
    """Threads fed by one queue. Every submitted item comes back through ``take`` exactly once.

    ``concurrent.futures`` is not used for two reasons: it costs more per item, and the
    interpreter joins its threads at exit, so one read stuck on a dead network share would block
    even a second Ctrl+C. Workers only read and compute, so daemon threads are safe here.
    """

    def __init__(self, work: Callable[[T], R], cancel: CancelToken, workers: int) -> None:
        self.in_flight = 0
        self._work = work
        self._cancel = cancel
        self._workers = workers
        self._skip = False
        self._threads: list[threading.Thread] = []
        # Items are boxed because None, which ends a worker, is also a valid item.
        self._tasks: SimpleQueue[tuple[T] | None] = SimpleQueue()
        self._results: SimpleQueue[Outcome[T, R] | _Stopped] = SimpleQueue()

    def submit(self, item: T) -> None:
        if len(self._threads) < self._workers:
            self._start_thread()
        self._tasks.put((item,))
        self.in_flight += 1

    def take(self, timeout: float) -> Outcome[T, R] | _Stopped | None:
        """Return a finished item, or None when there is none within ``timeout`` seconds."""
        try:
            if timeout > 0:
                result = self._results.get(timeout=timeout)
            else:
                result = self._results.get_nowait()
        except Empty:
            return None
        self.in_flight -= 1
        return result

    def skip_queued(self) -> None:
        self._skip = True

    def close(self, grace: float | None = None) -> None:
        """Skip what is queued, wait for what is running (up to ``grace``), end the threads."""
        self.skip_queued()
        for _ in self._threads:
            self._tasks.put(None)
        deadline = math.inf if grace is None else time.monotonic() + grace
        for thread in self._threads:
            # An abandoned generator can be closed by the garbage collector on one of its own
            # workers, or at interpreter exit when daemon threads no longer run.
            while (
                thread is not threading.current_thread()
                and not sys.is_finalizing()
                and thread.is_alive()
                and time.monotonic() < deadline
            ):
                thread.join(_POLL_SECONDS)

    def _start_thread(self) -> None:
        name = f"{THREAD_PREFIX}-{len(self._threads)}"
        thread = threading.Thread(target=self._loop, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _loop(self) -> None:
        # No item or result is kept in a local here: an idle worker would keep it alive.
        while self._step(self._tasks.get()):
            pass

    def _step(self, task: tuple[T] | None) -> bool:
        if task is None:
            return False
        if self._skip or self._cancel.cancelled:
            self._results.put(_Stopped())
        else:
            self._results.put(self._attempt(task[0]))
        return True

    def _attempt(self, item: T) -> Outcome[T, R] | _Stopped:
        try:
            return Outcome(item, value=self._work(item))
        except JobCancelled as error:
            self.skip_queued()  # at once: the caller may be busy for a while
            return _Stopped(_without_own_frame(error))
        except Exception as error:  # noqa: BLE001 - one bad item must not stop the job
            return Outcome(item, error=_without_own_frame(error))
        except BaseException as error:  # noqa: BLE001 - raised again on the caller's thread
            self.skip_queued()
            return _Stopped(error)


def _without_own_frame(error: E) -> E:
    """Drop this module's frame from the traceback; it only tells that the pool called work."""
    if error.__traceback__ is not None:
        error.__traceback__ = error.__traceback__.tb_next
    return error
