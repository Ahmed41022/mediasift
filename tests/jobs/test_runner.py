from __future__ import annotations

import gc
import itertools
import os
import subprocess
import sys
import textwrap
import threading
import traceback
import tracemalloc
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from mediasift.core.errors import JobCancelled
from mediasift.jobs import runner
from mediasift.jobs.context import CancelToken, JobContext, Progress
from mediasift.jobs.read import read_chunks
from mediasift.jobs.runner import Outcome, run_bounded

SRC = Path(__file__).resolve().parents[2] / "src"
TIMEOUT = 20.0


def wait(event: threading.Event) -> None:
    assert event.wait(TIMEOUT), "timed out"


class Pulls:
    """An endless source of 0, 1, 2, ... that says when the ``n``-th item has been pulled."""

    def __init__(self, n: int) -> None:
        self.count = 0
        self.reached = threading.Event()
        self._n = n

    def __iter__(self) -> Iterator[int]:
        return self

    def __next__(self) -> int:
        item = self.count
        self.count += 1
        if self.count == self._n:
            self.reached.set()
        return item


def collect_until(
    expected: type[BaseException], outcomes: Iterator[Outcome[int, int]]
) -> list[Outcome[int, int]]:
    got: list[Outcome[int, int]] = []
    with pytest.raises(expected):
        for outcome in outcomes:
            got.append(outcome)
    return got


# --- results -----------------------------------------------------------------------------------


@pytest.mark.parametrize("workers", [0, 1, 3])
@pytest.mark.parametrize("window", [0, 1, 7])
def test_every_item_is_worked_on_once(workers: int, window: int) -> None:
    outcomes = list(
        run_bounded(range(50), lambda x: x * x, job=JobContext(), workers=workers, window=window)
    )
    assert sorted((o.item, o.unwrap()) for o in outcomes) == [(x, x * x) for x in range(50)]


def test_no_items() -> None:
    assert list(run_bounded([], str, job=JobContext())) == []


def test_none_is_a_valid_item_and_value() -> None:
    outcomes = list(run_bounded([None, None], lambda x: x, job=JobContext(), workers=2))
    assert outcomes == [Outcome(None), Outcome(None)]
    assert all(o.ok and o.unwrap() is None for o in outcomes)


def test_completion_order() -> None:
    fast_seen = threading.Event()

    def work(name: str) -> str:
        if name == "slow":
            wait(fast_seen)
        return name

    order = []
    for outcome in run_bounded(["slow", "fast"], work, job=JobContext(), workers=2, window=2):
        order.append(outcome.unwrap())
        if outcome.item == "fast":
            fast_seen.set()
    assert order == ["fast", "slow"]


def test_pulls_on_the_calling_thread_and_works_on_others() -> None:
    me = threading.get_ident()
    pulled_on: set[int] = set()

    def source() -> Iterator[int]:
        for x in range(20):
            pulled_on.add(threading.get_ident())
            yield x

    outcomes = list(run_bounded(source(), lambda _: threading.get_ident(), job=JobContext()))
    assert pulled_on == {me}
    assert me not in {o.value for o in outcomes}


@pytest.mark.parametrize(("cpus", "expected"), [(3, 3), (64, 8), (None, 1)])
def test_auto_workers(cpus: int | None, expected: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    together = threading.Barrier(expected)

    def work(_: int) -> str:
        together.wait(TIMEOUT)
        return threading.current_thread().name

    outcomes = list(run_bounded(range(expected * 5), work, job=JobContext()))
    assert len({o.unwrap() for o in outcomes}) == expected


@pytest.mark.parametrize("kwargs", [{"workers": -1}, {"window": -1}])
def test_arguments_are_checked_at_the_call(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="negative"):
        run_bounded([1], str, job=JobContext(), **kwargs)


def test_one_bad_item_does_not_stop_the_job() -> None:
    def work(x: int) -> int:
        if x % 3 == 0:
            raise ValueError(f"bad {x}")
        return x

    outcomes = list(run_bounded(range(30), work, job=JobContext(), workers=4))
    assert sorted(o.item for o in outcomes) == list(range(30))
    for o in outcomes:
        assert o.ok == (o.item % 3 != 0)
        if not o.ok:
            assert isinstance(o.error, ValueError)
            assert str(o.error) == f"bad {o.item}"
            assert [f.name for f in traceback.extract_tb(o.error.__traceback__)] == ["work"]
            with pytest.raises(ValueError, match=f"bad {o.item}"):
                o.unwrap()


@pytest.mark.parametrize("error", [StopIteration, MemoryError, OSError, RuntimeError])
def test_any_exception_is_an_outcome(error: type[Exception]) -> None:
    def work(_: int) -> int:
        raise error

    (outcome,) = run_bounded([1], work, job=JobContext())
    assert isinstance(outcome.error, error)


# --- window, laziness, memory ------------------------------------------------------------------


@pytest.mark.parametrize(("workers", "window"), [(1, 1), (2, 5), (4, 3), (8, 32)])
def test_window_is_filled_and_never_exceeded(workers: int, window: int) -> None:
    items = Pulls(window)
    yielded = 0
    most = 0

    def source() -> Iterator[int]:
        nonlocal most
        for x in itertools.islice(items, 300):
            most = max(most, items.count - yielded)
            yield x

    def work(x: int) -> int:
        wait(items.reached)  # nothing completes before the window has been filled once
        return x

    for _ in run_bounded(source(), work, job=JobContext(), workers=workers, window=window):
        yielded += 1
        assert items.count - yielded <= window
    assert yielded == 300
    assert most == window


def test_waiting_for_slow_work_pulls_nothing_more() -> None:
    items = Pulls(3)
    waited = threading.Event()

    class Job(JobContext):
        turns = 0

        def deliver_progress(self) -> None:
            if items.reached.is_set():
                self.turns += 1
                if self.turns == 4:  # the caller has woken up more than once, with no result
                    waited.set()

    def work(x: int) -> int:
        wait(waited)
        return x

    outcomes = run_bounded(items, work, job=Job(), workers=1, window=3)
    next(outcomes)
    assert items.count == 3
    outcomes.close()


def test_a_slow_consumer_holds_the_workers_back() -> None:
    items = Pulls(4)
    done = threading.Semaphore(0)

    def work(x: int) -> int:
        wait(items.reached)
        done.release()
        return x

    outcomes = run_bounded(items, work, job=JobContext(), workers=2, window=4)
    next(outcomes)
    for _ in range(4):
        assert done.acquire(timeout=TIMEOUT)
    assert items.count == 4
    assert not done.acquire(blocking=False)
    outcomes.close()


def test_endless_source() -> None:
    items = Pulls(0)
    outcomes = run_bounded(items, lambda x: x, job=JobContext(), workers=2, window=8)
    assert items.count == 0
    assert len(list(itertools.islice(outcomes, 100))) == 100
    assert items.count <= 108
    outcomes.close()


def test_finished_work_does_not_wait_for_the_window_to_fill() -> None:
    first_is_done = threading.Event()
    pulled = 0

    class Pool(runner._Pool[int, int]):
        def _step(self, task: tuple[int] | None) -> bool:
            try:
                return super()._step(task)
            finally:
                first_is_done.set()

    def source() -> Iterator[int]:
        nonlocal pulled
        for pulled in range(1, 1000):
            if pulled == 2:
                wait(first_is_done)
            yield pulled

    pool = Pool(lambda x: x, CancelToken(), 1)
    outcomes = runner._outcomes(pool, source(), JobContext(), 1000)
    assert next(outcomes).item == 1
    assert pulled <= 2
    outcomes.close()


@pytest.mark.slow
def test_flat_memory_on_a_million_items() -> None:
    counts = []
    n = 0
    outcomes = run_bounded(range(1_000_000), lambda x: x + 1, job=JobContext(), workers=4)
    for _ in outcomes:
        n += 1
        if n % 100_000 == 0:
            counts.append(sys.getallocatedblocks())
    assert n == 1_000_000
    # Keeping even one object per item would add 900 000 blocks.
    assert max(counts) - min(counts) < 20_000


def test_flat_peak_memory() -> None:
    def run(count: int) -> None:
        for outcome in run_bounded(
            range(count), lambda x: bytes(100), job=JobContext(), workers=4, window=64
        ):
            assert outcome.ok

    run(1000)
    tracemalloc.start()
    try:
        run(100_000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 500_000  # the values alone add up to over 10 MB


def test_errors_are_freed_without_the_cycle_collector() -> None:
    def work(x: int) -> int:
        payload = bytes(10_000)
        raise ValueError(len(payload))

    gc.collect()
    gc.disable()
    tracemalloc.start()
    try:
        for outcome in run_bounded(range(2000), work, job=JobContext(), workers=2, window=16):
            assert not outcome.ok
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        gc.enable()
    assert peak < 2_000_000  # 20 MB if tracebacks were kept


# --- cancel ------------------------------------------------------------------------------------


def test_cancelled_before_the_start() -> None:
    job = JobContext()
    job.cancel.cancel()
    items = Pulls(0)
    assert collect_until(JobCancelled, run_bounded(items, lambda x: x, job=job)) == []
    assert items.count == 0


def test_cancel_skips_queued_items_and_keeps_running_ones() -> None:
    job = JobContext()
    items = Pulls(6)
    started = [threading.Event() for _ in range(6)]
    release = threading.Event()

    def work(x: int) -> int:
        started[x].set()
        if x == 0:
            wait(items.reached)
        else:
            wait(release)
        return x

    outcomes = run_bounded(items, work, job=job, workers=2, window=6)
    assert next(outcomes).item == 0
    wait(started[1])
    wait(started[2])
    job.cancel.cancel()
    release.set()
    rest = collect_until(JobCancelled, outcomes)
    assert sorted(o.unwrap() for o in rest) == [1, 2]
    assert [e.is_set() for e in started] == [True] * 3 + [False] * 3
    assert items.count == 6


def test_completed_outcomes_survive_a_cancel() -> None:
    job = JobContext()
    items = Pulls(8)
    done = threading.Semaphore(0)

    def work(x: int) -> int:
        wait(items.reached)
        done.release()
        if x == 3:
            raise ValueError("errors survive too")
        return x

    outcomes = run_bounded(items, work, job=job, workers=2, window=8)
    first = next(outcomes)
    for _ in range(8):
        assert done.acquire(timeout=TIMEOUT)
    job.cancel.cancel()
    rest = collect_until(JobCancelled, outcomes)
    assert sorted(o.item for o in [first, *rest]) == list(range(8))
    assert [o.item for o in [first, *rest] if not o.ok] == [3]


def test_cancel_after_the_last_item_still_raises() -> None:
    job = JobContext()
    got = []
    with pytest.raises(JobCancelled):
        for outcome in run_bounded([1], lambda x: x, job=job):
            got.append(outcome.item)
            job.cancel.cancel()
    assert got == [1]


class EndlessFile:
    """A file that never ends and counts the reads made after a cancel."""

    def __init__(self, cancel: CancelToken) -> None:
        self.reading = threading.Event()
        self.reads_after_cancel = 0
        self._cancel = cancel

    def read(self, size: int, /) -> bytes:
        if self._cancel.cancelled:
            self.reads_after_cancel += 1
        self.reading.set()
        return b"x"


def test_cancel_between_chunks_while_the_caller_is_waiting() -> None:
    job = JobContext()
    files = [EndlessFile(job.cancel) for _ in range(3)]

    def work(file: EndlessFile) -> int:
        return sum(len(chunk) for chunk in read_chunks(file, 1, job.cancel))

    def cancel_once_reading() -> None:
        for file in files[:2]:
            wait(file.reading)
        job.cancel.cancel()

    canceller = threading.Thread(target=cancel_once_reading)
    canceller.start()
    with pytest.raises(JobCancelled):
        for _ in run_bounded(files, work, job=job, workers=2, window=3):
            pytest.fail("an endless file cannot finish")
    canceller.join(TIMEOUT)
    # At most the read that was starting at that very moment.
    assert all(f.reads_after_cancel <= 1 for f in files[:2])
    assert not files[2].reading.is_set()


def test_job_cancelled_from_work_stops_the_run() -> None:
    job = JobContext()
    items = Pulls(4)
    ran = []
    cancelled = JobCancelled("from work")

    def work(x: int) -> int:
        wait(items.reached)
        ran.append(x)
        if x == 1:
            raise cancelled
        return x

    outcomes = run_bounded(items, work, job=job, workers=1, window=4)
    got = []
    with pytest.raises(JobCancelled) as info:
        got.extend(o.item for o in outcomes)
    assert info.value is cancelled
    assert got == [0]
    assert ran == [0, 1]
    assert items.count <= 5
    assert not job.cancelled


def test_job_cancelled_from_items_stops_the_run() -> None:
    started = threading.Event()
    skipped = threading.Event()
    ran = []
    cancelled = JobCancelled("from items")

    def source() -> Iterator[int]:
        yield from range(3)
        wait(started)
        raise cancelled

    def work(x: int) -> int:
        ran.append(x)
        started.set()
        wait(skipped)
        return x

    class Pool(runner._Pool[int, int]):
        def skip_queued(self) -> None:
            super().skip_queued()
            skipped.set()

    outcomes = runner._outcomes(Pool(work, CancelToken(), 1), source(), JobContext(), 10)
    got = []
    with pytest.raises(JobCancelled) as info:
        got.extend(o.item for o in outcomes)
    assert info.value is cancelled
    assert got == ran == [0]


# --- errors and early ends ---------------------------------------------------------------------


def test_an_error_from_items_is_raised_after_the_drain() -> None:
    raising = threading.Event()
    failure = OSError("the disk went away")

    def source() -> Iterator[int]:
        yield from range(5)
        raising.set()
        raise failure

    def work(x: int) -> int:
        wait(raising)
        return x

    got = []
    with pytest.raises(OSError) as info:
        for outcome in run_bounded(source(), work, job=JobContext(), workers=2, window=10):
            got.append(outcome.unwrap())
    assert info.value is failure
    assert sorted(got) == list(range(5))


def test_an_error_from_iter_is_raised_at_the_call() -> None:
    with pytest.raises(TypeError):
        run_bounded(3, str, job=JobContext())  # type: ignore[arg-type]


def test_break_and_consumer_errors_clean_up() -> None:
    for _ in run_bounded(itertools.count(), lambda x: x, job=JobContext(), workers=4):
        break
    assert threads_gone()

    def consume(error: BaseException) -> None:
        for _ in run_bounded(itertools.count(), lambda x: x, job=JobContext(), workers=4):
            raise error

    for error in (ValueError(), KeyboardInterrupt(), SystemExit()):
        with pytest.raises(type(error)):
            consume(error)
        assert threads_gone()


def threads_gone() -> bool:
    return not any(t.name.startswith(runner.THREAD_PREFIX) for t in threading.enumerate())


def test_close_skips_queued_items_and_waits_for_running_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = Pulls(5)
    release = threading.Event()
    started, finished = [], []
    second_started = threading.Event()

    def work(x: int) -> int:
        wait(items.reached)
        started.append(x)
        if x == 1:
            second_started.set()
            wait(release)
        finished.append(x)
        return x

    skip: Callable[[runner._Pool[int, int]], None] = runner._Pool.skip_queued

    def skip_then_release(pool: runner._Pool[int, int]) -> None:
        skip(pool)
        release.set()

    monkeypatch.setattr(runner._Pool, "skip_queued", skip_then_release)
    outcomes = run_bounded(items, work, job=JobContext(), workers=1, window=5)
    next(outcomes)
    wait(second_started)
    outcomes.close()
    assert threads_gone()
    assert started == finished == [0, 1]
    with pytest.raises(StopIteration):
        next(outcomes)


def test_closing_before_the_first_item_pulls_nothing() -> None:
    items = Pulls(0)
    run_bounded(items, lambda x: x, job=JobContext()).close()
    assert items.count == 0


def test_the_garbage_collector_may_close_it_on_a_worker() -> None:
    items = Pulls(2)
    first_taken, closed = threading.Event(), threading.Event()
    workers, problems = [], []

    def work(x: int) -> int:
        if x == 0:
            wait(items.reached)  # so that item 1 is queued when the generator is suspended
        else:
            workers.append(threading.current_thread())
            wait(first_taken)
            try:
                outcomes.close()
            except Exception as error:  # noqa: BLE001
                problems.append(error)
            closed.set()
        return x

    outcomes = run_bounded(items, work, job=JobContext(), workers=1, window=2)
    assert next(outcomes).item == 0
    first_taken.set()
    wait(closed)
    assert problems == []
    workers[0].join(TIMEOUT)


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_interrupt_on_the_calling_thread_cancels_running_work(error: type[BaseException]) -> None:
    job = JobContext()
    file = EndlessFile(job.cancel)

    def source() -> Iterator[int]:
        yield 0
        wait(file.reading)
        raise error

    def work(_: int) -> int:
        return sum(map(len, read_chunks(file, 1, job.cancel)))

    assert collect_until(error, run_bounded(source(), work, job=job, workers=1, window=4)) == []
    assert job.cancelled
    assert threads_gone()


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_interrupt_in_a_worker_is_raised_to_the_caller(error: type[BaseException]) -> None:
    job = JobContext()

    def work(x: int) -> int:
        if x == 5:
            raise error
        return x

    got = collect_until(error, run_bounded(itertools.count(), work, job=job, workers=2))
    assert 5 not in {o.item for o in got}
    assert job.cancelled


def test_a_thread_that_cannot_start_does_not_hang(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(thread: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    with pytest.raises(RuntimeError, match="start new thread"):
        list(run_bounded([1], lambda x: x, job=JobContext()))


def run_script(code: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONUTF8": "1"}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_an_abandoned_run_does_not_block_interpreter_exit() -> None:
    result = run_script(
        """
        import itertools
        from mediasift.jobs import JobContext, run_bounded
        outcomes = run_bounded(itertools.count(), lambda x: x, job=JobContext(), workers=4)
        print(next(outcomes).ok)
        """
    )
    assert (result.returncode, result.stdout.strip()) == (0, "True"), result.stderr


def test_second_ctrl_c_ends_the_program_even_with_a_stuck_worker() -> None:
    result = run_script(
        """
        import _thread, threading
        from mediasift.jobs import JobContext, cancel_on_sigint, run_bounded

        job = JobContext()
        stuck = threading.Event()

        def work(x):
            stuck.set()
            threading.Event().wait()

        def press_twice():
            stuck.wait()
            _thread.interrupt_main()
            while not job.cancelled:
                pass
            _thread.interrupt_main()

        threading.Thread(target=press_twice, daemon=True).start()
        with cancel_on_sigint(job.cancel):
            try:
                list(run_bounded([1], work, job=job))
            except KeyboardInterrupt:
                print("interrupted")
        """
    )
    assert (result.returncode, result.stdout.strip()) == (0, "interrupted"), result.stderr


# --- progress ----------------------------------------------------------------------------------


def test_progress_inside_a_long_item_is_reported_on_the_owning_thread() -> None:
    seen_midway = threading.Event()
    threads: set[int] = set()

    def on_progress(progress: Progress) -> None:
        threads.add(threading.get_ident())
        if progress.bytes_done == 7:
            seen_midway.set()

    job = JobContext(on_progress=on_progress)
    with job.stage("hash", total=1) as stage:

        def work(x: int) -> int:
            stage.advance(0, bytes=7)
            wait(seen_midway)  # reported while this item is still running
            return x

        assert [o.unwrap() for o in run_bounded([1], work, job=job)] == [1]
    assert threads == {threading.get_ident()}
