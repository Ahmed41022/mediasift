from __future__ import annotations

from collections.abc import Iterator

import pytest

from mediasift.core.errors import JobCancelled
from mediasift.jobs.batch import batched_commit
from mediasift.jobs.context import JobContext
from mediasift.jobs.runner import Outcome, run_bounded


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def failing(count: int, error: BaseException) -> Iterator[int]:
    yield from range(count)
    raise error


def test_by_size_with_remainder() -> None:
    batches: list[list[int]] = []
    assert batched_commit(range(7), batches.append, size=3, max_seconds=1e9) == 7
    assert batches == [[0, 1, 2], [3, 4, 5], [6]]


def test_nothing_is_flushed_for_nothing() -> None:
    batches: list[list[int]] = []
    assert batched_commit([], batches.append) == 0
    assert batched_commit(range(4), batches.append, size=2, max_seconds=1e9) == 4
    assert batches == [[0, 1], [2, 3]]


def test_by_time_since_the_first_result_of_the_batch() -> None:
    clock = Clock()
    batches: list[list[int]] = []

    def source() -> Iterator[int]:
        for x, seconds in enumerate([100, 0, 1.5, 0.5, 50, 1.9, 0]):
            clock.now += seconds
            yield x

    batched_commit(source(), batches.append, size=100, max_seconds=2, clock=clock)
    # A long wait before a batch starts does not count; the one before result 4 starts a batch.
    assert batches == [[0, 1, 2, 3], [4, 5, 6]]


def test_zero_seconds_flushes_every_result() -> None:
    batches: list[list[int]] = []
    batched_commit(range(3), batches.append, size=100, max_seconds=0)
    assert batches == [[0], [1], [2]]


@pytest.mark.parametrize("error", [JobCancelled(), OSError("gone"), KeyboardInterrupt()])
@pytest.mark.parametrize("count", [0, 2, 3, 5])
def test_collected_results_are_flushed_before_the_exception(
    error: BaseException, count: int
) -> None:
    flushed: list[int] = []
    with pytest.raises(type(error)) as info:
        batched_commit(failing(count, error), flushed.extend, size=3, max_seconds=1e9)
    assert info.value is error
    assert flushed == list(range(count))


def test_a_failed_flush_is_not_retried_and_closes_the_source() -> None:
    calls: list[list[int]] = []
    closed = False

    def source() -> Iterator[int]:
        nonlocal closed
        try:
            yield from range(10)
        finally:
            closed = True

    def flush(batch: list[int]) -> None:
        calls.append(batch)
        raise OSError("disk full")

    results = source()  # held here, so that only an explicit close ends it
    with pytest.raises(OSError, match="disk full"):
        batched_commit(results, flush, size=2, max_seconds=1e9)
    assert calls == [[0, 1]]
    assert closed


def test_a_failed_flush_of_the_remainder_keeps_the_first_error_as_context() -> None:
    calls: list[list[int]] = []

    def flush(batch: list[int]) -> None:
        calls.append(batch)
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full") as info:
        batched_commit(failing(1, JobCancelled()), flush)
    assert isinstance(info.value.__context__, JobCancelled)
    assert calls == [[0]]


def test_a_batch_belongs_to_flush() -> None:
    def flush(batch: list[int]) -> None:
        seen.append(tuple(batch))
        batch.clear()

    seen: list[tuple[int, ...]] = []
    assert batched_commit(range(3), flush, size=2, max_seconds=1e9) == 3
    assert seen == [(0, 1), (2,)]


@pytest.mark.parametrize("kwargs", [{"size": 0}, {"max_seconds": -1}])
def test_arguments(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must"):
        batched_commit([], lambda _: None, **kwargs)


def test_a_failed_flush_ends_the_pool_behind_it() -> None:
    def flush(batch: list[Outcome[int, int]]) -> None:
        raise OSError("disk full")

    # conftest checks that no worker is left, even while the traceback is still alive.
    outcomes = run_bounded(range(1000), lambda x: x, job=JobContext(), workers=2)
    with pytest.raises(OSError, match="disk full"):
        batched_commit(outcomes, flush, size=10)


def test_cancel_and_resume_do_every_item_exactly_once() -> None:
    worked: list[int] = []
    stored: list[int] = []

    def work(x: int) -> int:
        worked.append(x)
        return x

    def remaining() -> Iterator[int]:
        return (x for x in range(200) if x not in done)

    runs = 0
    while True:
        job = JobContext()
        done = set(stored)

        def rows(job: JobContext, stop_after: int) -> Iterator[int]:
            for n, outcome in enumerate(
                run_bounded(remaining(), work, job=job, workers=4, window=16), 1
            ):
                if n == stop_after:
                    job.cancel.cancel()
                yield outcome.unwrap()

        runs += 1
        try:
            batched_commit(rows(job, 45), stored.extend, size=10, max_seconds=1e9)
            break
        except JobCancelled:
            assert len(stored) > len(done)
    assert sorted(stored) == list(range(200))
    # Nothing that was finished before a cancel was done again.
    assert sorted(worked) == list(range(200))
    assert 3 <= runs <= 6
