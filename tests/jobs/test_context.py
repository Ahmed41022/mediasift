from __future__ import annotations

import dataclasses
import itertools
import logging
import threading
from collections.abc import Callable

import pytest

from mediasift.core.errors import JobCancelled
from mediasift.jobs.context import CancelToken, JobContext, Progress


class Clock:
    def __init__(self) -> None:
        self.now = 0.0  # so that only a forced delivery gets through at the start

    def __call__(self) -> float:
        return self.now


def on_thread(call: Callable[[], object]) -> BaseException | None:
    """Run ``call`` on another thread and return what it raised."""
    raised: list[BaseException] = []

    def run() -> None:
        try:
            call()
        except Exception as error:  # noqa: BLE001
            raised.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(20)
    assert not thread.is_alive()
    return raised[0] if raised else None


def test_token() -> None:
    token = CancelToken()
    assert not token.cancelled
    token.raise_if_cancelled()
    on_thread(token.cancel)
    token.cancel()
    assert token.cancelled
    with pytest.raises(JobCancelled):
        token.raise_if_cancelled()


def test_context_uses_the_given_token() -> None:
    token = CancelToken()
    job = JobContext(token)
    job.raise_if_cancelled()
    token.cancel()
    assert job.cancel is token
    assert job.cancelled
    with pytest.raises(JobCancelled):
        job.raise_if_cancelled()


def test_default_contexts_are_independent_and_silent() -> None:
    job, other = JobContext(), JobContext()
    job.cancel.cancel()
    assert not other.cancelled
    with other.stage("scan", total=1) as stage:
        stage.advance()
        assert stage.snapshot() == Progress("scan", done=1, total=1)


def test_progress_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Progress("scan").done = 1  # type: ignore[misc]


def test_counters() -> None:
    seen: list[Progress] = []
    with JobContext(on_progress=seen.append).stage("hash", 3, 100) as stage:
        stage.advance(current="a")
        stage.advance(1, 40, "b", 1)
        stage.advance(0, bytes=10)
    assert seen[0] == Progress("hash", 0, 3, 0, 100)
    assert seen[-1] == Progress("hash", 2, 3, 50, 100, "b", 1)


def test_set_total() -> None:
    stage = JobContext().stage("scan")
    assert stage.snapshot().total is None
    stage.set_total(5, 10)
    assert stage.snapshot() == Progress("scan", 0, 5, 0, 10)


def test_rate_limit() -> None:
    clock, seen = Clock(), []
    job = JobContext(on_progress=lambda p: seen.append((clock.now, p.done)), clock=clock)
    with job.stage("scan", total=10_000) as stage:
        for _ in range(10_000):
            clock.now += 0.001  # ten seconds in all
            stage.advance()
    assert 90 <= len(seen) <= 102
    assert seen[0] == (0.0, 0)
    assert seen[-1][1] == 10_000
    gaps = [b[0] - a[0] for a, b in itertools.pairwise(seen[:-1])]
    assert min(gaps) >= 0.1


def test_first_and_last_are_delivered_when_no_time_passes() -> None:
    seen: list[Progress] = []
    job = JobContext(on_progress=seen.append, clock=Clock())
    for name in ("scan", "hash"):
        with job.stage(name) as stage:
            for _ in range(100):
                stage.advance()
    assert [(p.stage, p.done) for p in seen] == [
        ("scan", 0),
        ("scan", 100),
        ("hash", 0),
        ("hash", 100),
    ]


def test_last_is_delivered_when_the_stage_fails() -> None:
    seen: list[Progress] = []
    job = JobContext(on_progress=seen.append, clock=Clock())
    with pytest.raises(JobCancelled), job.stage("scan") as stage:
        stage.advance()
        raise JobCancelled
    assert [p.done for p in seen] == [0, 1]


def test_custom_rate() -> None:
    clock, seen = Clock(), []
    job = JobContext(on_progress=lambda p: seen.append(p.done), max_rate=2, clock=clock)
    stage = job.stage("scan")
    for _ in range(4):
        clock.now += 0.25
        stage.advance()
    assert seen == [0, 2, 4]
    with pytest.raises(ValueError, match="max_rate"):
        JobContext(max_rate=0)


def test_nothing_is_delivered_twice_or_after_the_end() -> None:
    clock, seen = Clock(), []
    job = JobContext(on_progress=lambda p: seen.append(p.done), clock=clock)
    stage = job.stage("scan")
    clock.now += 1
    stage.advance()
    clock.now += 1
    job.deliver_progress()  # unchanged
    stage.finish()
    stage.finish()
    clock.now += 1
    stage.advance()
    job.deliver_progress()
    assert seen == [0, 1]


def test_overlapping_stages() -> None:
    clock, seen = Clock(), []
    job = JobContext(on_progress=lambda p: seen.append((p.stage, p.done)), clock=clock)
    with job.stage("scan") as scan, job.stage("hash") as hashing:
        clock.now += 1
        scan.advance()
        hashing.advance()
    assert seen == [("scan", 0), ("hash", 0), ("scan", 1), ("hash", 1)]


def test_workers_count_but_never_run_the_callback() -> None:
    clock, seen = Clock(), []
    job = JobContext(
        on_progress=lambda p: seen.append((threading.get_ident(), p.done)), clock=clock
    )
    with job.stage("hash") as stage:
        clock.now += 1
        assert on_thread(stage.advance) is None
        assert on_thread(job.deliver_progress) is None
        assert len(seen) == 1
        job.deliver_progress()
        assert len(seen) == 2
        assert isinstance(on_thread(stage.finish), RuntimeError)
    assert seen == [(threading.get_ident(), 0), (threading.get_ident(), 1)]


def test_only_the_owner_opens_stages() -> None:
    job = JobContext()
    job.stage("scan")
    assert isinstance(on_thread(lambda: job.stage("hash")), RuntimeError)


def test_no_advance_is_lost() -> None:
    stage = JobContext().stage("hash")
    barrier = threading.Barrier(4)

    def work() -> None:
        barrier.wait(20)
        for _ in range(5000):
            stage.advance(1, 2, errors=3)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert stage.snapshot() == Progress("hash", done=20_000, bytes_done=40_000, errors=60_000)


def test_a_failing_callback_is_logged_once_and_ignored(caplog: pytest.LogCaptureFixture) -> None:
    calls = 0

    def fail(progress: Progress) -> None:
        nonlocal calls
        calls += 1
        raise ValueError("broken display")

    with (
        caplog.at_level(logging.INFO, "mediasift.jobs.context"),
        JobContext(on_progress=fail, max_rate=1e9).stage("scan") as stage,
    ):
        stage.advance()
        stage.advance()
    assert calls >= 2
    assert len(caplog.records) == 1
    assert "broken display" in caplog.text


def test_a_callback_may_advance_and_cancel() -> None:
    job = JobContext(
        on_progress=lambda p: (stage.advance() if p.done else None, job.cancel.cancel())
    )
    with job.stage("scan") as stage:
        assert job.cancelled
        stage.advance()
    assert stage.snapshot().done == 2


def test_an_interrupt_in_a_callback_is_not_swallowed() -> None:
    def interrupt(progress: Progress) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        JobContext(on_progress=interrupt).stage("scan")
