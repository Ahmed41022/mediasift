from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from types import FrameType

import pytest

from mediasift.core.errors import JobCancelled
from mediasift.jobs.context import CancelToken, JobContext
from mediasift.jobs.runner import run_bounded
from mediasift.jobs.sigint import cancel_on_sigint


class Marker(Exception):
    pass


def marker_handler(signum: int, frame: FrameType | None) -> None:
    raise Marker


@pytest.fixture(autouse=True)
def own_handler() -> Iterator[None]:
    """Install a handler that shows up, so a missed restore cannot pass unnoticed."""
    previous = signal.signal(signal.SIGINT, marker_handler)
    yield
    signal.signal(signal.SIGINT, previous)


def test_first_cancels_and_second_interrupts() -> None:
    token = CancelToken()
    with pytest.raises(KeyboardInterrupt), cancel_on_sigint(token):
        assert not token.cancelled
        signal.raise_signal(signal.SIGINT)
        assert token.cancelled
        signal.raise_signal(signal.SIGINT)
        pytest.fail("the second one must interrupt")
    assert signal.getsignal(signal.SIGINT) is marker_handler


def test_handler_is_restored() -> None:
    with cancel_on_sigint(CancelToken()):
        assert signal.getsignal(signal.SIGINT) is not marker_handler
    with pytest.raises(Marker):
        signal.raise_signal(signal.SIGINT)
    with pytest.raises(ValueError, match="x"), cancel_on_sigint(CancelToken()):
        raise ValueError("x")
    assert signal.getsignal(signal.SIGINT) is marker_handler


def test_each_use_starts_afresh() -> None:
    for _ in range(2):
        token = CancelToken()
        with cancel_on_sigint(token):
            signal.raise_signal(signal.SIGINT)
        assert token.cancelled


def test_an_ignored_signal_stays_ignored() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    token = CancelToken()
    with cancel_on_sigint(token):
        assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
        signal.raise_signal(signal.SIGINT)
    assert not token.cancelled


def test_a_handler_that_cannot_be_restored_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signal, "getsignal", lambda _: None)
    token = CancelToken()
    with cancel_on_sigint(token), pytest.raises(Marker):
        signal.raise_signal(signal.SIGINT)
    assert not token.cancelled


def test_does_nothing_off_the_main_thread() -> None:
    seen: list[object] = []

    def run() -> None:
        with cancel_on_sigint(CancelToken()):
            seen.append(signal.getsignal(signal.SIGINT))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(20)
    assert seen == [marker_handler]


def test_ctrl_c_stops_a_run_cleanly() -> None:
    job = JobContext()
    got = []
    with pytest.raises(JobCancelled), cancel_on_sigint(job.cancel):
        for outcome in run_bounded(iter(int, 1), lambda x: x, job=job, workers=2, window=4):
            got.append(outcome)
            if len(got) == 10:
                signal.raise_signal(signal.SIGINT)
    assert 10 <= len(got) <= 14
