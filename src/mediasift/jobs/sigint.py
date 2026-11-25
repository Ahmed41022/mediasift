"""Turn Ctrl+C into a clean cancel."""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

from mediasift.jobs.context import CancelToken


@contextmanager
def cancel_on_sigint(token: CancelToken) -> Iterator[None]:
    """Cancel ``token`` at the first Ctrl+C; raise ``KeyboardInterrupt`` at the second.

    Does nothing off the main thread, where Python does not allow signal handlers, or when SIGINT
    is ignored (as under ``nohup``) or handled by code outside Python that could not be restored.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)
    if previous is None or previous is signal.SIG_IGN:
        yield
        return

    asked = False

    def handle(signum: int, frame: FrameType | None) -> None:
        nonlocal asked
        if asked:
            raise KeyboardInterrupt
        asked = True
        token.cancel()

    signal.signal(signal.SIGINT, handle)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
