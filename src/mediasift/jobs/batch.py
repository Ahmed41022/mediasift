"""Commit results in batches."""

from __future__ import annotations

import time
from collections.abc import Callable, Generator, Iterable
from typing import TypeVar

T = TypeVar("T")


def batched_commit(
    results: Iterable[T],
    flush: Callable[[list[T]], None],
    *,
    size: int = 500,
    max_seconds: float = 2.0,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Pass ``results`` to ``flush`` in lists, and return how many were flushed.

    A batch is flushed when it holds ``size`` results or when, as a result arrives, its first one
    is ``max_seconds`` old. Everything runs on the calling thread, so no flush happens while the
    source blocks.

    If ``results`` raises, whatever was collected is flushed before the exception continues. That
    is true of every exception, not only ``JobCancelled``: collected results are finished work,
    and whatever went wrong later does not make them invalid. It is what lets a job resume.

    If ``flush`` itself raises, its batch is not retried. A failed flush of the remainder hides
    the original exception, which stays visible as its context. A ``results`` generator is closed
    in every case, so a pool behind it does not outlive a failed flush.
    """
    if size < 1:
        raise ValueError("size must be at least 1")
    if max_seconds < 0:
        raise ValueError("max_seconds must not be negative")
    flushed = 0
    pending: list[T] = []
    started = 0.0
    iterator = iter(results)
    try:
        while True:
            try:
                result = next(iterator)
            except StopIteration:
                break
            except BaseException:
                if pending:
                    flush(pending)
                raise
            if not pending:
                started = clock()
            pending.append(result)
            if len(pending) >= size or clock() - started >= max_seconds:
                batch, pending = pending, []
                flushed += len(batch)  # before the call: the list is flush's to change
                flush(batch)
        if pending:
            flushed += len(pending)
            flush(pending)
    finally:
        if isinstance(iterator, Generator):
            iterator.close()
    return flushed
