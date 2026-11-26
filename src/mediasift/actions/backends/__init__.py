"""Removal backends: quarantine, Recycle Bin (Windows) and Trash (Linux).

Each one is named in the operation log, so ``BACKENDS.get(name)`` finds the backend that can
undo an operation.
"""

from __future__ import annotations

import sys

from mediasift.actions.backends.base import (
    Located,
    Location,
    PlannedRemoval,
    RemovalBackend,
    RemovalRequest,
    RestoredFile,
    RestoreRequest,
    StoredFile,
)
from mediasift.actions.backends.quarantine import QuarantineBackend
from mediasift.actions.backends.recycle_bin import RecycleBinBackend
from mediasift.actions.backends.trash import TrashBackend
from mediasift.core.plugins import Registry

BACKENDS: Registry[RemovalBackend] = Registry("removal backend")
BACKENDS.register(QuarantineBackend())
BACKENDS.register(RecycleBinBackend())
BACKENDS.register(TrashBackend())


def for_setting(setting: str) -> RemovalBackend:
    """The backend for the ``actions.backend`` setting: ``trash`` means this system's."""
    if setting == "trash":
        return BACKENDS.get("recycle_bin" if sys.platform == "win32" else "trash")
    return BACKENDS.get(setting)


__all__ = [
    "BACKENDS",
    "Located",
    "Location",
    "PlannedRemoval",
    "QuarantineBackend",
    "RecycleBinBackend",
    "RemovalBackend",
    "RemovalRequest",
    "RestoreRequest",
    "RestoredFile",
    "StoredFile",
    "TrashBackend",
    "for_setting",
]
