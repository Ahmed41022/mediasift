"""Shared fixtures: small generated media, built once per session."""

from __future__ import annotations

import pytest
from PIL import Image

from tests.media.images import scene


@pytest.fixture(scope="session")
def small_scene() -> Image.Image:
    """A 320x240 generated photo-like image (seed 1); treat it as read-only."""
    return scene(1, (320, 240))
