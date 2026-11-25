from __future__ import annotations

import pytest

from mediasift.core.errors import ConfigError
from mediasift.core.plugins import Registry
from mediasift.core.types import FileSignature, MediaType, media_type_for


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("a.JPG", MediaType.IMAGE),
        ("صورة.heic", MediaType.IMAGE),
        ("b.tar.mp4", MediaType.VIDEO),
        ("voice.amr", MediaType.AUDIO),
        (".jpg", None),
        ("notes.txt", None),
        ("jpg", None),
    ],
)
def test_media_type(name: str, expected: MediaType | None) -> None:
    assert media_type_for(name) is expected


def test_unknown_identity_is_never_the_same_file() -> None:
    assert not FileSignature(1, 1).same_file(FileSignature(1, 1))
    assert FileSignature(1, 1, 5, 9).same_file(FileSignature(2, 2, 5, 9))


def test_registry() -> None:
    class Plugin:
        name = "x"

    registry: Registry[Plugin] = Registry("plugin")
    item = registry.register(Plugin())
    assert registry.get("x") is item
    assert list(registry) == [item]
    with pytest.raises(ConfigError, match="twice"):
        registry.register(Plugin())
    with pytest.raises(ConfigError, match="unknown"):
        registry.get("y")
