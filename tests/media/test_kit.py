"""The media kit must itself be deterministic and produce what it claims."""

from __future__ import annotations

import hashlib
import io
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from tests.media import audio, corrupt, images, tree, video
from tests.media.dng import write_dng
from tests.media.ffmpeg_util import probe, requires_ffmpeg, streams, tags_of

SIZE = (320, 240)


def _arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float64)


def _mad(a: Image.Image, b: Image.Image) -> float:
    """Mean absolute pixel difference of two equally sized images."""
    return float(np.abs(_arr(a) - _arr(b)).mean())


def _thumb_mad(a: Image.Image, b: Image.Image) -> float:
    small = (32, 24)
    return _mad(a.resize(small, Image.Resampling.BOX), b.resize(small, Image.Resampling.BOX))


# ------------------------------------------------------------------ images


def test_scene_is_deterministic_and_seed_dependent() -> None:
    a, b = images.scene(5, SIZE), images.scene(5, SIZE)
    assert a.tobytes() == b.tobytes()
    assert images.scene(6, SIZE).tobytes() != a.tobytes()
    assert images.scene(5, SIZE).size == SIZE


def test_scene_is_not_flat_and_seeds_differ_in_composition() -> None:
    scenes = [images.scene(s, SIZE) for s in range(10, 16)]
    for img in scenes:
        assert _arr(img).std() > 20
    # Low-frequency structure (32x24 thumbnails) must differ clearly between seeds.
    diffs = [_thumb_mad(scenes[i], scenes[j]) for i in range(6) for j in range(i + 1, 6)]
    assert min(diffs) > 12


def test_similar_scene_is_close_to_its_base_and_far_from_others() -> None:
    base = images.scene(7, SIZE)
    near = images.similar_scene(7, 6, SIZE)
    assert near.size == SIZE
    assert near.tobytes() == images.similar_scene(7, 6, SIZE).tobytes()
    assert near.tobytes() != base.tobytes()
    assert _thumb_mad(base, near) < 0.5 * _thumb_mad(base, images.scene(8, SIZE))


def test_savers_roundtrip_through_pillow(tmp_path: Path) -> None:
    img = images.scene(2, (160, 120))
    savers = {
        "a.jpg": images.save_jpeg,
        "a.png": images.save_png,
        "a.webp": images.save_webp,
        "a.gif": images.save_gif,
        "a.bmp": images.save_bmp,
        "a.tif": images.save_tiff,
    }
    for name, save in savers.items():
        path = save(img, tmp_path / name)
        with Image.open(path) as back:
            back.load()
            assert back.size == (160, 120), name
    with Image.open(tmp_path / "a.png") as png:
        assert png.tobytes() == img.tobytes()  # lossless


@images.requires_heif
def test_heic_roundtrip(tmp_path: Path) -> None:
    img = images.scene(2, (160, 120))
    path = images.save_heic(img, tmp_path / "a.heic", exif=images.build_exif(make="Acme"))
    with Image.open(path) as back:
        assert back.size == (160, 120)
        assert images.read_exif(back)["make"] == "Acme"


def test_exif_roundtrip(tmp_path: Path) -> None:
    taken = datetime(2022, 5, 17, 14, 3, 9)
    exif = images.build_exif(
        taken=taken,
        offset="+03:00",
        make="Acme",
        model="Cam 9",
        gps=(30.0444, 31.2357),
        orientation=1,
    )
    path = images.save_jpeg(images.scene(3, (64, 48)), tmp_path / "e.jpg", exif=exif)
    with Image.open(path) as img:
        info = images.read_exif(img)
    assert info["taken"] == taken
    assert info["offset"] == "+03:00"
    assert (info["make"], info["model"], info["orientation"]) == ("Acme", "Cam 9", 1)
    assert info["gps"] == pytest.approx((30.0444, 31.2357), abs=1e-4)
    south_west = images.build_exif(gps=(-33.5, -70.25))
    back = images.read_exif(
        Image.open(io.BytesIO(images.to_bytes(images.scene(3, (64, 48)), "JPEG", exif=south_west)))
    )
    assert back["gps"] == pytest.approx((-33.5, -70.25), abs=1e-4)


def test_all_orientations_display_identically() -> None:
    base = images.scene(4, (200, 120))
    reference = ImageOps.exif_transpose(Image.open(io.BytesIO(images.to_bytes(base, "PNG"))))
    variants = images.all_orientations(base, quality=100)
    assert sorted(variants) == list(range(1, 9))
    for orientation, data in variants.items():
        with Image.open(io.BytesIO(data)) as stored:
            assert images.read_exif(stored)["orientation"] == orientation
            if orientation in (5, 6, 7, 8):
                assert stored.size == (120, 200)
            shown = ImageOps.exif_transpose(stored)
        assert shown.size == (200, 120), orientation
        assert _mad(shown, reference) < 4, orientation
    # The stored pixels of the rotated ones really are rotated.
    plain = Image.open(io.BytesIO(variants[1])).convert("RGB")
    rot = Image.open(io.BytesIO(variants[6])).convert("RGB")
    assert rot.size != plain.size


@pytest.mark.parametrize("name", sorted(images.VARIANTS))
def test_variant_dimensions_and_validity(name: str, small_scene: Image.Image) -> None:
    encoded = images.VARIANTS[name](small_scene)
    out = encoded.image()
    assert out.size == images.variant_size(name, small_scene.size)
    assert encoded.ext in {".jpg", ".png", ".webp"}
    if name == "whatsapp":
        assert "exif" not in out.info
        assert out.format == "JPEG"
    if name == "grayscale":
        assert out.mode == "L"


def test_variants_stay_close_to_the_original(small_scene: Image.Image) -> None:
    far = _thumb_mad(small_scene, images.scene(99, small_scene.size))
    for name in (
        "recompress_q80",
        "brightness_up",
        "slight_blur",
        "png_conversion",
        "webp_conversion",
    ):
        out = images.VARIANTS[name](small_scene).image()
        assert _thumb_mad(small_scene, out) < 0.5 * far, name
    assert far > 12


def test_whatsapp_size_rule() -> None:
    assert images.whatsapp_size((4000, 3000)) == (1600, 1200)
    assert images.whatsapp_size((1000, 500)) == (600, 300)


def test_exif_rotated_variant_displays_upright(small_scene: Image.Image) -> None:
    out = images.VARIANTS["exif_rotated"](small_scene).image()
    assert images.read_exif(out)["orientation"] == 6
    assert ImageOps.exif_transpose(out).size == small_scene.size


def test_negatives_differ_from_the_scene() -> None:
    base = images.scene(21, SIZE)
    shuffled = images.same_palette_shuffled(21, SIZE)
    assert np.array_equal(
        np.sort(_arr(shuffled).reshape(-1, 3), axis=0), np.sort(_arr(base).reshape(-1, 3), axis=0)
    )
    assert _thumb_mad(base, shuffled) > 8
    pages = [images.text_page(s, SIZE) for s in (1, 2)]
    assert pages[0].tobytes() != pages[1].tobytes()
    assert _arr(pages[0]).mean() > 200
    flat = images.flat_image(size=SIZE)
    assert _arr(flat).std(axis=(0, 1)).max() == 0
    h1, h2 = images.horizon_scene(30, 1, SIZE), images.horizon_scene(30, 2, SIZE)
    assert h1.tobytes() != h2.tobytes()
    # Same sky/ground composition: closer to each other than to an unrelated scene.
    assert _thumb_mad(h1, h2) < _thumb_mad(h1, images.horizon_scene(31, 1, SIZE))
    assert len(images.distinct_scenes(2, size=(64, 48))) == 2


# ------------------------------------------------------------------ corrupt


def test_corrupt_generators(tmp_path: Path) -> None:
    data = images.to_bytes(images.scene(1, (96, 72)), "JPEG")
    cut = corrupt.truncated(data, 0.5)
    assert len(cut) == len(data) // 2 and data.startswith(cut)
    assert 0 < len(corrupt.truncated(b"ab", 0.0)) < 2
    flipped = corrupt.bitflips(data, 5, seed=1)
    assert flipped != data and flipped[:32] == data[:32] and len(flipped) == len(data)
    assert flipped == corrupt.bitflips(data, 5, seed=1)
    assert corrupt.header_only(data, 10) == data[:10]
    assert corrupt.write_zero_bytes(tmp_path / "z.jpg").stat().st_size == 0
    junk = corrupt.write_garbage(tmp_path / "g.jpg", 500, seed=3)
    assert junk.stat().st_size == 500
    for bad in (cut, flipped[:200], corrupt.header_only(data), junk.read_bytes()):
        try:
            with Image.open(io.BytesIO(bad)) as im:
                im.load()
        except Exception:  # noqa: BLE001 - any decode failure is the point
            continue
    with pytest.raises(Exception):  # noqa: B017 - Pillow raises several types
        Image.open(junk).load()


# ------------------------------------------------------------------ dng


def test_dng_opens_with_rawpy(tmp_path: Path) -> None:
    rawpy = pytest.importorskip("rawpy")
    info = write_dng(tmp_path / "t.dng", seed=3)
    assert write_dng(tmp_path / "u.dng", seed=3).path.read_bytes() == info.path.read_bytes()
    with rawpy.imread(str(info.path)) as raw:
        assert (raw.sizes.raw_width, raw.sizes.raw_height) == (info.width, info.height)
        rgb = raw.postprocess()
        assert rgb.shape == (info.height, info.width, 3) and rgb.std() > 5
        thumb = raw.extract_thumb()
        assert Image.open(io.BytesIO(thumb.data)).size == (info.width, info.height)
    data = info.path.read_bytes()
    assert data[:4] == b"II*\x00"
    assert b"Test Camera 1" in data and b"2021:07:04 10:30:00" in data


def test_dng_without_preview(tmp_path: Path) -> None:
    rawpy = pytest.importorskip("rawpy")
    info = write_dng(tmp_path / "n.dng", preview=False)
    with rawpy.imread(str(info.path)) as raw:
        assert raw.postprocess().shape[2] == 3


# ------------------------------------------------------------------ audio / video


def test_wav_generation_without_ffmpeg(tmp_path: Path) -> None:
    m = audio.melody(4, 1.0)
    assert np.array_equal(m, audio.melody(4, 1.0))
    assert not np.array_equal(m, audio.melody(5, 1.0))
    path = audio.write_wav(tmp_path / "m.wav", m)
    back, rate = audio.read_wav(path)
    assert rate == audio.RATE and len(back) == len(m)
    assert np.abs(back - m).max() < 1e-3
    padded = audio.add_silence(m, 0.25, 0.25)
    assert len(padded) == len(m) + 2 * int(0.25 * audio.RATE)
    assert len(audio.trim_samples(m, 0.2, 0.3)) == int(0.3 * audio.RATE)
    assert abs(audio.tone(440, 0.5)).max() <= 0.5
    assert audio.noise(1, 0.2).std() > 0.1


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.parametrize("suffix", [".mp3", ".flac", ".ogg", ".m4a", ".opus"])
def test_audio_encodes(tmp_path: Path, suffix: str) -> None:
    wav = audio.write_wav(tmp_path / "s.wav", audio.melody(2, 1.5))
    out = audio.encode(wav, tmp_path / f"s{suffix}")
    assert float(probe(out)["format"]["duration"]) == pytest.approx(1.5, abs=0.3)


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.parametrize("suffix", [".mp3", ".ogg"])
def test_audio_tags_set_and_stripped(tmp_path: Path, suffix: str) -> None:
    wav = audio.write_wav(tmp_path / "s.wav", audio.melody(2, 1.0))
    tagged = audio.retag(
        audio.encode(wav, tmp_path / f"s{suffix}"),
        tmp_path / f"t{suffix}",
        {"title": "Hello", "artist": "Me"},
    )
    assert tags_of(tagged).get("title") == "Hello"
    assert "title" not in tags_of(audio.strip_tags(tagged, tmp_path / f"b{suffix}"))


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_audio_bitrates_differ(tmp_path: Path) -> None:
    wav = audio.write_wav(tmp_path / "s.wav", audio.melody(2, 2.0))
    low = audio.encode(wav, tmp_path / "low.mp3", "32k")
    high = audio.encode(wav, tmp_path / "high.mp3", "192k")
    assert low.stat().st_size < high.stat().st_size


@requires_ffmpeg
@pytest.mark.ffmpeg
def test_video_sources_and_variants_probe_ok(tmp_path: Path) -> None:
    sources = {
        s: video.make_video(tmp_path / f"{s}.mp4", s, duration=1.0, size=(96, 72), fps=10)
        for s in video.SOURCES
    }
    assert len({p.read_bytes() for p in sources.values()}) == 3
    src = sources["testsrc2"]
    again = video.make_video(
        tmp_path / "again.mp4", "testsrc2", duration=1.0, size=(96, 72), fps=10
    )
    assert again.read_bytes() == src.read_bytes()
    v = streams(src, "video")[0]
    assert (v["width"], v["height"]) == (96, 72) and streams(src, "audio")

    variants = {
        "reencode": video.reencode(src, tmp_path / "re.mp4"),
        "scaled": video.scaled(src, tmp_path / "sc.mp4", 48),
        "mkv": video.to_container(src, tmp_path / "c.mkv"),
        "trim": video.trim(src, tmp_path / "tr.mp4", 0.2, 0.5),
        "noaudio": video.no_audio(src, tmp_path / "na.mp4"),
        "rotated": video.rotated(src, tmp_path / "ro.mp4"),
        "vfr": video.variable_frame_rate(src, tmp_path / "vfr.mp4"),
    }
    for name, path in variants.items():
        assert streams(path, "video"), name
    assert streams(variants["scaled"], "video")[0]["width"] == 48
    assert not streams(variants["noaudio"], "audio")
    assert float(probe(variants["trim"])["format"]["duration"]) < 0.9
    side = streams(variants["rotated"], "video")[0].get("side_data_list", [])
    assert any(abs(int(s.get("rotation", 0))) == 90 for s in side)

    long_file, clip = video.clip_of_longer(
        tmp_path / "clip.mp4", duration=0.8, long_duration=2.0, start=0.5
    )
    assert float(probe(long_file)["format"]["duration"]) > float(probe(clip)["format"]["duration"])
    cut = video.truncated_video(src, tmp_path / "cut.mp4", 0.5)
    assert cut.stat().st_size < src.stat().st_size


# ------------------------------------------------------------------ trees


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def test_build_tree_matches_its_description(tmp_path: Path) -> None:
    exp = tree.build_tree(tmp_path / "lib", seed=1)
    root = exp.root
    on_disk = {p for p in root.rglob("*") if p.is_file()}
    assert on_disk == set(exp.all_paths)
    assert len(exp.all_paths) == len(set(exp.all_paths))

    # Hashing every file groups exactly as described (empty files aside).
    by_hash: dict[str, list[Path]] = {}
    for p in exp.media_paths + exp.non_media_paths:
        by_hash.setdefault(_sha(p), []).append(p)
    found = {frozenset(v) for v in by_hash.values() if len(v) > 1}
    assert found == {frozenset(g.paths) for g in exp.exact_groups}
    for g in exp.exact_groups:
        assert {_sha(p) for p in g.paths} == {g.sha256}
        assert {p.stat().st_size for p in g.paths} == {g.size}
    assert exp.reclaimable_bytes == sum(g.size * (len(g.paths) - 1) for g in exp.duplicate_groups)
    assert exp.reclaimable_bytes > 0

    assert all(p.stat().st_size == 0 for p in exp.empty_files) and len(exp.empty_files) == 3
    for group in exp.same_size_different:
        assert len({p.stat().st_size for p in group}) == 1
        assert len({_sha(p) for p in group}) == len(group)
    for group in exp.same_head_tail_different_middle:
        datas = [p.read_bytes() for p in group]
        assert all(len(d) > 256 * 1024 for d in datas)
        n = 100 * 1024
        assert len({d[:n] for d in datas}) == 1 and len({d[-n:] for d in datas}) == 1
        assert len({_sha(p) for p in group}) == len(group)

    names = {p.relative_to(root).as_posix() for p in on_disk}
    assert any("صور العائلة" in n for n in names)
    assert any(" " in n for n in names) and any("😀" in n for n in names)
    assert any(n.endswith((".JPG", ".MP4", ".Mp3", ".NEF")) for n in names)
    assert {p.suffix for p in exp.non_media_paths} >= {".txt", ".json", ".db", ""}


def test_build_tree_hardlinks(tmp_path: Path) -> None:
    exp = tree.build_tree(tmp_path / "lib", seed=2)
    links = exp.hardlink_groups
    if not links:
        pytest.skip("file system has no hard links")
    a, b = links[0].paths
    assert os.path.samefile(a, b) and a.stat().st_nlink >= 2
    assert not tree.build_tree(tmp_path / "nolinks", seed=2, hardlinks=False).hardlink_groups


def test_build_tree_is_deterministic(tmp_path: Path) -> None:
    one = _snapshot(tree.build_tree(tmp_path / "a", seed=3).root)
    two = _snapshot(tree.build_tree(tmp_path / "b", seed=3).root)
    other = _snapshot(tree.build_tree(tmp_path / "c", seed=4).root)
    assert one == two
    assert one != other
    assert set(one) == set(other)  # names are fixed, contents depend on the seed


def test_cheap_tree(tmp_path: Path) -> None:
    info = tree.cheap_tree(tmp_path / "big", 300, dup_every=10, per_dir=100)
    files = [p for p in info.root.rglob("*") if p.is_file()]
    assert len(files) == info.count == 300
    counts: dict[str, int] = {}
    for p in files:
        counts[_sha(p)] = counts.get(_sha(p), 0) + 1
    assert sum(1 for c in counts.values() if c == 2) == info.duplicate_groups == 29
    assert max(counts.values()) == 2
    with pytest.raises(ValueError, match="dup_every"):
        tree.cheap_tree(tmp_path / "x", 5, dup_every=1)
