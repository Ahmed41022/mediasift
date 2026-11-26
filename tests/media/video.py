"""Short generated videos and the variants named in the test plan (ffmpeg lavfi sources).

Each function takes an output path (and a source path for variants) and returns the
output path. Mark tests with :data:`requires_ffmpeg` so they skip cleanly without it.
"""

from __future__ import annotations

from pathlib import Path

from tests.media.corrupt import truncated
from tests.media.ffmpeg_util import requires_ffmpeg, run_ffmpeg

__all__ = [
    "SOURCES",
    "clip_of_longer",
    "make_video",
    "no_audio",
    "reencode",
    "requires_ffmpeg",
    "rotated",
    "scaled",
    "to_container",
    "trim",
    "truncated_video",
    "variable_frame_rate",
]

SOURCES = ("testsrc2", "mandelbrot", "life")
_TONES = {"testsrc2": 440, "mandelbrot": 330, "life": 550}


def make_video(
    out: Path,
    source: str = "testsrc2",
    *,
    duration: float = 2.0,
    size: tuple[int, int] = (160, 120),
    fps: int = 15,
    audio: bool = True,
    seed: int = 1,
) -> Path:
    """H.264 (+ AAC sine) clip from a lavfi source; ``seed`` picks the life / mandelbrot variant."""
    w, h = size
    if source == "testsrc2":
        video = f"testsrc2=size={w}x{h}:rate={fps}:duration={duration}"
    elif source == "mandelbrot":
        video = f"mandelbrot=size={w}x{h}:rate={fps}:start_scale=3.0:end_scale=0.05"
    elif source == "life":
        video = f"life=size={w}x{h}:rate={fps}:ratio=0.15:seed={seed}:mold=10:life_color=#00ff00"
    else:
        raise ValueError(f"unknown source {source!r}")
    args = ["-f", "lavfi", "-i", video]
    if audio:
        tone = _TONES[source] + 20 * (seed % 5)
        args += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={tone}:sample_rate=22050:duration={duration}",
        ]
    args += [
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(fps),
    ]
    args += ["-c:a", "aac", "-b:a", "48k"] if audio else ["-an"]
    return run_ffmpeg(args, out)


def reencode(src: Path, out: Path, *, crf: int = 32, codec: str = "libx264") -> Path:
    """Same picture, different bitrate (and codec when given)."""
    return run_ffmpeg(
        [
            "-i",
            str(src),
            "-c:v",
            codec,
            "-crf",
            str(crf),
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-b:a",
            "32k",
        ],
        out,
    )


def scaled(src: Path, out: Path, width: int = 80) -> Path:
    """Downscale keeping the aspect ratio (height rounded to even)."""
    return run_ffmpeg(
        [
            "-i",
            str(src),
            "-vf",
            f"scale={width}:-2",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "copy",
        ],
        out,
    )


def to_container(src: Path, out: Path) -> Path:
    """Re-mux without re-encoding; the container follows the suffix of ``out`` (mkv, mov...)."""
    return run_ffmpeg(["-i", str(src), "-c", "copy"], out)


def trim(src: Path, out: Path, start: float = 0.5, duration: float = 1.0) -> Path:
    """A re-encoded part of ``src`` (frame accurate)."""
    return run_ffmpeg(
        [
            "-ss",
            str(start),
            "-i",
            str(src),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
        ],
        out,
    )


def clip_of_longer(
    out: Path,
    source: str = "testsrc2",
    *,
    long_duration: float = 4.0,
    start: float = 1.0,
    duration: float = 1.5,
) -> tuple[Path, Path]:
    """Build a longer file and a clip cut from it; returns ``(long_file, clip)``."""
    long_file = make_video(
        out.with_name(out.stem + "_long" + out.suffix), source, duration=long_duration
    )
    return long_file, trim(long_file, out, start, duration)


def no_audio(src: Path, out: Path) -> Path:
    """Drop the audio stream, keep the video untouched."""
    return run_ffmpeg(["-i", str(src), "-an", "-c:v", "copy"], out)


def rotated(src: Path, out: Path, degrees: int = 90) -> Path:
    """Same pixels with a display-rotation side data entry (a phone video held sideways)."""
    return run_ffmpeg(["-display_rotation", str(degrees), "-i", str(src), "-c", "copy"], out)


def variable_frame_rate(src: Path, out: Path) -> Path:
    """Randomly drop about 40% of the frames so timestamps are irregular."""
    return run_ffmpeg(
        [
            "-i",
            str(src),
            "-vf",
            "select='lt(random(7),0.6)',setpts=N/FRAME_RATE/TB*1.0",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-an",
        ],
        out,
    )


def truncated_video(src: Path, out: Path, fraction: float = 0.6) -> Path:
    """The first part of the bytes of ``src``: a recording cut short."""
    out.write_bytes(truncated(src.read_bytes(), fraction))
    return out
