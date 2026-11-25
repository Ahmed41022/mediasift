"""Generated audio: pure NumPy WAV files plus ffmpeg-encoded variants.

``wav``-level functions need no ffmpeg. Encoding to mp3, flac, ogg, m4a and opus, tag
edits and bitrate variants use ffmpeg and should be guarded with :data:`requires_ffmpeg`.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tests.media.ffmpeg_util import requires_ffmpeg, run_ffmpeg, run_ffmpeg_with_tags

__all__ = [
    "RATE",
    "add_silence",
    "encode",
    "melody",
    "noise",
    "read_wav",
    "requires_ffmpeg",
    "retag",
    "strip_tags",
    "tone",
    "trim_samples",
    "write_wav",
]

RATE = 22050
Samples = NDArray[np.float32]

# Codec arguments per output suffix.
_CODECS: dict[str, list[str]] = {
    ".mp3": ["-c:a", "libmp3lame"],
    ".flac": ["-c:a", "flac"],
    ".ogg": ["-c:a", "libvorbis"],
    ".m4a": ["-c:a", "aac"],
    ".opus": ["-c:a", "libopus"],
    ".wav": ["-c:a", "pcm_s16le"],
}


def tone(freq: float = 440.0, duration: float = 1.5, rate: int = RATE) -> Samples:
    """A sine tone with short fades so it does not click."""
    t = np.arange(int(duration * rate), dtype=np.float64) / rate
    out = 0.5 * np.sin(2 * np.pi * freq * t)
    fade = min(len(t) // 10, int(0.02 * rate))
    if fade:
        ramp = np.linspace(0, 1, fade)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]
    return out.astype(np.float32)


def melody(seed: int, duration: float = 2.5, rate: int = RATE) -> Samples:
    """Notes from a pentatonic scale with harmonics and decay; the tune depends on the seed."""
    rng = np.random.Generator(np.random.PCG64(seed))
    scale = np.array([0, 2, 4, 7, 9]) + int(rng.integers(0, 12))
    out = np.zeros(int(duration * rate), dtype=np.float64)
    pos = 0
    while pos < len(out):
        length = int(rng.choice([0.15, 0.25, 0.4]) * rate)
        note = int(rng.choice(scale)) + 12 * int(rng.integers(0, 2))
        freq = 220.0 * 2 ** (note / 12)
        n = min(length, len(out) - pos)
        t = np.arange(n) / rate
        wave_ = sum(a * np.sin(2 * np.pi * freq * k * t) for k, a in ((1, 1.0), (2, 0.4), (3, 0.2)))
        out[pos : pos + n] += wave_ * np.exp(-4 * t / (length / rate)) * 0.3
        pos += n
    return np.clip(out, -1, 1).astype(np.float32)


def noise(seed: int, duration: float = 1.5, rate: int = RATE) -> Samples:
    """White noise at moderate level."""
    rng = np.random.Generator(np.random.PCG64(seed))
    return (rng.uniform(-0.4, 0.4, int(duration * rate))).astype(np.float32)


def add_silence(
    samples: Samples, lead: float = 0.5, trail: float = 0.5, rate: int = RATE
) -> Samples:
    """Pad with silence in front and behind."""
    return np.concatenate(
        [np.zeros(int(lead * rate), np.float32), samples, np.zeros(int(trail * rate), np.float32)]
    )


def trim_samples(samples: Samples, start: float, duration: float, rate: int = RATE) -> Samples:
    """A slice of the signal."""
    a = int(start * rate)
    return samples[a : a + int(duration * rate)]


def write_wav(path: Path, samples: Samples, rate: int = RATE) -> Path:
    """16-bit mono PCM WAV using only the standard library."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())
    return path


def read_wav(path: Path) -> tuple[Samples, int]:
    """Samples as float32 in -1..1 and the sample rate."""
    with wave.open(str(path), "rb") as f:
        raw = f.readframes(f.getnframes())
        rate = f.getframerate()
    return (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767), rate


def encode(src_wav: Path, out: Path, bitrate: str | None = None) -> Path:
    """Encode a WAV to the format implied by ``out``'s suffix (optionally at a given bitrate)."""
    args = ["-i", str(src_wav), *_CODECS[out.suffix.lower()]]
    if bitrate is not None:
        args += ["-b:a", bitrate]
    return run_ffmpeg(args, out)


def retag(src: Path, out: Path, tags: dict[str, str]) -> Path:
    """Copy the audio and write new tags (title, artist, album ...)."""
    return run_ffmpeg_with_tags(["-i", str(src), "-c", "copy"], out, tags)


def strip_tags(src: Path, out: Path) -> Path:
    """Copy the audio without any tags."""
    return run_ffmpeg(["-i", str(src), "-c", "copy"], out)
