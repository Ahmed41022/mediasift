"""EXIF parsing without files: dates, offsets, GPS, text, orientation."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pytest

from mediasift.readers.image import (
    ExifFields,
    display_size,
    estimate_jpeg_quality,
    parse_exif,
    parse_exif_datetime,
)
from tests.readers.helpers import FakeExif

ORIGINAL, DIGITIZED, DATETIME = 0x9003, 0x9004, 0x0132


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2021:07:04 10:30:00", datetime(2021, 7, 4, 10, 30, 0)),
        ("2021-07-04 10:30:00", datetime(2021, 7, 4, 10, 30, 0)),
        ("2021:07:04 10:30:00\x00", datetime(2021, 7, 4, 10, 30, 0)),
        (b"2021:07:04 10:30:00\x00", datetime(2021, 7, 4, 10, 30, 0)),
        ("  2021:07:04 10:30:00 ", datetime(2021, 7, 4, 10, 30, 0)),
        ("2021:07:04 10:30:00.250", datetime(2021, 7, 4, 10, 30, 0)),
        ("0000:00:00 00:00:00", None),
        ("2021:00:00 00:00:00", None),
        ("2021:13:40 25:61:61", None),
        ("2021:02:30 10:00:00", None),
        ("1800:01:01 00:00:00", None),
        ("    :  :     :  :  ", None),
        ("not a date", None),
        ("", None),
        (None, None),
        (20210704, None),
    ],
)
def test_datetime_parsing(text: Any, expected: datetime | None) -> None:
    assert parse_exif_datetime(text) == expected


def test_result_is_naive() -> None:
    parsed = parse_exif_datetime("2021:07:04 10:30:00")
    assert parsed is not None and parsed.tzinfo is None


def test_date_priority_original_then_digitized_then_datetime() -> None:
    exif = FakeExif(
        ifd0={DATETIME: "2020:03:03 03:03:03"},
        exif={ORIGINAL: "2021:01:01 01:01:01", DIGITIZED: "2022:02:02 02:02:02"},
    )
    assert parse_exif(exif).taken_at == datetime(2021, 1, 1, 1, 1, 1)
    exif = FakeExif(
        ifd0={DATETIME: "2020:03:03 03:03:03"},
        exif={ORIGINAL: "0000:00:00 00:00:00", DIGITIZED: "2022:02:02 02:02:02"},
    )
    assert parse_exif(exif).taken_at == datetime(2022, 2, 2, 2, 2, 2)
    exif = FakeExif(
        ifd0={DATETIME: "2020:03:03 03:03:03"},
        exif={ORIGINAL: "0000:00:00 00:00:00", DIGITIZED: "garbage"},
    )
    assert parse_exif(exif).taken_at == datetime(2020, 3, 3, 3, 3, 3)
    exif = FakeExif(ifd0={DATETIME: "0000:00:00 00:00:00"}, exif={ORIGINAL: "  "})
    assert parse_exif(exif).taken_at is None


def test_utc_offset_belongs_to_the_date_that_was_used() -> None:
    exif = FakeExif(
        exif={
            ORIGINAL: "0000:00:00 00:00:00",
            0x9011: "+09:00",
            DIGITIZED: "2022:02:02 02:02:02",
            0x9012: "-05:30",
        }
    )
    fields = parse_exif(exif)
    assert fields.taken_at == datetime(2022, 2, 2, 2, 2, 2)
    assert fields.utc_offset == "-05:30"


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        ("+02:00", "+02:00"),
        ("-0530", "-05:30"),
        (b"+01:00\x00", "+01:00"),
        ("+25:00", None),
        ("+02:75", None),
        ("Z", None),
        ("", None),
        (5, None),
    ],
)
def test_utc_offset(offset: Any, expected: str | None) -> None:
    exif = FakeExif(exif={ORIGINAL: "2021:07:04 10:30:00", 0x9011: offset})
    assert parse_exif(exif).utc_offset == expected


def test_make_and_model_lose_nuls_and_blanks() -> None:
    exif = FakeExif(ifd0={0x010F: "Canon\x00\x00\x00", 0x0110: b"  EOS 5D \x00", 0x0131: "  "})
    fields = parse_exif(exif)
    assert (fields.make, fields.model, fields.software) == ("Canon", "EOS 5D", None)
    assert parse_exif(FakeExif(ifd0={0x010F: "\x00\x00", 0x0110: 12})).make is None


@pytest.mark.parametrize(("value", "expected"), [(1, 1), (8, 8), (0, None), (9, None)])
def test_orientation_must_be_1_to_8(value: Any, expected: int | None) -> None:
    assert parse_exif(FakeExif(ifd0={0x0112: value})).orientation == expected
    assert parse_exif(FakeExif(ifd0={0x0112: "6"})).orientation is None


def _gps(lat: Any, lat_ref: Any, lon: Any, lon_ref: Any) -> FakeExif:
    return FakeExif(gps={1: lat_ref, 2: lat, 3: lon_ref, 4: lon})


@pytest.mark.parametrize(
    ("lat", "lat_ref", "lon", "lon_ref", "expected"),
    [
        ((40.0, 30.0, 0.0), "N", (74.0, 0.0, 36.0), "W", (40.5, -74.01)),
        ((33.0, 51.0, 0.0), "S", (151.0, 12.0, 0.0), "E", (-33.85, 151.2)),
        ((40.5,), "N", (10.25,), "E", (40.5, 10.25)),
        ((40.0, 30.0), b"N\x00", (10.0, 0.0, 0.0), "e", (40.5, 10.0)),
        ((90.0, 0.0, 0.0), "N", (180.0, 0.0, 0.0), "E", (90.0, 180.0)),
        ((0.0, 0.0, 0.0), "N", (0.0, 0.0, 0.0), "E", None),  # "no fix" as written by cameras
        ((0.0, 0.0, 0.0), "S", (10.0, 0.0, 0.0), "E", (0.0, 10.0)),
        ((91.0, 0.0, 0.0), "N", (10.0, 0.0, 0.0), "E", None),
        ((10.0, 0.0, 0.0), "N", (181.0, 0.0, 0.0), "E", None),
        ((90.0, 30.0, 0.0), "N", (10.0, 0.0, 0.0), "E", None),
        ((10.0, 75.0, 0.0), "N", (10.0, 0.0, 0.0), "E", None),
        ((10.0, 59.0, 60.0), "N", (10.0, 0.0, 0.0), "E", (11.0, 10.0)),  # writers that round up
        ((10.0, 0.0, 61.0), "N", (10.0, 0.0, 0.0), "E", None),
        ((-10.0, 0.0, 0.0), "N", (10.0, 0.0, 0.0), "E", None),
        ((10.0, 0.0, 0.0), "X", (10.0, 0.0, 0.0), "E", None),
        ((10.0, 0.0, 0.0), None, (10.0, 0.0, 0.0), "E", None),
        ((10.0, 0.0, 0.0), "N", (10.0, 0.0, 0.0), "N", None),
        ((math.nan, 0.0, 0.0), "N", (10.0, 0.0, 0.0), "E", None),  # zero denominator
        ((math.inf, 0.0, 0.0), "N", (10.0, 0.0, 0.0), "E", None),
        (("a", "b", "c"), "N", (10.0, 0.0, 0.0), "E", None),
        ((10.0, 0.0, 0.0, 0.0), "N", (10.0, 0.0, 0.0), "E", None),
        ((), "N", (10.0, 0.0, 0.0), "E", None),
        (None, "N", (10.0, 0.0, 0.0), "E", None),
    ],
)
def test_gps(lat: Any, lat_ref: Any, lon: Any, lon_ref: Any, expected: Any) -> None:
    got = parse_exif(_gps(lat, lat_ref, lon, lon_ref)).gps
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected, abs=1e-6)


def test_gps_with_a_missing_half_is_unknown() -> None:
    assert parse_exif(FakeExif(gps={1: "N", 2: (10.0, 0.0, 0.0)})).gps is None
    assert parse_exif(FakeExif()).gps is None


def test_one_bad_field_does_not_spoil_the_others() -> None:
    exif = FakeExif(
        ifd0={0x010F: "Canon", 0x0112: 6},
        exif={ORIGINAL: "2021:07:04 10:30:00"},
        gps={1: "N", 2: (math.nan,), 3: "E", 4: (1.0,)},
    )
    fields = parse_exif(exif)
    assert fields.make == "Canon" and fields.orientation == 6
    assert fields.taken_at == datetime(2021, 7, 4, 10, 30) and fields.gps is None


class _Exploding:
    def __contains__(self, tag: object, /) -> bool:
        raise RuntimeError("boom")

    def get(self, tag: int, default: Any = None, /) -> Any:
        raise RuntimeError("boom")

    def get_ifd(self, tag: int, /) -> Any:
        raise struct_error()


def struct_error() -> Exception:
    import struct

    return struct.error("bad offset")


def test_parsing_never_raises() -> None:
    assert parse_exif(_Exploding()) == ExifFields()  # type: ignore[arg-type]
    assert parse_exif(None) == ExifFields()


def test_has_exif_follows_the_tags() -> None:
    assert parse_exif(FakeExif()).has_exif is False
    assert parse_exif(FakeExif(ifd0={0x010F: "x"})).has_exif is True


@pytest.mark.parametrize(
    ("orientation", "expected"),
    [(None, (4, 3)), (1, (4, 3)), (4, (4, 3)), (5, (3, 4)), (6, (3, 4)), (8, (3, 4))],
)
def test_display_size(orientation: int | None, expected: tuple[int, int]) -> None:
    assert display_size((4, 3), orientation) == expected


def test_quality_estimate_needs_a_full_table() -> None:
    assert estimate_jpeg_quality(None) is None
    assert estimate_jpeg_quality({}) is None
    assert estimate_jpeg_quality({0: [1, 2, 3]}) is None
    assert estimate_jpeg_quality({1: [16] * 64}) is None
