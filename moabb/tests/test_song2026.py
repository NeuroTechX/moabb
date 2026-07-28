"""Focused regression tests for the Song2026 raw-data semantics."""

import struct

import pytest

from moabb.datasets.song2026 import Song2026, _label_from_marker, _parse_nef


def _event_frame(onset, marker, *, valid=True):
    frame = bytearray(1005)
    frame[0] = int(valid)
    struct.pack_into("<I", frame, 5, onset)
    encoded = marker.encode("ascii") + b"\x00\x16"
    frame[23 : 23 + len(encoded)] = encoded
    return frame


def test_parse_nef_uses_the_real_1005_byte_stride(tmp_path):
    header = bytearray(64)
    header[:2] = b"\xfe\xff"
    struct.pack_into("<I", header, 29, 3)
    path = tmp_path / "neuracle.nef"
    path.write_bytes(
        header
        + _event_frame(1_000, "23")
        + _event_frame(2_000, "20", valid=False)
        + _event_frame(3_000, "53")
    )

    assert _parse_nef(path) == [(1_000, "23"), (3_000, "53")]


def test_parse_nef_rejects_truncated_event_data(tmp_path):
    header = bytearray(64)
    header[:2] = b"\xfe\xff"
    struct.pack_into("<I", header, 29, 1)
    path = tmp_path / "neuracle.nef"
    path.write_bytes(header)

    with pytest.raises(ValueError, match="Truncated Neuracle event data"):
        _parse_nef(path)


def test_only_published_motor_attempt_block_is_labeled():
    expected = {
        "23": "fist_clench",
        "33": "pinch_grip",
        "43": "wrist_lift",
        "53": "elbow_flexion",
    }
    assert {marker: _label_from_marker(marker) for marker in expected} == expected
    assert all(
        _label_from_marker(marker) is None
        for marker in ("21", "31", "41", "51", "20", "14")
    )


def test_raw_release_exposes_all_fifty_subjects():
    dataset = Song2026()

    assert dataset.subject_list == list(range(1, 51))
    assert dataset.metadata.participants.n_subjects == 50
