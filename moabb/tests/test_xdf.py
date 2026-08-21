"""Tests for moabb.datasets._xdf, the built-in minimal XDF reader."""

import struct

import numpy as np

from moabb.datasets._xdf import read_xdf


def _chunk(tag, body):
    length = len(body) + 2
    return bytes([4]) + struct.pack("<I", length) + struct.pack("<H", tag) + body


def _write_xdf(path):
    eeg_xml = (
        b"<info><name>EEG</name><type>EEG</type><channel_count>2</channel_count>"
        b"<nominal_srate>500</nominal_srate>"
        b"<channel_format>float32</channel_format></info>"
    )
    mark_xml = (
        b"<info><name>Game</name><type>Markers</type><channel_count>1"
        b"</channel_count><nominal_srate>0</nominal_srate>"
        b"<channel_format>string</channel_format></info>"
    )
    # non-conforming footer, as AguileraRodriguez2025's recorder writes it:
    # sample_count is a float string, which crashes pyxdf 1.17.5
    footer_xml = b"<info><sample_count>4.0</sample_count></info>"

    eeg_samples = struct.pack("<I", 1) + bytes([1, 4])  # stream 1, 4 samples
    values = np.arange(8, dtype="<f4").reshape(4, 2)
    for i, row in enumerate(values):
        eeg_samples += bytes([8]) + struct.pack("<d", 10.0 + i / 500)
        eeg_samples += row.tobytes()

    marker_samples = struct.pack("<I", 2) + bytes([1, 2])  # stream 2, 2 samples
    for stamp, text in ((10.004, b"AVANZAR"), (10.006, b"Spoken AVANZAR")):
        marker_samples += bytes([8]) + struct.pack("<d", stamp)
        marker_samples += bytes([1, len(text)]) + text

    blob = b"XDF:"
    blob += _chunk(1, b"<info><version>1.0</version></info>")
    blob += _chunk(2, struct.pack("<I", 1) + eeg_xml)
    blob += _chunk(2, struct.pack("<I", 2) + mark_xml)
    blob += _chunk(3, eeg_samples)
    blob += _chunk(3, marker_samples)
    blob += _chunk(4, struct.pack("<I", 1) + struct.pack("<dd", 10.0, 0.001))
    blob += _chunk(6, struct.pack("<I", 1) + footer_xml)
    path.write_bytes(blob)


def test_read_xdf_numeric_string_and_quirky_footer(tmp_path):
    path = tmp_path / "rec.xdf"
    _write_xdf(path)

    streams = read_xdf(str(path))

    eeg = streams[1]
    assert eeg["info"]["type"] == "EEG"
    np.testing.assert_array_equal(eeg["series"], np.arange(8, dtype="<f4").reshape(4, 2))
    np.testing.assert_allclose(eeg["stamps"], 10.0 + np.arange(4) / 500)
    assert eeg["clock_offsets"] == [(10.0, 0.001)]

    markers = streams[2]
    assert [row[0] for row in markers["series"]] == ["AVANZAR", "Spoken AVANZAR"]
    np.testing.assert_allclose(markers["stamps"], [10.004, 10.006])
