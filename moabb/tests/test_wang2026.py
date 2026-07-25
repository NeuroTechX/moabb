"""Offline tests for the Wang2026 loaders.

The release ships four monolithic per-arm archives (8.7-25.3 GB each), so
``data_path`` pulls only one subject's members out of the *remote* ZIP over HTTP
range requests. That path is exercised here against a synthetic archive served by
a fake transport, so the wiring is verified without any network access. The label
conventions are pinned as well, because getting them wrong silently swaps classes
instead of raising.
"""

import io
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from moabb.datasets import wang2026
from moabb.datasets.wang2026 import (
    _BCI2000_2D_EVENT_MAP,
    _BCI2000_LR_EVENT_MAP,
    _BCI2000_UD_EVENT_MAP,
    _CHANNELS,
    _EEGNET_EVENT_MAP,
    _EVENTS,
    _SUBJECT_FILE_COUNTS,
    Wang2026JointLearning,
    _event_mapping,
    _mode_label,
    _path_sort_key,
    _RangeReader,
)


# ---------------------------------------------------------------------------
# A synthetic group archive, served over a fake range-request transport
# ---------------------------------------------------------------------------


def _build_archive(group, specs):
    """Return the bytes of a ZIP shaped like a released group archive.

    ``specs`` maps a subject number to ``(n_files, bytes_per_file)``. The payload
    is incompressible so that the stored sizes stay proportional to the requested
    ones, which is what makes the "only one subject crossed the wire" assertion
    below measure something.
    """
    rng = np.random.default_rng(0)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for subject, (n_files, size) in specs.items():
            name = f"S{subject:03d}"
            names = [f"{group}/{name}/{name}_run0.mat"]
            for index in range(1, n_files):
                session, run = divmod(index - 1, 6)
                names.append(
                    f"{group}/{name}/{name}_sess{session + 1:02d}_run{run + 1:02d}.mat"
                )
            for member in names:
                payload = rng.integers(0, 256, size=size, dtype=np.uint8).tobytes()
                zip_file.writestr(member, payload)
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, content, headers):
        self.content = content
        self.headers = headers

    def raise_for_status(self):
        return None


class _FakeServer:
    """Serve ``payload`` honouring the ``Range`` header, and count the traffic."""

    def __init__(self, payload):
        self.payload = payload
        self.bytes_served = 0
        self.requests = 0

    def get(self, url, headers=None, **kwargs):
        self.requests += 1
        header = (headers or {}).get("Range")
        if header is None:  # a whole-file download is exactly what we forbid
            raise AssertionError(f"non-range request issued for {url}")
        start, _, end = header.removeprefix("bytes=").partition("-")
        start = int(start)
        end = len(self.payload) - 1 if end == "" else int(end)
        chunk = self.payload[start : end + 1]
        self.bytes_served += len(chunk)
        return _FakeResponse(
            chunk, {"Content-Range": f"bytes {start}-{end}/{len(self.payload)}"}
        )


def _patch_transport(monkeypatch, payload, tmp_path=None):
    """Swap the module's ``requests`` for a fake server; optionally redirect I/O."""
    server = _FakeServer(payload)
    monkeypatch.setattr(wang2026, "requests", SimpleNamespace(get=server.get))
    if tmp_path is not None:
        monkeypatch.setattr(
            wang2026.dl, "get_dataset_path", lambda code, path: str(tmp_path)
        )
    return server


@pytest.fixture
def joint_learning_archive():
    """A JointLearning archive holding subject 1 plus a much larger subject 2.

    Subject 2 stands in for the other 14 subjects of the real archive: it is the
    bulk of the bytes, and none of them should be transferred to load subject 1.
    """
    counts = _SUBJECT_FILE_COUNTS["JointLearning"]
    return _build_archive(
        "JointLearning", {1: (counts[1], 512), 2: (counts[2], 32 * 1024)}
    )


# ---------------------------------------------------------------------------
# The graft: range-request extraction of a single subject
# ---------------------------------------------------------------------------


def test_range_reader_sizes_and_slices_a_remote_file(monkeypatch):
    payload = bytes(range(256))
    server = _patch_transport(monkeypatch, payload)

    reader = _RangeReader("https://example.invalid/file")
    assert reader.size == len(payload)
    assert server.bytes_served == 1  # the bytes=0-0 size probe

    reader.seek(-4, 2)  # zipfile locates the end-of-central-directory this way
    assert reader.tell() == len(payload) - 4
    assert reader.read(4) == payload[-4:]
    assert reader.seekable() and reader.readable()


def test_range_reader_rejects_a_server_without_range_support(monkeypatch):
    monkeypatch.setattr(
        wang2026, "requests", SimpleNamespace(get=lambda *a, **k: _FakeResponse(b"", {}))
    )
    with pytest.raises(OSError, match="does not support range requests"):
        _RangeReader("https://example.invalid/file")


def test_data_path_fetches_only_the_requested_subject(
    monkeypatch, tmp_path, joint_learning_archive
):
    """The graft must transfer one subject, not the whole group archive."""
    server = _patch_transport(monkeypatch, joint_learning_archive, tmp_path)

    paths = Wang2026JointLearning(subjects=[1]).data_path(1)

    expected = _SUBJECT_FILE_COUNTS["JointLearning"][1]
    assert len(paths) == expected
    assert all(path.endswith(".mat") for path in paths)
    # Subject 2 lives in the same archive and must not have been written out.
    assert not (tmp_path / "MNE-wang2026jointlearning-data/JointLearning/S002").exists()
    # The point of the graft: subject 2's bytes never crossed the wire, and what
    # did get there arrived by range request (a plain GET raises in _FakeServer).
    # A whole-archive download would land at 100% here.
    assert 0 < server.bytes_served < 0.2 * len(joint_learning_archive)
    assert server.requests > 1


def test_data_path_reuses_the_local_cache(monkeypatch, tmp_path, joint_learning_archive):
    """A second call must not touch the transport at all."""
    server = _patch_transport(monkeypatch, joint_learning_archive, tmp_path)
    dataset = Wang2026JointLearning(subjects=[1])
    dataset.data_path(1)
    after_first = server.requests
    assert after_first > 0
    dataset.data_path(1)
    assert server.requests == after_first


def test_data_path_rejects_an_incomplete_archive(monkeypatch, tmp_path):
    """A member-count mismatch must fail loudly rather than load partial data."""
    _patch_transport(monkeypatch, _build_archive("JointLearning", {1: (3, 64)}), tmp_path)
    with pytest.raises(RuntimeError, match="MAT files for S001"):
        Wang2026JointLearning(subjects=[1]).data_path(1)


def test_data_path_rejects_an_unknown_subject(monkeypatch, tmp_path):
    _patch_transport(monkeypatch, b"", tmp_path)
    with pytest.raises(ValueError, match="Invalid subject"):
        Wang2026JointLearning().data_path(99)


# ---------------------------------------------------------------------------
# Label conventions and filename parsing
# ---------------------------------------------------------------------------


def test_label_maps_land_on_declared_events():
    valid = set(_EVENTS.values())
    for mapping in (
        _EEGNET_EVENT_MAP,
        _BCI2000_LR_EVENT_MAP,
        _BCI2000_UD_EVENT_MAP,
        _BCI2000_2D_EVENT_MAP,
    ):
        assert set(mapping.values()) <= valid


def test_bci2000_and_eegnet_disagree_on_raw_label_2():
    """Under BCI2000 raw 2 is left hand; under EEGNet it is bilateral."""
    assert _BCI2000_LR_EVENT_MAP[2] == _EVENTS["left_hand"]
    assert _EEGNET_EVENT_MAP[2] == _EVENTS["hands"]


def test_bci2000_ud_runs_are_up_down_not_left_right():
    assert _BCI2000_UD_EVENT_MAP == {1: _EVENTS["hands"], 2: _EVENTS["rest"]}
    assert _BCI2000_LR_EVENT_MAP == {1: _EVENTS["right_hand"], 2: _EVENTS["left_hand"]}


def test_event_mapping_selects_by_field_then_filename():
    assert (
        _event_mapping("S001_sess01_run01.mat", "trialTargetClass") is _EEGNET_EVENT_MAP
    )
    assert (
        _event_mapping("S001_sess03_run04UD.mat", "trialTargetCode")
        is _BCI2000_UD_EVENT_MAP
    )
    assert (
        _event_mapping("S001_sess04_run01.mat", "trialTargetCode")
        is _BCI2000_2D_EVENT_MAP
    )
    assert (
        _event_mapping("S001_sess01_run01.mat", "trialTargetCode")
        is _BCI2000_LR_EVENT_MAP
    )


@pytest.mark.parametrize(
    "track, expected",
    [
        ([2, 2, 2], 2),
        ([np.nan, 3, 3], 3),
        # Documents the current mode-of-all-samples rule: a majority-zero
        # BCI2000 target track collapses to 0, which no BCI2000 map contains.
        ([0, 0, 0, 0, 2, 2], 0),
        ([0, 0, 0, 0], 0),
    ],
)
def test_mode_label_collapses_a_target_track(track, expected):
    assert _mode_label(np.asarray(track, dtype=float)) == expected


@pytest.mark.parametrize("track", [[], [np.nan], [1.5, 1.5]])
def test_mode_label_rejects_unusable_tracks(track):
    with pytest.raises(ValueError):
        _mode_label(np.asarray(track, dtype=float))


def test_path_sort_key_puts_baseline_first_and_lr_before_ud():
    names = [
        "S001_sess03_run01UD.mat",
        "S001_sess03_run01.mat",
        "S001_sess01_run02.mat",
        "S001_run0.mat",
    ]
    assert sorted(names, key=_path_sort_key) == [
        "S001_run0.mat",
        "S001_sess01_run02.mat",
        "S001_sess03_run01.mat",
        "S001_sess03_run01UD.mat",
    ]


def test_path_sort_key_rejects_an_unexpected_filename():
    with pytest.raises(ValueError, match="Unexpected Wang2026 filename"):
        _path_sort_key("S001_block1.mat")


def test_channel_names_resolve_against_the_declared_montage():
    from mne.channels import make_standard_montage

    montage = set(make_standard_montage("standard_1005").ch_names)
    assert len(_CHANNELS) == 62
    assert [name for name in _CHANNELS if name not in montage] == ["CB1", "CB2"]
