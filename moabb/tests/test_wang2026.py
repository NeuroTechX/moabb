"""Offline integrity tests for the Wang et al. (2026) dataset."""

import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import mne
import numpy as np
import pytest

import moabb.datasets as datasets
from moabb.datasets import wang2026
from moabb.datasets.metadata.schema import validate_metadata_against_dataset
from moabb.datasets.utils import _init_dataset, dataset_dict
from moabb.datasets.wang2026 import (
    _BCI2000_2D_EVENT_MAP,
    _BCI2000_LR_EVENT_MAP,
    _BCI2000_UD_EVENT_MAP,
    _CALIBRATION_ERROR,
    _CHANNELS,
    _EEGNET_EVENT_MAP,
    _EVENTS,
    _MANIFEST_FILENAME,
    _NOMINAL_TRIAL_SAMPLES,
    _SUBJECT_FILE_COUNTS,
    _TRIAL_GUARD_SAMPLES,
    Wang2026,
    _event_mapping,
    _make_raw,
    _path_sort_key,
    _RangeReader,
    _read_hdf5_trials,
    _strict_trial_label,
)


def _build_archive(group, specs, *, compression=zipfile.ZIP_DEFLATED):
    """Return a ZIP shaped like one released cohort archive."""
    rng = np.random.default_rng(0)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as zip_file:
        for source_subject, (n_files, size) in specs.items():
            subject_name = f"S{source_subject:03d}"
            names = [f"{group}/{subject_name}/{subject_name}_run0.mat"]
            for index in range(1, n_files):
                session, run = divmod(index - 1, 6)
                names.append(
                    f"{group}/{subject_name}/{subject_name}_sess"
                    f"{session + 1:02d}_run{run + 1:02d}.mat"
                )
            for member in names:
                payload = rng.integers(0, 256, size=size, dtype=np.uint8).tobytes()
                zip_file.writestr(member, payload)
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, content, headers, status_code=206):
        self._content = content
        self.headers = headers
        self.status_code = status_code
        self.closed = False
        self.iterated = False

    def iter_content(self, chunk_size):
        self.iterated = True
        for start in range(0, len(self._content), chunk_size):
            yield self._content[start : start + chunk_size]

    def close(self):
        self.closed = True


class _FakeServer:
    """Serve a byte payload with a strict HTTP Range implementation."""

    def __init__(self, payload):
        self.payload = payload
        self.bytes_served = 0
        self.requests = 0
        self.ranges = []
        self.urls = []

    def get(self, url, headers=None, **kwargs):
        assert kwargs["stream"] is True
        self.requests += 1
        self.urls.append(url)
        header = (headers or {}).get("Range")
        if header is None:
            raise AssertionError(f"non-range request issued for {url}")
        start_text, separator, end_text = header.removeprefix("bytes=").partition("-")
        assert separator == "-"
        start = int(start_text)
        end = len(self.payload) - 1 if end_text == "" else int(end_text)
        self.ranges.append((start, end))
        if start >= len(self.payload) or end < start:
            return _FakeResponse(
                b"", {"Content-Range": f"bytes */{len(self.payload)}"}, status_code=416
            )
        end = min(end, len(self.payload) - 1)
        chunk = self.payload[start : end + 1]
        self.bytes_served += len(chunk)
        return _FakeResponse(
            chunk, {"Content-Range": f"bytes {start}-{end}/{len(self.payload)}"}
        )


def _patch_transport(monkeypatch, payload, tmp_path=None):
    server = _FakeServer(payload)
    monkeypatch.setattr(wang2026, "requests", SimpleNamespace(get=server.get))
    if tmp_path is not None:
        monkeypatch.setattr(
            wang2026.dl, "get_dataset_path", lambda code, path: str(tmp_path)
        )
    return server


@pytest.fixture
def joint_learning_archive():
    counts = _SUBJECT_FILE_COUNTS["JointLearning"]
    return _build_archive(
        "JointLearning", {1: (counts[1], 512), 2: (counts[2], 32 * 1024)}
    )


def _write_fixed_mat(
    path,
    *,
    labels=(0, 1),
    task_type="EEGNet-style 1D control",
    samples=32,
    trial_info_labels=None,
    selected_channels=None,
    signal=None,
):
    labels = list(labels)
    if signal is None:
        signal = np.zeros((len(labels), len(_CHANNELS), samples), dtype=np.float64)
    with h5py.File(path, "w") as handle:
        run_data = handle.create_group("runData")
        meta = run_data.create_group("meta")
        meta.create_dataset("sampling_rate_hz", data=np.asarray([1000.0]))
        if task_type is not None:
            meta.create_dataset(
                "task_type", data=np.asarray(task_type, dtype=h5py.string_dtype())
            )
        channels = _CHANNELS if selected_channels is None else selected_channels
        meta.create_dataset(
            "selected_channels", data=np.asarray(channels, dtype=h5py.string_dtype())
        )
        run_data.create_dataset("trialSignal", data=signal)
        targets = np.repeat(np.asarray(labels, dtype=float)[:, None], samples, axis=1)
        run_data.create_dataset("trialTargetClass", data=targets)
        if trial_info_labels is not None:
            trial_info = run_data.create_group("trialInfo")
            trial_info.create_dataset(
                "target_label", data=np.asarray(trial_info_labels, dtype=float)
            )


def _write_variable_mat(
    path,
    *,
    labels=(1, 2),
    task_type="BCI2000 left/right control",
    samples=(32, 24),
    trial_info_labels=None,
):
    with h5py.File(path, "w") as handle:
        run_data = handle.create_group("runData")
        meta = run_data.create_group("meta")
        meta.create_dataset("sampling_rate_hz", data=np.asarray([1000.0]))
        meta.create_dataset(
            "task_type", data=np.asarray(task_type, dtype=h5py.string_dtype())
        )
        meta.create_dataset(
            "selected_channels", data=np.asarray(_CHANNELS, dtype=h5py.string_dtype())
        )
        signals = run_data.create_dataset(
            "trialSignal", shape=(len(labels), 1), dtype=h5py.ref_dtype
        )
        targets = run_data.create_dataset(
            "trialTargetCode", shape=(len(labels), 1), dtype=h5py.ref_dtype
        )
        for index, (label, n_samples) in enumerate(zip(labels, samples)):
            signal = handle.create_dataset(
                f"signal_{index}", data=np.full((len(_CHANNELS), n_samples), index + 1.0)
            )
            target = handle.create_dataset(
                f"target_{index}", data=np.full((1, n_samples), label, dtype=float)
            )
            signals[index, 0] = signal.ref
            targets[index, 0] = target.ref
        if trial_info_labels is not None:
            trial_info = run_data.create_group("trialInfo")
            trial_info.create_dataset(
                "target_label", data=np.asarray(trial_info_labels, dtype=float)
            )


# ---------------------------------------------------------------------------
# One public dataset identity and stable paper-first subject routing
# ---------------------------------------------------------------------------


def test_wang2026_is_the_only_public_dataset_identity():
    _init_dataset()
    assert datasets.Wang2026 is Wang2026
    assert dataset_dict["Wang2026"] is Wang2026
    for old_name in (
        "Wang2026JointLearning",
        "Wang2026TactileControl",
        "Wang2026EEGNetControl",
        "Wang2026BCI2000Control",
    ):
        assert not hasattr(datasets, old_name)
        assert old_name not in dataset_dict


def test_default_subject_mapping_is_global_unique_and_paper_first():
    dataset = Wang2026()
    assert dataset.subject_list == list(range(1, 40))
    assert dataset.all_subjects == list(range(1, 40))
    assert list(dataset.subject_mapping) == list(range(1, 40))

    expected = {}
    expected.update(
        {subject: ("JointLearning", f"S{subject:03d}") for subject in range(1, 16)}
    )
    expected.update(
        {subject: ("BCI2000Control", f"S{subject - 15:03d}") for subject in range(16, 24)}
    )
    expected.update(
        {subject: ("TactileControl", f"S{subject - 23:03d}") for subject in range(24, 32)}
    )
    expected.update(
        {subject: ("EEGNetControl", f"S{subject - 31:03d}") for subject in range(32, 40)}
    )
    assert dict(dataset.subject_mapping) == expected
    assert len(set(dataset.subject_mapping.values())) == 39
    with pytest.raises(TypeError):
        dataset.subject_mapping[1] = ("wrong", "S999")


@pytest.mark.parametrize(
    "group, expected_subjects",
    [
        ("all", list(range(1, 40))),
        ("joint_learning", list(range(1, 16))),
        ("bci2000_control", list(range(16, 24))),
        ("tactile_control", list(range(24, 32))),
        ("eegnet_control", list(range(32, 40))),
    ],
)
def test_group_filter_preserves_global_subject_ids(group, expected_subjects):
    dataset = Wang2026(group=group)
    assert dataset.subject_list == expected_subjects
    assert dataset.all_subjects == list(range(1, 40))
    assert dataset.code == "Wang2026"
    assert dataset.metadata.participants.n_subjects == 39
    assert not validate_metadata_against_dataset(dataset, dataset.metadata)


def test_group_filter_rejects_unknown_group_and_out_of_group_subject():
    with pytest.raises(ValueError, match="group must be one of"):
        Wang2026(group="four_datasets")
    with pytest.raises(ValueError, match="not in group"):
        Wang2026(group="joint_learning", subjects=[16])


# ---------------------------------------------------------------------------
# Defensive, coalesced range I/O
# ---------------------------------------------------------------------------


def test_range_reader_coalesces_sequential_small_reads(monkeypatch):
    payload = b"x" * (100 * 1024 * 1024)
    server = _patch_transport(monkeypatch, payload)
    reader = _RangeReader("https://example.invalid/file")

    reader.seek(16 * 1024 * 1024)
    chunks = [reader.read(64 * 1024) for _ in range(320)]
    assert sum(map(len, chunks)) == 20 * 1024 * 1024
    assert server.requests <= 4  # one probe plus three 8-MiB blocks
    assert server.bytes_served <= 24 * 1024 * 1024 + 1


def test_range_reader_seek_and_eof_contract(monkeypatch):
    payload = bytes(range(256))
    _patch_transport(monkeypatch, payload)
    reader = _RangeReader("https://example.invalid/file")

    assert reader.seek(10) == 10
    assert reader.seek(5, 1) == 15
    assert reader.seek(-4, 2) == 252
    assert reader.read(4) == payload[-4:]
    assert reader.read(1) == b""
    assert reader.read(0) == b""
    with pytest.raises(ValueError, match="before the start"):
        reader.seek(-1)
    with pytest.raises(ValueError, match="past end"):
        reader.seek(1, 2)
    with pytest.raises(ValueError, match="whence"):
        reader.seek(0, 99)


@pytest.mark.parametrize(
    "status, content_range, content, match",
    [
        (200, "bytes 0-0/100", b"x", "Expected HTTP 206"),
        (206, "not-a-range", b"x", "malformed Content-Range"),
        (206, "bytes 1-1/100", b"x", "does not match requested"),
        (206, "bytes 0-0/100", b"", "payload length"),
        (206, "bytes 0-0/100", b"xx", "payload length"),
        (416, "bytes */100", b"", "HTTP 416"),
    ],
)
def test_range_reader_rejects_invalid_probe_responses(
    monkeypatch, status, content_range, content, match
):
    response = _FakeResponse(
        content, {"Content-Range": content_range}, status_code=status
    )
    monkeypatch.setattr(
        wang2026, "requests", SimpleNamespace(get=lambda *args, **kwargs: response)
    )
    with pytest.raises(OSError, match=match):
        _RangeReader("https://example.invalid/file")
    assert response.closed
    if status == 200:
        assert not response.iterated


def test_range_reader_rejects_truncated_and_wrong_total_reads(monkeypatch):
    payload = b"x" * (9 * 1024 * 1024)
    server = _FakeServer(payload)

    def broken_get(url, headers=None, **kwargs):
        response = server.get(url, headers=headers, **kwargs)
        if server.requests == 2:
            response.headers["Content-Range"] = (
                f"bytes 0-{8 * 1024 * 1024 - 1}/{len(payload) + 1}"
            )
        return response

    monkeypatch.setattr(wang2026, "requests", SimpleNamespace(get=broken_get))
    reader = _RangeReader("https://example.invalid/file")
    with pytest.raises(OSError, match="total size changed"):
        reader.read(1)


def test_data_path_routes_global_subject_to_one_cohort(
    monkeypatch, tmp_path, joint_learning_archive
):
    monkeypatch.setattr(wang2026, "_RANGE_BLOCK_SIZE", 64 * 1024)
    server = _patch_transport(monkeypatch, joint_learning_archive, tmp_path)

    paths = Wang2026(group="joint_learning", subjects=[1]).data_path(1)

    assert len(paths) == _SUBJECT_FILE_COUNTS["JointLearning"][1]
    subject_dir = tmp_path / "MNE-wang2026-data/JointLearning/S001"
    assert all(str(subject_dir) in path for path in paths)
    assert not (tmp_path / "MNE-wang2026-data/JointLearning/S002").exists()
    assert (subject_dir / _MANIFEST_FILENAME).is_file()
    assert 0 < server.bytes_served < 0.35 * len(joint_learning_archive)


@pytest.mark.parametrize(
    "global_subject, group_filter, archive_group",
    [
        (1, "joint_learning", "JointLearning"),
        (16, "bci2000_control", "BCI2000Control"),
        (24, "tactile_control", "TactileControl"),
        (32, "eegnet_control", "EEGNetControl"),
    ],
)
def test_colliding_s001_routes_to_each_distinct_archive(
    monkeypatch, tmp_path, global_subject, group_filter, archive_group
):
    monkeypatch.setattr(wang2026, "_RANGE_BLOCK_SIZE", 64 * 1024)
    count = _SUBJECT_FILE_COUNTS[archive_group][1]
    payload = _build_archive(archive_group, {1: (count, 64)})
    server = _patch_transport(monkeypatch, payload, tmp_path)

    paths = Wang2026(group=group_filter, subjects=[global_subject]).data_path(
        global_subject
    )

    expected_dir = tmp_path / "MNE-wang2026-data" / archive_group / "S001"
    assert len(paths) == count
    assert all(str(expected_dir) in path for path in paths)
    expected_url = wang2026._FIGSHARE_FILE.format(
        file_id=wang2026._ARCHIVES[archive_group]["file_id"]
    )
    assert set(server.urls) == {expected_url}


def test_data_path_cache_manifest_detects_same_size_corruption(
    monkeypatch, tmp_path, joint_learning_archive
):
    monkeypatch.setattr(wang2026, "_RANGE_BLOCK_SIZE", 64 * 1024)
    server = _patch_transport(monkeypatch, joint_learning_archive, tmp_path)
    dataset = Wang2026(group="joint_learning", subjects=[1])
    paths = dataset.data_path(1)
    after_first = server.requests
    dataset.data_path(1)
    assert server.requests == after_first

    path = Path(paths[0])
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    dataset.data_path(1)
    assert server.requests > after_first
    manifest = json.loads(
        (
            tmp_path / "MNE-wang2026-data/JointLearning/S001" / _MANIFEST_FILENAME
        ).read_text()
    )
    assert manifest["archive"]["md5"] == wang2026._ARCHIVES["JointLearning"]["md5"]


def test_force_update_failure_preserves_previous_valid_cache(
    monkeypatch, tmp_path, joint_learning_archive
):
    monkeypatch.setattr(wang2026, "_RANGE_BLOCK_SIZE", 64 * 1024)
    _patch_transport(monkeypatch, joint_learning_archive, tmp_path)
    dataset = Wang2026(group="joint_learning", subjects=[1])
    paths = dataset.data_path(1)
    first_path = Path(paths[0])
    original = first_path.read_bytes()

    def interrupted_extract(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(wang2026, "safe_extract_zip", interrupted_extract)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        dataset.data_path(1, force_update=True)
    assert first_path.read_bytes() == original
    assert (first_path.parent / _MANIFEST_FILENAME).is_file()


def test_data_path_rejects_unknown_subject_before_network(monkeypatch, tmp_path):
    def unexpected_get(*args, **kwargs):
        raise AssertionError("network must not be touched")

    monkeypatch.setattr(wang2026, "requests", SimpleNamespace(get=unexpected_get))
    monkeypatch.setattr(wang2026.dl, "get_dataset_path", lambda code, path: str(tmp_path))
    with pytest.raises(ValueError, match="Invalid subject"):
        Wang2026().data_path(99)


def test_data_path_rejects_incomplete_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(wang2026, "_RANGE_BLOCK_SIZE", 64 * 1024)
    _patch_transport(monkeypatch, _build_archive("JointLearning", {1: (3, 64)}), tmp_path)
    with pytest.raises(RuntimeError, match="MAT files for S001"):
        Wang2026(group="joint_learning", subjects=[1]).data_path(1)


def test_data_path_rejects_member_crc_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(wang2026, "_RANGE_BLOCK_SIZE", 64 * 1024)
    count = _SUBJECT_FILE_COUNTS["JointLearning"][1]
    payload = bytearray(
        _build_archive("JointLearning", {1: (count, 64)}, compression=zipfile.ZIP_STORED)
    )
    with zipfile.ZipFile(io.BytesIO(payload)) as zip_file:
        member = zip_file.infolist()[0]
    payload_start = (
        member.header_offset
        + 30
        + len(member.filename.encode("utf-8"))
        + len(member.extra)
    )
    payload[payload_start] ^= 1
    _patch_transport(monkeypatch, bytes(payload), tmp_path)

    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        Wang2026(group="joint_learning", subjects=[1]).data_path(1)


# ---------------------------------------------------------------------------
# Authoritative task routing and strict labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task_type, target_name, filename, expected",
    [
        (
            "EEGNet-style 1D control",
            "trialTargetClass",
            "S001_sess01_run01.mat",
            _EEGNET_EVENT_MAP,
        ),
        (
            "BCI2000 left/right control",
            "trialTargetCode",
            "S001_sess01_run01.mat",
            _BCI2000_LR_EVENT_MAP,
        ),
        (
            "BCI2000 up/down control",
            "trialTargetCode",
            "S001_sess03_run01UD.mat",
            _BCI2000_UD_EVENT_MAP,
        ),
        (
            "BCI2000 2D control",
            "trialTargetCode",
            "S001_sess04_run01.mat",
            _BCI2000_2D_EVENT_MAP,
        ),
    ],
)
def test_event_mapping_uses_authoritative_task_type(
    task_type, target_name, filename, expected
):
    assert _event_mapping(filename, target_name, task_type) is expected


def test_event_mapping_rejects_metadata_filename_disagreement():
    with pytest.raises(ValueError, match="disagrees"):
        _event_mapping(
            "S001_sess03_run01UD.mat", "trialTargetCode", "BCI2000 left/right control"
        )


def test_event_mapping_rejects_missing_authoritative_task_type():
    with pytest.raises(ValueError, match="no runData.meta.task_type"):
        _event_mapping("S001_sess01_run01.mat", "trialTargetCode")


@pytest.mark.parametrize(
    "track, allowed, expected", [([0, 0, 0], {0, 1, 2, 3}, 0), ([2, 2, 2], {1, 2}, 2)]
)
def test_strict_trial_label_accepts_one_allowed_label(track, allowed, expected):
    assert _strict_trial_label(track, allowed) == expected


@pytest.mark.parametrize(
    "track, allowed, match",
    [
        ([], {1, 2}, "empty"),
        ([np.nan], {1, 2}, "non-finite"),
        ([1.5], {1, 2}, "non-integer"),
        ([0], {1, 2}, "not valid"),
        ([1, 2], {1, 2}, "mixed"),
    ],
)
def test_strict_trial_label_rejects_ambiguous_tracks(track, allowed, match):
    with pytest.raises(ValueError, match=match):
        _strict_trial_label(track, allowed)


def test_read_hdf5_fixed_trials_maps_and_cross_checks_labels(tmp_path):
    path = tmp_path / "S001_sess01_run01.mat"
    _write_fixed_mat(path, labels=(0, 3), trial_info_labels=(0, 3))
    trials, event_codes = _read_hdf5_trials(path)
    assert len(trials) == 2
    assert event_codes == [_EVENTS["left_hand"], _EVENTS["rest"]]


def test_read_hdf5_variable_trials_maps_labels(tmp_path):
    path = tmp_path / "S001_sess01_run01.mat"
    _write_variable_mat(path, labels=(1, 2), trial_info_labels=(1, 2))
    trials, event_codes = _read_hdf5_trials(path)
    assert [trial.shape[1] for trial in trials] == [32, 24]
    assert event_codes == [_EVENTS["right_hand"], _EVENTS["left_hand"]]


def test_read_hdf5_rejects_trial_info_disagreement(tmp_path):
    path = tmp_path / "S001_sess01_run01.mat"
    _write_fixed_mat(path, labels=(0, 1), trial_info_labels=(0, 2))
    with pytest.raises(ValueError, match="trialInfo.target_label disagrees"):
        _read_hdf5_trials(path)


def test_read_hdf5_rejects_channel_order_disagreement(tmp_path):
    path = tmp_path / "S001_sess01_run01.mat"
    channels = list(_CHANNELS)
    channels[0], channels[1] = channels[1], channels[0]
    _write_fixed_mat(path, selected_channels=channels)
    with pytest.raises(ValueError, match="channel order"):
        _read_hdf5_trials(path)


# ---------------------------------------------------------------------------
# Reconstruction policy and explicit calibration blocker
# ---------------------------------------------------------------------------


def test_raw_construction_is_blocked_without_authoritative_calibration(tmp_path):
    with pytest.raises(RuntimeError, match="physical calibration"):
        _make_raw(tmp_path / "does-not-need-to-exist.mat")
    assert "10.1184/R1/32293995.v1" in _CALIBRATION_ERROR


def test_get_data_hits_calibration_guard_before_download(monkeypatch):
    dataset = Wang2026(subjects=[1])
    monkeypatch.setattr(
        dataset,
        "data_path",
        lambda *args, **kwargs: pytest.fail("download must not start"),
    )
    with pytest.raises(RuntimeError, match="physical calibration"):
        dataset._get_single_subject_data(1)


def test_reconstruction_centers_trials_uses_exact_epoch_and_marks_padding(tmp_path):
    path = tmp_path / "S001_sess01_run01.mat"
    # HDF5 fixed arrays cannot vary trial length, so use the reference/cell layout.
    _write_variable_mat(
        path,
        labels=(1, 2),
        samples=(_NOMINAL_TRIAL_SAMPLES + 1, 1000),
        trial_info_labels=(1, 2),
    )
    with h5py.File(path, "r+") as handle:
        first = handle["signal_0"]
        first[...] = 10.0
        first[0, :_NOMINAL_TRIAL_SAMPLES] = 10.0 + np.linspace(
            -1.0, 1.0, _NOMINAL_TRIAL_SAMPLES
        )
        # This inclusive-endpoint sample must neither shift the next event nor
        # influence centering of the exact [0, 4.999] s analysis window.
        first[0, _NOMINAL_TRIAL_SAMPLES] = 1e6

    raw = _make_raw(path, _eeg_scale=1.0)
    events = mne.find_events(raw, shortest_event=1, verbose=False)
    assert np.diff(events[:, 0]).tolist() == [
        _NOMINAL_TRIAL_SAMPLES + _TRIAL_GUARD_SAMPLES
    ]
    assert Wang2026().interval == [0.0, 4.999]
    first_epoch = raw.get_data(
        picks=[0], start=events[0, 0], stop=events[0, 0] + _NOMINAL_TRIAL_SAMPLES
    )
    assert first_epoch.mean() == pytest.approx(0.0, abs=1e-12)
    descriptions = list(raw.annotations.description)
    assert "BAD_WANG2026_ZERO_PADDED" in descriptions
    assert descriptions.count("BAD_ACQ_SKIP") == 2
    padding_index = descriptions.index("BAD_WANG2026_ZERO_PADDED")
    assert raw.annotations.duration[padding_index] == pytest.approx(4.0)
    guard_indices = [
        index
        for index, description in enumerate(descriptions)
        if description == "BAD_ACQ_SKIP"
    ]
    assert raw.annotations.duration[guard_indices] == pytest.approx([2.0, 2.0])
    assert raw.info["line_freq"] == 60.0
    epochs = mne.Epochs(
        raw, events, tmin=0.0, tmax=4.999, baseline=None, preload=True, verbose=False
    )
    assert len(epochs) == 1


def test_guard_prevents_filter_leakage_into_neighboring_trial(tmp_path):
    impulse_path = tmp_path / "S001_sess01_run01.mat"
    zero_path = tmp_path / "S002_sess01_run01.mat"
    for path in (impulse_path, zero_path):
        _write_variable_mat(
            path,
            labels=(1, 2),
            samples=(_NOMINAL_TRIAL_SAMPLES, _NOMINAL_TRIAL_SAMPLES),
            trial_info_labels=(1, 2),
        )
    with h5py.File(impulse_path, "r+") as handle:
        handle["signal_0"][0, _NOMINAL_TRIAL_SAMPLES // 2] = 10.0

    impulse = _make_raw(impulse_path, _eeg_scale=1.0).pick(["Fp1"])
    control = _make_raw(zero_path, _eeg_scale=1.0).pick(["Fp1"])
    event_sample = mne.find_events(
        _make_raw(impulse_path, _eeg_scale=1.0), shortest_event=1, verbose=False
    )[1, 0]
    impulse.filter(4.0, 40.0, verbose=False)
    control.filter(4.0, 40.0, verbose=False)
    difference = impulse.get_data(
        start=event_sample, stop=event_sample + _NOMINAL_TRIAL_SAMPLES
    ) - control.get_data(start=event_sample, stop=event_sample + _NOMINAL_TRIAL_SAMPLES)
    assert np.max(np.abs(difference)) < 1e-12


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


def test_channel_names_resolve_against_declared_montage():
    montage = set(mne.channels.make_standard_montage("standard_1005").ch_names)
    assert len(_CHANNELS) == 62
    assert [name for name in _CHANNELS if name not in montage] == ["CB1", "CB2"]
