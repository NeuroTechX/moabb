"""Offline integrity tests for the Wang et al. (2026) dataset."""

import io
import zipfile
from types import SimpleNamespace

import h5py
import mne
import numpy as np
import pytest

import moabb.datasets as datasets
from moabb.datasets import wang2026
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
    _RangeReader,
    _read_hdf5_trials,
    _strict_trial_label,
)


_MAT_NAME = "S001_sess01_run01.mat"
_SMALL_RANGE_BLOCK = 64 * 1024
_EEGNET_TASK = "EEGNet-style 1D control"
_BCI_LR_TASK = "BCI2000 left/right control"
_BCI_UD_TASK = "BCI2000 up/down control"
_BCI_2D_TASK = "BCI2000 2D control"
_CLASS_TARGET = "trialTargetClass"
_CODE_TARGET = "trialTargetCode"
_COHORTS = (
    ("joint_learning", "JointLearning", range(1, 16)),
    ("bci2000_control", "BCI2000Control", range(16, 24)),
    ("tactile_control", "TactileControl", range(24, 32)),
    ("eegnet_control", "EEGNetControl", range(32, 40)),
)


def _build_archive(group, counts, *, size=64, compression=zipfile.ZIP_DEFLATED):
    rng = np.random.default_rng(0)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as zip_file:
        for source_subject, n_files in counts.items():
            subject_name = f"S{source_subject:03d}"
            for index in range(n_files):
                if index:
                    session, run = divmod(index - 1, 6)
                    suffix = f"sess{session + 1:02d}_run{run + 1:02d}"
                else:
                    suffix = "run0"
                zip_file.writestr(
                    f"{group}/{subject_name}/{subject_name}_{suffix}.mat", rng.bytes(size)
                )
    return buffer.getvalue()


class _FakeResponse(io.BytesIO):
    def __init__(self, content, headers, status_code=206):
        super().__init__(content)
        self.headers = headers
        self.status_code = status_code
        self.iterated = False

    def iter_content(self, chunk_size):
        self.iterated = True
        while chunk := self.read(chunk_size):
            yield chunk


class _FakeServer:
    """Serve a byte payload with a strict HTTP Range implementation."""

    def __init__(self, payload):
        self.payload = payload
        self.bytes_served = 0
        self.requests = 0
        self.urls = []

    def get(self, url, headers=None, **kwargs):
        assert kwargs["stream"] is True
        self.requests += 1
        self.urls.append(url)
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
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


def _serve_ranges(monkeypatch, payload):
    server = _FakeServer(payload)
    monkeypatch.setattr(wang2026, "requests", SimpleNamespace(get=server.get))
    return server


@pytest.fixture
def archive_transport(monkeypatch, tmp_path):
    """Configure selective archive extraction against an in-memory server."""
    monkeypatch.setattr(wang2026, "_RANGE_BLOCK_SIZE", _SMALL_RANGE_BLOCK)
    monkeypatch.setattr(wang2026.dl, "get_dataset_path", lambda code, path: str(tmp_path))
    return lambda payload: _serve_ranges(monkeypatch, payload)


@pytest.fixture
def joint_subject(archive_transport, tmp_path):
    count = _SUBJECT_FILE_COUNTS["JointLearning"][1]
    server = archive_transport(_build_archive("JointLearning", {1: count}, size=512))
    dataset = Wang2026(group="joint_learning", subjects=[1])
    dataset.data_path(1)
    return dataset, server, tmp_path / "MNE-wang2026-data/JointLearning/S001"


def _write_mat(
    path,
    *,
    variable=False,
    labels=None,
    samples=None,
    sfreq=1000.0,
    selected_channels=_CHANNELS,
    trial_info_labels=None,
):
    labels = list(labels or ((1, 2) if variable else (0, 1)))
    with h5py.File(path, "w") as handle:
        run_data = handle.create_group("runData")
        meta = run_data.create_group("meta")
        text_dtype = h5py.string_dtype()
        meta["sampling_rate_hz"] = [sfreq]
        meta.create_dataset(
            "task_type", data=_BCI_LR_TASK if variable else _EEGNET_TASK, dtype=text_dtype
        )
        meta.create_dataset("selected_channels", data=selected_channels, dtype=text_dtype)
        if variable:
            sample_counts = samples or (32, 24)
            signals = run_data.create_dataset(
                "trialSignal", shape=(len(labels), 1), dtype=h5py.ref_dtype
            )
            targets = run_data.create_dataset(
                _CODE_TARGET, shape=(len(labels), 1), dtype=h5py.ref_dtype
            )
            for index, (label, n_samples) in enumerate(zip(labels, sample_counts)):
                trial = handle.create_dataset(
                    f"signal_{index}",
                    data=np.full((len(_CHANNELS), n_samples), index + 1.0),
                )
                target = handle.create_dataset(
                    f"target_{index}", data=np.full(n_samples, label, dtype=float)
                )
                signals[index, 0] = trial.ref
                targets[index, 0] = target.ref
        else:
            n_samples = samples or 32
            signal = np.zeros((len(labels), len(_CHANNELS), n_samples), dtype=np.float64)
            run_data.create_dataset("trialSignal", data=signal)
            targets = np.repeat(
                np.asarray(labels, dtype=float)[:, None], n_samples, axis=1
            )
            run_data.create_dataset(_CLASS_TARGET, data=targets)
        run_data.create_group("trialInfo")["target_label"] = np.asarray(
            labels if trial_info_labels is None else trial_info_labels, float
        )


@pytest.fixture
def mat_file(tmp_path):
    """Create one synthetic public-release MAT run."""

    def write(name=_MAT_NAME, **kwargs):
        path = tmp_path / name
        _write_mat(path, **kwargs)
        return path

    return write


def test_wang2026_is_one_dataset_with_global_subject_ids():
    _init_dataset()
    dataset = Wang2026()
    subjects = list(range(1, 40))
    assert datasets.Wang2026 is Wang2026
    assert dataset_dict["Wang2026"] is Wang2026
    assert dataset.subject_list == dataset.all_subjects == subjects
    assert list(dataset.subject_mapping) == subjects
    expected_mapping = [
        (archive, f"S{source:03d}")
        for _, archive, cohort_subjects in _COHORTS
        for source in range(1, len(cohort_subjects) + 1)
    ]
    assert list(dataset.subject_mapping.values()) == expected_mapping
    assert dataset.metadata.participants.n_subjects == 39
    for _, archive, _ in _COHORTS:
        old_name = f"Wang2026{archive}"
        assert not hasattr(datasets, old_name)
        assert old_name not in dataset_dict


def test_range_reader_coalesces_sequential_small_reads(monkeypatch):
    payload = b"x" * (100 * 1024 * 1024)
    server = _serve_ranges(monkeypatch, payload)
    reader = _RangeReader("https://example.invalid/file")

    reader.seek(16 * 1024 * 1024)
    chunks = [reader.read(64 * 1024) for _ in range(320)]
    assert sum(map(len, chunks)) == 20 * 1024 * 1024
    assert server.requests <= 4  # one probe plus three 8-MiB blocks
    assert server.bytes_served <= 24 * 1024 * 1024 + 1


@pytest.mark.parametrize(
    "status, content_range, content, match",
    [
        (200, "bytes 0-0/100", b"x", "Expected HTTP 206"),
        (206, "not-a-range", b"x", "malformed Content-Range"),
        (206, "bytes 1-1/100", b"x", "does not match requested"),
        (206, "bytes 0-0/100", b"", "payload length"),
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


@pytest.mark.parametrize("group_filter, archive_group, subjects", _COHORTS)
def test_colliding_s001_routes_to_each_distinct_archive(
    tmp_path, archive_transport, group_filter, archive_group, subjects
):
    global_subject = subjects.start
    counts = _SUBJECT_FILE_COUNTS[archive_group]
    payload = _build_archive(archive_group, {1: counts[1], 2: counts[2]})
    server = archive_transport(payload)
    assert Wang2026(group=group_filter).subject_list == list(subjects)
    dataset = Wang2026(group=group_filter, subjects=[global_subject])
    paths = dataset.data_path(global_subject)

    expected_dir = tmp_path / "MNE-wang2026-data" / archive_group / "S001"
    assert dataset.subject_list == [global_subject]
    assert dataset.subject_mapping[global_subject] == (archive_group, "S001")
    assert len(paths) == counts[1]
    assert all(str(expected_dir) in path for path in paths)
    assert not expected_dir.with_name("S002").exists()
    assert (expected_dir / _MANIFEST_FILENAME).is_file()
    expected_url = wang2026._FIGSHARE_FILE.format(
        file_id=wang2026._ARCHIVES[archive_group]["file_id"]
    )
    assert set(server.urls) == {expected_url}


def test_cache_detects_corruption_and_preserves_valid_data_on_failure(
    monkeypatch, joint_subject
):
    dataset, server, directory = joint_subject
    after_first = server.requests
    dataset.data_path(1)
    assert server.requests == after_first

    path = directory / "S001_run0.mat"
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    dataset.data_path(1)
    assert server.requests > after_first
    assert path.read_bytes() == original

    def interrupted_extract(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(wang2026, "safe_extract_zip", interrupted_extract)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        dataset.data_path(1, force_update=True)
    assert path.read_bytes() == original
    assert (directory / _MANIFEST_FILENAME).is_file()


def test_data_path_rejects_member_crc_failure(archive_transport):
    count = _SUBJECT_FILE_COUNTS["JointLearning"][1]
    payload = bytearray(
        _build_archive("JointLearning", {1: count}, compression=zipfile.ZIP_STORED)
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
    archive_transport(bytes(payload))

    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        Wang2026(group="joint_learning", subjects=[1]).data_path(1)


@pytest.mark.parametrize(
    "filename, target_name, task_type, expected, match",
    [
        (_MAT_NAME, _CLASS_TARGET, _EEGNET_TASK, _EEGNET_EVENT_MAP, None),
        (_MAT_NAME, _CODE_TARGET, _BCI_LR_TASK, _BCI2000_LR_EVENT_MAP, None),
        (
            "S001_sess03_run01UD.mat",
            _CODE_TARGET,
            _BCI_UD_TASK,
            _BCI2000_UD_EVENT_MAP,
            None,
        ),
        (
            "S001_sess04_run01.mat",
            _CODE_TARGET,
            _BCI_2D_TASK,
            _BCI2000_2D_EVENT_MAP,
            None,
        ),
        ("S001_sess03_run01UD.mat", _CODE_TARGET, _BCI_LR_TASK, None, "disagrees"),
        (_MAT_NAME, _CODE_TARGET, None, None, "no runData.meta.task_type"),
    ],
)
def test_event_mapping_uses_authoritative_metadata(
    filename, target_name, task_type, expected, match
):
    if match:
        with pytest.raises(ValueError, match=match):
            _event_mapping(filename, target_name, task_type)
    else:
        assert _event_mapping(filename, target_name, task_type) is expected


@pytest.mark.parametrize(
    "track, allowed, expected, match",
    [
        ([0, 0, 0], {0, 1, 2, 3}, 0, None),
        ([2, 2, 2], {1, 2}, 2, None),
        ([], {1, 2}, None, "empty"),
        ([np.nan], {1, 2}, None, "non-finite"),
        ([1.5], {1, 2}, None, "non-integer"),
        ([0], {1, 2}, None, "not valid"),
        ([1, 2], {1, 2}, None, "mixed"),
    ],
)
def test_strict_trial_label(track, allowed, expected, match):
    if match:
        with pytest.raises(ValueError, match=match):
            _strict_trial_label(track, allowed)
    else:
        assert _strict_trial_label(track, allowed) == expected


@pytest.mark.parametrize(
    "variable, labels, expected_lengths, expected_events",
    [
        (False, (0, 3), [32, 32], [_EVENTS["left_hand"], _EVENTS["rest"]]),
        (True, (1, 2), [32, 24], [_EVENTS["right_hand"], _EVENTS["left_hand"]]),
    ],
    ids=("fixed", "reference"),
)
def test_read_hdf5_trial_layouts(
    mat_file, variable, labels, expected_lengths, expected_events
):
    trials, event_codes = _read_hdf5_trials(mat_file(variable=variable, labels=labels))
    assert [trial.shape[1] for trial in trials] == expected_lengths
    assert event_codes == expected_events


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"sfreq": 500.0}, "sampling rate"),
        ({"selected_channels": _CHANNELS[::-1]}, "channel order"),
        ({"trial_info_labels": (0, 2)}, "target_label disagrees"),
    ],
)
def test_read_hdf5_rejects_inconsistent_metadata(mat_file, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _read_hdf5_trials(mat_file(**kwargs))


def test_calibration_guard_blocks_raw_construction_and_download(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="physical calibration"):
        _make_raw(tmp_path / "does-not-need-to-exist.mat")
    assert "10.1184/R1/32293995.v1" in _CALIBRATION_ERROR

    dataset = Wang2026(subjects=[1])
    monkeypatch.setattr(
        dataset,
        "data_path",
        lambda *args, **kwargs: pytest.fail("download must not start"),
    )
    with pytest.raises(RuntimeError, match="physical calibration"):
        dataset._get_single_subject_data(1)


def test_reconstruction_centers_trials_uses_exact_epoch_and_marks_padding(mat_file):
    # HDF5 fixed arrays cannot vary trial length, so use the reference/cell layout.
    path = mat_file(
        variable=True, labels=(1, 2), samples=(_NOMINAL_TRIAL_SAMPLES + 1, 1000)
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
    annotations = sorted(zip(raw.annotations.description, raw.annotations.duration))
    assert annotations == [
        ("BAD_ACQ_SKIP", 2.0),
        ("BAD_ACQ_SKIP", 2.0),
        ("BAD_WANG2026_ZERO_PADDED", 4.0),
    ]
    epochs = mne.Epochs(
        raw, events, tmin=0.0, tmax=4.999, baseline=None, preload=True, verbose=False
    )
    assert len(epochs) == 1


def test_guard_prevents_filter_leakage_into_neighboring_trial(mat_file):
    kwargs = {
        "variable": True,
        "labels": (1, 2),
        "samples": (_NOMINAL_TRIAL_SAMPLES, _NOMINAL_TRIAL_SAMPLES),
    }
    impulse_path = mat_file(**kwargs)
    zero_path = mat_file("S002_sess01_run01.mat", **kwargs)
    with h5py.File(impulse_path, "r+") as handle:
        handle["signal_0"][0, _NOMINAL_TRIAL_SAMPLES // 2] = 10.0

    impulse = _make_raw(impulse_path, _eeg_scale=1.0)
    event_sample = mne.find_events(impulse, shortest_event=1, verbose=False)[1, 0]
    impulse.pick(["Fp1"])
    control = _make_raw(zero_path, _eeg_scale=1.0).pick(["Fp1"])
    impulse.filter(4.0, 40.0, verbose=False)
    control.filter(4.0, 40.0, verbose=False)
    difference = impulse.get_data(
        start=event_sample, stop=event_sample + _NOMINAL_TRIAL_SAMPLES
    ) - control.get_data(start=event_sample, stop=event_sample + _NOMINAL_TRIAL_SAMPLES)
    assert np.max(np.abs(difference)) < 1e-12
