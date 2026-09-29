"""Offline raw-mirror routing, real SDK HTTP-transport and loader regressions.

The ``OpenNeuroMirrorMixin`` provider policy is unit-tested with the shared
helpers (``test_loader_utils.py``); here the SDK transport and the no-prefetch
contract run once through a real loader, via ``MIXIN_CASE``.
"""

import hashlib
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import mne
import numpy as np
import pytest

from moabb.datasets import Iwama2023, Lee2022, Lioi2020, LioiXP1


CASES = [
    (Iwama2023, "on004444", 30, "030"),
    (Lee2022, "on004022", 1, "01"),
    (Lioi2020, "on002338", 17, "xp222"),
    (LioiXP1, "on002336", 10, "xp110"),
]
MIXIN_CASE = CASES[0]


@pytest.mark.parametrize("cls,nemar_id,subject,label", CASES)
def test_mirror_flags(cls, nemar_id, subject, label, monkeypatch, tmp_path):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    transport = Mock()
    monkeypatch.setattr("moabb.datasets.download.nemar.download", transport)
    ds = cls()
    assert ds.nemar_id == nemar_id
    ds.download([subject], tmp_path, True, False, verbose="ERROR")
    transport.assert_called_once_with(
        dataset=nemar_id,
        target_dir=tmp_path / f"MNE-{ds.code.lower()}-data" / nemar_id,
        subject=label,
        trust_existing=False,
        scope="raw",
        datatype="eeg",
    )
    # Raw mirror loaders must never prefetch converted sourcedata.
    monkeypatch.setattr(ds, "sourcedata_path", Mock(side_effect=AssertionError))
    ds._prefetch_nemar_sourcedata([subject])


_MIXIN_PREFIX = f"sub-{MIXIN_CASE[3]}/eeg/sub-{MIXIN_CASE[3]}_task-test"
_PAYLOADS = {
    "dataset_description.json": json.dumps(
        {"Name": "Synthetic", "BIDSVersion": "1.9.0"}
    ).encode(),
    f"{_MIXIN_PREFIX}_eeg.edf": b"synthetic EEG transport fixture, not a recording",
    f"{_MIXIN_PREFIX}_events.tsv": b"onset\tduration\ttrial_type\n0\t1\tright_hand\n",
    f"{_MIXIN_PREFIX}_channels.tsv": b"name\ttype\tunits\tstatus\nC3\tEEG\tuV\tbad\n",
    f"sub-{MIXIN_CASE[3]}/eeg/sub-{MIXIN_CASE[3]}_electrodes.tsv": (
        b"name\tx\ty\tz\nC3\t0\t0\t0\n"
    ),
    "sub-other/eeg/sub-other_task-test_eeg.edf": b"exclude other subject",
    "sourcedata/original.edf": b"exclude sourcedata",
}


def _nemar_handler(request):
    """Serve the synthetic NEMAR index, manifest and bytes for ``MIXIN_CASE``."""
    nemar_id, path = MIXIN_CASE[1], request.url.path
    if path == f"/{nemar_id}/":
        return httpx.Response(
            200,
            json={
                "dataset_id": nemar_id,
                "latest": "v1.0.0",
                "versions": [
                    {"version": "v1.0.0", "manifest_url": f"/{nemar_id}/manifest.json"}
                ],
            },
        )
    if path == f"/{nemar_id}/manifest.json":
        return httpx.Response(
            200,
            json=[
                {
                    "path": p,
                    "url": f"https://data.nemar.org/bytes/{p}",
                    "size": len(b),
                    "sha256": hashlib.sha256(b).hexdigest(),
                }
                for p, b in _PAYLOADS.items()
            ],
        )
    if path.startswith("/bytes/"):
        return httpx.Response(200, content=_PAYLOADS[path.removeprefix("/bytes/")])
    raise AssertionError(f"Unexpected HTTP request: {request.url}")


def test_http_only_raw_selection(monkeypatch, tmp_path):
    """Leave SDK selection, transfer and hash verification real; no live HTTP."""
    cls, _, subject, _ = MIXIN_CASE
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    handler = Mock(side_effect=_nemar_handler)
    monkeypatch.setattr(
        httpx, "Client", partial(httpx.Client, transport=httpx.MockTransport(handler))
    )
    ds = cls()
    # Select the SDK HTTPS backend explicitly; retain real selection, transfer
    # and verification while mocking its HTTP transport only.
    ds.nemar_bids_filters = dict(ds.nemar_bids_filters, downloader="python")
    root = Path(ds._mirror_root(subject, tmp_path, False, False, None))
    seen = [call.args[0].url.path for call in handler.call_args_list]
    for path, data in _PAYLOADS.items():
        target = root / path
        if path.startswith(("sub-other/", "sourcedata/")):
            assert not target.exists()
            assert f"/bytes/{path}" not in seen
        else:
            assert target.read_bytes() == data


def test_get_data_does_not_prefetch_sourcedata(monkeypatch):
    cls, _, subject, _ = MIXIN_CASE
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    ds = cls()
    monkeypatch.setattr(ds, "sourcedata_path", Mock(side_effect=AssertionError))
    monkeypatch.setattr(ds, "_get_selected_subject_data", Mock(return_value={}))
    assert ds.get_data([subject]) == {subject: {}}


def _raw(ch_names, types, annotations=()):
    raw = mne.io.RawArray(
        np.zeros((len(ch_names), 6000)),
        mne.create_info(ch_names, 100.0, types),
        verbose=False,
    )
    onsets, descriptions = zip(*annotations) if annotations else ((), ())
    raw.set_annotations(mne.Annotations(onsets, [0.0] * len(onsets), descriptions))
    return raw


def test_iwama2023_ignores_the_native_status_trigger(tmp_path, monkeypatch):
    """Events come from the ms-onset events.tsv, never from the EDF Status
    channel, whose code 1 also marks a point ~1 s before each task onset."""
    raw = _raw(["C3", "C4", "Status"], ["eeg", "eeg", "stim"])
    raw._data[2, [500, 1900, 2500, 3900]] = 1  # rest, pre-task, rest, pre-task
    (tmp_path / "sub-001_ses-01_task-smrbmi_events.tsv").write_text(
        "onset\tduration\ttrial\tvalue\tinstruction\n"
        "5000\t6\t1\t1\trest\n20000\t6\t1\t3\ttask\n"
        "25000\t6\t2\t1\trest\n40000\t6\t2\t3\ttask\n"
    )
    edf = tmp_path / "sub-001_ses-01_task-smrbmi_eeg.edf"
    ds = Iwama2023(subjects=[1])
    bids_path = SimpleNamespace(fpath=str(edf), session="01", run=None)
    monkeypatch.setattr(ds, "bids_paths", Mock(return_value=[bids_path]))
    monkeypatch.setattr(
        "moabb.datasets.iwama2023.mne.io.read_raw_edf", Mock(return_value=raw)
    )
    loaded = ds._get_single_subject_data(1)["01"]["0"]
    assert "Status" not in loaded.ch_names
    events = mne.find_events(loaded, shortest_event=0, verbose=False)
    np.testing.assert_array_equal(
        events[:, [0, 2]], [[500, 1], [2000, 3], [2500, 1], [4000, 3]]
    )


def test_lioixp1_maps_block_markers_and_skips_absent_runs(monkeypatch):
    markers = ["Stimulus/S 99", "Stimulus/S  2", "Response/R128", "Stimulus/S  1"]
    raw = _raw(["Cz", "ECG"], ["eeg", "eeg"], zip([1, 21, 22, 41], markers))
    ds = LioiXP1()
    runs = ["sub-xp102_task-eegNF_eeg.vhdr", "sub-xp102_task-MIpost_eeg.vhdr"]
    monkeypatch.setattr(ds, "data_path", Mock(return_value=runs))
    monkeypatch.setattr(
        "moabb.datasets.lioixp1.mne.io.read_raw_brainvision",
        Mock(side_effect=[raw.copy() for _ in range(2)]),
    )
    loaded = ds._get_single_subject_data(2)["0"]
    assert list(loaded) == ["1eegNF", "4MIpost"]
    assert loaded["1eegNF"].get_channel_types() == ["eeg", "ecg", "stim"]
    events = mne.find_events(loaded["1eegNF"], shortest_event=0, verbose=False)
    np.testing.assert_array_equal(events[:, [0, 2]], [[100, 1], [2100, 2]])


def test_lioi2020_upstream_manifest_transport_only(monkeypatch, tmp_path):
    """Real per-run S3 manifest and BIDS stub; only requests.get is mocked."""
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    get = Mock(return_value=Mock(status_code=404))
    monkeypatch.setattr("moabb.datasets.lioi2020.requests.get", get)
    ds = Lioi2020(imagery_only=True)
    root = Path(ds._download_subject(4, str(tmp_path), False, None, None))
    description = json.loads((root / "dataset_description.json").read_text())
    assert description["DatasetDOI"] == "doi:10.18112/openneuro.ds002338.v2.0.1"
    assert description["Authors"] == ds.METADATA.documentation.investigators
    urls = [call.args[0].split("/ds002338/")[1] for call in get.call_args_list]
    stem = "sub-xp204/eeg/sub-xp204_task-{}_eeg"
    assert urls == [
        name.format(task)
        for task in ("MIpre", "MIpost")
        for name in (
            *(stem + ext for ext in (".vhdr", ".vmrk", ".eeg")),
            "task-{}_events.tsv",
            "task-{}_eeg.json",
        )
    ]
