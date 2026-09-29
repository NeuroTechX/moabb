"""Offline raw-mirror routing and real SDK HTTP-transport regressions."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from moabb.datasets import Iwama2023, Lee2022, Lioi2020, LioiXP1
from moabb.datasets.download import NemarDownloadError


CASES = [
    (Iwama2023, "on004444", 30, "030"),
    (Lee2022, "on004022", 1, "01"),
    (Lioi2020, "on002338", 17, "xp222"),
    (LioiXP1, "on002336", 10, "xp110"),
]


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


@pytest.mark.parametrize("cls,nemar_id,subject,label", CASES)
def test_provider_policy(cls, nemar_id, subject, label, monkeypatch, tmp_path):
    ds = cls()
    transport = Mock(side_effect=NemarDownloadError("offline"))
    monkeypatch.setattr(ds, "_download_nemar", transport)
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    with pytest.raises(NemarDownloadError):
        ds._mirror_root(subject, tmp_path, False, False, None)
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "auto")
    with pytest.warns(RuntimeWarning, match="OpenNeuro"):
        assert ds._mirror_root(subject, tmp_path, False, False, None) is None
    transport.reset_mock()
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    assert ds._mirror_root(subject, tmp_path, False, False, None) is None
    transport.assert_not_called()


@pytest.mark.parametrize("cls,nemar_id,subject,label", CASES)
def test_http_only_raw_selection(cls, nemar_id, subject, label, monkeypatch, tmp_path):
    """Leave SDK selection, transfer and hash verification real; no live HTTP."""
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    prefix = f"sub-{label}/eeg/sub-{label}_task-test"
    payloads = {
        "dataset_description.json": json.dumps(
            {"Name": "Synthetic", "BIDSVersion": "1.9.0"}
        ).encode(),
        f"{prefix}_eeg.edf": b"synthetic EEG transport fixture, not a recording",
        f"{prefix}_events.tsv": b"onset\tduration\ttrial_type\n0\t1\tright_hand\n",
        f"{prefix}_channels.tsv": b"name\ttype\tunits\tstatus\nC3\tEEG\tuV\tbad\n",
        f"sub-{label}/eeg/sub-{label}_electrodes.tsv": b"name\tx\ty\tz\nC3\t0\t0\t0\n",
        "sub-other/eeg/sub-other_task-test_eeg.edf": b"exclude other subject",
        "sourcedata/original.edf": b"exclude sourcedata",
    }
    manifest = [
        {
            "path": p,
            "url": f"https://data.nemar.org/bytes/{p}",
            "size": len(b),
            "sha256": hashlib.sha256(b).hexdigest(),
        }
        for p, b in payloads.items()
    ]
    seen = []

    def handler(request):
        path = request.url.path
        seen.append(path)
        if path == f"/{nemar_id}/":
            return httpx.Response(
                200,
                json={
                    "dataset_id": nemar_id,
                    "latest": "v1.0.0",
                    "versions": [
                        {
                            "version": "v1.0.0",
                            "manifest_url": f"/{nemar_id}/manifest.json",
                        }
                    ],
                },
            )
        if path == f"/{nemar_id}/manifest.json":
            return httpx.Response(200, json=manifest)
        if path.startswith("/bytes/"):
            return httpx.Response(200, content=payloads[path.removeprefix("/bytes/")])
        raise AssertionError(f"Unexpected HTTP request: {request.url}")

    original = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda *a, **kw: original(*a, **dict(kw, transport=httpx.MockTransport(handler))),
    )
    ds = cls()
    # Select the SDK HTTPS backend explicitly; retain real selection, transfer
    # and verification while mocking its HTTP transport only.
    ds.nemar_bids_filters = dict(ds.nemar_bids_filters, downloader="python")
    root = Path(ds._mirror_root(subject, tmp_path, False, False, None))
    for path, data in payloads.items():
        target = root / path
        if path.startswith(("sub-other/", "sourcedata/")):
            assert not target.exists()
            assert f"/bytes/{path}" not in seen
        else:
            assert target.read_bytes() == data


@pytest.mark.parametrize("cls,nemar_id,subject,label", CASES)
def test_get_data_does_not_prefetch_sourcedata(
    cls, nemar_id, subject, label, monkeypatch
):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    ds = cls()
    monkeypatch.setattr(ds, "sourcedata_path", Mock(side_effect=AssertionError))
    monkeypatch.setattr(ds, "_get_selected_subject_data", lambda *args: {})
    assert ds.get_data([subject]) == {subject: {}}
