"""Nieto2022 and TrianaGuzman2024 read their NEMAR raw mirror first (offline)."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from moabb.datasets import Nieto2022, TrianaGuzman2024


CASES = [(Nieto2022, "on003626", 1, "01"), (TrianaGuzman2024, "on005342", 1, "001")]


@pytest.mark.parametrize("cls,nemar_id,subject,label", CASES)
def test_download_goes_to_the_raw_mirror(
    cls, nemar_id, subject, label, monkeypatch, tmp_path
):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    transport = Mock()
    monkeypatch.setattr("moabb.datasets.download.nemar.download", transport)
    ds = cls()
    assert ds.nemar_id == nemar_id
    ds.download([subject], path=tmp_path, force_update=True, verbose="ERROR")
    assert transport.call_count >= 1
    for call in transport.call_args_list:
        assert call.kwargs == {
            "dataset": nemar_id,
            "target_dir": tmp_path / f"MNE-{ds.code.lower()}-data" / nemar_id,
            "subject": label,
            "trust_existing": False,
            "scope": "raw",
            "datatype": "eeg",
        }
    # Raw mirror loaders never prefetch converted sourcedata.
    monkeypatch.setattr(ds, "sourcedata_path", Mock(side_effect=AssertionError))
    ds._prefetch_nemar_sourcedata([subject])


def test_nieto_reads_the_session_bdf_from_the_mirror(monkeypatch, tmp_path):
    monkeypatch.delenv("MOABB_DOWNLOAD_PROVIDER", raising=False)
    monkeypatch.setattr("moabb.datasets.download.nemar.download", Mock())
    monkeypatch.setattr(
        "moabb.datasets.nieto2022.dl.data_dl", Mock(side_effect=AssertionError)
    )
    path = Nieto2022().data_path(2, session=3, path=tmp_path)
    assert Path(path) == (
        tmp_path
        / "MNE-nieto2022-data"
        / "on003626"
        / "sub-02/ses-03/eeg/sub-02_ses-03_task-innerspeech_eeg.bdf"
    )


def test_nieto_upstream_provider_keeps_openneuro(monkeypatch, tmp_path):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    monkeypatch.setattr(
        "moabb.datasets.download.nemar.download", Mock(side_effect=AssertionError)
    )
    data_dl = Mock(return_value="local.bdf")
    monkeypatch.setattr("moabb.datasets.nieto2022.dl.data_dl", data_dl)
    assert Nieto2022().data_path(1, session=1, path=tmp_path) == "local.bdf"
    assert data_dl.call_args.args[0].startswith("https://s3.amazonaws.com/openneuro.org/")


def test_triana_mirror_root_gets_description_and_decimal_fix(monkeypatch, tmp_path):
    monkeypatch.delenv("MOABB_DOWNLOAD_PROVIDER", raising=False)
    root = tmp_path / "MNE-trianaguzman2024-data" / "on005342"
    eeg = root / "sub-001" / "eeg"

    def fake_download(**kwargs):
        eeg.mkdir(parents=True)
        (eeg / "sub-001_task-sitstand_events.tsv").write_text("onset\tvalue\n5,004\t1\n")

    monkeypatch.setattr("moabb.datasets.download.nemar.download", fake_download)
    monkeypatch.setattr(
        "moabb.datasets.triana_guzman2024.requests.get", Mock(side_effect=AssertionError)
    )
    returned = TrianaGuzman2024()._download_subject(1, tmp_path, False, None, None)
    assert Path(returned) == root
    assert (root / "dataset_description.json").is_file()
    assert "5.004" in (eeg / "sub-001_task-sitstand_events.tsv").read_text()
