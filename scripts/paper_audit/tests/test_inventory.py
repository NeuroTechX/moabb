"""Tests for the dataset metadata inventory."""

import json
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.paper_audit.inventory import (
    DatasetRecord,
    build_inventory,
    extract_docstring_dois,
    normalize_doi,
)


def test_doi_normalization():
    spellings = [
        "https://doi.org/10.1002/HBM.23730",
        "doi:10.1002/hbm.23730",
        "10.1002/hbm.23730",
        "http://dx.doi.org/10.1002/Hbm.23730",
        "  DOI: 10.1002/hbm.23730  ",
    ]
    assert {normalize_doi(s) for s in spellings} == {"10.1002/hbm.23730"}
    assert normalize_doi(None) is None
    assert normalize_doi("") is None
    assert normalize_doi("hal-01234567") is None
    assert normalize_doi("not a doi") is None


def test_docstring_doi_extraction_strips_rst_noise():
    doc = """
    See https://doi.org/10.1234/ABC.5 and `paper <https://doi.org/10.1234/xyz>`_.
    Also doi:10.5281/zenodo.123456. and 10.1002/hbm.23730/abstract
    """
    assert extract_docstring_dois(doc) == [
        "10.1234/abc.5",
        "10.1234/xyz",
        "10.5281/zenodo.123456",
        "10.1002/hbm.23730",
    ]


def _write_fake_checkout(root: Path) -> None:
    pkg = root / "moabb"
    (pkg / "datasets").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "datasets" / "__init__.py").write_text(
        textwrap.dedent(
            '''
            from types import SimpleNamespace


            class FakeAuditDataset:
                """A fake dataset.

                Paper: https://doi.org/10.9999/FAKE.paper and doi:10.5281/zenodo.1
                """

                METADATA = SimpleNamespace(
                    acquisition=SimpleNamespace(
                        sampling_rate=512.0,
                        channel_types={"eeg": 64, "eog": 2},
                        reference="Cz",
                        ground="AFz",
                        hardware="BioSemi ActiveTwo",
                        filters={"highpass": 0.1},
                        line_freq=50.0,
                    ),
                    participants=SimpleNamespace(n_subjects=20),
                    experiment=SimpleNamespace(
                        paradigm="imagery",
                        class_labels=["left_hand", "right_hand"],
                        events={"left_hand": 1, "right_hand": 2},
                        trials_per_class={"left_hand": 72, "right_hand": 72},
                    ),
                    documentation=SimpleNamespace(
                        doi="https://doi.org/10.9999/FAKE.paper",
                        associated_paper_doi=None,
                        related_paper_dois=["10.9999/related"],
                        data_url="https://example.org/data",
                        license="CC-BY-4.0",
                        institution="Fake University",
                        country="FR",
                        publication_year=2020,
                        investigators=["A. Person", "B. Person"],
                    ),
                    sessions_per_subject=2,
                    runs_per_session=6,
                    data_structure=SimpleNamespace(n_trials=288),
                )

                def __init__(self):
                    self.doi = "10.9999/FAKE.paper"
                    self.interval = [2, 6]
                    self.subject_list = list(range(1, 21))
                    self.n_sessions = 2
                    self.paradigm = "imagery"
                    self.event_id = {"left_hand": 1, "right_hand": 2}


            class NoMetadata:
                pass
            '''
        )
    )


def test_inventory_uses_checkout_path(tmp_path):
    _write_fake_checkout(tmp_path)
    before = {k for k in sys.modules if k == "moabb" or k.startswith("moabb.")}

    records = build_inventory(tmp_path)

    after = {k for k in sys.modules if k == "moabb" or k.startswith("moabb.")}
    assert after == before, "inventory must not import moabb into the parent process"
    assert [r.name for r in records] == ["FakeAuditDataset"]
    rec = records[0]
    assert isinstance(rec, DatasetRecord)
    assert rec.module == "moabb.datasets"
    assert rec.primary_doi == "10.9999/fake.paper"
    assert rec.paper_dois == ["10.9999/related"]
    assert rec.docstring_dois == ["10.9999/fake.paper", "10.5281/zenodo.1"]
    assert rec.data_url == "https://example.org/data"
    assert rec.declared["n_subjects"] == 20
    assert rec.declared["sessions_per_subject"] == 2
    assert rec.declared["runs_per_session"] == 6
    assert rec.declared["sampling_rate"] == 512.0
    assert rec.declared["channel_types"] == {"eeg": 64, "eog": 2}
    assert rec.declared["reference"] == "Cz"
    assert rec.declared["ground"] == "AFz"
    assert rec.declared["hardware"] == "BioSemi ActiveTwo"
    assert rec.declared["filters"] == {"highpass": 0.1}
    assert rec.declared["line_freq"] == 50.0
    assert rec.declared["interval"] == [2, 6]
    assert rec.declared["n_trials"] == 288
    assert rec.declared["class_labels"] == ["left_hand", "right_hand"]
    assert rec.declared["paradigm"] == "imagery"
    assert rec.declared["license"] == "CC-BY-4.0"
    assert rec.declared["institution"] == "Fake University"
    assert rec.declared["country"] == "FR"
    assert rec.declared["publication_year"] == 2020
    assert rec.declared["investigators"] == ["A. Person", "B. Person"]


def test_inventory_class_filter(tmp_path):
    _write_fake_checkout(tmp_path)
    assert build_inventory(tmp_path, classes=["NoSuchClass"]) == []
    assert [r.name for r in build_inventory(tmp_path, classes=["FakeAuditDataset"])] == [
        "FakeAuditDataset"
    ]


def test_inventory_missing_checkout_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_inventory(tmp_path / "nope")


def test_record_roundtrip_json(tmp_path):
    _write_fake_checkout(tmp_path)
    rec = build_inventory(tmp_path)[0]
    payload = json.loads(json.dumps(rec.to_dict()))
    assert DatasetRecord.from_dict(payload) == rec
