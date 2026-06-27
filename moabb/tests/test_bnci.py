from types import SimpleNamespace

import numpy as np

from moabb.datasets import BNCI2014_001, BNCI2014_008
from moabb.datasets.bnci.base import _BNCI_ARTIFACT_ANNOTATION_DESCRIPTION, _convert_run
from moabb.datasets.preprocessing import _is_preserved_annotation


def _fake_mi_run():
    """Return a minimal Graz MI run.

    The fake run mirrors the fields used by ``_convert_run``: ``X`` is samples by
    channels, ``fs`` is the sampling frequency, ``trial`` contains source
    1-indexed event sample positions, ``y`` contains class codes, ``classes``
    contains class names, and ``artifacts`` contains one flag per trial.
    """
    return SimpleNamespace(
        X=np.zeros((2500, 2)),
        fs=250,
        trial=np.array([1, 251, 501, 751]),
        y=np.array([1, 2, 1, 2]),
        classes=np.array(["left hand", "right hand"]),
        artifacts=np.array([0, 1, 0, 1]),
    )


def test_convert_run_default_ignores_artifact_annotations():
    """Default BNCI conversion keeps events and does not add artifact annotations."""
    raw, event_id = _convert_run(_fake_mi_run(), ch_names=["C3", "C4"])

    stim = raw.get_data(picks="stim")[0]
    assert np.array_equal(np.flatnonzero(stim), np.array([0, 250, 500, 750]))
    assert np.array_equal(stim[np.flatnonzero(stim)], np.array([1, 2, 1, 2]))
    assert event_id == {"left hand": 1, "right hand": 2}
    assert len(raw.annotations) == 0


def test_convert_run_adds_bnci_artifact_annotations():
    """BNCI artifact annotation mode preserves flagged trials without rejection."""
    raw, _ = _convert_run(
        _fake_mi_run(),
        ch_names=["C3", "C4"],
        artifact_handling="annotate",
        artifact_interval=(2, 6),
    )

    assert len(raw.annotations) == 2
    assert raw.annotations.description.tolist() == ["bnci_artifact", "bnci_artifact"]
    assert np.allclose(raw.annotations.onset, [3.0, 5.0])
    assert np.allclose(raw.annotations.duration, [4.0, 4.0])
    assert raw.annotations.extras == [
        {"trial": 2, "sample": 251, "artifact": 1},
        {"trial": 4, "sample": 751, "artifact": 1},
    ]


def test_convert_run_adds_bad_artifact_annotations():
    """BNCI bad artifact mode marks flagged trials for MNE epoch rejection."""
    raw, _ = _convert_run(
        _fake_mi_run(),
        ch_names=["C3", "C4"],
        artifact_handling="annotate_bad",
        artifact_interval=(2, 6),
    )

    assert raw.annotations.description.tolist() == ["BAD_artifact", "BAD_artifact"]


def test_bnci_artifact_markers_survive_event_rederivation():
    """Every BNCI artifact marker must satisfy the pipeline's preservation rule.

    Ties the producer (``bnci/base.py`` artifact descriptions) to the consumer
    (``SetRawAnnotations`` via :func:`_is_preserved_annotation`). If a marker is
    ever renamed to something not preserved, ``SetRawAnnotations`` would silently
    drop it and ``reject_by_annotation`` would become a no-op again.
    """
    for description in _BNCI_ARTIFACT_ANNOTATION_DESCRIPTION.values():
        assert _is_preserved_annotation(description)


def test_bnci2014_001_metadata():
    """Test metadata for BNCI2014-001."""
    dataset = BNCI2014_001()
    subject = 1
    session_name = list(dataset.get_data(subjects=[subject])[subject].keys())[0]
    raw = dataset.get_data(subjects=[subject])[subject][session_name]["0"]

    assert "birthday" in raw.info["subject_info"]
    assert "sex" in raw.info["subject_info"]
    assert "hand" in raw.info["subject_info"]
    assert raw.info["subject_info"]["sex"] in [1, 2]
    assert raw.info["meas_date"].year == 2008
    assert raw.get_montage() is not None
    assert raw.info["line_freq"] == 50.0


def test_bnci2014_008_metadata():
    """Test metadata for BNCI2014-008."""
    dataset = BNCI2014_008()
    subject = 1
    session_name = list(dataset.get_data(subjects=[subject])[subject].keys())[0]
    raw = dataset.get_data(subjects=[subject])[subject][session_name]["0"]

    assert "birthday" in raw.info["subject_info"]
    assert "sex" in raw.info["subject_info"]
    assert "hand" in raw.info["subject_info"]
    assert raw.info["subject_info"]["sex"] in [1, 2]
    assert raw.info["meas_date"].year == 2012
    assert raw.get_montage() is not None
    assert raw.info["line_freq"] == 50.0
    assert "ALSfrs" in raw.info["description"]
