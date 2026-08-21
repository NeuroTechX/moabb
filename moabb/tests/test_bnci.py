from types import SimpleNamespace

import numpy as np
from scipy.io import savemat

from moabb.datasets import BNCI2014_001, BNCI2014_008, BNCI2022_001
from moabb.datasets.bnci.base import _BNCI_ARTIFACT_ANNOTATION_DESCRIPTION, _convert_run
from moabb.datasets.bnci.bnci_2022_001 import _convert_run_001_2022
from moabb.datasets.preprocessing import SetRawAnnotations, _is_preserved_annotation


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


def _fake_2022_001_mat(path, sfreq=128, n_trajectories=3, pulse_samples=13):
    """Write a synthetic BNCI2022-001 task MAT file and return layout info.

    Mimics the public release structure (``EEG``, ``EOG``, ``Trigger``,
    ``Header``): each ~90 s trajectory starts with a trigger pulse of code 1,
    contains 4 waypoint pulses (codes 48/16 alternating) and ends with a code
    255 pulse. As in the real recordings, the hardware trigger holds each code
    for several consecutive samples (``pulse_samples``).
    """
    traj_samples = 92 * sfreq  # ~90 s trajectory + 2 s gap
    n_samples = n_trajectories * traj_samples
    trigger = np.zeros(n_samples)
    for k in range(n_trajectories):
        t0 = k * traj_samples
        trigger[t0 : t0 + pulse_samples] = 1
        for w in range(4):
            p = t0 + (w + 1) * 10 * sfreq
            trigger[p : p + pulse_samples] = 48 if w % 2 == 0 else 16
        end = t0 + 90 * sfreq
        trigger[end : end + pulse_samples] = 255
    rng = np.random.RandomState(42)
    savemat(
        path,
        {
            "EEG": rng.standard_normal((n_samples, 64)) * 10.0,  # microvolts
            "EOG": rng.standard_normal((n_samples, 3)) * 10.0,
            "Trigger": trigger,
            "Header": {"fs": float(sfreq)},
        },
    )
    return n_trajectories


def _load_fake_2022_001_raw(tmp_path):
    mat_path = str(tmp_path / "s1w.mat")
    n_traj = _fake_2022_001_mat(mat_path)
    ch_names = [f"EEG{i:02d}" for i in range(1, 65)] + ["EOG1", "EOG2", "EOG3"]
    ch_types = ["eeg"] * 64 + ["eog"] * 3
    raw = _convert_run_001_2022(mat_path, ch_names, ch_types, subject_id=1)
    return raw, n_traj


def test_bnci2022_001_declares_only_trajectory_trials():
    """Point events must not be declared as 90 s trials.

    The waypoint hit/miss and trajectory-end triggers are instantaneous
    markers occurring ~1000 times per subject; combined with the 90 s trial
    interval they made default paradigm epoching try to allocate hundreds of
    GiB (gh-1143, defect 1). Only the ~90 s trajectory is a trial.
    """
    dataset = BNCI2022_001()
    assert dataset.event_id == {"trajectory_start": 1}
    assert dataset.interval == [0, 90]


def test_bnci2022_001_trigger_pulses_annotated_once(tmp_path):
    """Each held trigger pulse must yield exactly one annotation.

    The 8-bit hardware trigger holds each event code over several consecutive
    samples; annotating every non-zero sample multiplied each real event into
    dozens of duplicates (33k+ events per subject on develop).
    """
    raw, n_traj = _load_fake_2022_001_raw(tmp_path)
    desc = raw.annotations.description
    assert np.sum(desc == "trajectory_start") == n_traj
    assert np.sum(desc == "waypoint_hit") == 2 * n_traj
    assert np.sum(desc == "waypoint_miss") == 2 * n_traj
    assert np.sum(desc == "trajectory_end") == n_traj


def test_bnci2022_001_default_epoching_is_bounded(tmp_path):
    """Default trial derivation must stay within the recording's duration.

    Applies the dataset's own event_id/interval the way both
    ``BaseDataset.get_data`` and the paradigm pipelines do (via
    ``SetRawAnnotations``) and checks that the resulting trials are the
    n_trajectories 90 s trajectories, not thousands of overlapping 90 s
    epochs anchored on instantaneous waypoint markers.
    """
    dataset = BNCI2022_001()
    raw, n_traj = _load_fake_2022_001_raw(tmp_path)
    transform = SetRawAnnotations(dataset.event_id, interval=tuple(dataset.interval))
    raw = transform.transform(raw)

    trial_annotations = raw.annotations
    assert set(trial_annotations.description) == {"trajectory_start"}
    assert len(trial_annotations) == n_traj
    # total epoched time cannot exceed the recording length
    assert trial_annotations.duration.sum() <= raw.times[-1]


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
