"""Synthetic regression tests for the Beetl2021 label maps and trial window (#1241).

``final_MI_label.txt`` holds the competition's *three-class* test labels, whose
class 2 ("other") merges two of the four training tasks. The loaders used to
annotate it with the four-class training map, which filed every "other" trial
under ``right_hand`` (dataset A) or ``feet`` (dataset B). They also attached
slices of that final-phase file to the leaderboard test trials, whose labels
were never released, and their ``interval=[0, 4]`` made MNE's inclusive
``tmax`` drop the last 4 s trial of every run.
"""

import mne
import numpy as np
import pytest

from moabb.datasets.beetl import (
    FINAL_TEST_EVENT_DESC_A,
    FINAL_TEST_EVENT_DESC_B,
    Beetl2021_A,
    Beetl2021_B,
    _trials_to_raw,
)
from moabb.paradigms import LeftRightImagery, MotorImagery


mne.set_log_level("ERROR")

_SFREQ = {"A": 500, "B": 200}
_N_CHANNELS = {"A": 63, "B": 32}
_TRIAL_S = 4
# Two training races (dataset A) or one training file (dataset B), each with the
# four classes twice, in the files' own label dtypes (A ints, B floats).
_TRAIN_LABELS_A = np.array([0, 1, 2, 3])
_TRAIN_LABELS_B = np.array([0.0, 1.0, 2.0, 3.0] * 2)
_TRAIN_NAMES = {
    "A": ["rest", "left_hand", "right_hand", "feet"],
    "B": ["left_hand", "right_hand", "feet", "rest"],
}
# Test trials per subject; the label file holds one block per subject, S1-S5.
_N_TEST = 8
_TEST_BLOCK = np.array([0, 1, 2, 0, 1, 2, 2, 2])


def _test_block(subject):
    """Labels of one subject's block, distinct per subject so a wrong slice shows."""
    return np.roll(_TEST_BLOCK, subject - 1)


@pytest.fixture(scope="module")
def beetl_root(tmp_path_factory):
    """Write tiny files in the real ``finalMI``/``leaderboardMI`` layout.

    Dataset A: subject 1 in both phases; dataset B: subject 4 (final) and 3
    (leaderboard). Trials are exactly 4 s so the epoching path is exercised.
    """
    root = tmp_path_factory.mktemp("beetl2021_synth")
    rng = np.random.default_rng(0)

    def trials(kind, n, scale):
        shape = (n, _N_CHANNELS[kind], _TRIAL_S * _SFREQ[kind])
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    for phase_dir in ("finalMI", "leaderboardMI"):
        d = root / phase_dir / "S1"
        (d / "training").mkdir(parents=True)
        (d / "testing").mkdir()
        for race in (1, 2):
            np.save(d / "training" / f"race{race}_padsData.npy", trials("A", 4, 1e-5))
            np.save(d / "training" / f"race{race}_padsLabel.npy", _TRAIN_LABELS_A)
        for race in (6, 7):
            np.save(d / "testing" / f"race{race}_padsData.npy", trials("A", 4, 1e-5))

    for phase_dir, subject in (("finalMI", 4), ("leaderboardMI", 3)):
        d = root / phase_dir / f"S{subject}"
        (d / "training").mkdir(parents=True)
        (d / "testing").mkdir()
        np.save(d / "training" / f"training_s{subject}X.npy", trials("B", 8, 10.0))
        np.save(d / "training" / f"training_s{subject}y.npy", _TRAIN_LABELS_B)
        np.save(d / "testing" / f"testing_s{subject}X.npy", trials("B", _N_TEST, 10.0))

    labels = np.concatenate([_test_block(s) for s in range(1, 6)])
    np.savetxt(root / "final_MI_label.txt", labels, fmt="%d")
    return root


@pytest.fixture
def offline_datasets(monkeypatch, beetl_root):
    """Serve the synthetic tree and keep both loaders off the network."""
    for cls in (Beetl2021_A, Beetl2021_B):
        monkeypatch.setattr(
            cls, "data_path", lambda self, subject, **kwargs: [str(beetl_root)]
        )
        monkeypatch.setattr(cls, "nemar_id", None)
    return beetl_root


@pytest.mark.parametrize(
    "cls, kind, subject, test_desc",
    [
        (Beetl2021_A, "A", 1, FINAL_TEST_EVENT_DESC_A),
        (Beetl2021_B, "B", 4, FINAL_TEST_EVENT_DESC_B),
    ],
)
def test_final_test_run_uses_three_class_map(
    offline_datasets, cls, kind, subject, test_desc
):
    """The test run is annotated with the competition's map, not the training one."""
    dataset = cls()
    runs = dataset._get_single_subject_data(subject)["0"]
    assert list(runs) == ["0finaltrain", "1finaltest"]

    train = runs["0finaltrain"].annotations.description.tolist()
    assert train == _TRAIN_NAMES[kind] * 2

    test = runs["1finaltest"].annotations.description.tolist()
    assert test == [test_desc[code] for code in _test_block(subject)]

    other = test_desc[2]
    assert other in dataset.event_id and other not in _TRAIN_NAMES[kind]
    assert list(dataset.event_id)[:4] == _TRAIN_NAMES[kind]


@pytest.mark.parametrize(
    "cls, kind, other",
    [(Beetl2021_A, "A", "right_hand_or_feet"), (Beetl2021_B, "B", "feet_or_rest")],
)
def test_merged_class_is_selected_only_by_name(cls, kind, other):
    """Paradigms pick the merged class only when asked for it by name."""
    dataset = cls()
    assert set(LeftRightImagery().used_events(dataset)) == {"left_hand", "right_hand"}
    four = MotorImagery(events=_TRAIN_NAMES[kind], n_classes=4).used_events(dataset)
    assert list(four) == _TRAIN_NAMES[kind]
    competition = _competition_names(kind)
    three = MotorImagery(events=competition, n_classes=3).used_events(dataset)
    assert list(three) == competition
    assert other in MotorImagery().used_events(dataset)


def _competition_names(kind):
    desc = FINAL_TEST_EVENT_DESC_A if kind == "A" else FINAL_TEST_EVENT_DESC_B
    return [desc[code] for code in sorted(desc)]


@pytest.mark.parametrize("cls, subject", [(Beetl2021_A, 1), (Beetl2021_B, 4)])
def test_interval_keeps_every_trial(offline_datasets, cls, subject):
    """Back-to-back 4 s trials all survive epoching (``interval=[0, 4]`` lost one per run)."""
    dataset = cls()
    n_trials = 8 + _N_TEST
    X, y, _ = MotorImagery().get_data(dataset, subjects=[subject])
    assert len(y) == n_trials
    assert X.shape[-1] == _TRIAL_S * _SFREQ["A" if cls is Beetl2021_A else "B"]


@pytest.mark.parametrize(
    "cls, kind, subject", [(Beetl2021_A, "A", 1), (Beetl2021_B, "B", 3)]
)
def test_leaderboard_phase_serves_training_run_only(offline_datasets, cls, kind, subject):
    """No leaderboard test labels exist, so none are invented from the final file."""
    dataset = cls(phase="leaderboard")
    runs = dataset._get_single_subject_data(subject)["0"]
    assert list(runs) == ["0leaderboardtrain"]
    assert runs["0leaderboardtrain"].annotations.description.tolist() == (
        _TRAIN_NAMES[kind] * 2
    )


def test_trials_to_raw_rejects_mismatched_labels():
    info = mne.create_info(["C3", "C4"], sfreq=100.0, ch_types="eeg")
    trials = np.zeros((3, 2, 400), dtype=np.float32)
    with pytest.raises(ValueError, match="2 labels for 3 trials"):
        _trials_to_raw(trials, [0, 1], info, {0: "a", 1: "b"})
    with pytest.raises(ValueError, match=r"Labels \[2\] are not in the event map"):
        _trials_to_raw(trials, [0, 1, 2], info, {0: "a", 1: "b"})
