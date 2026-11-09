"""Trial counts of the dataset summary tables (``moabb/datasets/summary_*.csv``).

``#Trials / class`` counts the trials of one subject over all of that subject's
sessions and runs. ``_compute_n_trials`` therefore returns trials per subject,
the bubble size of :func:`moabb.analysis.plotting.dataset_bubble_plot` must not
multiply it by ``#Sessions`` again, and ``Total_trials`` (all subjects together)
must stay close to ``#Subj x #Classes x #Trials / class``.
"""

import math

import pytest

import moabb.datasets as moabb_datasets
from moabb.analysis._utils import _compute_n_trials
from moabb.analysis.plotting import _get_dataset_parameters, get_bubble_size
from moabb.datasets.base import get_summary_table


# Median number of trials per subject (all sessions and runs together, classes
# of the summary table) and median number of sessions per subject, measured on
# every subject with MOABB's own paradigms in the October 2026 benchmark rerun:
# within-session result rows (trials of each session, summed per subject) for
# motor imagery and SSVEP, epoch counts of ``P300().get_data`` for P300.
# ``#Sessions`` starts with the sessions of that median subject.
VERIFIED_TRIALS_PER_SUBJECT = {
    # motor imagery
    "AlexMI": (60, 1),
    "BNCI2014_001": (576, 2),
    "BNCI2014_002": (160, 1),
    "BNCI2014_004": (720, 5),
    "BNCI2015_001": (400, 2),
    "Cho2017": (200, 1),
    "GrosseWentrup2009": (300, 1),
    "Lee2019_MI": (200, 2),
    "PhysionetMI": (90, 1),
    "Schirrmeister2017": (1040, 1),
    "Shin2017A": (60, 3),
    "Weibo2014": (560, 1),
    "Zhou2016": (443, 3),
    # SSVEP
    "Kalunga2016": (64, 1),
    "Lee2019_SSVEP": (200, 2),
    "MAMEM1": (115, 1),
    "MAMEM2": (125, 1),
    "MAMEM3": (125, 1),
    "Nakanishi2015": (180, 1),
    "Wang2016": (240, 1),
    # P300 (non-target + target epochs)
    "BI2012": (766, 1),
    # subjects 1-7: 3,840 epochs over 8 sessions; subjects 8-24: 480 in 1
    "BI2013a": (480, 1),
    # 420 to 1,788 epochs per subject (subject 1: 1,188)
    "BI2014a": (892, 1),
    "BI2014b": (228, 1),
    # 972 to 4,956 epochs per subject (subject 1)
    "BI2015a": (1512, 3),
    "BI2015b": (2640, 1),
    "BNCI2014_008": (4200, 1),
    "BNCI2014_009": (1728, 3),
    "BNCI2015_003": (1800, 1),
    "Cattan2019_VR": (1440, 2),
    "EPFLP300": (3290, 4),
    "Huebner2017": (12850, 3),
    "Huebner2018": (14280, 3),
    "Lee2019_ERP": (8280, 2),
    # one pseudo-session per trial of 90 tones: 60 trials (80 for subject 1)
    "Sosulski2019": (5400, 60),
}

# Rows whose Total_trials is still not about #Subj x #Classes x #Trials / class.
# Neither the data nor the dataset's documentation gives their counts per
# subject yet; remove a row from this set once fixed.
UNVERIFIED_TOTAL_TRIALS = {
    "BNCI2003_004",  # the loader keeps only the labelled cues
    "BNCI2020_001",  # rest events per subject are not documented
    "Beetl2021_A",  # training trials per subject are not documented
    "Kaya2018",  # variable number of trials per session
    "Stieger2021",  # 450 trials per session for all classes together
}


def _dataset_parameters(name):
    return _get_dataset_parameters(getattr(moabb_datasets, name)())


@pytest.mark.parametrize("name", sorted(VERIFIED_TRIALS_PER_SUBJECT))
def test_compute_n_trials_is_trials_per_subject(name):
    expected, sessions = VERIFIED_TRIALS_PER_SUBJECT[name]
    dataset = getattr(moabb_datasets, name)()
    n_trials = _compute_n_trials(dataset._summary_table, dataset.paradigm)
    assert n_trials == pytest.approx(expected, rel=0.1), (
        f"{name}: the summary table gives {n_trials} trials per subject, "
        f"the data {expected} over {sessions} session(s)"
    )


@pytest.mark.parametrize("name", sorted(VERIFIED_TRIALS_PER_SUBJECT))
def test_bubble_size_counts_each_trial_once(name):
    expected, sessions = VERIFIED_TRIALS_PER_SUBJECT[name]
    _, _, _, n_sessions, n_trials, trial_len = _dataset_parameters(name)
    count = get_bubble_size("count", n_sessions, n_trials, trial_len)
    duration = get_bubble_size("duration", n_sessions, n_trials, trial_len)
    assert count == n_trials
    assert count == pytest.approx(expected, rel=0.1), (
        f"{name}: bubble of {count} trials, the data {expected} per subject "
        f"over {sessions} session(s)"
    )
    assert duration == pytest.approx(n_trials * trial_len)


@pytest.mark.parametrize("name", sorted(VERIFIED_TRIALS_PER_SUBJECT))
def test_sessions_is_that_of_a_typical_subject(name):
    # The plots read the first integer of ``#Sessions``: when the number of
    # sessions varies, the cell must start with that of the median subject,
    # e.g. "1 (8 for subjects 1-7)".
    _, sessions = VERIFIED_TRIALS_PER_SUBJECT[name]
    _, _, _, n_sessions, _, _ = _dataset_parameters(name)
    assert n_sessions == sessions, (
        f"{name}: #Sessions gives {n_sessions} sessions, the median subject "
        f"has {sessions}"
    )


def test_bubble_size_does_not_depend_on_sessions():
    # BNCI2014_004: 720 trials per subject over 5 sessions, not 5 x 720.
    sizes = {
        get_bubble_size("count", n_sessions, n_trials=720, trial_len=4.5)
        for n_sessions in (1, 2, 5)
    }
    assert sizes == {720}
    assert get_bubble_size("duration", 5, n_trials=720, trial_len=4.5) == 3240.0
    with pytest.raises(ValueError):
        get_bubble_size("area", 1, n_trials=720, trial_len=4.5)


@pytest.mark.parametrize("paradigm", ["imagery", "cognitive", "imagined_speech"])
def test_total_trials_matches_trials_per_subject(paradigm):
    table = get_summary_table(paradigm)
    inconsistent = {}
    columns = ("#Subj", "#Classes", "#Trials / class", "Total_trials")
    for name, row in table.iterrows():
        try:
            subj, n_classes, per_class, total = (float(row[col]) for col in columns)
        except (TypeError, ValueError):  # "varies", ranges
            continue
        if any(math.isnan(v) for v in (subj, n_classes, per_class, total)):
            continue
        ratio = total / (subj * n_classes * per_class)
        if not 0.8 <= ratio <= 1.25:
            inconsistent[name] = round(ratio, 3)
    unexpected = {
        k: v for k, v in inconsistent.items() if k not in UNVERIFIED_TOTAL_TRIALS
    }
    assert not unexpected, (
        "Total_trials must be about #Subj x #Classes x #Trials / class "
        f"(Total_trials / product): {unexpected}"
    )
    fixed = (UNVERIFIED_TOTAL_TRIALS & set(table.index)) - set(inconsistent)
    assert not fixed, f"Now consistent, remove from UNVERIFIED_TOTAL_TRIALS: {fixed}"


def test_generate_summary_table_from_metadata():
    """Summary rows and tables can be generated directly from class METADATA."""
    from moabb.analysis.plotting import _get_dataset_parameters
    from moabb.datasets import (
        BNCI2014_001,
        BNCI2014_008,
        Cattan2019_PHMD,
        Kalunga2016,
        Shin2017B,
        Thielen2015,
    )
    from moabb.datasets.base import (
        _SUMMARY_COLUMNS_BY_PARADIGM,
        generate_summary_table,
        get_summary_table,
        summary_row_from_metadata,
    )

    for cls in (
        BNCI2014_001,
        BNCI2014_008,
        Kalunga2016,
        Thielen2015,
        Cattan2019_PHMD,
        Shin2017B,
    ):
        ds = cls()
        row = summary_row_from_metadata(ds)
        assert row["#Subj"] == len(ds.all_subjects)
        assert row["#Sessions"] == ds.n_sessions
        assert row["#Chan"] == ds.metadata.acquisition.channel_types["eeg"]
        assert row["Freq (Hz)"] == int(ds.metadata.acquisition.sampling_rate)
        if ds.paradigm != "rstate" and row.get("#Trials / class") != "varies":
            n_trials = _compute_n_trials(row, ds.paradigm)
            assert n_trials > 0

    df_mi = get_summary_table("imagery", from_metadata=True)
    assert list(df_mi.columns) == _SUMMARY_COLUMNS_BY_PARADIGM["imagery"]
    assert "BNCI2014_001" in df_mi.index
    assert df_mi.loc["BNCI2014_001", "#Subj"] == 9
    assert df_mi.loc["BNCI2014_001", "#Chan"] == 22
    assert df_mi.loc["BNCI2014_001", "#Trials / class"] == 144
    assert df_mi.loc["BNCI2014_001", "Total_trials"] == 5184

    df_sub = generate_summary_table("p300", datasets=[BNCI2014_008])
    assert list(df_sub.index) == ["BNCI2014_008"]
    assert df_sub.loc["BNCI2014_008", "#Subj"] == 8
    assert _get_dataset_parameters(BNCI2014_001()) == (
        "BNCI2014_001",
        "imagery",
        9,
        2,
        576,
        4.0,
    )


def test_metaclass_synthesizes_summary_table_from_metadata():
    """A custom BaseDataset with METADATA gets _summary_table without a CSV entry."""
    from moabb.analysis.plotting import _get_dataset_parameters
    from moabb.datasets.base import BaseDataset
    from moabb.datasets.metadata.schema import (
        AcquisitionMetadata,
        DatasetMetadata,
        DataStructureMetadata,
        ExperimentMetadata,
        ParticipantMetadata,
    )

    class CustomMetadataDataset(BaseDataset):
        """Custom dataset docstring."""

        METADATA = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=250.0, channel_types={"eeg": 16, "eog": 2}
            ),
            participants=ParticipantMetadata(n_subjects=5),
            experiment=ExperimentMetadata(
                paradigm="imagery",
                n_classes=2,
                events={"left_hand": 1, "right_hand": 2},
                trial_duration=4.0,
            ),
            data_structure=DataStructureMetadata(
                n_trials_per_class={"left_hand": 60, "right_hand": 60}
            ),
            sessions_per_subject=2,
            runs_per_session=3,
        )

        def __init__(self):
            super().__init__(
                subjects=[1, 2, 3, 4, 5],
                sessions_per_subject=2,
                events={"left_hand": 1, "right_hand": 2},
                code="CustomMetadataDataset",
                interval=[0, 4],
                paradigm="imagery",
            )

        def _get_single_subject_data(self, subject):
            return {}

        def data_path(
            self, subject, path=None, force_update=False, update_path=None, verbose=None
        ):
            return []

    assert hasattr(CustomMetadataDataset, "_summary_table")
    assert ".. admonition:: Dataset summary" in (CustomMetadataDataset.__doc__ or "")
    st = CustomMetadataDataset._summary_table
    assert st["#Subj"] == 5
    assert st["#Chan"] == 16
    assert st["#Classes"] == 2
    assert st["#Trials / class"] == 60
    assert st["Total_trials"] == 600
    assert _get_dataset_parameters(CustomMetadataDataset()) == (
        "CustomMetadataDataset",
        "imagery",
        5,
        2,
        120,
        4.0,
    )


def test_summary_row_from_bids_sidecars(tmp_path):
    """summary_row_from_bids computes exact summary stats from BIDS events & sidecars."""
    from moabb.datasets.base import summary_row_from_bids

    for sub in ("01", "02"):
        eeg_dir = tmp_path / f"sub-{sub}" / "ses-01" / "eeg"
        eeg_dir.mkdir(parents=True)
        stem = f"sub-{sub}_ses-01_task-P300_run-01"
        (eeg_dir / f"{stem}_eeg.json").write_text('{"SamplingFrequency": 512.0}')
        (eeg_dir / f"{stem}_channels.tsv").write_text(
            "name\ttype\tunits\nFz\tEEG\tuV\nCz\tEEG\tuV\nPz\tEEG\tuV\nVEOG\tEOG\tuV\n"
        )
        rows = ["onset\tduration\ttrial_type\tvalue"]
        rows.extend(f"{i * 1.0}\t0.8\tTarget\t1" for i in range(1, 11))
        rows.extend(f"{i * 1.0}\t0.8\tNonTarget\t2" for i in range(11, 61))
        (eeg_dir / f"{stem}_events.tsv").write_text("\n".join(rows) + "\n")

    class _DummyP300:
        paradigm = "p300"
        event_id = {"Target": 1, "NonTarget": 2}
        interval = [0, 0.8]
        all_subjects = [1, 2]
        n_sessions = 1

    row = summary_row_from_bids(tmp_path, dataset=_DummyP300())
    assert row == {
        "#Subj": 2,
        "#Chan": 3,
        "#Trials / class": "50 NT / 10 T",
        "Trials length (s)": 0.8,
        "Freq (Hz)": 512,
        "#Sessions": 1,
    }
    assert _compute_n_trials(row, "p300") == 60
