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
