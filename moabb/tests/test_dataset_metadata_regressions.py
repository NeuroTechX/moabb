"""Regression tests for dataset definitions that the 2026 metadata enrichment (#868, #974) got wrong, found by
diffing every dataset definition between v1.4.3 and develop and checking each change against the datasets'
own descriptions and papers (October 2026).

No data is downloaded: the tests read the class definitions only.
"""

import pytest

from moabb.datasets import (
    BI2015b,
    BNCI2014_002,
    BNCI2014_004,
    BNCI2014_008,
    BNCI2014_009,
    BNCI2015_004,
    Wang2016,
)
from moabb.paradigms import MotorImagery


def test_bnci2015_004_tasks_and_window():
    """BNCI 2015-004 (Scherer et al. 2015): five mental tasks WORD, SUB, NAV, HAND, FEET, coded 1-5 in that
    order in the .mat files; the cue is shown at 3 s and imagery lasts until 10 s of each trial."""
    ds = BNCI2015_004()
    assert ds.event_id == {
        "word_ass": 1,
        "subtraction": 2,
        "navigation": 3,
        "right_hand": 4,
        "feet": 5,
    }
    assert ds.interval == [3, 10]
    assert BNCI2015_004.METADATA.experiment.events == ds.event_id
    # the right-hand vs feet motor-imagery task of the MOABB benchmark must accept the dataset again
    assert MotorImagery(events=["right_hand", "feet"], n_classes=2).is_valid(ds)
    assert "word association" in BNCI2015_004.__doc__
    assert "multiplication" not in BNCI2015_004.__doc__


def test_wang2016_has_35_subjects():
    """Wang et al. 2017: 35 subjects, files S01.mat to S35.mat on the Zenodo record."""
    ds = Wang2016()
    assert ds.subject_list == list(range(1, 36))
    assert Wang2016.METADATA.participants.n_subjects == 35


def test_bnci2014_002_doi():
    """BNCI 2014-002 is Steyrl et al. 2016, Biomed. Tech. 61(1), not the 2012 Soft Computing review."""
    assert BNCI2014_002().doi == "10.1515/bmt-2014-0117"


@pytest.mark.parametrize(
    "dataset_cls", [BNCI2014_002, BNCI2014_004, BNCI2014_008, BNCI2014_009, BI2015b]
)
def test_metadata_events_match_constructor(dataset_cls):
    """METADATA.experiment.events must name and code the classes the loader exposes (the constructor's
    event_id), not the raw GDF codes or another spelling of the class names."""
    ds = dataset_cls()
    assert dataset_cls.METADATA.experiment.events == ds.event_id
