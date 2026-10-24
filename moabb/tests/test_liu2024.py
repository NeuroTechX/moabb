"""Liu2024 subjects derived from another subject's recording (gh-1248).

Subjects 5, 15 and 49 are copies (EDF) or noise-added copies (raw MAT) of
subjects 4, 11 and 47. They are excluded unless ``include_derived_subjects=True``.
No data is downloaded.
"""

import warnings

import pytest

from moabb.datasets import Liu2024
from moabb.datasets.liu2024 import LIU2024_DERIVED_SUBJECTS


def test_derived_mapping():
    assert LIU2024_DERIVED_SUBJECTS == {5: 4, 15: 11, 49: 47}


def test_default_excludes_derived_subjects():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dataset = Liu2024()
    assert len(dataset.subject_list) == 47
    assert dataset.subject_list == dataset.all_subjects
    assert not set(dataset.subject_list) & set(LIU2024_DERIVED_SUBJECTS)
    assert set(LIU2024_DERIVED_SUBJECTS.values()) <= set(dataset.subject_list)
    assert dataset.metadata.participants.n_subjects == len(dataset.subject_list)


@pytest.mark.parametrize("subject", sorted(LIU2024_DERIVED_SUBJECTS))
def test_requesting_derived_subject_without_flag_raises(subject):
    with pytest.raises(ValueError, match="include_derived_subjects=True"):
        Liu2024(subjects=[1, subject])


def test_opt_in_loads_all_50_and_warns():
    with pytest.warns(UserWarning, match=r"5 \(from 4\), 15 \(from 11\), 49 \(from 47\)"):
        dataset = Liu2024(include_derived_subjects=True)
    assert dataset.subject_list == list(range(1, 51))
    with pytest.warns(UserWarning):
        dataset = Liu2024(subjects=[5], include_derived_subjects=True)
    assert dataset.subject_list == [5]


def test_source_subjects_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        dataset = Liu2024(subjects=[4, 11, 47])
    assert dataset.subject_list == [4, 11, 47]
