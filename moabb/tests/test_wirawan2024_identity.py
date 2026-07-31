"""Regression tests for duplicate participant labels in Wirawan2024."""

import pytest

from moabb.datasets.wirawan2024 import (
    WIRAWAN2024_DUPLICATE_SUBJECTS,
    WIRAWAN2024_UNIQUE_SUBJECTS,
    Wirawan2024,
)


def test_wirawan2024_exposes_only_unique_imagery_recordings():
    """P24-P30 must not be presented as independent subjects."""
    dataset = Wirawan2024()

    assert WIRAWAN2024_UNIQUE_SUBJECTS == tuple(range(1, 24))
    assert WIRAWAN2024_DUPLICATE_SUBJECTS == {
        24: 17,
        25: 18,
        26: 19,
        27: 20,
        28: 21,
        29: 22,
        30: 23,
    }
    assert dataset.subject_list == list(WIRAWAN2024_UNIQUE_SUBJECTS)
    assert dataset.METADATA.participants.n_subjects == len(dataset.subject_list)


@pytest.mark.parametrize("subject", WIRAWAN2024_DUPLICATE_SUBJECTS)
def test_wirawan2024_rejects_duplicate_labels_before_download(monkeypatch, subject):
    """Invalid duplicate labels fail before archive access."""
    dataset = Wirawan2024()

    def fail_if_called():
        pytest.fail("archive access must not occur for a duplicate participant label")

    monkeypatch.setattr(dataset, "_extract_root", fail_if_called)

    with pytest.raises(ValueError, match="Invalid subject number"):
        dataset.data_path(subject)
