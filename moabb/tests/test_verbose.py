import logging

import pytest

from moabb.datasets.fake import FakeDataset
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import MotorImagery


def test_verbose_warning(caplog):
    # Setup
    dataset = FakeDataset(n_sessions=1, n_subjects=2)
    paradigm = MotorImagery()

    # We expect a warning because n_sessions=1 < 2 required for CrossSessionEvaluation
    # And subsequently an Exception because no datasets left

    with pytest.raises(Exception, match="No datasets left"):
        with caplog.at_level(logging.WARNING):
            CrossSessionEvaluation(paradigm=paradigm, datasets=[dataset])

    # Check if warning was logged
    assert "not compatible with evaluation" in caplog.text


def test_verbose_error_suppression(caplog):
    # Setup
    dataset = FakeDataset(n_sessions=1, n_subjects=2)
    paradigm = MotorImagery()

    # We expect an Exception because no datasets left, but NO warning if verbose='ERROR'
    with pytest.raises(Exception, match="No datasets left"):
        with caplog.at_level(logging.WARNING):
            # Passing verbose="ERROR" should suppress the warning
            CrossSessionEvaluation(paradigm=paradigm, datasets=[dataset], verbose="ERROR")

    # Check if warning was suppressed
    assert "not compatible with evaluation" not in caplog.text


def test_verbose_false_warning(caplog):
    # Setup
    dataset = FakeDataset(n_sessions=1, n_subjects=2)
    paradigm = MotorImagery()

    # MNE style: verbose=False implies WARNING level, so warning should STILL appear
    with pytest.raises(Exception, match="No datasets left"):
        with caplog.at_level(
            logging.INFO
        ):  # Set to INFO to see if behavior is consistent
            CrossSessionEvaluation(paradigm=paradigm, datasets=[dataset], verbose=False)

    # Check if warning was logged (since verbose=False -> WARNING)
    assert "not compatible with evaluation" in caplog.text
