"""Tests to ensure that datasets download correctly using pytest."""

import mne
import pytest

from moabb.datasets.bbci_eeg_fnirs import BaseShin2017
from moabb.datasets.utils import dataset_list


def _get_events(raw):
    """Helper function to extract events from a raw object."""
    stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
    if len(stim_channels) > 0:
        events = mne.find_events(raw, shortest_event=0, verbose=False)
    else:
        events, _ = mne.events_from_annotations(raw, verbose=False)
    return events


@pytest.mark.parametrize("dataset", dataset_list)
def test_dataset_download(dl_data, dataset):
    """
    Test that a dataset downloads and returns data with the correct structure.

    For datasets of type BaseShin2017, we need to pass an "accept" flag.
    We then test that:

    - The returned data is a dict.
    - The keys of the dict match the subject list.
    - For each subject, sessions are provided as a dict and their number is at least as expected.
    - Each session contains runs that are dicts and each run is a valid MNE Raw object
      that contains at least one event.
    """
    if not dl_data:
        pytest.skip(
            "Skipping download tests by default. "
            "Run the test with option --dl-data to execute these tests."
        )

    # Some datasets (e.g., BaseShin2017) require explicit acceptance of terms.
    if isinstance(dataset(), BaseShin2017):
        obj = dataset(accept=True)
    else:
        obj = dataset()

    # Use only a subset of subjects for faster testing.
    subj = (0, 1)
    obj.subject_list = obj.subject_list[subj[0] : subj[1]]
    data = obj.get_data(obj.subject_list)

    # Check that the returned data is a dict.
    assert isinstance(data, dict), "Data returned by get_data is not a dict."

    # Check that the dictionary keys match the subject_list.
    assert (
        list(data.keys()) == obj.subject_list
    ), "Data keys do not match the subject_list."

    # For each subject, check the structure of sessions and runs.
    for subject, sessions in data.items():
        assert isinstance(
            sessions, dict
        ), f"Sessions for subject {subject} is not a dict."
        assert (
            len(sessions) >= obj.n_sessions
        ), f"Number of sessions for subject {subject} is less than expected."

        for session, runs in sessions.items():
            assert isinstance(runs, dict), f"Runs for session {session} is not a dict."

            for run, raw in runs.items():
                assert isinstance(
                    raw, mne.io.BaseRaw
                ), f"Data for run {run} in session {session} is not an instance of mne.io.BaseRaw."
                events = _get_events(raw)
                assert (
                    len(events) != 0
                ), f"No events found in run {run} of session {session}."
