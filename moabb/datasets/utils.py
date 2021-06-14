"""
Utils for easy database selection
"""

import inspect

import moabb.datasets as db
from moabb.datasets.base import BaseDataset


dataset_list = []
for ds in inspect.getmembers(db, inspect.isclass):
    if issubclass(ds[1], BaseDataset):
        dataset_list.append(ds[1])


def dataset_search(  # noqa: C901
    paradigm,
    multi_session=False,
    events=None,
    has_all_events=False,
    interval=None,
    min_subjects=1,
    channels=(),
):
    """
    Returns a list of datasets that match a given criteria

    Parameters
    ----------
    paradigm: str
        'imagery', 'p300', 'ssvep'

    multi_session: bool
        if True only returns datasets with more than one session per subject.
        If False return all

    events: list of strings
        events to select

    has_all_events: bool
        skip datasets that don't have all events in events

    interval:
        Length of motor imagery interval, in seconds. Only used in imagery
        paradigm

    min_subjects: int,
        minimum subjects in dataset

    channels: list of str
        list or set of channels
    """
    channels = set(channels)
    out_data = []
    if events is not None and has_all_events:
        n_classes = len(events)
    else:
        n_classes = None
    assert paradigm in ["imagery", "p300", "ssvep"]

    for type_d in dataset_list:
        d = type_d()
        skip_dataset = False
        if multi_session and d.n_sessions < 2:
            continue

        if len(d.subject_list) < min_subjects:
            continue

        if paradigm != d.paradigm:
            continue

        if interval is not None and d.interval[1] - d.interval[0] < interval:
            continue

        keep_event_dict = {}
        if events is None:
            keep_event_dict = d.event_id.copy()
        else:
            n_events = 0
            for e in events:
                if n_classes is not None:
                    if n_events == n_classes:
                        break
                if e in d.event_id.keys():
                    keep_event_dict[e] = d.event_id[e]
                    n_events += 1
                else:
                    if has_all_events:
                        skip_dataset = True
        if keep_event_dict and not skip_dataset:
            if len(channels) > 0:
                s1 = d.get_data([1])[1]
                sess1 = s1[list(s1.keys())[0]]
                raw = sess1[list(sess1.keys())[0]]
                raw.pick_types(eeg=True)
                if channels <= set(raw.info["ch_names"]):
                    out_data.append(d)
            else:
                out_data.append(d)
    return out_data


def find_intersecting_channels(datasets, verbose=False):
    """
    Given a list of dataset instances return a list of channels shared by all
    datasets.
    Skip datasets which have 0 overlap with the others

    returns: set of common channels, list of datasets with valid channels
    """
    allchans = set()
    dset_chans = []
    keep_datasets = []
    for d in datasets:
        print("Searching dataset: {:s}".format(type(d).__name__))
        s1 = d.get_data([1])[1]
        sess1 = s1[list(s1.keys())[0]]
        raw = sess1[list(sess1.keys())[0]]
        raw.pick_types(eeg=True)
        processed = []
        for ch in raw.info["ch_names"]:
            ch = ch.upper()
            if ch.find("EEG") == -1:
                # TODO: less hacky way of finding poorly labeled datasets
                processed.append(ch)
        allchans.update(processed)
        if len(processed) > 0:
            if verbose:
                print("Found EEG channels: {}".format(processed))
            dset_chans.append(processed)
            keep_datasets.append(d)
        else:
            print(
                "Dataset {:s} has no recognizable EEG channels".format(type(d).__name__)
            )  # noqa
    allchans.intersection_update(*dset_chans)
    allchans = [s.replace("Z", "z") for s in allchans]
    return allchans, keep_datasets


def _download_all(update_path=True, verbose=None):
    """Download all data.

    This function is mainly used to generate the data cache.
    """

    # iterate over dataset
    for ds in dataset_list:
        # call download
        ds().download(update_path=True, verbose=verbose, accept=True)
