"""Utils for easy database selection."""

import inspect

import numpy as np
from mne import create_info
from mne.io import RawArray

import moabb.datasets as db
from moabb.datasets.base import BaseDataset
from moabb.utils import aliases_list


dataset_list = []


def _init_dataset_list():
    for ds in inspect.getmembers(db, inspect.isclass):
        if issubclass(ds[1], BaseDataset):
            dataset_list.append(ds[1])


def dataset_search(  # noqa: C901
    paradigm=None,
    multi_session=False,
    events=None,
    has_all_events=False,
    interval=None,
    min_subjects=1,
    channels=(),
):
    """Returns a list of datasets that match a given criteria.

    Parameters
    ----------
    paradigm: str | None
        'imagery', 'p300', 'ssvep', 'cvep', None

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
    if len(dataset_list) == 0:
        _init_dataset_list()
    deprecated_names, _, _ = zip(*aliases_list)

    channels = set(channels)
    out_data = []
    if events is not None and has_all_events:
        n_classes = len(events)
    else:
        n_classes = None
    assert paradigm in ["imagery", "p300", "ssvep", "cvep", None]

    for type_d in dataset_list:
        if type_d.__name__ in deprecated_names:
            continue

        d = type_d()
        skip_dataset = False
        if multi_session and d.n_sessions < 2:
            continue

        if len(d.subject_list) < min_subjects:
            continue

        if paradigm is not None and paradigm != d.paradigm:
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
    """Given a list of dataset instances return a list of channels shared by
    all datasets. Skip datasets which have 0 overlap with the others.

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


def block_rep(block: int, rep: int, n_rep: int):
    idx = block * n_rep + rep
    return f"{idx}block{block}rep{rep}"


def blocks_reps(blocks: list, reps: list, n_rep: int):
    return [block_rep(b, r, n_rep) for b in blocks for r in reps]


def add_stim_channel_trial(raw, onsets, labels, offset=200, ch_name="stim_trial"):
    """
    Add a stimulus channel with trial onsets and their labels.

    Parameters
    ----------
    raw: mne.Raw
        The raw object to add the stimulus channel to.
    onsets: List | np.ndarray
        The onsets of the trials in sample numbers.
    labels: List | np.ndarray
        The labels of the trials.
    offset: int (default: 200)
        The integer value to start markers with. For instance, if 200, then label 0 will be marker 200, label 1
        will be marker 201, etc.
    ch_name: str (default: "stim_trial")
        The name of the added stimulus channel.

    Returns
    -------
    mne.Raw
        The raw object with the added stimulus channel.

    Notes
    -----
    .. versionadded:: 1.1.0
    """
    stim_chan = np.zeros((1, len(raw)))
    for onset, label in zip(onsets, labels):
        stim_chan[0, onset] = offset + label
    info = create_info(
        ch_names=[ch_name],
        ch_types=["stim"],
        sfreq=raw.info["sfreq"],
        verbose=False,
    )
    raw = raw.add_channels([RawArray(data=stim_chan, info=info, verbose=False)])
    return raw


def add_stim_channel_epoch(
    raw,
    onsets,
    labels,
    codes=None,
    presentation_rate=None,
    offset=100,
    ch_name="stim_epoch",
):
    """
    Add a stimulus channel with epoch onsets and their labels, which are the values of the presented code for each
    of the trials.

    Parameters
    ----------
    raw: mne.Raw
        The raw object to add the stimulus channel to.
    onsets: List | np.ndarray
        The onsets of the trials in sample numbers.
    labels: List | np.ndarray
        The labels of the trials.
    codes: np.ndarray (default: None)
        The codebook containing each presented code of shape (nr_bits, nr_codes), sampled at the presentation rate.
        If None, the labels information is used directly.
    presentation_rate: int (default: None):
        The presentation rate (e.g., frame rate) at which the codes were presented in Hz.
        If None, the raw object's sampling frequency is used.
    offset: int (default: 100)
        The integer value to start markers with. For instance, if 100, then label 0 will be marker 100, label 1
        will be marker 101, etc.
    ch_name: str (default: "stim_epoch")
        The name of the added stimulus channel.

    Returns
    -------
    mne.Raw
        The raw object with the added stimulus channel.

    Notes
    -----
    .. versionadded:: 1.1.0
    """
    if presentation_rate is None:
        presentation_rate = raw.info["sfreq"]
    stim_chan = np.zeros((1, len(raw)))
    for onset, label in zip(onsets, labels):
        if codes is None:
            stim_chan[0, int(onset * presentation_rate)] = offset + label
        else:
            idx = np.round(
                onset + np.arange(codes.shape[0]) / presentation_rate * raw.info["sfreq"]
            ).astype("int")
            stim_chan[0, idx] = offset + codes[:, label]

    info = create_info(
        ch_names=[ch_name],
        ch_types=["stim"],
        sfreq=raw.info["sfreq"],
        verbose=False,
    )
    raw = raw.add_channels([RawArray(data=stim_chan, info=info, verbose=False)])
    return raw
