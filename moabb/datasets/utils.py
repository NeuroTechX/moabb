'''
Utils for easy database selection
'''

import inspect
import moabb.datasets as db
from moabb.datasets.base import BaseDataset

dataset_list = []
for ds in inspect.getmembers(db, inspect.isclass):
    if issubclass(ds[1], BaseDataset):
        dataset_list.append(ds[1])


def dataset_search(paradigm, multi_session=False, events=None,
                   has_all_events=False, total_classes=None, interval=None,
                   min_subjects=1, channels=()):
    '''
    Function that returns a list of datasets that match given criteria. Valid
    criteria are:

    Parameters
    ----------
    paradigm: str
        'imagery','p300','ssvep'

    multi_session: bool
        if True only returns datasets with more than one session per subject.
        If False return all

    events: list of strings
        events to select

    has_all_events: bool
        skip datasets that don't have all events in events

    total_classes: int or None
        total number of classes (returns all if None)
        will either truncate or choose rom events.

    interval:
        Length of motor imagery interval, in seconds. Only used in imagery
        paradigm

    min_subjects: int,
        minimum subjects in dataset

    channels: list of str
        list or set of channels

    '''
    channels = set(channels)
    out_data = []
    n_classes = total_classes
    if events is not None and has_all_events:
        n_classes = len(events)
    assert paradigm in ['imagery', 'p300', 'ssvep']
    if paradigm == 'p300':
        raise Exception('SORRY NOBDOYS GOTTEN AROUND TO THIS YET')

    for type_d in dataset_list:
        d = type_d()
        skip_dataset = False

        if multi_session and d.n_sessions < 2:
            continue

        if len(d.subject_list) < min_subjects:
            continue

        if paradigm == d.paradigm:
            if interval is not None:
                if d.interval[1] - d.interval[0] < interval:
                    continue
            keep_event_dict = {}
            if events is None:
                # randomly keep n_classes events
                if n_classes is None:
                    keep_event_dict = d.event_id.copy()
                else:
                    for k in d.event_id.keys():
                        if len(keep_event_dict) < n_classes:
                            keep_event_dict[k] = d.event_id[k]
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
                # don't want to use datasets with less than total number of
                # labels
                if n_classes is not None:
                    if n_events < n_classes:
                        skip_dataset = True
            if keep_event_dict and not skip_dataset:
                if len(channels) > 0:
                    s1 = d.get_data([1], False)[0][0][0]
                    if channels <= set(s1.info['ch_names']):
                        out_data.append(d)
                else:
                    out_data.append(d)
    return out_data


def find_intersecting_channels(datasets, verbose=False):
    '''
    Given a list of dataset instances return a list of channels shared by all
    datasets.
    Skip datasets which have 0 overlap with the others

    returns: set of common channels, list of datasets with valid channels
    '''
    allchans = set()
    dset_chans = []
    keep_datasets = []
    for d in datasets:
        print('Searching dataset: {:s}'.format(type(d).__name__))
        s1 = d.get_data([1], False)[0][0][0]
        s1.pick_types(eeg=True)
        processed = []
        for ch in s1.info['ch_names']:
            ch = ch.upper()
            if ch.find('EEG') == -1:
                # TODO: less hacky way of finding poorly labeled datasets
                processed.append(ch)
        allchans.update(processed)
        if len(processed) > 0:
            if verbose:
                print('Found EEG channels: {}'.format(processed))
            dset_chans.append(processed)
            keep_datasets.append(d)
        else:
            print('Dataset {:s} has no recognizable EEG channels'.
                  format(type(d).__name__))  # noqa
    for d in dset_chans:
        allchans.intersection_update(d)
    allchans = [s.replace('Z', 'z') for s in allchans]
    return allchans, d


def _download_all(update_path=True, verbose=None):
    """Download all data.

    This function is mainly used to generate the data cache.
    """

    # iterate over dataset
    for ds in dataset_list:
        # call download
        ds().download(update_path=True, verbose=verbose)
