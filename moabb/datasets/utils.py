'''
Utils for easy database selection
'''

import inspect
import moabb.datasets as dsets
from moabb.datasets.base import BaseDataset
import moabb.database as db

for ds in inspect.getmembers(dsets, inspect.isclass):
    if issubclass(ds[1], BaseDataset):
        db.add_dataset(ds[1]())


def dataset_search(paradigm, multi_session=False, events=[],
                   has_all_events=False, total_classes=2, interval=None,
                   min_subjects=2, channels=()):
    '''
    Function that returns a list of datasets that match given criteria. Valid
    criteria are:

    Parameters
    ----------
    paradigm: str
        'imagery','p300',(more to come)

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
    events = set(events)
    out_data = []

    nsessions = 0
    if multi_session:
        nsessions = 1

    dataset_list = []
    for entry in db.session.query(db.DatasetEntry).\
            filter(db.DatasetEntry.nsessions > nsessions).\
            filter(db.DatasetEntry.nsubjects >= min_subjects).\
            filter(db.DatasetEntry.paradigm == paradigm):

        # test events
        dset_events = set([e.name for e in entry.events])
        if has_all_events and events <= dset_events:
            dataset_list.append(getattr(dsets, entry.classname))
        elif len(events) == 0 and len(dset_events) >= total_classes:
            dataset_list.append(getattr(dsets, entry.classname))
        elif len(dset_events & events) >= total_classes:
            dataset_list.append(getattr(dsets, entry.classname))
            
    for type_d in dataset_list:
        d = type_d()
        if interval is not None:
            if d.interval[1] - d.interval[0] < interval:
                continue
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
    for name in db.session.query(db.DatasetEntry.name):
        ds = getattr(dsets, name)
        # call download
        ds().download(update_path=True, verbose=verbose)
