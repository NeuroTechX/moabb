'''
Utils for easy database selection
'''

import inspect
import moabb.datasets as db
from moabb.datasets.base import BaseDataset

dataset_list = []
for ds in inspect.getmembers(db, inspect.isclass):
    if issubclass(db, BaseDataset):
        dataset_list.append(ds)

def dataset_search(paradigm, multi_session=False, events=None, two_class=True, channels=None):
    '''
    Function that returns a list of datasets that match given criteria. Valid
    criteria are:

    events: list of strings
    two_class: bool, uses first two classes in events list
    multi_session: bool, if True only returns datasets with more than one
    session per subject. If False return all
    paradigm: 'imagery','p300',(more to come)
    channels: list, returns datasets where all designated channels are present
    
    '''
    out_data = []
    max_events = 100
    if two_class:
        max_events = 2
    assert paradigm in ['imagery','p300']
    if paradigm=='p300':
        raise Exception('SORRY NOBDOYS GOTTEN AROUND TO THIS YET')
    for d in dataset_list:
        if multi_session:
            if d.n_sessions < 2:
                continue
        if paradigm == d.paradigm:
            keep_event_dict = {}
            if events is None:
                keep_event_dict = d.event_id
            else:
                n_events = 0
                for e in events:
                    if n_events > max_events:
                        break
                    if e in d.event_id.keys():
                        keep_event_dict[e] = d[e]
                        n_events+=1
                if len(keep_event_dict.keys()) == 0:
                    continue
                d.selected_events = keep_event_dict
                out_data.append(d)
    return out_data


def find_intersecting_channels(datasets):
    '''
    Given a list of datasets return a list of channels shared by all datasets.
    Skip datasets which have 0 overlap with the others
    '''
    pass
