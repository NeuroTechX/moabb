import inspect
from typing import List

import moabb.datasets.compound_dataset as db
from moabb.datasets.base import BaseDataset
from moabb.datasets.compound_dataset.base import CompoundDataset


compound_dataset_list = []


def _init_compound_dataset_list():
    for ds in inspect.getmembers(db, inspect.isclass):
        if issubclass(ds[1], CompoundDataset) and not ds[0] == "CompoundDataset":
            compound_dataset_list.append(ds[1])


def compound(*datasets: List[BaseDataset], interval=[0, 1.0]):
    subjects_list = [
        (d, subject, None, None) for d in datasets for subject in d.subject_list
    ]
    code = "".join([d.code for d in datasets])
    ret = CompoundDataset(
        subjects_list=subjects_list,
        code=code,
        interval=interval,
    )
    return ret
