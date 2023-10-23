import inspect

import moabb.datasets.compound_dataset as db
from moabb.datasets.compound_dataset.base import CompoundDataset


compound_dataset_list = []


def _init_compound_dataset_list():
    for ds in inspect.getmembers(db, inspect.isclass):
        if issubclass(ds[1], CompoundDataset) and not ds[0] == "CompoundDataset":
            compound_dataset_list.append(ds[1])
