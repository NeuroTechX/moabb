# flake8: noqa
from .base import CompoundDataset
from .bi_illiteracy import (
    VirtualReality_il,
    bi2014a_il,
    bi2014b_il,
    bi2015a_il,
    bi2015b_il,
    biIlliteracy,
)
from .utils import _init_compound_dataset_list


_init_compound_dataset_list()
del _init_compound_dataset_list
