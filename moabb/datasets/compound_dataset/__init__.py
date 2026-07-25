# flake8: noqa
from .base import CompoundDataset
from .bi_illiteracy import (
    BI2014a_Il,
    BI2014b_Il,
    BI2015a_Il,
    BI2015b_Il,
    BI_Il,
    Cattan2019_VR_Il,
)
from .utils import _init_compound_dataset_list, compound  # noqa: F401


_init_compound_dataset_list()
del _init_compound_dataset_list
