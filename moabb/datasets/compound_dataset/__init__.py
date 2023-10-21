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

# Depreciated datasets (not added to dataset_list):
from .bi_illiteracy import VirtualReality_il  # noqa: F401
from .bi_illiteracy import bi2014a_il  # noqa: F401
from .bi_illiteracy import bi2014b_il  # noqa: F401
from .bi_illiteracy import bi2015a_il  # noqa: F401
from .bi_illiteracy import bi2015b_il  # noqa: F401
from .bi_illiteracy import biIlliteracy  # noqa: F401
