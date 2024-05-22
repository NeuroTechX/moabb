from moabb.utils import depreciated_alias

from ..braininvaders import BI2014a, BI2014b, BI2015a, BI2015b, Cattan2019_VR
from .base import CompoundDataset


class _base_bi_il(CompoundDataset):
    def __init__(self, subjects_list, dataset=None, code=None):
        if code is None and dataset is None:
            raise ValueError("Either code or dataset must be provided")
        if code is None:
            code = f"{dataset.code}-Il"
        CompoundDataset.__init__(
            self,
            subjects_list=subjects_list,
            code=code,
            interval=[0, 1.0],
        )


@depreciated_alias("bi2014a_il", "1.1")
class BI2014a_Il(_base_bi_il):
    """A selection of subject from BI2014a with AUC < 0.7 with pipeline:
    ERPCovariances(estimator="lwf"), MDM(metric="riemann")
    """

    def __init__(self):
        dataset = BI2014a()
        subjects_list = [
            (dataset, 4, None, None),
            (dataset, 7, None, None),
            (dataset, 33, None, None),
            (dataset, 34, None, None),
            (dataset, 36, None, None),
            (dataset, 38, None, None),
            (dataset, 42, None, None),
            (dataset, 45, None, None),
            (dataset, 46, None, None),
            (dataset, 47, None, None),
            (dataset, 48, None, None),
            (dataset, 50, None, None),
            (dataset, 51, None, None),
            (dataset, 52, None, None),
            (dataset, 53, None, None),
            (dataset, 55, None, None),
            (dataset, 61, None, None),
        ]
        _base_bi_il.__init__(self, subjects_list=subjects_list, dataset=dataset)


@depreciated_alias("bi2014b_il", "1.1")
class BI2014b_Il(_base_bi_il):
    """A selection of subject from BI2014b with AUC < 0.7 with pipeline:
    ERPCovariances(estimator="lwf"), MDM(metric="riemann")
    """

    def __init__(self):
        dataset = BI2014b()
        subjects_list = [
            (dataset, 2, None, None),
            (dataset, 7, None, None),
            (dataset, 10, None, None),
            (dataset, 13, None, None),
            (dataset, 14, None, None),
            (dataset, 17, None, None),
            (dataset, 23, None, None),
            (dataset, 26, None, None),
            (dataset, 33, None, None),
            (dataset, 35, None, None),
            (dataset, 36, None, None),
        ]
        _base_bi_il.__init__(self, subjects_list=subjects_list, dataset=dataset)


@depreciated_alias("bi2015a_il", "1.1")
class BI2015a_Il(_base_bi_il):
    """A selection of subject from BI2015a with AUC < 0.7 with pipeline:
    ERPCovariances(estimator="lwf"), MDM(metric="riemann")
    """

    def __init__(self):
        dataset = BI2015a()
        subjects_list = [
            (dataset, 1, ["0", "1", "2"], None),
            (dataset, 39, ["1", "2"], None),
        ]
        _base_bi_il.__init__(self, subjects_list=subjects_list, dataset=dataset)


@depreciated_alias("bi2015b_il", "1.1")
class BI2015b_Il(_base_bi_il):
    """A selection of subject from BI2015b with AUC < 0.7 with pipeline:
    ERPCovariances(estimator="lwf"), MDM(metric="riemann")
    """

    def __init__(self):
        dataset = BI2015b()
        subjects_list = [
            (dataset, 2, None, None),
            (dataset, 4, None, None),
            (dataset, 6, None, None),
            (dataset, 8, None, None),
            (dataset, 10, None, None),
            (dataset, 12, None, None),
            (dataset, 14, None, None),
            (dataset, 16, None, None),
            (dataset, 18, None, None),
            (dataset, 20, None, None),
            (dataset, 22, None, None),
            (dataset, 24, None, None),
            (dataset, 26, None, None),
            (dataset, 28, None, None),
            (dataset, 30, None, None),
            (dataset, 32, None, None),
            (dataset, 33, None, None),
            (dataset, 34, None, None),
            (dataset, 35, None, None),
            (dataset, 36, None, None),
            (dataset, 38, None, None),
            (dataset, 40, None, None),
            (dataset, 41, None, None),
            (dataset, 42, None, None),
            (dataset, 44, None, None),
        ]
        _base_bi_il.__init__(self, subjects_list=subjects_list, dataset=dataset)


@depreciated_alias("VirtualReality_il", "1.1")
class Cattan2019_VR_Il(_base_bi_il):
    """A selection of subject from Cattan2019_VR with AUC < 0.7 with pipeline:
    ERPCovariances(estimator="lwf"), MDM(metric="riemann")
    """

    def __init__(self):
        dataset = Cattan2019_VR(virtual_reality=True, screen_display=True)
        subjects_list = [
            (dataset, 4, None, None),
            (dataset, 10, None, None),
            (dataset, 13, "0VR", None),
            (dataset, 15, "0VR", None),
        ]
        _base_bi_il.__init__(self, subjects_list=subjects_list, dataset=dataset)


@depreciated_alias("biIlliteracy", "1.1")
class BI_Il(_base_bi_il):
    """Subjects from braininvaders datasets with AUC < 0.7 with pipeline:
    ERPCovariances(estimator="lwf"), MDM(metric="riemann")
    """

    def __init__(self):
        subjects_list = [
            BI2014a_Il(),
            BI2014b_Il(),
            BI2015a_Il(),
            BI2015b_Il(),
            Cattan2019_VR_Il(),
        ]
        _base_bi_il.__init__(self, subjects_list=subjects_list, code="BrainInvaders-Il")
