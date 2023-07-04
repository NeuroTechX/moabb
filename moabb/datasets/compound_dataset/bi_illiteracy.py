from ..braininvaders import VirtualReality, bi2014a, bi2014b, bi2015a, bi2015b
from .base import CompoundDataset


class _base_bi_il(CompoundDataset):
    def __init__(self, subjects_list, dataset=None):
        code = "Illiteracy" if dataset is None else f"{dataset.code}+IL"
        CompoundDataset.__init__(
            self,
            subjects_list=subjects_list,
            events=dict(Target=2, NonTarget=1),
            code=code,
            interval=[0, 1.0],
            paradigm="p300",
        )


class bi2014a_il(_base_bi_il):
    """A selection of subject from bi2014a with AUC < 0.7 with pipeline:
    ERPCovariances(estimator="lwf"), MDM(metric="riemann")
    """

    def __init__(self):
        dataset = bi2014a()
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


class bi2014b_il(_base_bi_il):
    """A selection of subject from bi2014b with AUC < 0.7 with pipeline:
    ERPCovariances(estimator="lwf"), MDM(metric="riemann")
    """

    def __init__(self):
        dataset = bi2014b()
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


class bi2015a_il(_base_bi_il):
    """A selection of subject from bi2015a with AUC < 0.7 with pipeline:
    ERPCovariances(estimator="lwf"), MDM(metric="riemann")
    """

    def __init__(self):
        dataset = bi2015a()
        subjects_list = [
            (dataset, 1, ["session_1", "session_2", "session_3"], None),
            (dataset, 39, ["session_2", "session_3"], None),
        ]
        _base_bi_il.__init__(self, subjects_list=subjects_list, dataset=dataset)


class bi2015b_il(_base_bi_il):
    """A selection of subject from bi2015b with AUC < 0.7 with pipeline:
    ERPCovariances(estimator="lwf"), MDM(metric="riemann")
    """

    def __init__(self):
        dataset = bi2015b()
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


class VirtualReality_il(_base_bi_il):
    """A selection of subject from VirtualReality with AUC < 0.7 with pipeline:
    ERPCovariances(estimator="lwf"), MDM(metric="riemann")
    """

    def __init__(self):
        dataset = VirtualReality(virtual_reality=True, screen_display=True)
        subjects_list = [
            (dataset, 4, None, None),
            (dataset, 10, None, None),
            (dataset, 13, "VR", None),
            (dataset, 15, "VR", None),
        ]
        _base_bi_il.__init__(self, subjects_list=subjects_list, dataset=dataset)


class biIlliteracy(_base_bi_il):
    """Subjects from braininvaders datasets with AUC < 0.7 with pipeline:
    ERPCovariances(estimator="lwf"), MDM(metric="riemann")
    """

    def __init__(self):
        subjects_list = [
            bi2014a_il(),
            bi2014b_il(),
            bi2015a_il(),
            bi2015b_il(),
            VirtualReality_il(),
        ]
        _base_bi_il.__init__(self, subjects_list=subjects_list)
