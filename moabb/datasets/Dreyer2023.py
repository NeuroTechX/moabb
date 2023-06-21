"""
A large EEG right-left hand motor imagery dataset.
It is organized into three A, B, C datasets.
https://zenodo.org/record/7554429
"""

from functools import partialmethod
import logging
import os
import shutil
from moabb.datasets import download as dl
from pooch import Unzip, retrieve
import zipfile

from .base import BaseDataset
from .download import get_dataset_path
from pathlib import Path


DREYER2023_URL = "https://zenodo.org/record/7554429/files/BCI Database.zip"


class Dreyer2023(BaseDataset):
    def __init__(self, db_id='A', subjects=[]):
        assert db_id in ['A', 'B', 'C'], "Invalid dataset selection! Existing Dreyer2023 datasets: A, B, and C."
        self.db_id = db_id
        self.db_off = dict(A=0, B=60, C=81)
        super().__init__(subjects,
                         sessions_per_subject=1,
                         events=dict(left_hand=1, right_hand=2), 
                         code='Dreyer2023', 
                         interval=[3, 8],
                         paradigm='imagery')

    def _get_single_subject_data(self, subject):

        subj_dir_path = self.data_path(subject)
        subj_files = os.listdir(subj_dir_path)
        print(subj_files)


    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        path = get_dataset_path("DREYER", path)
        basepath = os.path.join(path, "MNE-dreyer-2023")
        if not os.path.isdir(basepath):
            os.makedirs(basepath)

        #dlpath = dl.data_dl(DREYER2023_URL, "DREYER_2023", path)
        dlpath = '/home/sara/mne_data_test/MNE-dreyer_2023-data/record/7554429/files/BCI Database.zip'
        if not os.path.exists(os.path.join(os.path.dirname(dlpath), 'BCI Database')):
            with zipfile.ZipFile(dlpath) as zip_file:
                zip_file.extractall(os.path.dirname(dlpath))

        subj_template_path = os.path.join(os.path.dirname(dlpath), 'BCI Database', 'Signals/DATA {0}/{0}{1}')
        return subj_template_path.format(self.db_id, subject + self.db_off[self.db_id])

class Dreyer2023A(Dreyer2023):
    __init__ = partialmethod(Dreyer2023.__init__, "A", subjects=list(range(1, 61)))

class Dreyer2023B(Dreyer2023):
    __init__ = partialmethod(Dreyer2023.__init__, "B", subjects=list(range(1, 22)))

class Dreyer2023C(Dreyer2023):
    __init__ = partialmethod(Dreyer2023.__init__, "C", subjects=list(range(1, 7)))




