"""
A large EEG right-left hand motor imagery dataset.
It is organized into three A, B, C datasets.
https://zenodo.org/record/7554429
"""

import logging
import os
import shutil
import zipfile
from functools import partialmethod
from os.path import dirname, join
from pathlib import Path

import mne
from mne.channels import make_standard_montage
from mne.io import read_raw_gdf
from pooch import Unzip, retrieve

from moabb.datasets import download as dl

from .base import BaseDataset
from .download import get_dataset_path


DREYER2023_URL = "https://zenodo.org/record/7554429/files/BCI Database.zip"


class Dreyer2023(BaseDataset):
    """Class for Dreyer2023 dataset management. MI dataset.

    .. admonition:: Dataset summary

        ==========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate    #Sessions
        ==========  =======  =======  ==========  =================  ============  ===============  ===========
        Dreyer2023    87       27         2                                            512 Hz           6
        ==========  =======  =======  ==========  =================  ============  ===============  ===========

    ===================
    Dataset description
    ===================
    A large EEG database with users' profile information for motor imagery
    Brain-Computer Interface research

    Data collectors : Appriou Aurélien; Caselli Damien; Benaroch Camille;
                      Yamamoto Sayu Maria; Roc Aline; Lotte Fabien;
                      Dreyer Pauline; Pillette Léa
    Data manager : Dreyer Pauline
    Project leader : Lotte Fabien
    Project members : Rimbert Sébastien; Monseigne Thibaut

    """
    def __init__(self, db_id='A', subjects=[]):
        assert db_id in ['A', 'B', 'C'], \
               "Invalid dataset selection! Existing Dreyer2023 datasets: A, B, and C."
        self.db_id = db_id
        self.db_idx_off = dict(A=0, B=60, C=81)
        super().__init__(subjects, sessions_per_subject=1,
                         events=dict(left_hand=1, right_hand=2),
                         code='Dreyer2023',  interval=[3, 8], paradigm='imagery',
                         doi='10.5281/zenodo.7554429')

    def _get_single_subject_data(self, subject):

        subj_dir = self.data_path(subject)
        subj_id = self.db_id + str(subject + self.db_idx_off[self.db_id])

        ch_names = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5',
                    'C2', 'C4', 'C6', 'EOG1', 'EOG2', 'EOG3', 'EMGg', 'EMGd',
                    'F4', 'FC2', 'FC4', 'FC6', 'CP2', 'CP4', 'CP6', 'P4',
                    'F3', 'FC1', 'FC3', 'FC5', 'CP1', 'CP3', 'CP5', 'P3']

        ch_types = ["eeg"] * 11 + ["eog"] * 3 + ["emg"] * 2 + ["eeg"] * 16

        montage = make_standard_montage("biosemi32")

        # Closed and open eyes baselines
        baselines = {}
        baselines['ce'] = \
            read_raw_gdf(join(subj_dir, subj_id + '_{0}_baseline.gdf').format('CE'),
                         eog=['EOG1', 'EOG2', 'EOG3'], misc=['EMGg', 'EMGd'])
        baselines['oe'] = \
            read_raw_gdf(join(subj_dir, subj_id + '_{0}_baseline.gdf').format('OE'),
                         eog=['EOG1', 'EOG2', 'EOG3'], misc=['EMGg', 'EMGd'])

        # Recordings
        recordings = {}
        # i - index, n - name, t - type
        for r_i, (r_n, r_t) in enumerate(zip(['R1', 'R2', 'R3', 'R4', 'R5', 'R6'],
                                             ['acquisition'] * 2 + ['onlineT'] * 4)):

            # One subject of dataset A has 4 recordings
            if r_i > 3 and self.db_id == 'A' and subject == 59:
                continue

            recordings['run_%d' % r_i] = \
                read_raw_gdf(join(subj_dir, subj_id + '_{0}_{1}.gdf'.format(r_n, r_t)),
                             eog=['EOG1', 'EOG2', 'EOG3'], misc=['EMGg', 'EMGd'])

        return {"session_0": recordings}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        path = get_dataset_path("DREYER", path)
        basepath = join(path, "MNE-dreyer-2023")
        if not os.path.isdir(basepath):
            os.makedirs(basepath)

        dlpath = dl.data_dl(DREYER2023_URL, "DREYER_2023", path)
        if not os.path.exists(join(dirname(dlpath), 'BCI Database')):

        subj_temp_path = join(dirname(dlpath), 'BCI Database', 'Signals/DATA {0}/{0}{1}')
        return subj_temp_path.format(self.db_id, subject + self.db_idx_off[self.db_id])


class Dreyer2023A(Dreyer2023):
    __init__ = partialmethod(Dreyer2023.__init__, "A", subjects=list(range(1, 61)))


class Dreyer2023B(Dreyer2023):
    __init__ = partialmethod(Dreyer2023.__init__, "B", subjects=list(range(1, 22)))


class Dreyer2023C(Dreyer2023):
    __init__ = partialmethod(Dreyer2023.__init__, "C", subjects=list(range(1, 7)))
