'''
Tests to ensure that datasets download correctly
'''
from moabb.datasets.gigadb import Cho2017
from moabb.datasets.alex_mi import AlexMI
from moabb.datasets.physionet_mi import PhysionetMI
from moabb.datasets.bnci import (BNCI2014001, BNCI2014002, BNCI2014004,
                                 BNCI2015001, BNCI2015004)
from moabb.datasets.openvibe_mi import OpenvibeMI
from moabb.datasets.bbci_eeg_fnirs import Shin2017A, Shin2017B
from moabb.datasets.upper_limb import Ofner2017
import unittest
import mne


class Test_Downloads(unittest.TestCase):

    def run_dataset(self, dataset):
        obj = dataset()
        obj.subject_list = obj.subject_list[:2]
        data = obj.get_data(obj.subject_list)

        # get data return a dict
        self.assertTrue(isinstance(data, dict))

        # keys must corresponds to subjects list
        self.assertTrue(list(data.keys()) == obj.subject_list)

        # session must be a dict, and the length must match
        for _, sessions in data.items():
            self.assertTrue(isinstance(sessions, dict))
            self.assertTrue(len(sessions) == obj.n_sessions)

            # each session is a dict, with multiple runs
            for _, runs in sessions.items():
                self.assertTrue(isinstance(runs, dict))

                for _, raw in runs.items():
                    self.assertTrue(isinstance(raw, mne.io.BaseRaw))

    def test_cho2017(self):
        self.run_dataset(Cho2017)

    def test_bnci_1401(self):
        self.run_dataset(BNCI2014001)

    def test_bnci_1402(self):
        self.run_dataset(BNCI2014002)

    def test_bnci_1404(self):
        self.run_dataset(BNCI2014004)

    def test_bnci_1501(self):
        self.run_dataset(BNCI2015001)

    def test_bnci_1504(self):
        self.run_dataset(BNCI2015004)

    def test_alexmi(self):
        self.run_dataset(AlexMI)

    def test_ovmi(self):
        self.run_dataset(OpenvibeMI)

    def test_physionet(self):
        self.run_dataset(PhysionetMI)

    def test_eegfnirs(self):
        self.run_dataset(Shin2017A)
        self.run_dataset(Shin2017B)

    def test_upper_limb(self):
        self.run_dataset(Ofner2017)


if __name__ == '__main__':
    unittest.main()
