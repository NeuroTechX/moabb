'''
Tests to ensure that datasets download correctly
'''
from moabb.datasets.gigadb import GigaDbMI
from moabb.datasets.alex_mi import AlexMI
from moabb.datasets.physionet_mi import PhysionetMI
from moabb.datasets.bnci import BNCI2014001, BNCI2014002, BNCI2014004, BNCI2015001, BNCI2015004
from moabb.datasets.openvibe_mi import OpenvibeMI
from moabb.datasets.bbci_eeg_fnirs import BBCIEEGfNIRS
import unittest
import mne


class Test_Downloads(unittest.TestCase):

    def run_dataset(self, data, stack_sessions=False):
        obj = data()
        obj.subject_list = obj.subject_list[:2]
        data = obj.get_data(obj.subject_list, stack_sessions)
        self.assertTrue(type(data) is list, type(data))
        self.assertTrue(type(data[0]) is list, type(data[0]))
        self.assertTrue(type(data[0][0]) is list or issubclass(type(data[0][0]), mne.io.BaseRaw),
                        type(data[0][0]))

    def test_gigadb(self):
        self.run_dataset(GigaDbMI)

    def test_bnci_1401(self):
        self.run_dataset(BNCI2014001)
        self.run_dataset(BNCI2014001, stack_sessions=True)

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
        self.run_dataset(OpenvibeMI, stack_sessions=True)

    def test_physionet(self):
        self.run_dataset(PhysionetMI)

    def test_eegfnirs(self):
        self.run_dataset(BBCIEEGfNIRS)

if __name__ == '__main__':
    unittest.main()
