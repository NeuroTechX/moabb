'''
Tests to ensure that datasets download correctly
'''
from moabb.datasets.gigadb import Cho2017
from moabb.datasets.alex_mi import AlexMI
from moabb.datasets.physionet_mi import PhysionetMI
from moabb.datasets.bnci import (BNCI2014001, BNCI2014002, BNCI2014004,
                                 BNCI2014008, BNCI2014009, BNCI2015001,
                                 BNCI2015003, BNCI2015004)
from moabb.datasets.bbci_eeg_fnirs import Shin2017A, Shin2017B
from moabb.datasets.upper_limb import Ofner2017
from moabb.datasets.mpi_mi import MunichMI
from moabb.datasets.schirrmeister2017 import Schirrmeister2017
from moabb.datasets.Weibo2014 import Weibo2014
from moabb.datasets.Zhou2016 import Zhou2016
from moabb.datasets.ssvep_exo import SSVEPExo
from moabb.datasets.braininvaders import bi2013a
import unittest
import mne


class Test_Downloads(unittest.TestCase):

    def run_dataset(self, dataset):
        def _get_events(raw):
            stim_channels = mne.utils._get_stim_channel(
                None, raw.info, raise_error=False)
            if len(stim_channels) > 0:
                events = mne.find_events(raw, shortest_event=0, verbose=False)
            else:
                events, _ = mne.events_from_annotations(raw, verbose=False)
            return events

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
            self.assertEqual(len(sessions), obj.n_sessions)
            self.assertTrue(len(sessions) >= obj.n_sessions)

            # each session is a dict, with multiple runs
            for _, runs in sessions.items():
                self.assertTrue(isinstance(runs, dict))

                for _, raw in runs.items():
                    self.assertTrue(isinstance(raw, mne.io.BaseRaw))

                # each raw should contains events
                for _, raw in runs.items():
                    self.assertTrue(len(_get_events(raw) != 0))

    def test_cho2017(self):
        self.run_dataset(Cho2017)

    def test_bnci_1401(self):
        self.run_dataset(BNCI2014001)

    def test_bnci_1402(self):
        self.run_dataset(BNCI2014002)

    def test_bnci_1404(self):
        self.run_dataset(BNCI2014004)

    def test_bnci_1408(self):
        self.run_dataset(BNCI2014008)

    def test_bnci_1409(self):
        self.run_dataset(BNCI2014009)

    def test_bnci_1501(self):
        self.run_dataset(BNCI2015001)

    def test_bnci_1503(self):
        self.run_dataset(BNCI2015003)

    def test_bnci_1504(self):
        self.run_dataset(BNCI2015004)

    def test_alexmi(self):
        self.run_dataset(AlexMI)

    def test_physionet(self):
        self.run_dataset(PhysionetMI)

    def test_eegfnirs(self):
        self.run_dataset(Shin2017A)
        self.run_dataset(Shin2017B)

    def test_upper_limb(self):
        self.run_dataset(Ofner2017)

    def test_mpi_mi(self):
        self.run_dataset(MunichMI)

    def test_schirrmeister2017(self):
        self.run_dataset(Schirrmeister2017)

    def test_Weibo2014(self):
        self.run_dataset(Weibo2014)

    def test_Zhou2016(self):
        self.run_dataset(Zhou2016)

    def test_ssvep_exo(self):
        self.run_dataset(SSVEPExo)

    def test_bi2013a(self):
        self.run_dataset(bi2013a)


if __name__ == '__main__':
    unittest.main()
