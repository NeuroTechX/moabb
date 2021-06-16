"""
Tests to ensure that datasets download correctly
"""
import unittest

import mne

from moabb.datasets.bbci_eeg_fnirs import Shin2017


# from moabb.datasets.gigadb import Cho2017
# from moabb.datasets.alex_mi import AlexMI
# from moabb.datasets.physionet_mi import PhysionetMI
# from moabb.datasets.bnci import (BNCI2014001, BNCI2014002, BNCI2014004,
#                                  BNCI2014008, BNCI2014009, BNCI2015001,
#                                  BNCI2015003, BNCI2015004)
# from moabb.datasets.bbci_eeg_fnirs import Shin2017A, Shin2017B
# from moabb.datasets.upper_limb import Ofner2017
# from moabb.datasets.mpi_mi import MunichMI
# from moabb.datasets.schirrmeister2017 import Schirrmeister2017
# from moabb.datasets.Weibo2014 import Weibo2014
# from moabb.datasets.Zhou2016 import Zhou2016
# from moabb.datasets.ssvep_exo import SSVEPExo
# from moabb.datasets.braininvaders import bi2013a
# from moabb.datasets.epfl import EPFLP300
# from moabb.datasets.Lee2019 import Lee2019_MI
# from moabb.datasets.neiry import DemonsP300
# from moabb.datasets.physionet_mi import PhysionetMI
# from moabb.datasets.ssvep_mamem import MAMEM1, MAMEM2, MAMEM3
# from moabb.datasets.ssvep_nakanishi import Nakanishi2015
# from moabb.datasets.ssvep_wang import Wang2016


class Test_Downloads(unittest.TestCase):
    def run_dataset(self, dataset, subj=(0, 2)):
        def _get_events(raw):
            stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
            if len(stim_channels) > 0:
                events = mne.find_events(raw, shortest_event=0, verbose=False)
            else:
                events, _ = mne.events_from_annotations(raw, verbose=False)
            return events

        if isinstance(dataset(), Shin2017):
            obj = dataset(accept=True)
        else:
            obj = dataset()
        obj.subject_list = obj.subject_list[subj[0] : subj[1]]
        data = obj.get_data(obj.subject_list)

        # get data return a dict
        self.assertTrue(isinstance(data, dict))

        # keys must corresponds to subjects list
        self.assertTrue(list(data.keys()) == obj.subject_list)

        # session must be a dict, and the length must match
        for _, sessions in data.items():
            self.assertTrue(isinstance(sessions, dict))
            self.assertTrue(len(sessions) >= obj.n_sessions)

            # each session is a dict, with multiple runs
            for _, runs in sessions.items():
                self.assertTrue(isinstance(runs, dict))

                for _, raw in runs.items():
                    self.assertTrue(isinstance(raw, mne.io.BaseRaw))

                # each raw should contains events
                for _, raw in runs.items():
                    self.assertTrue(len(_get_events(raw) != 0))

    # def test_cho2017(self):
    #     self.run_dataset(Cho2017)

    # def test_bnci(self):
    #     self.run_dataset(BNCI2014001)
    #     self.run_dataset(BNCI2014002)
    #     self.run_dataset(BNCI2014004)
    #     self.run_dataset(BNCI2014008)
    #     self.run_dataset(BNCI2014009)
    #     self.run_dataset(BNCI2015001)
    #     self.run_dataset(BNCI2015003)
    #     self.run_dataset(BNCI2015004)

    # def test_alexmi(self):
    #     self.run_dataset(AlexMI)

    # def test_physionet(self):
    #     self.run_dataset(PhysionetMI)

    # def test_eegfnirs(self):
    #     self.run_dataset(Shin2017A)
    #     self.run_dataset(Shin2017B)

    # def test_upper_limb(self):
    #     self.run_dataset(Ofner2017)

    # def test_mpi_mi(self):
    #     self.run_dataset(MunichMI)

    # def test_schirrmeister2017(self):
    #     self.run_dataset(Schirrmeister2017, subj=(0, 1))

    # def test_Weibo2014(self):
    #     self.run_dataset(Weibo2014)

    # def test_Zhou2016(self):
    #     self.run_dataset(Zhou2016)

    # def test_ssvep_exo(self):
    #     self.run_dataset(SSVEPExo)

    # def test_bi2013a(self):
    #     self.run_dataset(bi2013a)

    # def test_epflp300(self):
    #     self.run_dataset(EPFLP300)

    # def test_lee2019_MI(self):
    #     self.run_dataset(Lee2019_MI)

    # def test_demonsp300(self):
    #     self.run_dataset(DemonsP300)

    # def test_physionetmi(self):
    #     self.run_dataset(PhysionetMI)

    # def test_mamem(self):
    #     self.run_dataset(MAMEM1)
    #     self.run_dataset(MAMEM2)
    #     self.run_dataset(MAMEM3)

    # def test_nakanishi2015(self):
    #     self.run_dataset(Nakanishi2015)

    # def test_wang2016(self):
    #     self.run_dataset(Wang2016)


if __name__ == "__main__":
    unittest.main()
