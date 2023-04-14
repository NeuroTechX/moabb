import unittest

import mne

from moabb.datasets import Shin2017A, Shin2017B, VirtualReality
from moabb.datasets.fake import FakeDataset


_ = mne.set_log_level("CRITICAL")


def _run_tests_on_dataset(d):
    for s in d.subject_list:
        data = d.get_data(subjects=[s])

        # we should get a dict
        assert isinstance(data, dict)

        # We should get a raw array at the end
        rawdata = data[s]["session_0"]["run_0"]
        assert issubclass(type(rawdata), mne.io.BaseRaw), type(rawdata)

        # print events
        print(mne.find_events(rawdata))
        print(d.event_id)


class Test_Datasets(unittest.TestCase):
    def test_fake_dataset(self):
        """this test will insure the basedataset works"""
        n_subjects = 3
        n_sessions = 2
        n_runs = 2

        for paradigm in ["imagery", "p300", "ssvep"]:
            ds = FakeDataset(
                n_sessions=n_sessions,
                n_runs=n_runs,
                n_subjects=n_subjects,
                paradigm=paradigm,
            )
            data = ds.get_data()

            # we should get a dict
            self.assertTrue(isinstance(data, dict))

            # we get the right number of subject
            self.assertEqual(len(data), n_subjects)

            # right number of session
            self.assertEqual(len(data[1]), n_sessions)

            # right number of run
            self.assertEqual(len(data[1]["session_0"]), n_runs)

            # We should get a raw array at the end
            self.assertEqual(type(data[1]["session_0"]["run_0"]), mne.io.RawArray)

            # bad subject id must raise error
            self.assertRaises(ValueError, ds.get_data, [1000])

    def test_dataset_accept(self):
        """verify that accept licence is working"""
        # Only Shin2017 (bbci_eeg_fnirs) for now
        for ds in [Shin2017A(), Shin2017B()]:
            # if the data is already downloaded:
            if mne.get_config("MNE_DATASETS_BBCIFNIRS_PATH") is None:
                self.assertRaises(AttributeError, ds.get_data, [1])


class Test_VirtualReality_Dataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestingClass, self).__init__(*args, **kwargs)
        self.generate_mock_data()

    def generate_mock_data(self):
        sessions = {}
        for session in range(1, 10 + 1):
            sessions[session] = {}
            for block in range(1, 5 + 1):
                for repetition in range(1, 12 + 1):
                    sessions[session][
                        "block_" + str(block) + "-repetition_" + repetition
                    ] = (str(session) + str(block) + str(repetition))
        self.mock_data = sessions

    def test_canary(self):
        assert not VirtualReality() == None

    def test_get_block_repetition(self):
        ds = VirtualReality()
        ds._get_single_subject_data = lambda: self.mock_data
        ret = ds.get_block_repetition(P300(), [1], [2], [3])
        assert ret == "123"
