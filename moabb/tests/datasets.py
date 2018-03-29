import unittest

import mne
from moabb.datasets.fake import FakeDataset


class Test_Datasets(unittest.TestCase):

    def test_fake_dataset(self):
        """this test will insure the basedataset works"""
        n_subjects = 3
        n_sessions = 2
        n_runs = 2
        ds = FakeDataset(n_sessions=n_sessions, n_runs=n_runs,
                         n_subjects=n_subjects)
        data = ds.get_data()

        # we should get a dict
        self.assertTrue(isinstance(data, dict))

        # we get the right number of subject
        self.assertEqual(len(data), n_subjects)

        # right number of session
        self.assertEqual(len(data[1]), n_sessions)

        # right number of run
        self.assertEqual(len(data[1]['session_0']), n_runs)

        # We should get a raw array at the end
        self.assertEqual(type(data[1]['session_0']['run_0']),
                         mne.io.RawArray)
