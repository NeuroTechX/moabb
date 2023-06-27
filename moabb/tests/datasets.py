import unittest

import mne

from moabb.datasets import Shin2017A, Shin2017B, VirtualReality
from moabb.datasets.fake import FakeDataset, FakeVirtualRealityDataset
from moabb.datasets.shopping import GoShoppingDataset
from moabb.datasets.utils import block_rep
from moabb.paradigms import P300


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
        super().__init__(*args, **kwargs)

    def test_canary(self):
        assert VirtualReality() is not None

    def test_warning_if_parameters_false(self):
        with self.assertWarns(UserWarning):
            VirtualReality(virtual_reality=False, screen_display=False)

    def test_data_path(self):
        ds = VirtualReality(virtual_reality=True, screen_display=True)
        data_path = ds.data_path(1)
        assert len(data_path) == 2
        assert "subject_01_VR.mat" in data_path[0]
        assert "subject_01_PC.mat" in data_path[1]

    def test_get_block_repetition(self):
        ds = FakeVirtualRealityDataset()
        subject = 5
        block = 3
        repetition = 4
        _, _, ret = ds.get_block_repetition(P300(), [subject], [block], [repetition])
        assert ret.subject.unique()[0] == subject
        assert ret.run.unique()[0] == block_rep(block, repetition)


class Test_GoShoppingDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.paradigm = "p300"
        self.n_sessions = 2
        self.n_subjects = 2
        self.n_runs = 2
        self.ds = FakeDataset(
            n_sessions=self.n_sessions,
            n_runs=self.n_runs,
            n_subjects=self.n_subjects,
            event_list=["Target", "NonTarget"],
            paradigm=self.paradigm,
        )
        super().__init__(*args, **kwargs)

    def test_fake_dataset(self):
        """this test will insure the basedataset works"""
        param_list = [(None, None), ("session_0", "run_0"), (["session_0"], ["run_0"])]
        for sessions, runs in param_list:
            with self.subTest():
                shopping_list = [(self.ds, 1, sessions, runs)]
                shopping_data = GoShoppingDataset(
                    shopping_list,
                    events=dict(Target=2, NonTarget=1),
                    code="GoShoppingTest",
                    interval=[0, 1],
                    paradigm=self.paradigm,
                )

                data = shopping_data.get_data()

                # Check data type
                self.assertTrue(isinstance(data, dict))
                self.assertEqual(type(data[1]["session_0"]["run_0"]), mne.io.RawArray)

                # Check data size
                self.assertEqual(len(data), 1)
                expected_session_number = self.n_sessions if sessions is None else 1
                self.assertEqual(len(data[1]), expected_session_number)
                expected_runs_number = self.n_runs if runs is None else 1
                self.assertEqual(len(data[1]["session_0"]), expected_runs_number)

                # bad subject id must raise error
                self.assertRaises(ValueError, shopping_data.get_data, [1000])

    def test_shopping_dataset_composition(self):
        # Test we can compound two instance of GoShoppingDataset into a new one.

        # Create an instance of GoShoppingDataset with one subject
        shopping_list = [(self.ds, 1, None, None)]
        shopping_dataset = GoShoppingDataset(
            shopping_list,
            events=dict(Target=2, NonTarget=1),
            code="D1",
            interval=[0, 1],
            paradigm=self.paradigm,
        )

        # Add it two time to a shopping_list
        shopping_list = [shopping_dataset, shopping_dataset]
        shopping_data = GoShoppingDataset(
            shopping_list,
            events=dict(Target=2, NonTarget=1),
            code="GoShoppingTest",
            interval=[0, 1],
            paradigm=self.paradigm,
        )

        # Assert that the coumpouned dataset has two times more subject than the original one.
        data = shopping_data.get_data()
        self.assertEqual(len(data), 2)

    def test_get_sessions_per_subject(self):
        # define a new fake dataset with two times more sessions:
        self.ds2 = FakeDataset(
            n_sessions=self.n_sessions * 2,
            n_runs=self.n_runs,
            n_subjects=self.n_subjects,
            event_list=["Target", "NonTarget"],
            paradigm=self.paradigm,
        )

        # Add the two datasets to a goshoppingdataset
        shopping_list = [(self.ds, 1, None, None), (self.ds2, 1, None, None)]
        shopping_dataset = GoShoppingDataset(
            shopping_list,
            events=dict(Target=2, NonTarget=1),
            code="GoShoppingTest",
            interval=[0, 1],
            paradigm=self.paradigm,
        )

        # Test private method _get_sessions_per_subject returns the minimum number of sessions per subjects
        self.assertEqual(shopping_dataset._get_sessions_per_subject(), self.n_sessions)
