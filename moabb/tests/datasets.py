import inspect
import shutil
import tempfile
import unittest

import mne

import moabb.datasets as db
from moabb.datasets import Shin2017A, Shin2017B, VirtualReality
from moabb.datasets.base import BaseDataset
from moabb.datasets.compound_dataset import CompoundDataset
from moabb.datasets.fake import FakeDataset, FakeVirtualRealityDataset
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
        """This test will insure the basedataset works."""
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
            self.assertIsInstance(data[1]["session_0"]["run_0"], mne.io.BaseRaw)

            # bad subject id must raise error
            self.assertRaises(ValueError, ds.get_data, [1000])

    def test_cache_dataset(self):
        tempdir = tempfile.mkdtemp()
        for paradigm in ["imagery", "p300", "ssvep"]:
            dataset = FakeDataset(paradigm=paradigm)
            # Save cache:
            with self.assertLogs(
                logger="moabb.datasets.bids_interface", level="INFO"
            ) as cm:
                _ = dataset.get_data(
                    subjects=[1],
                    cache_config=dict(
                        save_raw=True,
                        use=True,
                        overwrite_raw=False,
                        path=tempdir,
                    ),
                )
            print("\n".join(cm.output))
            expected = [
                "Attempting to retrieve cache .* datatype-eeg",  # empty pipeline
                "No cache found at",
                "Starting caching .* datatype-eeg",
                "Finished caching .* datatype-eeg",
            ]
            self.assertEqual(len(expected), len(cm.output))
            for i, regex in enumerate(expected):
                self.assertRegex(cm.output[i], regex)

            # Load cache:
            with self.assertLogs(
                logger="moabb.datasets.bids_interface", level="INFO"
            ) as cm:
                _ = dataset.get_data(
                    subjects=[1],
                    cache_config=dict(
                        save_raw=True,
                        use=True,
                        overwrite_raw=False,
                        path=tempdir,
                    ),
                )
            print("\n".join(cm.output))
            expected = [
                "Attempting to retrieve cache .* datatype-eeg",
                "Finished reading cache .* datatype-eeg",
            ]
            self.assertEqual(len(expected), len(cm.output))
            for i, regex in enumerate(expected):
                self.assertRegex(cm.output[i], regex)

            # Overwrite cache:
            with self.assertLogs(
                logger="moabb.datasets.bids_interface", level="INFO"
            ) as cm:
                _ = dataset.get_data(
                    subjects=[1],
                    cache_config=dict(
                        save_raw=True,
                        use=True,
                        overwrite_raw=True,
                        path=tempdir,
                    ),
                )
            print("\n".join(cm.output))
            expected = [
                "Starting erasing cache .* datatype-eeg",
                "Finished erasing cache .* datatype-eeg",
                "Starting caching .* datatype-eeg",
                "Finished caching .* datatype-eeg",
            ]
            self.assertEqual(len(expected), len(cm.output))
            for i, regex in enumerate(expected):
                self.assertRegex(cm.output[i], regex)
        shutil.rmtree(tempdir)

    def test_dataset_accept(self):
        """Verify that accept licence is working."""
        # Only Shin2017 (bbci_eeg_fnirs) for now
        for ds in [Shin2017A(), Shin2017B()]:
            # if the data is already downloaded:
            if mne.get_config("MNE_DATASETS_BBCIFNIRS_PATH") is None:
                self.assertRaises(AttributeError, ds.get_data, [1])

    def test_datasets_init(self):
        for ds in inspect.getmembers(db, inspect.isclass):
            if issubclass(ds[1], BaseDataset):
                kwargs = {}
                if inspect.signature(ds[1]).parameters.get("accept"):
                    kwargs["accept"] = True
                self.assertIsNotNone(ds[1](**kwargs))


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


class Test_CompoundDataset(unittest.TestCase):
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
        """This test will insure the basedataset works."""
        param_list = [(None, None), ("session_0", "run_0"), (["session_0"], ["run_0"])]
        for sessions, runs in param_list:
            with self.subTest():
                subjects_list = [(self.ds, 1, sessions, runs)]
                compound_data = CompoundDataset(
                    subjects_list,
                    events=dict(Target=2, NonTarget=1),
                    code="CompoundTest",
                    interval=[0, 1],
                    paradigm=self.paradigm,
                )

                data = compound_data.get_data()

                # Check data type
                self.assertTrue(isinstance(data, dict))
                self.assertIsInstance(data[1]["session_0"]["run_0"], mne.io.BaseRaw)

                # Check data size
                self.assertEqual(len(data), 1)
                expected_session_number = self.n_sessions if sessions is None else 1
                self.assertEqual(len(data[1]), expected_session_number)
                expected_runs_number = self.n_runs if runs is None else 1
                self.assertEqual(len(data[1]["session_0"]), expected_runs_number)

                # bad subject id must raise error
                self.assertRaises(ValueError, compound_data.get_data, [1000])

    def test_compound_dataset_composition(self):
        # Test we can compound two instance of CompoundDataset into a new one.

        # Create an instance of CompoundDataset with one subject
        subjects_list = [(self.ds, 1, None, None)]
        compound_dataset = CompoundDataset(
            subjects_list,
            events=dict(Target=2, NonTarget=1),
            code="D1",
            interval=[0, 1],
            paradigm=self.paradigm,
        )

        # Add it two time to a subjects_list
        subjects_list = [compound_dataset, compound_dataset]
        compound_data = CompoundDataset(
            subjects_list,
            events=dict(Target=2, NonTarget=1),
            code="CompoundTest",
            interval=[0, 1],
            paradigm=self.paradigm,
        )

        # Assert that the coumpouned dataset has two times more subject than the original one.
        data = compound_data.get_data()
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

        # Add the two datasets to a CompoundDataset
        subjects_list = [(self.ds, 1, None, None), (self.ds2, 1, None, None)]
        compound_dataset = CompoundDataset(
            subjects_list,
            events=dict(Target=2, NonTarget=1),
            code="CompoundTest",
            interval=[0, 1],
            paradigm=self.paradigm,
        )

        # Test private method _get_sessions_per_subject returns the minimum number of sessions per subjects
        self.assertEqual(compound_dataset._get_sessions_per_subject(), self.n_sessions)
