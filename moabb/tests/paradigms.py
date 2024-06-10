import logging
import shutil
import tempfile
import unittest
from math import ceil

import numpy as np
import pandas as pd
import pytest
from mne import BaseEpochs
from mne.io import BaseRaw

from moabb.datasets import BNCI2014_001
from moabb.datasets.fake import FakeDataset
from moabb.paradigms import (
    CVEP,
    P300,
    SSVEP,
    BaseCVEP,
    BaseMotorImagery,
    BaseP300,
    BaseSSVEP,
    FakeCVEPParadigm,
    FilterBankCVEP,
    FilterBankFixedIntervalWindowsProcessing,
    FilterBankLeftRightImagery,
    FilterBankMotorImagery,
    FilterBankSSVEP,
    FixedIntervalWindowsProcessing,
    LeftRightImagery,
    MotorImagery,
    RestingStateToP300Adapter,
)


log = logging.getLogger(__name__)
log.setLevel(logging.ERROR)


class SimpleMotorImagery(BaseMotorImagery):  # Needed to assess BaseImagery
    def used_events(self, dataset):
        return dataset.event_id


class Test_MotorImagery(unittest.TestCase):
    def test_BaseImagery_paradigm(self):
        paradigm = SimpleMotorImagery()
        dataset = FakeDataset(paradigm="imagery")
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # we should have all the same length
        self.assertEqual(len(X), len(labels), len(metadata))
        # X must be a 3D Array
        self.assertEqual(len(X.shape), 3)
        # labels must contain 3 values
        self.assertEqual(len(np.unique(labels)), 3)
        # metadata must have subjets, sessions, runs
        self.assertTrue("subject" in metadata.columns)
        self.assertTrue("session" in metadata.columns)
        self.assertTrue("run" in metadata.columns)
        # we should have only one subject in the metadata
        self.assertEqual(np.unique(metadata.subject), 1)
        # we should have two sessions in the metadata
        self.assertEqual(len(np.unique(metadata.session)), 2)
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)
        # should return raws
        raws, _, _ = paradigm.get_data(dataset, subjects=[1], return_raws=True)
        for raw in raws:
            self.assertIsInstance(raw, BaseRaw)
        # should raise error
        self.assertRaises(
            ValueError,
            paradigm.get_data,
            dataset,
            subjects=[1],
            return_epochs=True,
            return_raws=True,
        )

    def test_BaseImagery_channel_order(self):
        """Test if paradigm return correct channel order, see issue #227."""
        datasetA = FakeDataset(paradigm="imagery", channels=["C3", "Cz", "C4"])
        datasetB = FakeDataset(paradigm="imagery", channels=["Cz", "C4", "C3"])
        paradigm = SimpleMotorImagery(channels=["C4", "C3", "Cz"])

        ep1, _, _ = paradigm.get_data(datasetA, subjects=[1], return_epochs=True)
        ep2, _, _ = paradigm.get_data(datasetB, subjects=[1], return_epochs=True)
        self.assertEqual(ep1.info["ch_names"], ep2.info["ch_names"])

    def test_BaseImagery_tmintmax(self):
        self.assertRaises(ValueError, SimpleMotorImagery, tmin=1, tmax=0)

    def test_BaseImagery_filters(self):
        # can work with filter bank
        paradigm = SimpleMotorImagery(filters=[[7, 12], [12, 24]])
        dataset = FakeDataset(paradigm="imagery")
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # X must be a 4D Array
        self.assertEqual(len(X.shape), 4)
        self.assertEqual(X.shape[-1], 2)
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)

    def test_baseImagery_wrongevent(self):
        # test process_raw return empty list if raw does not contain any
        # selected event. certain runs in dataset are event specific.
        paradigm = SimpleMotorImagery(filters=[[7, 12], [12, 24]])
        dataset = FakeDataset(paradigm="imagery")
        epochs_pipeline = paradigm._get_epochs_pipeline(
            return_epochs=True, return_raws=False, dataset=dataset
        )
        # no stim channel after loading cache
        raw = dataset.get_data([1], cache_config=dict(use=False, save_raw=False))[1]["0"][
            "0"
        ]
        raw.load_data()
        self.assertEqual("stim", raw.ch_names[-1])
        # add something on the event channel
        raw._data[-1] *= 10
        with self.assertRaises(ValueError, msg="No events found"):
            epochs_pipeline.transform(raw)
        # zeros it out
        raw._data[-1] *= 0
        with self.assertRaises(ValueError, msg="No events found"):
            epochs_pipeline.transform(raw)

    def test_BaseImagery_noevent(self):
        # Assert error if events from paradigm and dataset dont overlap
        paradigm = SimpleMotorImagery(events=["left_hand", "right_hand"])
        dataset = FakeDataset(paradigm="imagery")
        self.assertRaises(AssertionError, paradigm.get_data, dataset)

    def test_BaseImagery_droppedevent(self):
        dataset = FakeDataset(paradigm="imagery")
        tmax = dataset.interval[1]
        # with regular windows, all epochs should be valid:
        paradigm1 = SimpleMotorImagery(tmax=tmax)
        # with large windows, some epochs will have to be dropped:
        paradigm2 = SimpleMotorImagery(tmax=10 * tmax)
        # with epochs:
        epochs1, labels1, metadata1 = paradigm1.get_data(dataset, return_epochs=True)
        epochs2, labels2, metadata2 = paradigm2.get_data(dataset, return_epochs=True)
        self.assertEqual(len(epochs1), len(labels1), len(metadata1))
        self.assertEqual(len(epochs2), len(labels2), len(metadata2))
        self.assertGreater(len(epochs1), len(epochs2))
        # with np.array:
        X1, labels1, metadata1 = paradigm1.get_data(dataset)
        X2, labels2, metadata2 = paradigm2.get_data(dataset)
        self.assertEqual(len(X1), len(labels1), len(metadata1))
        self.assertEqual(len(X2), len(labels2), len(metadata2))
        self.assertGreater(len(X1), len(X2))

    def test_BaseImagery_epochsmetadata(self):
        dataset = FakeDataset(paradigm="imagery")
        paradigm = SimpleMotorImagery()
        epochs, _, metadata = paradigm.get_data(dataset, return_epochs=True)
        # does not work with multiple filters:
        self.assertTrue(metadata.equals(epochs.metadata))

    def test_BaseImagery_cache(self):
        tempdir = tempfile.mkdtemp()
        dataset = FakeDataset(paradigm="imagery", n_sessions=1, n_runs=1)
        paradigm = SimpleMotorImagery()
        # We save the full cache (raws, epochs, arrays):
        from moabb import set_log_level

        set_log_level("INFO")
        with self.assertLogs(logger="moabb.datasets.bids_interface", level="INFO") as cm:
            _ = paradigm.get_data(
                dataset,
                subjects=[1],
                cache_config=dict(
                    use=True,
                    path=tempdir,
                    save_raw=True,
                    save_epochs=True,
                    save_array=True,
                    overwrite_raw=False,
                    overwrite_epochs=False,
                    overwrite_array=False,
                ),
            )
        print("\n".join(cm.output))
        expected = [
            "Attempting to retrieve cache .* datatype-array",
            "No cache found at",
            "Attempting to retrieve cache .* datatype-epo",
            "No cache found at",
            "Attempting to retrieve cache .* datatype-eeg",  # raw_pipeline
            "No cache found at",
            "Attempting to retrieve cache .* datatype-eeg",
            # SetRawAnnotations pipeline
            "No cache found at",
            "Starting caching .* datatype-eeg",
            "Finished caching .* datatype-eeg",
            "Starting caching .* datatype-epo",
            "Finished caching .* datatype-epo",
            "Starting caching .* datatype-array",
            "Finished caching .* datatype-array",
        ]
        self.assertEqual(len(expected), len(cm.output))
        for i, regex in enumerate(expected):
            self.assertRegex(cm.output[i], regex)

        # Test loading the array cache:
        with self.assertLogs(logger="moabb.datasets.bids_interface", level="INFO") as cm:
            _ = paradigm.get_data(
                dataset,
                subjects=[1],
                cache_config=dict(
                    use=True,
                    path=tempdir,
                    save_raw=False,
                    save_epochs=False,
                    save_array=False,
                    overwrite_raw=False,
                    overwrite_epochs=False,
                    overwrite_array=False,
                ),
            )
        print("\n".join(cm.output))
        expected = [
            "Attempting to retrieve cache .* datatype-array",
            "Finished reading cache .* datatype-array",
        ]
        self.assertEqual(len(expected), len(cm.output))
        for i, regex in enumerate(expected):
            self.assertRegex(cm.output[i], regex)

        # Test loading the epochs cache:
        with self.assertLogs(logger="moabb.datasets.bids_interface", level="INFO") as cm:
            _ = paradigm.get_data(
                dataset,
                subjects=[1],
                cache_config=dict(
                    use=True,
                    path=tempdir,
                    save_raw=False,
                    save_epochs=False,
                    save_array=False,
                    overwrite_raw=False,
                    overwrite_epochs=False,
                    overwrite_array=True,
                ),
            )
        print("\n".join(cm.output))
        expected = [
            "Starting erasing cache .* datatype-array",
            "Finished erasing cache .* datatype-array",
            "Attempting to retrieve cache .* datatype-epo",
            "Finished reading cache .* datatype-epo",
        ]
        self.assertEqual(len(expected), len(cm.output))
        for i, regex in enumerate(expected):
            self.assertRegex(cm.output[i], regex)

        # Test loading the raw cache:
        with self.assertLogs(logger="moabb.datasets.bids_interface", level="INFO") as cm:
            _ = paradigm.get_data(
                dataset,
                subjects=[1],
                cache_config=dict(
                    use=True,
                    path=tempdir,
                    save_raw=False,
                    save_epochs=False,
                    save_array=False,
                    overwrite_raw=False,
                    overwrite_epochs=True,
                    overwrite_array=False,
                ),
            )
        print("\n".join(cm.output))
        expected = [
            "Attempting to retrieve cache .* datatype-array",
            "No cache found at",
            "Starting erasing cache .* datatype-epo",
            "Finished erasing cache .* datatype-epo",
            "Attempting to retrieve cache .* datatype-eeg",
            "Finished reading cache .* datatype-eeg",
        ]
        self.assertEqual(len(expected), len(cm.output))
        for i, regex in enumerate(expected):
            self.assertRegex(cm.output[i], regex)
        shutil.rmtree(tempdir)

    def test_LeftRightImagery_paradigm(self):
        # with a good dataset
        paradigm = LeftRightImagery()
        dataset = FakeDataset(event_list=["left_hand", "right_hand"], paradigm="imagery")
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        self.assertEqual(len(np.unique(labels)), 2)
        self.assertEqual(list(np.unique(labels)), ["left_hand", "right_hand"])
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)

    def test_LeftRightImagery_noevent(self):
        # we can't pass event to this class
        self.assertRaises(ValueError, LeftRightImagery, events=["a"])

    def test_LeftRightImagery_badevents(self):
        paradigm = LeftRightImagery()
        # does not accept dataset with bad event
        dataset = FakeDataset(paradigm="imagery")
        self.assertRaises(AssertionError, paradigm.get_data, dataset)

    def test_FilterBankMotorImagery_paradigm(self):
        # can work with filter bank
        paradigm = FilterBankMotorImagery()
        dataset = FakeDataset(paradigm="imagery")
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # X must be a 4D Array
        self.assertEqual(len(X.shape), 4)
        self.assertEqual(X.shape[-1], 6)
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)

    def test_FilterBankMotorImagery_moreclassesthanevent(self):
        self.assertRaises(
            AssertionError, FilterBankMotorImagery, n_classes=3, events=["hands", "feet"]
        )

    def test_FilterBankLeftRightImagery_paradigm(self):
        # can work with filter bank
        paradigm = FilterBankLeftRightImagery()
        dataset = FakeDataset(event_list=["left_hand", "right_hand"], paradigm="imagery")
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # X must be a 4D Array
        self.assertEqual(len(X.shape), 4)
        self.assertEqual(X.shape[-1], 6)
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)


class SimpleP300(BaseP300):  # Needed to assess BaseP300
    def used_events(self, dataset):
        return dataset.event_id


class Test_P300(unittest.TestCase):
    def test_match_all(self):
        # Note: the match all property is implemented in the base paradigm.
        # Thus, although it is located in the P300 section, this test stands for all paradigms.
        paradigm = SimpleP300()
        dataset1 = FakeDataset(
            paradigm="p300",
            event_list=["Target", "NonTarget"],
            channels=["C3", "Cz", "Fz"],
            sfreq=64,
        )
        dataset2 = FakeDataset(
            paradigm="p300",
            event_list=["Target", "NonTarget"],
            channels=["C3", "C4", "Cz"],
            sfreq=256,
        )
        dataset3 = FakeDataset(
            paradigm="p300",
            event_list=["Target", "NonTarget"],
            channels=["C3", "Cz", "Fz", "C4"],
            sfreq=512,
        )
        shift = -0.5

        paradigm.match_all(
            [dataset1, dataset2, dataset3], shift=shift, channel_merge_strategy="union"
        )
        # match_all should returns the smallest frequency minus 0.5.
        # See comment inside the match_all method
        self.assertEqual(paradigm.resample, 64 + shift)
        self.assertEqual(paradigm.channels.sort(), ["C3", "Cz", "Fz", "C4"].sort())
        self.assertEqual(paradigm.interpolate_missing_channels, True)
        X, _, _ = paradigm.get_data(dataset1, subjects=[1])
        n_channels, _ = X[0].shape
        self.assertEqual(n_channels, 4)

        paradigm.match_all(
            [dataset1, dataset2, dataset3],
            shift=shift,
            channel_merge_strategy="intersect",
        )
        self.assertEqual(paradigm.resample, 64 + shift)
        self.assertEqual(paradigm.channels.sort(), ["C3", "Cz"].sort())
        self.assertEqual(paradigm.interpolate_missing_channels, False)
        X, _, _ = paradigm.get_data(dataset1, subjects=[1])
        n_channels, _ = X[0].shape
        self.assertEqual(n_channels, 2)

    def test_BaseP300_paradigm(self):
        paradigm = SimpleP300()
        dataset = FakeDataset(paradigm="p300", event_list=["Target", "NonTarget"])
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # we should have all the same length
        self.assertEqual(len(X), len(labels), len(metadata))
        # X must be a 3D Array
        self.assertEqual(len(X.shape), 3)
        # labels must contain 2 values (Target/NonTarget)
        self.assertEqual(len(np.unique(labels)), 2)
        # metadata must have subjets, sessions, runs
        self.assertTrue("subject" in metadata.columns)
        self.assertTrue("session" in metadata.columns)
        self.assertTrue("run" in metadata.columns)
        # we should have only one subject in the metadata
        self.assertEqual(np.unique(metadata.subject), 1)
        # we should have two sessions in the metadata
        self.assertEqual(len(np.unique(metadata.session)), 2)
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)
        # should return raws
        raws, _, _ = paradigm.get_data(dataset, subjects=[1], return_raws=True)
        for raw in raws:
            self.assertIsInstance(raw, BaseRaw)
        # should raise error
        self.assertRaises(
            ValueError,
            paradigm.get_data,
            dataset,
            subjects=[1],
            return_epochs=True,
            return_raws=True,
        )

    def test_BaseP300_channel_order(self):
        """Test if paradigm return correct channel order, see issue #227."""
        datasetA = FakeDataset(
            paradigm="p300",
            channels=["C3", "Cz", "C4"],
            event_list=["Target", "NonTarget"],
        )
        datasetB = FakeDataset(
            paradigm="p300",
            channels=["Cz", "C4", "C3"],
            event_list=["Target", "NonTarget"],
        )
        paradigm = SimpleP300(channels=["C4", "C3", "Cz"])

        ep1, _, _ = paradigm.get_data(datasetA, subjects=[1], return_epochs=True)
        ep2, _, _ = paradigm.get_data(datasetB, subjects=[1], return_epochs=True)
        self.assertEqual(ep1.info["ch_names"], ep2.info["ch_names"])

    def test_BaseP300_tmintmax(self):
        self.assertRaises(ValueError, SimpleP300, tmin=1, tmax=0)

    def test_BaseP300_filters(self):
        # can work with filter bank
        paradigm = SimpleP300(filters=[[1, 12], [12, 24]])
        dataset = FakeDataset(paradigm="p300", event_list=["Target", "NonTarget"])
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # X must be a 4D Array
        self.assertEqual(len(X.shape), 4)
        self.assertEqual(X.shape[-1], 2)
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)

    def test_BaseP300_wrongevent(self):
        # test process_raw return empty list if raw does not contain any
        # selected event. certain runs in dataset are event specific.
        paradigm = SimpleP300(filters=[[1, 12], [12, 24]])
        dataset = FakeDataset(paradigm="p300", event_list=["Target", "NonTarget"])
        epochs_pipeline = paradigm._get_epochs_pipeline(
            return_epochs=True, return_raws=False, dataset=dataset
        )
        # no stim channel after loading cache
        raw = dataset.get_data([1], cache_config=dict(use=False, save_raw=False))[1]["0"][
            "0"
        ]
        raw.load_data()
        self.assertEqual("stim", raw.ch_names[-1])
        # add something on the event channel
        raw._data[-1] *= 10
        with self.assertRaises(ValueError, msg="No events found"):
            epochs_pipeline.transform(raw)
        # zeros it out
        raw._data[-1] *= 0
        with self.assertRaises(ValueError, msg="No events found"):
            epochs_pipeline.transform(raw)

    def test_BaseP300_droppedevent(self):
        dataset = FakeDataset(paradigm="p300", event_list=["Target", "NonTarget"])
        tmax = dataset.interval[1]
        # with regular windows, all epochs should be valid:
        paradigm1 = SimpleP300(tmax=tmax)
        # with large windows, some epochs will have to be dropped:
        paradigm2 = SimpleP300(tmax=10 * tmax)
        # with epochs:
        epochs1, labels1, metadata1 = paradigm1.get_data(dataset, return_epochs=True)
        epochs2, labels2, metadata2 = paradigm2.get_data(dataset, return_epochs=True)
        self.assertEqual(len(epochs1), len(labels1), len(metadata1))
        self.assertEqual(len(epochs2), len(labels2), len(metadata2))
        self.assertGreater(len(epochs1), len(epochs2))
        # with np.array:
        X1, labels1, metadata1 = paradigm1.get_data(dataset)
        X2, labels2, metadata2 = paradigm2.get_data(dataset)
        self.assertEqual(len(X1), len(labels1), len(metadata1))
        self.assertEqual(len(X2), len(labels2), len(metadata2))
        self.assertGreater(len(X1), len(X2))

    def test_BaseP300_epochsmetadata(self):
        dataset = FakeDataset(paradigm="p300", event_list=["Target", "NonTarget"])
        paradigm = SimpleP300()
        epochs, _, metadata = paradigm.get_data(dataset, return_epochs=True)
        # does not work with multiple filters:
        self.assertTrue(metadata.equals(epochs.metadata))

    def test_P300_specifyevent(self):
        # we can't pass event to this class
        self.assertRaises(ValueError, P300, events=["a"])

    def test_P300_wrongevent(self):
        # does not accept dataset with bad event
        paradigm = P300()
        dataset = FakeDataset(paradigm="p300")
        self.assertRaises(AssertionError, paradigm.get_data, dataset)

    def test_P300_paradigm(self):
        # with a good dataset
        paradigm = P300()
        dataset = FakeDataset(event_list=["Target", "NonTarget"], paradigm="p300")
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertEqual(list(np.unique(labels)), sorted(["Target", "NonTarget"]))
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)


class Test_RestingState(unittest.TestCase):
    def test_RestingState_paradigm(self):
        event_list = ["Open", "Close"]
        paradigm = RestingStateToP300Adapter(events=event_list)
        dataset = FakeDataset(paradigm="rstate", event_list=event_list)
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # we should have all the same length
        self.assertEqual(len(X), len(labels), len(metadata))
        # X must be a 3D Array
        self.assertEqual(len(X.shape), 3)
        # labels must contain 2 values (Open/Close)
        self.assertEqual(len(np.unique(labels)), 2)
        # metadata must have subjets, sessions, runs
        self.assertTrue("subject" in metadata.columns)
        self.assertTrue("session" in metadata.columns)
        self.assertTrue("run" in metadata.columns)
        # we should have only one subject in the metadata
        self.assertEqual(np.unique(metadata.subject), 1)
        # we should have two sessions in the metadata
        self.assertEqual(len(np.unique(metadata.session)), 2)
        # should return epochs
        epochs, _, _ = paradigm.get_data(
            dataset,
            subjects=[1],
            return_epochs=True,
        )
        self.assertIsInstance(epochs, BaseEpochs)
        # should return raws
        raws, _, _ = paradigm.get_data(
            dataset,
            subjects=[1],
            return_raws=True,
        )
        for raw in raws:
            self.assertIsInstance(raw, BaseRaw)
        # should raise error
        self.assertRaises(
            ValueError,
            paradigm.get_data,
            dataset,
            subjects=[1],
            return_epochs=True,
            return_raws=True,
        )

    def test_RestingState_default_values(self):
        paradigm = RestingStateToP300Adapter()
        assert paradigm.tmin == 10
        assert paradigm.tmax == 50
        assert paradigm.fmin == 1
        assert paradigm.fmax == 35
        assert paradigm.resample == 128


class Test_SSVEP(unittest.TestCase):
    def test_BaseSSVEP_paradigm(self):
        paradigm = BaseSSVEP(n_classes=None)
        dataset = FakeDataset(paradigm="ssvep")
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # Verify that they have the same length
        self.assertEqual(len(X), len(labels), len(metadata))
        # X must be a 3D array
        self.assertEqual(len(X.shape), 3)
        # labels must contain 3 values
        self.assertEqual(len(np.unique(labels)), 3)
        # metadata must have subjets, sessions, runs
        self.assertTrue("subject" in metadata.columns)
        self.assertTrue("session" in metadata.columns)
        self.assertTrue("run" in metadata.columns)
        # Only one subject in the metadata
        self.assertEqual(np.unique(metadata.subject), 1)
        # we should have two sessions in the metadata, n_classes = 2 as default
        self.assertEqual(len(np.unique(metadata.session)), 2)
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)
        # should return raws
        raws, _, _ = paradigm.get_data(dataset, subjects=[1], return_raws=True)
        for raw in raws:
            self.assertIsInstance(raw, BaseRaw)
        # should raise error
        self.assertRaises(
            ValueError,
            paradigm.get_data,
            dataset,
            subjects=[1],
            return_epochs=True,
            return_raws=True,
        )

    def test_BaseSSVEP_channel_order(self):
        """Test if paradigm return correct channel order, see issue #227."""
        datasetA = FakeDataset(paradigm="ssvep", channels=["C3", "Cz", "C4"])
        datasetB = FakeDataset(paradigm="ssvep", channels=["Cz", "C4", "C3"])
        paradigm = BaseSSVEP(channels=["C4", "C3", "Cz"])

        ep1, _, _ = paradigm.get_data(datasetA, subjects=[1], return_epochs=True)
        ep2, _, _ = paradigm.get_data(datasetB, subjects=[1], return_epochs=True)
        self.assertEqual(ep1.info["ch_names"], ep2.info["ch_names"])

    def test_baseSSVEP_tmintmax(self):
        # Verify that tmin < tmax
        self.assertRaises(ValueError, BaseSSVEP, tmin=1, tmax=0)

    def test_BaseSSVEP_filters(self):
        # Accept filters
        paradigm = BaseSSVEP(filters=[(10.5, 11.5), (12.5, 13.5)])
        dataset = FakeDataset(paradigm="ssvep")
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # X must be a 4D array
        self.assertEqual(len(X.shape), 4)
        # Last dim should be 2 as the number of filters
        self.assertEqual(X.shape[-1], 2)
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)

    def test_BaseSSVEP_nclasses_default(self):
        # Default is with 3 classes
        paradigm = BaseSSVEP()
        dataset = FakeDataset(paradigm="ssvep")
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # labels must contain all 3 classes of dataset,
        # as n_classes is "None" by default (taking all classes)
        self.assertEqual(len(np.unique(labels)), 3)

    def test_BaseSSVEP_specified_nclasses(self):
        # Set the number of classes
        paradigm = BaseSSVEP(n_classes=3)
        dataset = FakeDataset(event_list=["13", "15", "17", "19"], paradigm="ssvep")
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # labels must contain 3 values
        self.assertEqual(len(np.unique(labels)), 3)

    def test_BaseSSVEP_toomany_nclasses(self):
        paradigm = BaseSSVEP(n_classes=4)
        dataset = FakeDataset(event_list=["13", "15"], paradigm="ssvep")
        self.assertRaises(ValueError, paradigm.get_data, dataset)

    def test_BaseSSVEP_moreclassesthanevent(self):
        self.assertRaises(AssertionError, BaseSSVEP, n_classes=3, events=["13.", "14."])

    def test_BaseSSVEP_droppedevent(self):
        dataset = FakeDataset(paradigm="ssvep")
        tmax = dataset.interval[1]
        # with regular windows, all epochs should be valid:
        paradigm1 = BaseSSVEP(tmax=tmax)
        # with large windows, some epochs will have to be dropped:
        paradigm2 = BaseSSVEP(tmax=10 * tmax)
        # with epochs:
        epochs1, labels1, metadata1 = paradigm1.get_data(dataset, return_epochs=True)
        epochs2, labels2, metadata2 = paradigm2.get_data(dataset, return_epochs=True)
        self.assertEqual(len(epochs1), len(labels1), len(metadata1))
        self.assertEqual(len(epochs2), len(labels2), len(metadata2))
        self.assertGreater(len(epochs1), len(epochs2))
        # with np.array:
        X1, labels1, metadata1 = paradigm1.get_data(dataset)
        X2, labels2, metadata2 = paradigm2.get_data(dataset)
        self.assertEqual(len(X1), len(labels1), len(metadata1))
        self.assertEqual(len(X2), len(labels2), len(metadata2))
        self.assertGreater(len(X1), len(X2))

    def test_BaseSSVEP_epochsmetadata(self):
        dataset = FakeDataset(paradigm="ssvep")
        paradigm = BaseSSVEP()
        epochs, _, metadata = paradigm.get_data(dataset, return_epochs=True)
        # does not work with multiple filters:
        self.assertTrue(metadata.equals(epochs.metadata))

    def test_SSVEP_noevent(self):
        # Assert error if events from paradigm and dataset dont overlap
        paradigm = SSVEP(events=["11", "12"], n_classes=2)
        dataset = FakeDataset(event_list=["13", "14"], paradigm="ssvep")
        self.assertRaises(AssertionError, paradigm.get_data, dataset)

    def test_SSVEP_paradigm(self):
        paradigm = SSVEP(n_classes=None)
        dataset = FakeDataset(event_list=["13", "15", "17", "19"], paradigm="ssvep")
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # Verify that they have the same length
        self.assertEqual(len(X), len(labels), len(metadata))
        # X must be a 3D array
        self.assertEqual(len(X.shape), 3)
        # labels must contain 4 values, defined in the dataset
        self.assertEqual(len(np.unique(labels)), 4)
        # metadata must have subjets, sessions, runs
        self.assertTrue("subject" in metadata.columns)
        self.assertTrue("session" in metadata.columns)
        self.assertTrue("run" in metadata.columns)
        # Only one subject in the metadata
        self.assertEqual(np.unique(metadata.subject), 1)
        # We should have two sessions in the metadata
        self.assertEqual(len(np.unique(metadata.session)), 2)
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)

    def test_SSVEP_singlepass(self):
        # Accept only single pass filter
        paradigm = SSVEP(fmin=2, fmax=25)
        dataset = FakeDataset(paradigm="ssvep")
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # Verify that they have the same length
        self.assertEqual(len(X), len(labels), len(metadata))
        # X must be a 3D array
        self.assertEqual(len(X.shape), 3)
        # labels must contain all 3 classes of dataset,
        # as n_classes is "None" by default (taking all classes)
        self.assertEqual(len(np.unique(labels)), 3)
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)

    def test_SSVEP_filter(self):
        # Do not accept multiple filters
        self.assertRaises(ValueError, SSVEP, filters=[(10.5, 11.5), (12.5, 13.5)])

    def test_FilterBankSSVEP_paradigm(self):
        # FilterBankSSVEP with all events
        paradigm = FilterBankSSVEP(n_classes=None)
        dataset = FakeDataset(event_list=["13", "15", "17", "19"], paradigm="ssvep")
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # X must be a 4D array
        self.assertEqual(len(X.shape), 4)
        # X must be a 4D array with d=4 as last dimension for the 4 events
        self.assertEqual(X.shape[-1], 4)
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)

    def test_FilterBankSSVEP_filters(self):
        # can work with filter bank
        paradigm = FilterBankSSVEP(filters=[(10.5, 11.5), (12.5, 13.5)])
        dataset = FakeDataset(event_list=["13", "15", "17"], paradigm="ssvep")
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # X must be a 4D array with d=2 as last dimension for the 2 filters
        self.assertEqual(len(X.shape), 4)
        self.assertEqual(X.shape[-1], 2)
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)


class Test_FixedIntervalWindowsProcessing(unittest.TestCase):
    def test_processing(self):
        processings = [
            FixedIntervalWindowsProcessing(length=0.51, stride=0.27, resample=99),
            FilterBankFixedIntervalWindowsProcessing(
                length=0.51, stride=0.27, resample=99, filters=[[8, 35]]
            ),
        ]
        for processing in processings:
            for paradigm_name in ["ssvep", "p300", "imagery"]:
                dataset = FakeDataset(
                    paradigm=paradigm_name,
                    n_sessions=1,
                    n_runs=1,
                    duration=55.4,
                    sfreq=128,
                )
                X, labels, metadata = processing.get_data(dataset, subjects=[1])

                # Verify that they have the same length
                self.assertEqual(len(X), len(labels), len(metadata))
                # X must be a 3D array
                self.assertEqual(len(X.shape), 3)
                # labels must contain 3 values
                self.assertTrue(all(label == "Window" for label in labels))
                # metadata must have subjets, sessions, runs
                self.assertTrue("subject" in metadata.columns)
                self.assertTrue("session" in metadata.columns)
                self.assertTrue("run" in metadata.columns)
                # Only one subject in the metadata
                self.assertEqual(np.unique(metadata.subject), 1)
                self.assertEqual(len(np.unique(metadata.session)), 1)
                # should return epochs
                epochs, _, _ = processing.get_data(
                    dataset, subjects=[1], return_epochs=True
                )
                self.assertIsInstance(epochs, BaseEpochs)
                # should return raws
                raws, _, _ = processing.get_data(dataset, subjects=[1], return_raws=True)
                for raw in raws:
                    self.assertIsInstance(raw, BaseRaw)
                n_times = int(55.4 * 128)
                n_epochs = ceil(
                    (n_times - int(processing.length * 128))
                    / int(processing.stride * 128)
                )  # no start/stop offset
                self.assertEqual(n_epochs, len(epochs))
                # should raise error
                self.assertRaises(
                    ValueError,
                    processing.get_data,
                    dataset,
                    subjects=[1],
                    return_epochs=True,
                    return_raws=True,
                )


# Define the FakeLogger class outside of the test class
class FakeLogger:
    def __init__(self):
        self.messages = []

    def warning(self, message):
        self.messages.append(message)

    def format_warning(self, message):
        return f"WARNING: {message}"


class Test_CVEP:
    @pytest.fixture
    def fix_paradigm(self):
        return BaseCVEP()

    @pytest.fixture
    def fix_fake_dataset(self):
        return FakeDataset(event_list=["1.0", "0.0"], paradigm="cvep")

    @pytest.fixture
    def fix_cvep_paradigm(self):
        return CVEP()

    @pytest.fixture(params=[BaseCVEP(), CVEP()])
    def fix_test_paradigm(self, request):
        return request.param

    @pytest.fixture(params=[BaseCVEP, FilterBankCVEP])
    def fix_test_filters_paradigm(self, request):
        return request.param

    @pytest.fixture
    def fix_two_fake_dataset(self):
        datasetA = FakeDataset(paradigm="cvep", channels=["C3", "Cz", "C4"])
        datasetB = FakeDataset(paradigm="cvep", channels=["Cz", "C4", "C3"])
        paradigm = BaseCVEP(channels=["C4", "C3", "Cz"])

        ep1, _, _ = paradigm.get_data(datasetA, subjects=[1], return_epochs=True)
        ep2, _, _ = paradigm.get_data(datasetB, subjects=[1], return_epochs=True)
        return ep1, ep2

    def test_base_cvep_init(self):
        # Test without events
        base_cvep = BaseCVEP(n_classes=2)
        assert base_cvep

        # Test with events
        base_cvep = BaseCVEP(events=["event1", "event2"], n_classes=2)
        assert base_cvep

        # Test with more classes than events (should raise an exception)
        with pytest.raises(AssertionError):
            _ = BaseCVEP(events=["event1"], n_classes=2)

    def test_base_cvep_is_valid(self, fix_fake_dataset) -> None:
        base_cvep = BaseCVEP()
        assert base_cvep.is_valid(fix_fake_dataset)

    def test_is_valid_with_valid_dataset(self, fix_fake_dataset):
        base_cvep = BaseCVEP(events=["1.0", "0.0"], n_classes=2)
        assert base_cvep.is_valid(fix_fake_dataset)

    def test_is_valid_with_invalid_paradigm(self):
        invalid_dataset = FakeDataset(paradigm="ssvep")
        base_cvep = BaseCVEP(events=["1.0", "0.0"], n_classes=2)
        assert not base_cvep.is_valid(invalid_dataset)

    def test_is_valid_with_missing_events(self):
        valid_dataset = FakeDataset(paradigm="cvep", event_list=["1.0"])
        base_cvep = BaseCVEP(events=["1.0", "0.0"], n_classes=2)
        assert not base_cvep.is_valid(valid_dataset)

    def test_cvep_init(self):
        cvep = CVEP()
        assert cvep

    def test_used_events_with_all_events(self, fix_fake_dataset):
        base_cvep = BaseCVEP(events=None, n_classes=2)
        used_events = base_cvep.used_events(fix_fake_dataset)
        assert used_events == {"1.0": 1, "0.0": 2}

    def test_used_events_with_specific_events(self, fix_fake_dataset):
        base_cvep = BaseCVEP(events=["1.0"], n_classes=1)
        used_events = base_cvep.used_events(fix_fake_dataset)
        assert used_events == {"1.0": 1}

    def test_used_events_with_insufficient_events(self, fix_fake_dataset):
        base_cvep = BaseCVEP(events=["1.0", "0.0", "2.0"], n_classes=3)
        with pytest.raises(ValueError):
            base_cvep.used_events(fix_fake_dataset)

    def test_filterbank_cvep_init(self):
        filterbank_cvep = FilterBankCVEP()
        assert filterbank_cvep

    def test_fake_cvep_paradigm_datasets(self) -> None:
        fake_cvep = FakeCVEPParadigm()
        datasets = fake_cvep.datasets
        assert len(datasets) == 1
        assert isinstance(datasets[0], FakeDataset)

    def test_get_data_with_paradigm(self, fix_test_paradigm, fix_fake_dataset) -> None:
        X, labels, metadata = fix_test_paradigm.get_data(fix_fake_dataset, subjects=[1])

        # Verify that they have the same length
        assert len(X) == len(labels) == len(metadata)
        # X must be a 3D array
        assert len(X.shape) == 3
        # labels must contain 2 values
        assert len(np.unique(labels)) == 2
        # metadata must have subjects, sessions, runs
        assert "subject" in metadata.columns
        assert "session" in metadata.columns
        assert "run" in metadata.columns
        # Only one subject in the metadata
        assert len(np.unique(metadata.subject)) == 1
        # we should have two sessions in the metadata, n_classes = 2 as default
        assert len(np.unique(metadata.session)) == 2

        epochs, _, _ = fix_test_paradigm.get_data(
            fix_fake_dataset, subjects=[1], return_epochs=True
        )
        assert isinstance(epochs, BaseEpochs)

        raws, _, _ = fix_test_paradigm.get_data(
            fix_fake_dataset, subjects=[1], return_raws=True
        )
        for raw in raws:
            assert isinstance(raw, BaseRaw)

        with pytest.raises(ValueError):
            fix_test_paradigm.get_data(
                fix_fake_dataset, subjects=[1], return_epochs=True, return_raws=True
            )

    def test_channel_order_consistency(self, fix_two_fake_dataset) -> None:
        ep1, ep2 = fix_two_fake_dataset
        assert ep1.info["ch_names"] == ep2.info["ch_names"]

    def test_channel_location_consistency(self, fix_two_fake_dataset) -> None:
        """Checking if location vector is equal"""
        ep1, ep2 = fix_two_fake_dataset

        locs_ep1 = np.array([ch["loc"] for ch in ep1.info["chs"]])
        locs_ep2 = np.array([ch["loc"] for ch in ep2.info["chs"]])

        for ch1, ch2 in zip(locs_ep1, locs_ep2):
            assert (ch1[:6] == ch2[:6]).all()

    def test_invalid_tmin_tmax_raises_error(self) -> None:
        with pytest.raises(ValueError):
            BaseCVEP(tmin=1, tmax=0)

    def test_get_data_with_filters(
        self, fix_fake_dataset, fix_test_filters_paradigm
    ) -> None:
        filters = ((1.0, 45.0), (12.0, 45.0))
        paradigm = fix_test_filters_paradigm(filters=filters)
        X, _, _ = paradigm.get_data(fix_fake_dataset, subjects=[1])

        # X must be a 4D array
        assert len(X.shape) == 4
        # Last dim should be 2 as the number of filters
        assert X.shape[-1] == len(filters)
        # should return MNE epochs format
        epochs, _, _ = paradigm.get_data(
            fix_fake_dataset, subjects=[1], return_epochs=True
        )
        assert isinstance(epochs, BaseEpochs)

    def test_default_classes_extraction(self, fix_fake_dataset, fix_paradigm) -> None:
        X, labels, _ = fix_paradigm.get_data(fix_fake_dataset, subjects=[1])

        assert len(np.unique(labels)) == 2

    def test_specified_classes_extraction(self) -> None:
        # Set the number of classes
        n_classes = 3
        paradigm = BaseCVEP(n_classes=3)
        dataset = FakeDataset(
            event_list=["0.0", "0.25", "0.5", "0.75", "1.0"], paradigm="cvep"
        )
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])
        # labels must contain 3 values
        assert len(np.unique(labels)) == n_classes

    def test_too_many_classes_error(self) -> None:
        paradigm = BaseCVEP(n_classes=4)
        dataset = FakeDataset(event_list=["1.0", "0.0"], paradigm="cvep")
        with pytest.raises(ValueError):
            paradigm.get_data(dataset)

    def test_more_classes_than_events_error(self) -> None:
        with pytest.raises(AssertionError):
            BaseCVEP(n_classes=3, events=["1.0", "0.0"])

    def test_dropped_events_handling(self, fix_fake_dataset) -> None:
        tmax = fix_fake_dataset.interval[1]
        paradigm_regular = BaseCVEP(tmax=tmax)
        paradigm_large = BaseCVEP(tmax=10 * tmax)

        # Test with epochs
        epochs_regular, labels_regular, metadata_regular = paradigm_regular.get_data(
            fix_fake_dataset, return_epochs=True
        )
        epochs_large, labels_large, metadata_large = paradigm_large.get_data(
            fix_fake_dataset, return_epochs=True
        )

        assert len(epochs_regular) == len(labels_regular) == len(metadata_regular)
        assert len(epochs_large) == len(labels_large) == len(metadata_large)
        assert len(epochs_regular) > len(epochs_large)

        # Test with np.array
        X_regular, labels_regular, metadata_regular = paradigm_regular.get_data(
            fix_fake_dataset
        )
        X_large, labels_large, metadata_large = paradigm_large.get_data(fix_fake_dataset)

        assert len(X_regular) == len(labels_regular) == len(metadata_regular)
        assert len(X_large) == len(labels_large) == len(metadata_large)
        assert len(X_regular) > len(X_large)

    def test_epochs_metadata_equality(self, fix_fake_dataset, fix_paradigm) -> None:
        epochs, _, metadata = fix_paradigm.get_data(fix_fake_dataset, return_epochs=True)
        # does not work with multiple filters:
        assert metadata.equals(epochs.metadata)

    def test_no_event_overlap_error(self) -> None:
        # Assert error if events from paradigm and dataset dont overlap
        paradigm = CVEP(events=["1.0", "0.0"], n_classes=2)
        dataset = FakeDataset(event_list=["13", "14"], paradigm="cvep")
        with pytest.raises(AssertionError):
            paradigm.get_data(dataset)

    def test_single_pass_filter_configuration(self, fix_fake_dataset) -> None:
        # Accept only single pass filter
        paradigm = CVEP(fmin=2.0, fmax=40.0)
        X, labels, metadata = paradigm.get_data(fix_fake_dataset, subjects=[1])

        # Verify that they have the same length
        assert len(X) == len(labels) == len(metadata)
        # X must be a 3D array
        assert len(X.shape) == 3
        # labels must contain all 3 classes of dataset,
        # as n_classes is "None" by default (taking all classes)
        assert len(np.unique(labels)) == 2
        # should return epochs
        epochs, _, _ = paradigm.get_data(
            fix_fake_dataset, subjects=[1], return_epochs=True
        )
        isinstance(epochs, BaseEpochs)

    def test_multiple_filters_error(self) -> None:
        # Do not accept multiple filters
        with pytest.raises(ValueError):
            CVEP(filters=[(1.0, 45.0), (12.0, 45.0)])

    def test_filter_bank_data_generation(self, fix_fake_dataset) -> None:
        # can work with filter bank
        paradigm = FilterBankCVEP(filters=((1.0, 45.0), (12.0, 45.0)))
        X, labels, metadata = paradigm.get_data(fix_fake_dataset, subjects=[1])

        # X must be a 4D array with d=2 as last dimension for the 2 filters
        assert len(X.shape) == 4
        assert X.shape[-1] == 2
        # should return epochs
        epochs, _, _ = paradigm.get_data(
            fix_fake_dataset, subjects=[1], return_epochs=True
        )
        isinstance(epochs, BaseEpochs)

    def test_default_scoring_values(self):
        assert CVEP().scoring == "accuracy"
        assert CVEP(n_classes=3).scoring == "accuracy"
        assert CVEP(n_classes=2).scoring == "roc_auc"
        assert CVEP(events=["1.0", "0.0"], n_classes=2).scoring == "roc_auc"
        assert CVEP(events=["1.0", "0.5", "0.0"], n_classes=3).scoring == "accuracy"
        assert CVEP(events=["1.0", "0.5", "0.0"], n_classes=2).scoring == "roc_auc"


class TestParadigm:
    @pytest.fixture(
        params=[
            (SimpleMotorImagery, "imagery"),
            (SimpleP300, "p300"),
            (SSVEP, "ssvep"),
            (CVEP, "cvep"),
        ]
    )
    def paradigm_cls_name(self, request):
        paradigm, paradigm_name = request.param
        return paradigm, paradigm_name

    @pytest.mark.parametrize(
        "stim,annotations", [(True, False), (False, True), (True, True)]
    )
    def test_no_events(self, stim, annotations, paradigm_cls_name):
        # the paradigms should still be able to process the data
        # even if some runs have no events
        paradigm = paradigm_cls_name[0]()
        dataset = FakeDataset(
            stim=stim,
            annotations=annotations,
            paradigm=paradigm_cls_name[1],
            n_sessions=2,
            n_runs=2,
            n_events=[0, 10],
        )
        X, y, metadata = paradigm.get_data(dataset, subjects=[1])
        assert len(X) == len(y) == len(metadata) == 10 * 2


class Test_Data:
    @pytest.fixture
    def dataset(self):
        return BNCI2014_001()

    @pytest.fixture
    def paradigm(self):
        return MotorImagery(tmin=0.1, tmax=3)

    @pytest.fixture
    def epochs_labels_metadata(self, dataset, paradigm):
        return paradigm.get_data(dataset, subjects=[1], return_epochs=True)

    @pytest.fixture
    def X_labels_metadata(self, dataset, paradigm):
        return paradigm.get_data(dataset, subjects=[1])

    def test_compare_X_epochs(self, epochs_labels_metadata, X_labels_metadata, dataset):
        epo, labelsEpo, metadataEpo = epochs_labels_metadata
        X, labelsX, metadataX = X_labels_metadata
        assert len(labelsEpo) == len(labelsX)
        assert all(le == lx for le, lx in zip(labelsEpo, labelsX))
        pd.testing.assert_frame_equal(metadataX, metadataEpo)
        np.testing.assert_array_almost_equal(X, epo.get_data() * dataset.unit_factor)

    def test_epochs(self, epochs_labels_metadata, dataset):
        epo, labelsEpo, metadataEpo = epochs_labels_metadata
        # values computed form moabb 0.5:
        assert len(epo) == 576
        events = np.array(
            [
                [250, 0, 4],
                [2253, 0, 3],
                [4171, 0, 2],
            ]
        )
        np.testing.assert_array_equal(epo.events[:3], events)
        assert epo.tmin == 2.1
        assert epo.tmax == 5.0
        X = np.array(
            [
                7.49779224,
                3.07656416,
                5.59544847,
                6.19830887,
                8.04925351,
                6.97703295,
                3.48397992,
                3.61332516,
                5.17954447,
                4.32935335,
                6.10852621,
                4.81854389,
                5.04162646,
                3.48391445,
                3.50507861,
                1.35462057,
                2.47525825,
                2.67810322,
                1.38214336,
                0.25897466,
                0.24160032,
                -1.96250978,
            ]
        )
        np.testing.assert_array_almost_equal(
            epo.get_data()[0, :, 0] * dataset.unit_factor, X
        )
