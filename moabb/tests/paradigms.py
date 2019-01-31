import unittest

from moabb.datasets.fake import FakeDataset
from moabb.paradigms import (LeftRightImagery, BaseMotorImagery,
                             FilterBankMotorImagery,
                             FilterBankLeftRightImagery,
                             BaseP300, P300)

import numpy as np


class SimpleMotorImagery(BaseMotorImagery):  # Needed to assess BaseImagery
    def used_events(self, dataset):
        return dataset.event_id


class Test_MotorImagery(unittest.TestCase):

    def test_BaseImagery_paradigm(self):
        paradigm = SimpleMotorImagery()
        dataset = FakeDataset(paradigm='imagery')
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # we should have all the same length
        self.assertEqual(len(X), len(labels), len(metadata))
        # X must be a 3D Array
        self.assertEqual(len(X.shape), 3)
        # labels must contain 3 values
        self.assertEqual(len(np.unique(labels)), 3)

        # metadata must have subjets, sessions, runs
        self.assertTrue('subject' in metadata.columns)
        self.assertTrue('session' in metadata.columns)
        self.assertTrue('run' in metadata.columns)

        # we should have only one subject in the metadata
        self.assertEqual(np.unique(metadata.subject), 1)

        # we should have two sessions in the metadata
        self.assertEqual(len(np.unique(metadata.session)), 2)

    def test_BaseImagery_tmintmax(self):
        self.assertRaises(ValueError, SimpleMotorImagery, tmin=1, tmax=0)

    def test_BaseImagery_filters(self):
        # can work with filter bank
        paradigm = SimpleMotorImagery(filters=[[7, 12], [12, 24]])
        dataset = FakeDataset(paradigm='imagery')
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # X must be a 4D Array
        self.assertEqual(len(X.shape), 4)
        self.assertEqual(X.shape[-1], 2)

    def test_baseImagery_wrongevent(self):
        # test process_raw return empty list if raw does not contain any
        # selected event. cetain runs in dataset are event specific.
        paradigm = SimpleMotorImagery(filters=[[7, 12], [12, 24]])
        dataset = FakeDataset(paradigm='imagery')
        raw = dataset.get_data([1])[1]['session_0']['run_0']
        # add something on the event channel
        raw._data[-1] *= 10
        self.assertIsNone(paradigm.process_raw(raw, dataset))
        # zeros it out
        raw._data[-1] *= 0
        self.assertIsNone(paradigm.process_raw(raw, dataset))

    def test_BaseImagery_noevent(self):
        # Assert error if events from paradigm and dataset dont overlap
        paradigm = SimpleMotorImagery(events=['left_hand', 'right_hand'])
        dataset = FakeDataset(paradigm='imagery')
        self.assertRaises(AssertionError, paradigm.get_data, dataset)

    def test_LeftRightImagery_paradigm(self):
        # with a good dataset
        paradigm = LeftRightImagery()
        dataset = FakeDataset(event_list=['left_hand', 'right_hand'],
                              paradigm='imagery')
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertEqual(list(np.unique(labels)), ['left_hand', 'right_hand'])

    def test_LeftRightImagery_noevent(self):
        # we cant pass event to this class
        self.assertRaises(ValueError, LeftRightImagery, events=['a'])

    def test_LeftRightImagery_badevents(self):
        paradigm = LeftRightImagery()
        # does not accept dataset with bad event
        dataset = FakeDataset(paradigm='imagery')
        self.assertRaises(AssertionError, paradigm.get_data, dataset)

    def test_FilterBankMotorImagery_paradigm(self):
        # can work with filter bank
        paradigm = FilterBankMotorImagery()
        dataset = FakeDataset(paradigm='imagery')
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # X must be a 4D Array
        self.assertEqual(len(X.shape), 4)
        self.assertEqual(X.shape[-1], 6)

    def test_FilterBankMotorImagery_moreclassesthanevent(self):
        self.assertRaises(AssertionError, FilterBankMotorImagery, n_classes=3,
                          events=['hands', 'feet'])

    def test_FilterBankLeftRightImagery_paradigm(self):
        # can work with filter bank
        paradigm = FilterBankLeftRightImagery()
        dataset = FakeDataset(event_list=['left_hand', 'right_hand'],
                              paradigm='imagery')
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # X must be a 4D Array
        self.assertEqual(len(X.shape), 4)
        self.assertEqual(X.shape[-1], 6)


class SimpleP300(BaseP300):  # Needed to assess BaseP300
    def used_events(self, dataset):
        return dataset.event_id


class Test_P300(unittest.TestCase):

    def test_BaseP300_paradigm(self):
        paradigm = SimpleP300()
        dataset = FakeDataset(paradigm='p300')
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # we should have all the same lenght
        self.assertEquals(len(X), len(labels), len(metadata))
        # X must be a 3D Array
        self.assertEquals(len(X.shape), 3)
        # labels must contain 3 values
        self.assertEquals(len(np.unique(labels)), 3)

        # metadata must have subjets, sessions, runs
        self.assertTrue('subject' in metadata.columns)
        self.assertTrue('session' in metadata.columns)
        self.assertTrue('run' in metadata.columns)

        # we should have only one subject in the metadata
        self.assertEquals(np.unique(metadata.subject), 1)

        # we should have two sessions in the metadata
        self.assertEquals(len(np.unique(metadata.session)), 2)

    def test_BaseP300_tmintmax(self):
        self.assertRaises(ValueError, SimpleP300, tmin=1, tmax=0)

    def test_BaseP300_filters(self):
        # can work with filter bank
        paradigm = SimpleP300(filters=[[1, 12], [12, 24]])
        dataset = FakeDataset(paradigm='p300')
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # X must be a 4D Array
        self.assertEqual(len(X.shape), 4)
        self.assertEqual(X.shape[-1], 2)

    def test_BaseP300_wrongevent(self):
        # test process_raw return empty list if raw does not contain any
        # selected event. cetain runs in dataset are event specific.
        paradigm = SimpleP300(filters=[[1, 12], [12, 24]])
        dataset = FakeDataset(paradigm='p300')
        raw = dataset.get_data([1])[1]['session_0']['run_0']
        # add something on the event channel
        raw._data[-1] *= 10
        self.assertIsNone(paradigm.process_raw(raw, dataset))
        # zeros it out
        raw._data[-1] *= 0
        self.assertIsNone(paradigm.process_raw(raw, dataset))

    def test_P300_specifyevent(self):
        # we cant pass event to this class
        self.assertRaises(ValueError, P300, events=['a'])

    def test_P300_wrongevent(self):
        # does not accept dataset with bad event
        paradigm = P300()
        dataset = FakeDataset(paradigm='p300')
        self.assertRaises(AssertionError, paradigm.get_data, dataset)

    def test_P300_paradigm(self):
        # with a good dataset
        paradigm = P300()
        dataset = FakeDataset(event_list=['Target', 'NonTarget'],
                              paradigm='p300')
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])
        self.assertEquals(len(np.unique(labels)), 2)
        self.assertEquals(list(np.unique(labels)),
                          sorted(['Target', 'NonTarget']))
