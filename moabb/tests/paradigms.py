import unittest

from moabb.datasets.fake import FakeDataset
from moabb.paradigms import (BaseSSVEP,
                             LeftRightImagery, BaseMotorImagery,
                             FilterBankMotorImagery,
                             FilterBankLeftRightImagery)

import numpy as np


class Test_MotorImagery(unittest.TestCase):

    def test_BaseImagery_paradigm(self):

        class SimpleMotorImagery(BaseMotorImagery):

            def used_events(self, dataset):
                return dataset.event_id

        self.assertRaises(ValueError, SimpleMotorImagery, tmin=1, tmax=0)

        paradigm = SimpleMotorImagery()
        dataset = FakeDataset()
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

        # can work with filter bank
        paradigm = SimpleMotorImagery(filters=[[7, 12], [12, 24]])
        dataset = FakeDataset()
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # X must be a 3D Array
        self.assertEquals(len(X.shape), 4)
        self.assertEquals(X.shape[-1], 2)

        # test process_raw return empty list if raw does not contain any
        # selected event. cetain runs in dataset are event specific.
        dataset = FakeDataset()
        raw = dataset.get_data([1])[1]['session_0']['run_0']
        # add something on the event channel
        raw._data[-1] *= 10
        self.assertIsNone(paradigm.process_raw(raw, dataset))
        # zeros it out
        raw._data[-1] *= 0
        self.assertIsNone(paradigm.process_raw(raw, dataset))

    def test_leftright_paradigm(self):
        # we cant pass event to this class
        paradigm = LeftRightImagery()
        self.assertRaises(ValueError, LeftRightImagery, events=['a'])

        # does not accept dataset with bad event
        dataset = FakeDataset()
        self.assertRaises(AssertionError, paradigm.get_data, dataset)

        # with a good dataset
        dataset = FakeDataset(event_list=['left_hand', 'right_hand'])
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])
        self.assertEquals(len(np.unique(labels)), 2)
        self.assertEquals(list(np.unique(labels)), ['left_hand', 'right_hand'])

    def test_filter_bank_mi(self):
        # can work with filter bank
        paradigm = FilterBankMotorImagery()
        dataset = FakeDataset()
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # X must be a 3D Array
        self.assertEquals(len(X.shape), 4)
        self.assertEquals(X.shape[-1], 6)

        # can work with filter bank
        paradigm = FilterBankLeftRightImagery()
        dataset = FakeDataset(event_list=['left_hand', 'right_hand'])
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # X must be a 3D Array
        self.assertEquals(len(X.shape), 4)
        self.assertEquals(X.shape[-1], 6)


class Test_SSVEP(unittest.TestCase):

    def test_BaseSSVEP_paradigm(self):

        class SimpleSSVEP(BaseSSVEP):

            def used_events(self, dataset):
                return dataset.event_id

        self.assertRaises(ValueError, SimpleSSVEP, tmin=1, tmax=0)

        paradigm = SimpleSSVEP()
        dataset = FakeDataset(paradigm='ssvep')
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # Verify that they have the same length
        self.assertEquals(len(X), len(labels), len(metadata))
        # X must be a 3D Array
        self.assertEquals(len(X.shape), 3)
        # labels must contain 3 values
        self.assertEquals(len(np.unique(labels)), 3)

        # metadata must have subjets, sessions, runs
        self.assertTrue('subject' in metadata.columns)
        self.assertTrue('session' in metadata.columns)
        self.assertTrue('run' in metadata.columns)

        # Only one subject in the metadata
        self.assertEquals(np.unique(metadata.subject), 1)

        # we should have two sessions in the metadata
        self.assertEquals(len(np.unique(metadata.session)), 2)

        # Accept filters
        paradigm = SimpleSSVEP(filters=[[10.5, 11.5], [12.5, 13.5]])
        dataset = FakeDataset(paradigm='ssvep')
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])
        # X must be a 3D Array
        self.assertEquals(len(X.shape), 4)
        self.assertEquals(X.shape[-1], 2)
