import unittest

import numpy as np
import pytest
from braindecode.datasets import BaseConcatDataset, create_from_X_y
from mne import EpochsArray, create_info
from sklearn.preprocessing import LabelEncoder

from moabb.datasets.fake import FakeDataset
from moabb.pipelines.utils_pytorch import BraindecodeDatasetLoader
from moabb.tests import SimpleMotorImagery


@pytest.fixture(scope="module")
def data():
    """Return EEG data from dataset to test transformer"""
    paradigm = SimpleMotorImagery()
    dataset = FakeDataset(paradigm="imagery")
    X, labels, metadata = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
    y = LabelEncoder().fit_transform(labels)
    return X, y, labels, metadata


class TestTransformer:
    def test_transform_input_and_output_shape(self, data):
        X, y, _, info = data
        transformer = BraindecodeDatasetLoader()
        braindecode_dataset = transformer.fit_transform(X, y=y)
        assert (
            len(braindecode_dataset) == X.get_data().shape[0]
            and braindecode_dataset[0][0].shape[0] == X.get_data().shape[1]
            and braindecode_dataset[0][0].shape[1] == X.get_data().shape[2]
        )

    def test_sklearn_is_fitted(self, data):
        transformer = BraindecodeDatasetLoader()
        assert transformer.__sklearn_is_fitted__()

    def test_transformer_fit(self, data):
        """Test whether transformer can fit to some training data"""
        X_train, y_train, _, _ = data
        transformer = BraindecodeDatasetLoader()
        assert transformer.fit(X_train, y_train) == transformer

    def test_transformer_transform_returns_dataset(self, data):
        """Test whether the output of the transform method is a BaseConcatDataset"""
        X_train, y_train, _, _ = data
        transformer = BraindecodeDatasetLoader()
        dataset = transformer.fit(X_train, y_train).transform(X_train, y_train)
        assert isinstance(dataset, BaseConcatDataset)

    def test_transformer_transform_contents(self, data):
        """Test whether the contents and metadata of a transformed dataset are correct"""
        X_train, y_train, _, _ = data
        transformer = BraindecodeDatasetLoader()
        dataset = transformer.fit(X_train, y_train).transform(X_train, y_train)
        assert len(dataset) == len(X_train)
        # test the properties of one epoch - that they match the input MNE Epoch object
        sample_epoch = dataset.datasets[0][0]
        # assert with approximately equal values
        assert np.allclose(sample_epoch[0], X_train.get_data()[0])
        assert sample_epoch[1] == y_train[0]

    def test_sfreq_passed_through(self, data):
        """Test if the sfreq parameter makes it through the transformer"""
        sfreq = 128.0
        info = create_info(ch_names=["test"], sfreq=sfreq, ch_types=["eeg"])
        data = np.random.normal(size=(2, 1, 10 * int(sfreq))) * 1e-6
        # create some noise data in a 10s window
        epochs = EpochsArray(data, info=info)
        y_train = np.array([0])
        transformer = BraindecodeDatasetLoader()
        dataset = transformer.fit(epochs, y_train).transform(epochs, y_train)
        assert dataset.datasets[0].windows.info["sfreq"] == sfreq

    def test_kw_args_initialization(self):
        """Test initializing the transformer with kw_args"""
        kw_args = {"sampling_rate": 128}
        transformer = BraindecodeDatasetLoader(kw_args=kw_args)
        assert transformer.kw_args == kw_args

    def test_is_fitted_method(self):
        """Test __sklearn_is_fitted__ returns True"""
        transformer = BraindecodeDatasetLoader()
        is_fitter = transformer.__sklearn_is_fitted__()
        assert is_fitter

    def test_assert_raises_value_error(self, data):
        """Test that an invalid argument gives a ValueError"""
        X_train, y_train, _, _ = data
        transformer = BraindecodeDatasetLoader()
        invalid_param_name = "invalid"
        with pytest.raises(TypeError):
            transformer.fit(X_train, y=y_train, **{invalid_param_name: None})

    def test_type_create_from_X_y_vs_transfomer(self, data):
        """Test the type from create_from_X_y() and the transfomer"""
        X_train, y_train, _, _ = data

        dataset = create_from_X_y(
            X_train.get_data(),
            y=y_train,
            window_size_samples=X_train.get_data().shape[2],
            window_stride_samples=X_train.get_data().shape[2],
            drop_last_window=False,
            sfreq=X_train.info["sfreq"],
        )
        transformer = BraindecodeDatasetLoader()
        dataset_trans = transformer.fit(X=X_train.get_data(), y=y_train).transform(
            X_train
        )
        assert isinstance(dataset_trans, BaseConcatDataset)
        assert type(dataset_trans) == type(dataset)

    def test_wrong_input(self):
        """Test that an invalid input raises a ValueError"""
        transformer = BraindecodeDatasetLoader()
        with pytest.raises(ValueError):
            transformer.fit_transform(np.random.normal(size=(2, 1, 10)), y=np.array([0]))


if __name__ == "__main__":
    unittest.main()
