from braindecode.datasets import WindowsDataset, create_from_X_y
from mne.epochs import BaseEpochs
from sklearn.base import BaseEstimator, TransformerMixin


class CreateBraindecodeDataset(BaseEstimator, TransformerMixin):
    """
    Wrapper to create a Braindecode Dataset from an MNE epoched
    object.

    This is a transformer function that allow used to use the
    dataset as a sklearn pipeline.
    """

    def __init__(self, kw_args: dict = None):
        """

        Parameters
        ----------
        kw_args: dict

        """
        self.kw_args = kw_args

    def fit(self, X: BaseEpochs, y=None):
        self.y = y
        return self

    def transform(self, X: BaseEpochs, y=None) -> WindowsDataset:
        """

        Parameters
        ----------
        X: BaseEpochs object from mne
        y: list|array of labels

        Returns
        -------
        WindowsDataset: Braindecode Dataset
        """
        dataset = create_from_X_y(
            X.get_data(),
            y=self.y,
            window_size_samples=X.get_data().shape[2],
            window_stride_samples=X.get_data().shape[2],
            drop_last_window=False,
            sfreq=X.info["sfreq"],
        )

        return dataset

    def __sklearn_is_fitted__(self) -> bool:
        """
        Return True since CreateBraindecodeDataset is stateless.
        """
        return True
