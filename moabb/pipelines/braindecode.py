from braindecode.datasets import WindowsDataset, create_from_X_y
from mne.epochs import BaseEpochs
from numpy import array, unique
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


class CreateBraindecodeDataset(BaseEstimator, TransformerMixin):
    """
    Wrapper to create a Braindecode Dataset from a mne Epoched
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


class ClassifierModel(BaseEstimator, ClassifierMixin):
    def __init__(self, clf: BaseEstimator, kw_args: dict = None):
        self.clf = clf
        self.classes_ = None
        self.kw_args = kw_args

    def fit(self, X: WindowsDataset, y=None) -> BaseEstimator:
        self.clf.fit(X, y=y, **self.kw_args)
        self.classes_ = unique(y)

        return self.clf

    def predict(self, X: WindowsDataset) -> array:
        return self.clf.predict(X)

    def predict_proba(self, X: WindowsDataset) -> array:
        return self.clf.predict_proba(X)
