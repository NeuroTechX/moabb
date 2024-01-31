import mne
import numpy as np
import scipy.signal as signal
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class LogVariance(BaseEstimator, TransformerMixin):
    """LogVariance transformer."""

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform."""
        assert X.ndim == 3
        return np.log(np.var(X, -1))


class FM(BaseEstimator, TransformerMixin):
    """Transformer to scale sampling frequency."""

    def __init__(self, freq=128):
        """Init function for FM transformer.

        Instantaneous frequencies require a sampling frequency to be
        properly scaled, which is helpful for some algorithms.

        This assumes 128 if not told otherwise.

        Parameters
        ----------
        freq: int
            Sampling frequency of the signal. This is used to scale
            the instantaneous frequency.
        """
        self.freq = freq

    def fit(self, X, y):
        """Only for scikit-learn compatibility."""
        return self

    def transform(self, X):
        """transform."""
        xphase = np.unwrap(np.angle(signal.hilbert(X, axis=-1)))
        return np.median(self.freq * np.diff(xphase, axis=-1) / (2 * np.pi), axis=-1)


class ExtendedSSVEPSignal(BaseEstimator, TransformerMixin):
    """Prepare FilterBank SSVEP EEG signal for estimating extended covariances.

    Riemannian approaches on SSVEP rely on extended covariances
    matrices, where the filtered signals are contenated to estimate a
    large covariance matrice.

    FilterBank SSVEP EEG are of shape (n_trials, n_channels, n_times,
    n_freqs) and should be convert in (n_trials, n_channels*n_freqs,
    n_times) to estimate covariance matrices of (n_channels*n_freqs,
    n_channels*n_freqs).
    """

    def __init__(self):
        """Empty init for ExtendedSSVEPSignal."""
        pass

    def fit(self, X, y):
        """No need to fit for ExtendedSSVEPSignal."""
        return self

    def transform(self, X):
        """Transpose and reshape EEG for extended covmat estimation."""
        out = X.transpose((0, 3, 1, 2))
        n_trials, n_freqs, n_channels, n_times = out.shape
        out = out.reshape((n_trials, n_channels * n_freqs, n_times))
        return out


class AugmentedDataset(BaseEstimator, TransformerMixin):
    """Dataset augmentation methods in a higher dimensional space.

    This transformation allow to create an embedding version of the current
    dataset. The implementation and the application is described in [1]_.

    References
    ----------
    .. [1] Carrara, I., & Papadopoulo, T. (2023). Classification of BCI-EEG based on augmented covariance matrix.
           arXiv preprint arXiv:2302.04508.
           https://doi.org/10.48550/arXiv.2302.04508
    """

    def __init__(self, order: int = 1, lag: int = 1):
        self.order = order
        self.lag = lag

    def fit(self, X: ndarray, y: ndarray):
        return self

    def transform(self, X: ndarray):
        if self.order == 1:
            X_new: ndarray = X
        else:
            X_new = np.concatenate(
                [
                    X[:, :, p * self.lag : -(self.order - p) * self.lag]
                    for p in range(0, self.order)
                ],
                axis=1,
            )

        return X_new


class StandardScaler_Epoch(BaseEstimator, TransformerMixin):
    """Function to standardize the X raw data for the DeepLearning Method."""

    def __init__(self):
        """Init."""

    def fit(self, X, y):
        return self

    def transform(self, X):
        X_fin = []

        for i in np.arange(X.shape[0]):
            X_p = StandardScaler().fit_transform(X[i])
            X_fin.append(X_p)
        X_fin = np.array(X_fin)

        return X_fin


class Resampler_Epoch(BaseEstimator, TransformerMixin):
    """Function that copies and resamples an epochs object."""

    def __init__(self, sfreq):
        self.sfreq = sfreq

    def fit(self, X, y):
        return self

    def transform(self, X: mne.Epochs):
        X = X.copy()
        X.resample(self.sfreq)
        return X


class Convert_Epoch_Array(BaseEstimator, TransformerMixin):
    """Function that copies and resamples an epochs object."""

    def __init__(self):
        """Init."""

    def fit(self, X, y):
        return self

    def transform(self, X: mne.Epochs):
        return X.get_data()
