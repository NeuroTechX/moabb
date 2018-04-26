from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import scipy.signal as signal


class LogVariance(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform"""
        assert X.ndim == 3
        return np.log(np.var(X, -1))


class FM(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. Note however that without the
        sampling rate these values are unnormalized."""
        xphase = np.unwrap(np.angle(signal.hilbert(X, axis=-1)))
        return np.median(np.diff(xphase, axis=-1) / (2 * np.pi), axis=-1)
