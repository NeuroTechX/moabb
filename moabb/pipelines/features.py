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

    def __init__(self, freq=128):
        '''instantaneous frequencies require a sampling frequency to be properly
        scaled,
        which is helpful for some algorithms. This assumes 128 if not told
        otherwise.

        '''
        self.freq = freq

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. """
        xphase = np.unwrap(np.angle(signal.hilbert(X, axis=-1)))
        return np.median(self.freq * np.diff(xphase, axis=-1) / (2 * np.pi),
                         axis=-1)
