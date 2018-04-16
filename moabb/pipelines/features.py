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

    def __init__(self, sr=128):
        '''

        A sampling rate is required to compute the instantanous frequency. We assume 128Hz if nothing else is given
        '''
        self.sr = sr

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. To deal with normalization we assume a given sampling rate."""
        xphase = np.unwrap(np.angle(signal.hilbert(X, axis=-1)))
        return np.median(np.diff(xphase, axis=-1) / (2 * np.pi) * self.sr, axis=-1)
