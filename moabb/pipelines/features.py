from sklearn.base import BaseEstimator, TransformerMixin, clone
from pyriemann.spatialfilters import CSP
from pyriemann.estimation import Covariances as Cov
import functools
import numpy as np
import logging

log = logging.getLogger()

class EEGFeatures(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        pass


class LogVariance(EEGFeatures):

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform, concatenating all frequency bands."""
        return np.concatenate([np.log(np.var(X[..., i], -1)) for i in range(X.shape[-1])],axis=1)


class MultibandCovariances(EEGFeatures):

    def __init__(self, estimator='scm', est=None):
        """Init."""
        if est is None:
            self.est = Cov(estimator=estimator)
        else:
            self.est = est

    def transform(self, X):
        if X.shape[-1] == 1:
            return self.est.transform(X[..., 0])[..., None]
        else:
            return np.stack([self.est.transform(X[..., i]) for i in range(X.shape[-1])], axis=3)


class AverageCovariances(MultibandCovariances):

    def transform(self, X):
        return super().transform(X).mean(axis=-1)


class FBCSP(BaseEstimator, TransformerMixin):
    """Implementation of the CSP spatial Filtering with Covariance as input.

    Implementation of the famous Common Spatial Pattern Algorithm, but with
    covariance matrices as input. In addition, the implementation allow
    different metric for the estimation of the class-related mean covariance
    matrices, as described in [3].

    This implementation support multiclass CSP by means of approximate joint
    diagonalization. In this case, the spatial filter selection is achieved
    according to [4].

    Parameters
    ----------
    nfilter : int (default 10)
        The number of components to decompose M/EEG signals.
    metric : str (default "euclid")
        The metric for the estimation of mean covariance matrices
    log : bool (default True)
        If true, return the log variance, otherwise return the spatially
        filtered covariance matrices.

    Attributes
    ----------
    filters_ : ndarray
        If fit, the CSP spatial filters, else None.
    patterns_ : ndarray
        If fit, the CSP spatial patterns, else None.


    See Also
    --------
    MDM, SPoC

    References
    ----------
    """

    def __init__(self, filter_obj, feature_selector):
        """Init."""
        self.filter_obj = filter_obj
        self.feature_selector = feature_selector

    def fit(self, X, y):
        assert X.ndim == 4, 'insufficient dimensions in given data'
        self.filters = [clone(self.filter_obj) for i in range(X.shape[-1])]
        [self.filters[i].fit(X[..., i], y) for i in range(X.shape[-1])]
        self.feature_selector.fit(self._transform(X), y)
        return self

    def _transform(self, X):
        assert X.shape[-1] == len(self.filters), "given X has {} band passes while trained model has {}".format(X.shape[-1], len(self.filters))
        return np.concatenate([self.filters[i].transform(X[..., i])
                            for i in range(X.shape[-1])])

    def transform(self, X):
        return self.feature_selector.transform(self._transform(X))
