from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FilterBank(BaseEstimator, TransformerMixin):
    """Apply a given indentical pipeline over a bank of filter.

    The pipeline provided with the constrictor will be appield on the 4th
    axis of the input data. This pipeline should be used with a FilterBank
    paradigm.

    This can be used to build a filterbank CSP, for example::

        pipeline = make_pipeline(FilterBank(estimator=CSP()), LDA())

    Parameters
    ----------
    estimator: sklean Estimator
        the sklearn pipeline to apply on each band of the filter bank.
    flatten: bool (True)
        If True, output of each band are concatenated together on the feature
        axis. if False, output are stacked.
    """

    def __init__(self, estimator, flatten=True):
        self.estimator = estimator
        self.flatten = flatten

    def fit(self, X, y=None):
        assert X.ndim == 4
        self.models = [deepcopy(self.estimator).fit(X[..., i], y)
                       for i in range(X.shape[-1])]
        return self

    def transform(self, X):
        assert X.ndim == 4
        out = [self.models[i].transform(X[..., i]) for i in range(X.shape[-1])]
        assert out[0].ndim == 2, 'Each band must return a two dimensional matrix, currently have {}'.format(
            out[0].ndim)
        if self.flatten:
            return np.concatenate(out, axis=1)
        else:
            return np.stack(out, axis=2)

    def __repr__(self):
        return '{}(estimator={}, flatten={})'.format(type(self).__name__,
                                                     self.estimator.get_params(),
                                                     self.flatten)
