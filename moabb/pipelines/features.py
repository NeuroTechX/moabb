import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LogVariance(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform."""
        return np.log(np.var(X, -1))
