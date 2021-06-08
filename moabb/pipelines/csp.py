import numpy as np
from pyriemann.spatialfilters import CSP
from pyriemann.utils.mean import mean_covariance
from scipy import linalg


class TRCSP(CSP):
    """
    Weighted Tikhonov-regularized CSP as described in Lotte and Guan 2011
    """

    def __init__(self, nfilter=4, metric="euclid", log=True, alpha=1):
        super().__init__(nfilter, metric, log)
        self.alpha = alpha

    def fit(self, X, y):
        """
        Train spatial filters. Only deals with two class
        """

        if not isinstance(X, (np.ndarray, list)):
            raise TypeError("X must be an array.")
        if not isinstance(y, (np.ndarray, list)):
            raise TypeError("y must be an array.")
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 3:
            raise ValueError("X must be n_trials * n_channels * n_channels")
        if len(y) != len(X):
            raise ValueError("X and y must have the same length.")
        if np.squeeze(y).ndim != 1:
            raise ValueError("y must be of shape (n_trials,).")

        Nt, Ne, Ns = X.shape
        classes = np.unique(y)
        assert len(classes) == 2, "Can only do 2-class TRCSP"
        # estimate class means
        C = []
        for c in classes:
            C.append(mean_covariance(X[y == c], self.metric))
        C = np.array(C)

        # regularize CSP
        evals = [[], []]
        evecs = [[], []]
        Creg = C[1] + np.eye(C[1].shape[0]) * self.alpha
        evals[1], evecs[1] = linalg.eigh(C[0], Creg)
        Creg = C[0] + np.eye(C[0].shape[0]) * self.alpha
        evals[0], evecs[0] = linalg.eigh(C[1], Creg)
        # sort eigenvectors
        filters = []
        patterns = []
        for i in range(2):
            ix = np.argsort(evals[i])[::-1]  # in descending order
            # sort eigenvectors
            evecs[i] = evecs[i][:, ix]
            # spatial patterns
            A = np.linalg.pinv(evecs[i].T)
            filters.append(evecs[i][:, : (self.nfilter // 2)])
            patterns.append(A[:, : (self.nfilter // 2)])
        self.filters_ = np.concatenate(filters, axis=1).T
        self.patterns_ = np.concatenate(patterns, axis=1).T

        return self
