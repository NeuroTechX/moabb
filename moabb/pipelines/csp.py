import numpy as np
from pyriemann.geometry.mean import gmean
from pyriemann.spatialfilters import CSP
from scipy import linalg


class TRCSP(CSP):
    """Weighted Tikhonov-regularized CSP as described in Lotte and Guan
    2011."""

    def __init__(self, nfilter=4, metric="euclid", log=True, alpha=1):
        super().__init__(nfilter, metric, log)
        self.alpha = alpha

    def fit(self, X, y):
        """Train spatial filters.

        Only deals with two class
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
        if len(classes) != 2:  # was assert, now raises properly
            raise ValueError(f"TRCSP only supports 2 classes, got {len(classes)}")
        # estimate class means
        C = []
        for c in classes:
            C.append(gmean(X[y == c], metric=self.metric))
        C = np.array(C)

        # regularize CSP
        n_half = self.nfilter // 2
        if n_half == 0:
            self.filters_ = np.empty((0, Ne), dtype=C.dtype)
            self.patterns_ = np.empty((0, Ne), dtype=C.dtype)
            return self

        eye = np.eye(Ne) * self.alpha
        Creg = (C[0] + eye, C[1] + eye)
        subset = [Ne - n_half, Ne - 1]
        filters = []
        patterns = []
        for i in range(2):
            _, evec = linalg.eigh(
                C[1 - i], Creg[i], subset_by_index=subset, check_finite=False
            )
            W = evec[:, ::-1]
            filters.append(W)
            # Since W.T @ Creg[i] @ W = I, the top columns of pinv(evecs[i].T) equal Creg[i] @ W
            patterns.append(Creg[i] @ W)
        self.filters_ = np.concatenate(filters, axis=1).T
        self.patterns_ = np.concatenate(patterns, axis=1).T

        return self

    def transform(self, X):
        """Apply spatial filters."""
        if not isinstance(X, (np.ndarray, list)):
            raise TypeError("X must be an array.")
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("X must be n_trials * n_channels * n_channels")
        # Diagonal of W @ X @ W.T without computing off-diagonal entries
        WX = self.filters_ @ X
        out = np.einsum("tfc,fc->tf", WX, self.filters_)
        return np.log(out) if self.log else out
