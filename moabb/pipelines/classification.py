import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import CCA


class SSVEP_CCA(BaseEstimator, ClassifierMixin):
    """Classifier based on Canonical Correlation Analysis for SSVEP

    A CCA is computed from the set of training signals and some pure
    sinusoids to act as reference.
    Classification is made by taking the frequency with the max correlation,
    as proposed in [1]_.

    References
    ----------

    .. [1] Bin, G., Gao, X., Yan, Z., Hong, B., & Gao, S. (2009). An online
           multi-channel SSVEP-based brain-computer interface using a
           canonical correlation analysis method. Journal of neural
           engineering, 6(4), 046002.
           https://doi.org/10.1088/1741-2560/6/4/046002
    """

    def __init__(self, interval, freqs, n_harmonics=3):
        self.Yf = dict()
        self.cca = CCA(n_components=1)
        self.interval = interval
        self.slen = interval[1] - interval[0]
        self.freqs = freqs
        self.n_harmonics = n_harmonics
        self.one_hot = {}
        for i, k in enumerate(freqs.keys()):
            self.one_hot[k] = i

    def fit(self, X, y, sample_weight=None):
        """Compute reference sinusoid signal

        These sinusoid are generated for each frequency in the dataset
        """
        n_times = X.shape[2]

        for f in self.freqs:
            if f.replace(".", "", 1).isnumeric():
                freq = float(f)
                yf = []
                for h in range(1, self.n_harmonics + 1):
                    yf.append(
                        np.sin(2 * np.pi * freq * h * np.linspace(0, self.slen, n_times))
                    )
                    yf.append(
                        np.cos(2 * np.pi * freq * h * np.linspace(0, self.slen, n_times))
                    )
                self.Yf[f] = np.array(yf)
        return self

    def predict(self, X):
        """Predict is made by taking the maximum correlation coefficient"""
        y = []
        for x in X:
            corr_f = {}
            for f in self.freqs:
                if f.replace(".", "", 1).isnumeric():
                    S_x, S_y = self.cca.fit_transform(x.T, self.Yf[f].T)
                    corr_f[f] = np.corrcoef(S_x.T, S_y.T)[0, 1]
            y.append(self.one_hot[max(corr_f, key=lambda k: corr_f[k])])
        return y

    def predict_proba(self, X):
        """Probabilty could be computed from the correlation coefficient"""
        P = np.zeros(shape=(len(X), len(self.freqs)))
        for i, x in enumerate(X):
            for j, f in enumerate(self.freqs):
                if f.replace(".", "", 1).isnumeric():
                    S_x, S_y = self.cca.fit_transform(x.T, self.Yf[f].T)
                    P[i, j] = np.corrcoef(S_x.T, S_y.T)[0, 1]
        return P / np.resize(P.sum(axis=1), P.T.shape).T
