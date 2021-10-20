import numpy as np
import scipy.linalg as linalg
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import CCA
from sklearn.utils.validation import check_is_fitted

from .utils import filterbank


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


class SSVEP_TRCA(BaseEstimator, ClassifierMixin):
    """Classifier based on the Task-Related Component Analysis method [1]_ for SSVEP.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the data to be analyzed.

    freqs : dict with n_classes keys
        Frequencies corresponding to the SSVEP components. These are
        necessary to design the filterbank bands.

    n_fbands : int, default=5
        Number of sub-bands considered for filterbank analysis.

    downsample: int, default=1
        Factor by which downsample the data. A downsample value of N will result
        on a sampling frequency of (sfreq // N) by taking one sample every N of
        the original data. In the original TRCA paper [1]_ data are at 250Hz.

    is_ensemble: bool, default=False
        If True, predict on new data using the Ensemble-TRCA method described
        in [1]_.

    method: str, default='original'
        'original' computes euclidean mean for S as in the original paper [1]_.
        'riemann' variation computes geodesic mean instead. This geodesic
        mean is more robust to outlier but negatively impacted by ill-conditioned
        matrices (when only few samples are available for training for instance).
        If the geometric mean can't be estimated, please consider trying 'logeuclid'.
        It computes log-euclidean mean instead of the affine-invariant one and is more robust
        computationally.
        'riemann' and 'logeuclid' variations are useful when lots of noisy
        training data are available. With few training data 'original' is more
        appropriate.

    estimator: str
        For both methods, regularization to use for covariance matrices estimations.
        Consider 'schaefer', 'lwf', 'oas' or 'scm' for no regularization.
        In the original implementation from TRCA paper [1]_, no regularization
        is used. So method='original' and regul='scm' is similar to original
        implementation.



    Attributes
    ----------
    fb_coefs : list of len (n_fbands)
        Alpha coefficients for the fusion of the filterbank sub-bands.

    classes_ : ndarray of shape (n_classes,)
        Array with the class labels extracted at fit time.

    n_classes: int
        Number of unique labels/classes.

    templates_ : ndarray of shape (n_classes, n_bands, n_channels, n_samples)
        Template data obtained by averaging all training trials for a given
        class. Each class templates is divided in n_fbands sub-bands extracted
        from the filterbank approach.

    weights_ : ndarray of shape (n_fbands, n_classes, n_channels)
        Weight coefficients for the different electrodes which are used
        as spatial filters for the data.

    Reference
    ----------

    .. [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
          "Enhancing detection of SSVEPs for a high-speed brain speller using
          task-related component analysis",
          IEEE Trans. Biomed. Eng, 65(1):104-112, 2018.

    Code based on the Matlab implementation from authors of [1]_
    (https://github.com/mnakanishi/TRCA-SSVEP).

        .. versionadded:: 0.4.4
    """

    def __init__(
        self,
        interval,
        freqs,
        n_fbands=5,
        downsample=1,
        is_ensemble=True,
        method="original",
        estimator="scm",
    ):
        self.freqs = freqs
        self.peaks = np.array([float(f) for f in freqs.keys()])
        self.n_fbands = n_fbands
        self.downsample = downsample
        self.interval = interval
        self.slen = interval[1] - interval[0]
        self.is_ensemble = is_ensemble
        self.fb_coefs = [(x + 1) ** (-1.25) + 0.25 for x in range(self.n_fbands)]
        self.estimator = estimator
        self.method = method

    def _Q_S_estim(self, data):
        # Check if X is a single trial (test data) or not
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        # Get data shape
        n_trials, n_channels, n_samples = data.shape

        X = np.concatenate((data, data), axis=1)

        # Initialize S matrix
        S = np.zeros((n_channels, n_channels))

        # Estimate covariance between every trial and the rest of the trials (excluding itself)
        for trial_i in range(n_trials - 1):
            x1 = np.squeeze(data[trial_i, :, :])

            # Mean centering for the selected trial
            x1 -= np.mean(x1, 0)

            # Select a second trial that is different
            for trial_j in range(trial_i + 1, n_trials):
                x2 = np.squeeze(data[trial_j, :, :])

                # Mean centering for the selected trial
                x2 -= np.mean(x2, 0)

                # Put the two trials together
                X = np.concatenate((x1, x2))

                if n_channels == 1:
                    X = X.reshape((n_channels, len(X)))

                # Regularized covariance estimate
                cov = Covariances(estimator=self.estimator).fit_transform(
                    X[np.newaxis, ...]
                )
                cov = np.squeeze(cov)

                # Compute empirical covariance betwwen the two selected trials and sum it
                if n_channels > 1:
                    S = S + cov[:n_channels, n_channels:] + cov[n_channels:, :n_channels]

                else:
                    S = S + cov + cov

        # Concatenate all the trials
        UX = np.empty((n_channels, n_samples * n_trials))

        for trial_n in range(n_trials):
            UX[:, trial_n * n_samples : (trial_n + 1) * n_samples] = data[trial_n, :, :]

        # Mean centering
        UX -= np.mean(UX, 1)[:, None]
        cov = Covariances(estimator=self.estimator).fit_transform(UX[np.newaxis, ...])
        Q = np.squeeze(cov)

        return S, Q

    def _Q_S_estim_riemann(self, data):
        # Check if X is a single trial (test data) or not
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        # Get data shape
        n_trials, n_channels, n_samples = data.shape

        X = np.concatenate((data, data), axis=1)

        # Concatenate all the trials
        UX = np.empty((n_channels, n_samples * n_trials))

        for trial_n in range(n_trials):
            UX[:, trial_n * n_samples : (trial_n + 1) * n_samples] = data[trial_n, :, :]

        # Mean centering
        UX -= np.mean(UX, 1)[:, None]

        # Compute empirical variance of all data (to be bounded)
        cov = Covariances(estimator=self.estimator).fit_transform(UX[np.newaxis, ...])
        Q = np.squeeze(cov)

        cov = Covariances(estimator=self.estimator).fit_transform(X)
        S = cov[:, :n_channels, n_channels:] + cov[:, n_channels:, :n_channels]

        S = mean_covariance(S, metric=self.method)

        return S, Q

    def _compute_trca(self, X):
        """Computation of TRCA spatial filters.

        Parameters
        ----------
        X: ndarray of shape (n_trials, n_channels, n_samples)
            Training data

        Returns
        -------
        W: ndarray of shape (n_channels)
            Weight coefficients for electrodes which can be used as
            a spatial filter.
        """

        if self.method == "original":
            S, Q = self._Q_S_estim(X)
        elif self.method == "riemann" or self.method == "logeuclid":
            S, Q = self._Q_S_estim_riemann(X)
        else:
            raise ValueError(
                "Method should be either 'original', 'riemann' or 'logeuclid'."
            )

        # Compute eigenvalues and vectors
        lambdas, W = linalg.eig(S, Q, left=True, right=False)

        # Sort eigenvectors by eigenvalue
        arr1inds = lambdas.argsort()
        W = W[:, arr1inds[::-1]]

        return W[:, 0], W

    def fit(self, X, y):
        """Extract spatial filters and templates from the given calibration data.

        Parameters
        ----------
        X : ndarray of shape (n_trials, n_channels, n_samples)
            Training data. Trials are grouped by class, divided in n_fbands bands by
            the filterbank approach and then used to calculate weight vectors and
            templates for each class and band.

        y : ndarray of shape (n_trials,)
            Label vector in respect to X.

        Returns
        -------
        self: CCA object
            Instance of classifier.
        """
        # Downsample data
        X = X[:, :, :: self.downsample]

        # Get shape of X and labels
        n_trials, n_channels, n_samples = X.shape

        self.sfreq = int(n_samples / self.slen)
        self.sfreq = self.sfreq / self.downsample

        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)

        # Initialize the final arrays
        self.templates_ = np.zeros((self.n_classes, self.n_fbands, n_channels, n_samples))
        self.weights_ = np.zeros((self.n_fbands, self.n_classes, n_channels))

        for class_idx in self.classes_:
            X_cal = X[y == class_idx]  # Select data with a specific label
            # Filterbank approach
            for band_n in range(self.n_fbands):
                # Filter the data and compute TRCA
                X_filter = filterbank(X_cal, self.sfreq, band_n, self.peaks)
                w_best, _ = self._compute_trca(X_filter)

                # Get template by averaging trials and take the best filter for this band
                self.templates_[class_idx, band_n, :, :] = np.mean(X_filter, axis=0)
                self.weights_[band_n, class_idx, :] = w_best

        return self

    def predict(self, X):
        """Make predictions on unseen data.

        The new data observation X will be filtered
        with weights previously extracted and compared to the templates to assess
        similarity with each of them and select a class based on the maximal value.

        Parameters
        ----------
        X : ndarray of shape (n_trials, n_channels, n_samples)
            Testing data. This will be divided in self.n_fbands using the filter- bank approach,
            then it will be transformed by the different spatial filters and compared to the
            previously fit templates according to the selected method for analysis (ensemble or
            not). Finally, correlation scores for all sub-bands of each class will be combined,
            resulting on a single correlation score per class, from which the maximal one is
            identified as the predicted class of the data.

        Returns
        -------
        y_pred : ndarray of shape (n_trials,)
            Prediction vector in respect to X.
        """

        # Check is fit had been called
        check_is_fitted(self)

        # Check if X is a single trial or not
        if X.ndim == 2:
            X = X[np.newaxis, ...]

        # Downsample data
        X = X[:, :, :: self.downsample]

        # Get test data shape
        n_trials, _, _ = X.shape

        # Initialize pred array
        y_pred = []

        for trial_n in range(n_trials):
            # Pick trial
            X_test = X[trial_n, :, :]

            # Initialize correlations array
            corr_array = np.zeros((self.n_fbands, self.n_classes))

            # Filter the data in the corresponding band
            for band_n in range(self.n_fbands):
                X_filter = filterbank(X_test, self.sfreq, band_n, self.peaks)

                # Compute correlation with all the templates and bands
                for class_idx in range(self.n_classes):
                    # Get the corresponding template
                    template = np.squeeze(self.templates_[class_idx, band_n, :, :])

                    if self.is_ensemble:
                        w = np.squeeze(
                            self.weights_[band_n, :, :]
                        ).T  # (n_classes, n_channel)
                    else:
                        w = np.squeeze(
                            self.weights_[band_n, class_idx, :]
                        ).T  # (n_channel,)

                    # Compute 2D correlation of spatially filtered testdata with ref
                    r = np.corrcoef(
                        np.dot(X_filter.T, w).flatten(),
                        np.dot(template.T, w).flatten(),
                    )
                    corr_array[band_n, class_idx] = r[0, 1]

            # Fusion for the filterbank analysis
            rho = np.dot(self.fb_coefs, corr_array)

            # Select the maximal value and append to preddictions
            tau = np.argmax(rho)
            y_pred.append(tau)

        return y_pred

    def predict_proba(self, X):
        """Make predictions on unseen data with the asociated probabilities.

        The new data observation X will be filtered
        with weights previously extracted and compared to the templates to assess
        similarity with each of them and select a class based on the maximal value.

        Parameters
        ----------
        X : ndarray of shape (n_trials, n_channels, n_samples)
            Testing data. This will be divided in self.n_fbands using the filter-bank approach,
            then it will be transformed by the different spatial filters and compared to the
            previously fit templates according to the selected method for analysis (ensemble or
            not). Finally, correlation scores for all sub-bands of each class will be combined,
            resulting on a single correlation score per class, from which the maximal one is
            identified as the predicted class of the data.

        Returns
        -------
        y_pred : ndarray of shape (n_trials,)
            Prediction vector in respect to X.
        """

        # Check is fit had been called
        check_is_fitted(self)

        # Check if X is a single trial or not
        if X.ndim == 2:
            X = X[np.newaxis, ...]

        # Downsample data
        X = X[:, :, :: self.downsample]

        # Get test data shape
        n_trials, _, _ = X.shape

        # Initialize pred array
        y_pred = np.zeros((n_trials, len(self.peaks)))

        for trial_n in range(n_trials):
            # Pick trial
            X_test = X[trial_n, :, :]

            # Initialize correlations array
            corr_array = np.zeros((self.n_fbands, self.n_classes))

            # Filter the data in the corresponding band
            for band_n in range(self.n_fbands):
                X_filter = filterbank(X_test, self.sfreq, band_n, self.peaks)

                # Compute correlation with all the templates and bands
                for class_idx in range(self.n_classes):
                    # Get the corresponding template
                    template = np.squeeze(self.templates_[class_idx, band_n, :, :])

                    if self.is_ensemble:
                        w = np.squeeze(
                            self.weights_[band_n, :, :]
                        ).T  # (n_class, n_channel)
                    else:
                        w = np.squeeze(
                            self.weights_[band_n, class_idx, :]
                        ).T  # (n_channel,)

                    # Compute 2D correlation of spatially filtered testdata with ref
                    r = np.corrcoef(
                        np.dot(X_filter.T, w).flatten(),
                        np.dot(template.T, w).flatten(),
                    )
                    corr_array[band_n, class_idx] = r[0, 1]

            normalized_coefs = self.fb_coefs / (np.sum(self.fb_coefs))
            # Fusion for the filterbank analysis
            rho = np.dot(normalized_coefs, corr_array)

            rho /= sum(rho)
            y_pred[trial_n] = rho

        return y_pred
