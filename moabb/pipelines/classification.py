import numpy as np
import scipy.linalg as linalg
from joblib import Parallel, delayed
from mne import BaseEpochs
from pyriemann.estimation import Covariances
from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_covariance
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from .utils import filterbank


class SSVEP_CCA(BaseEstimator, ClassifierMixin):
    """Classifier based on Canonical Correlation Analysis for SSVEP.

    A CCA is computed from the set of training signals and some pure
    sinusoids to act as reference.
    Classification is made by taking the frequency with the max correlation,
    as proposed in [1]_.

    Parameters
    ----------
    n_harmonics: int, default=3
        Number of stimulation frequency's harmonics to be used in the generation
        of the CCA reference signal.

    Attributes
    ----------
    classes_: list of int
        List of unique classes present in the training data.


    References
    ----------

    .. [1] Bin, G., Gao, X., Yan, Z., Hong, B., & Gao, S. (2009). An online
           multi-channel SSVEP-based brain-computer interface using a
           canonical correlation analysis method. Journal of neural
           engineering, 6(4), 046002.
           https://doi.org/10.1088/1741-2560/6/4/046002

    Notes
    -----
    .. versionchanged:: 1.1.0
       Use MNE Epochs object as input data instead of numpy array, fix label encoding.
    """

    def __init__(self, n_harmonics=3):
        self.Yf = dict()
        self.cca = CCA(n_components=1)
        self.n_harmonics = n_harmonics
        self.classes_ = []
        self.one_hot_ = {}
        self.le_ = self.slen_ = self.freqs_ = None

    def fit(self, X, y, sample_weight=None):
        """Compute reference sinusoid signal.

        These sinusoid are generated for each frequency in the dataset

        Parameters
        ----------
        X : MNE Epochs
            The training data as MNE Epochs object.
        y : unused, only for compatibility with scikit-learn

        Returns
        -------
        self: SSVEP_CCA object
            Instance of classifier.
        """
        if not isinstance(X, BaseEpochs):
            raise ValueError("X should be an MNE Epochs object.")

        self.slen_ = X.times[-1] - X.times[0]
        n_times = len(X.times)
        self.freqs_ = list(X.event_id.keys())
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.classes_ = self.le_.transform(self.freqs_)
        for i, k in zip(self.freqs_, self.le_.transform(self.freqs_)):
            self.one_hot_[i] = k

        for f in self.freqs_:
            if f.replace(".", "", 1).isnumeric():
                freq = float(f)
                yf = []
                for h in range(1, self.n_harmonics + 1):
                    yf.append(
                        np.sin(2 * np.pi * freq * h * np.linspace(0, self.slen_, n_times))
                    )
                    yf.append(
                        np.cos(2 * np.pi * freq * h * np.linspace(0, self.slen_, n_times))
                    )
                self.Yf[f] = np.array(yf)
        return self

    def predict(self, X):
        """Predict is made by taking the maximum correlation coefficient.

        Parameters
        ----------
        X : MNE Epochs
            The data to predict as MNE Epochs object.

        Returns
        -------
        y : list of int
            Predicted labels.
        """
        y = []
        for x in X:
            corr_f = {}
            for f in self.freqs_:
                if f.replace(".", "", 1).isnumeric():
                    S_x, S_y = self.cca.fit_transform(x.T, self.Yf[f].T)
                    corr_f[f] = np.corrcoef(S_x.T, S_y.T)[0, 1]
            y.append(self.one_hot_[max(corr_f, key=corr_f.get)])
        return y

    def predict_proba(self, X):
        """Probability could be computed from the correlation coefficient.

        Parameters
        ----------
        X : MNE Epochs
            The data to predict as MNE Epochs object.

        Returns
        -------
        proba : ndarray of shape (n_trials, n_classes)
            probability of each class for each trial.
        """
        P = np.zeros(shape=(len(X), len(self.freqs_)))
        for i, x in enumerate(X):
            for j, f in enumerate(self.freqs_):
                if f.replace(".", "", 1).isnumeric():
                    S_x, S_y = self.cca.fit_transform(x.T, self.Yf[f].T)
                    P[i, j] = np.corrcoef(S_x.T, S_y.T)[0, 1]
        return P / np.resize(P.sum(axis=1), P.T.shape).T


class SSVEP_TRCA(BaseEstimator, ClassifierMixin):
    """Task-Related Component Analysis method [1]_ for SSVEP.

    Parameters
    ----------
    n_fbands: int, default=5
        Number of sub-bands to divide the SSVEP frequencies, with filterbank
        approach.

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

    freqs_: list of str
        List of unique frequencies present in the training data.

    Reference
    ---------

    .. [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
          "Enhancing detection of SSVEPs for a high-speed brain speller using
          task-related component analysis",
          IEEE Trans. Biomed. Eng, 65(1):104-112, 2018.

    Code based on the Matlab implementation from authors of [1]_
    (https://github.com/mnakanishi/TRCA-SSVEP).


    Notes
    -----
    .. versionadded:: 0.4.4

    .. versionchanged:: 1.1.1
       TRCA implementation works with MNE Epochs object, fix labels encoding issue.
    """

    def __init__(
        self,
        n_fbands=5,
        is_ensemble=True,
        method="original",
        estimator="scm",
    ):
        self.is_ensemble = is_ensemble
        self.estimator = estimator
        self.method = method
        self.n_fbands = n_fbands
        self.fb_coefs = [(x + 1) ** (-1.25) + 0.25 for x in range(self.n_fbands)]
        self.one_hot_, self.one_inv_ = {}, {}
        self.sfreq_, self.freqs_, self.peaks_ = None, None, None
        self.le_, self.classes_, self.n_classes = None, None, None
        self.templates_, self.weights_ = None, None

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

                # Compute empirical covariance between the two selected trials and sum it
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
        """Extract spatial filters and templates from the given calibration
        data.

        Parameters
        ----------
        X : MNE Epochs
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
        if not isinstance(X, BaseEpochs):
            raise ValueError("X should be an MNE Epochs object.")

        n_channels, n_samples = X.info["nchan"], len(X.times)
        self.sfreq_ = X.info["sfreq"]
        self.freqs_ = list(X.event_id.keys())
        self.peaks_ = np.array([float(f) for f in self.freqs_])
        self.fb_coefs = [(x + 1) ** (-1.25) + 0.25 for x in range(self.n_fbands)]
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.classes_ = self.le_.transform(self.freqs_)
        self.n_classes = len(self.classes_)
        for i, k in zip(self.freqs_, self.classes_):
            self.one_hot_[i] = k
            self.one_inv_[k] = i
        if self.n_fbands > len(self.peaks_):
            raise ValueError(
                "Number of filterbank bands should be less or equal to the number of peaks."
            )

        # Initialize the final arrays
        self.templates_ = np.zeros((self.n_classes, self.n_fbands, n_channels, n_samples))
        self.weights_ = np.zeros((self.n_fbands, self.n_classes, n_channels))

        # for class_idx in self.classes_:
        for freq, k in self.one_hot_.items():
            X_cal = X[freq]  # Select data with a specific label

            # Filterbank approach
            for band_n in range(self.n_fbands):
                print(self.sfreq_, band_n, self.peaks_)
                # Filter the data and compute TRCA
                X_filter = filterbank(
                    X_cal.get_data(copy=False), self.sfreq_, band_n, self.peaks_
                )
                w_best, _ = self._compute_trca(X_filter)

                # Get template by averaging trials and take the best filter for this band
                self.templates_[k, band_n, :, :] = np.mean(X_filter, axis=0)
                self.weights_[band_n, k, :] = w_best

        return self

    def predict(self, X):
        """Make predictions on unseen data.

        The new data observation X will be filtered
        with weights previously extracted and compared to the templates to assess
        similarity with each of them and select a class based on the maximal value.

        Parameters
        ----------
        X : MNE Epochs
            Testing data. This will be divided in self.n_fbands using the filterbank approach,
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

        # Initialize pred array
        y_pred = []

        for X_test in X:
            # Initialize correlations array
            corr_array = np.zeros((self.n_fbands, self.n_classes))

            # Filter the data in the corresponding band
            for band_n in range(self.n_fbands):
                X_filter = filterbank(X_test, self.sfreq_, band_n, self.peaks_)

                # Compute correlation with all the templates and bands
                for freq, k in self.one_hot_.items():
                    # Get the corresponding template
                    template = np.squeeze(self.templates_[k, band_n, :, :])

                    if self.is_ensemble:
                        w = np.squeeze(
                            self.weights_[band_n, :, :]
                        ).T  # (n_classes, n_channel)
                    else:
                        w = np.squeeze(
                            # self.weights_[band_n, class_idx, :]
                            self.weights_[band_n, k, :]
                        ).T  # (n_channel,)

                    # Compute 2D correlation of spatially filtered testdata with ref
                    r = np.corrcoef(
                        np.dot(X_filter.T, w).flatten(),
                        np.dot(template.T, w).flatten(),
                    )
                    corr_array[band_n, k] = r[0, 1]

            # Fusion for the filterbank analysis
            self.rho = np.dot(self.fb_coefs, corr_array)

            # Select the maximal value and append to predictions
            self.tau = np.argmax(self.rho)
            y_pred.append(self.one_hot_[self.one_inv_[self.tau]])

        return y_pred

    def predict_proba(self, X):
        """Make predictions on unseen data with the associated probabilities.

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
        n_trials = len(X)

        # Initialize pred array
        y_pred = np.zeros((n_trials, self.n_classes))

        for trial_n, X_test in enumerate(X):
            # Initialize correlations array
            corr_array = np.zeros((self.n_fbands, self.n_classes))

            # Filter the data in the corresponding band
            for band_n in range(self.n_fbands):
                X_filter = filterbank(X_test, self.sfreq_, band_n, self.peaks_)

                # Compute correlation with all the templates and bands
                for freq, k in self.one_hot_.items():
                    # Get the corresponding template
                    template = np.squeeze(self.templates_[k, band_n, :, :])

                    if self.is_ensemble:
                        w = np.squeeze(
                            self.weights_[band_n, :, :]
                        ).T  # (n_class, n_channel)
                    else:
                        w = np.squeeze(self.weights_[band_n, k, :]).T  # (n_channel,)

                    # Compute 2D correlation of spatially filtered testdata with ref
                    r = np.corrcoef(
                        np.dot(X_filter.T, w).flatten(),
                        np.dot(template.T, w).flatten(),
                    )
                    corr_array[band_n, k] = r[0, 1]

            normalized_coefs = self.fb_coefs / (np.sum(self.fb_coefs))
            # Fusion for the filterbank analysis
            rho = np.dot(normalized_coefs, corr_array)

            rho /= sum(rho)
            y_pred[trial_n] = rho

        return y_pred


def _whitening(X):
    """Utility function to whiten EEG signal.

    Parameters
    ----------
    X: ndarray of shape (n_channels, n_samples)
    """
    n_channels, n_samples = X.shape
    X_white = X.copy()

    X_white = X_white - np.mean(X_white, axis=1, keepdims=True)
    C = covariances(X_white.reshape((1, n_channels, n_samples)), estimator="sch")[
        0
    ]  # Shrunk covariance matrix
    eig_val, eig_vec = linalg.eigh(C)
    V = (np.abs(eig_val) ** -0.5)[:, np.newaxis] * eig_vec.T
    X_white = V @ X_white
    return X_white


class SSVEP_MsetCCA(BaseEstimator, ClassifierMixin):
    """Classifier based on MsetCCA for SSVEP.

     The MsetCCA method learns multiple linear transforms to extract
     SSVEP common features from multiple sets of EEG data. These are then used
     to compute the reference signal used in CCA [1]_.

    Parameters
    ----------
    n_filters: int, default=1
        Number of multisets spatial filters used per sample data.
        It corresponds to the number of eigen vectors taken the solution of the
        MAXVAR objective function as formulated in Eq.5 in [1]_.

    n_jobs: int, default=1
        Number of jobs to run whitening in parallel.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Array with the class labels extracted at fit time.

    freqs_: list of str
        List of unique frequencies present in the training data.

    References
    ----------

    .. [1] Zhang, Y.U., Zhou, G., Jin, J., Wang, X. and Cichocki, A. (2014). Frequency
           recognition in SSVEP-based BCI using multiset canonical correlation analysis.
           International journal of neural systems, 24(04), p.1450013.
           https://doi.org/10.1142/S0129065714500130

    Notes
    -----
    .. versionadded:: 0.5.0
    """

    def __init__(self, n_filters=1, n_jobs=1):
        self.n_jobs = n_jobs
        self.n_filters = n_filters
        self.cca = CCA(n_components=1)
        self.freqs_, self.le_, self.classes_ = None, None, None
        self.one_hot_, self.Ym = {}, {}

    def fit(self, X, y, sample_weight=None):
        """Compute the optimized reference signal at each stimulus frequency.

        Parameters
        ----------
        X : MNE Epochs
            The training data as MNE Epochs object.

        y : np.ndarray of shape (n_trials,)
            The target labels for each trial.

        Returns
        -------
        self: SSVEP_MsetCCA object
            Instance of classifier.
        """
        if not isinstance(X, BaseEpochs):
            raise ValueError("X should be an MNE Epochs object.")

        self.freqs_ = list(X.event_id.keys())
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.classes_ = self.le_.transform(self.freqs_)
        for i, k in zip(self.freqs_, self.le_.transform(self.freqs_)):
            self.one_hot_[i] = k
        n_trials, n_channels, n_times = len(X), X.info["nchan"], len(X.times)

        # Whiten signal in parallel
        if self.n_jobs == 1:
            X_white = [_whitening(X_i) for X_i in X]
        else:
            X_white = Parallel(n_jobs=self.n_jobs)(delayed(_whitening)(X_i) for X_i in X)
        X_white = np.stack(X_white, axis=0)

        Y = X_white.transpose(2, 0, 1).reshape(-1, n_times)
        # R = np.cov(Y)
        # or more similar to the article:
        R = Y @ Y.T
        # S = np.diag(np.diag(R)) # This does not match the definition in the article
        # S exactly as defined in the article
        mask = np.kron(
            np.eye(n_trials), np.ones((n_channels, n_channels))
        )  # block diagonal mask
        S = R * mask

        # Get W
        _, tempW = linalg.eigh(
            R - S, S, subset_by_index=[R.shape[1] - self.n_filters, R.shape[1] - 1]
        )
        W = np.reshape(tempW, (n_trials, n_channels, self.n_filters))

        # Normalise W
        W = W / np.linalg.norm(W, axis=0, keepdims=True)

        Z = W.transpose((0, 2, 1)) @ X_white

        # Get Ym
        for m_class in self.classes_:
            self.Ym[m_class] = Z[y == m_class].transpose(2, 0, 1).reshape(-1, n_times)

        return self

    def predict(self, X):
        """Predict is made by taking the maximum correlation coefficient.

        Parameters
        ----------
        X : MNE Epochs
            The data to predict as MNE Epochs object.

        Returns
        -------
        y : list of int
            Predicted labels.
        """

        # Check is fit had been called
        check_is_fitted(self)

        y = []
        for x in X:
            corr_f = {}
            for f in self.classes_:
                S_x, S_y = self.cca.fit_transform(x.T, self.Ym[f].T)
                corr_f[f] = np.corrcoef(S_x.T, S_y.T)[0, 1]
            y.append(max(corr_f, key=corr_f.get))
        return y

    def predict_proba(self, X):
        """Probability could be computed from the correlation coefficient.

        Parameters
        ----------
        X : MNE Epochs
            The data to predict as MNE Epochs object.

        Returns
        -------
        P : ndarray of shape (n_trials, n_classes)
            Probability of each class for each trial.
        """

        # Check is fit had been called
        check_is_fitted(self)

        P = np.zeros(shape=(len(X), len(self.classes_)))
        for i, x in enumerate(X):
            for j, f in enumerate(self.classes_):
                S_x, S_y = self.cca.fit_transform(x.T, self.Ym[f].T)
                P[i, j] = np.corrcoef(S_x.T, S_y.T)[0, 1]
        return P / np.resize(P.sum(axis=1), P.T.shape).T
