import logging

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
from sklearn.utils.validation import NotFittedError, check_is_fitted

from .utils import filterbank


log = logging.getLogger(__name__)


class SSVEP_CCA(BaseEstimator, ClassifierMixin):
    """Classifier based on Canonical Correlation Analysis for SSVEP.

    Canonical Correlation Analysis (CCA) is a multivariate statistical method
    used to find linear relationships between two sets of variables. For SSVEP
    detection, CCA finds spatial filters that maximize the correlation between
    multi-channel EEG signals and predefined sinusoidal reference signals at
    stimulation frequencies [1]_.

    **Mathematical Formulation**

    Given multi-channel EEG signal :math:`\\mathbf{X} \\in \\mathbb{R}^{N_c \\times N_s}`
    and reference signal :math:`\\mathbf{Y}_f \\in \\mathbb{R}^{2N_h \\times N_s}`, CCA
    finds weight vectors :math:`\\mathbf{w}_x` and :math:`\\mathbf{w}_y` that maximize
    the correlation between linear combinations :math:`x = \\mathbf{X}^T \\mathbf{w}_x`
    and :math:`y = \\mathbf{Y}_f^T \\mathbf{w}_y`:

    .. math::

        \\max_{\\mathbf{w}_x, \\mathbf{w}_y}
        \\rho(x, y) = \\frac{E[\\mathbf{w}_x^T \\mathbf{X} \\mathbf{Y}_f^T \\mathbf{w}_y]}
        {\\sqrt{E[\\mathbf{w}_x^T \\mathbf{X} \\mathbf{X}^T \\mathbf{w}_x]
        E[\\mathbf{w}_y^T \\mathbf{Y}_f \\mathbf{Y}_f^T \\mathbf{w}_y]}}

    **Reference Signal Construction**

    The reference signals :math:`\\mathbf{Y}_f` for stimulus frequency :math:`f` consist
    of sine and cosine pairs at the fundamental frequency and its harmonics:

    .. math::

        \\mathbf{Y}_f = \\begin{bmatrix}
        \\sin(2\\pi f t) \\\\
        \\cos(2\\pi f t) \\\\
        \\sin(2\\pi \\cdot 2f \\cdot t) \\\\
        \\cos(2\\pi \\cdot 2f \\cdot t) \\\\
        \\vdots \\\\
        \\sin(2\\pi \\cdot N_h \\cdot f \\cdot t) \\\\
        \\cos(2\\pi \\cdot N_h \\cdot f \\cdot t)
        \\end{bmatrix}

    where :math:`N_h` is the number of harmonics and :math:`t` is the time vector.

    **Classification Rule**

    For a test signal :math:`\\mathbf{X}`, the predicted class is the stimulus
    frequency that yields the maximum canonical correlation:

    .. math::

        \\hat{f} = \\arg\\max_{f \\in \\mathcal{F}} \\rho_f

    where :math:`\\mathcal{F}` is the set of stimulus frequencies and :math:`\\rho_f`
    is the canonical correlation between the test signal and reference :math:`\\mathbf{Y}_f`.

    Parameters
    ----------
    n_harmonics : int, default=3
        Number of harmonics :math:`N_h` to include in the reference signal.
        Higher values capture more harmonic components of the SSVEP response.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Encoded class labels (0 to n_classes-1).
    freqs_ : list of str
        List of stimulus frequencies extracted from training data.
    one_hot_ : dict
        Mapping from frequency strings to encoded class labels.
    slen_ : float
        Signal length in seconds.
    le_ : LabelEncoder
        Fitted label encoder for frequency strings.
    Yf : dict
        Dictionary mapping frequency strings to reference signals
        :math:`\\mathbf{Y}_f` of shape ``(2 * n_harmonics, n_times)``.

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
        self._le, self._slen, self._freqs = None, None, []

    def fit(self, X, y, sample_weight=None):
        """Compute reference sinusoid signal.

        These sinusoid are generated for each frequency in the dataset

        Parameters
        ----------
        X : MNE Epochs
            The training data as MNE Epochs object.
        y : Unused,
            Only for compatibility with scikit-learn
        sample_weight : Unused,
            Only for compatibility with scikit-learn

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
        check_is_fitted(self, ["freqs_", "classes_", "one_hot_", "slen_", "le_"])
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
        check_is_fitted(self, ["freqs_", "classes_", "one_hot_", "slen_", "le_"])
        P = np.zeros(shape=(len(X), len(self.freqs_)))
        for i, x in enumerate(X):
            for j, f in enumerate(self.freqs_):
                if f.replace(".", "", 1).isnumeric():
                    S_x, S_y = self.cca.fit_transform(x.T, self.Yf[f].T)
                    P[i, j] = np.corrcoef(S_x.T, S_y.T)[0, 1]
        return P / np.resize(P.sum(axis=1), P.T.shape).T


class SSVEP_TRCA(BaseEstimator, ClassifierMixin):
    """Task-Related Component Analysis (TRCA) method for SSVEP detection [1]_.

    TRCA is a data-driven spatial filtering approach that enhances SSVEP detection
    by maximizing the reproducibility of task-related EEG components across multiple
    trials. Unlike CCA which uses predefined sinusoidal references, TRCA learns
    optimal spatial filters directly from the training data.

    **Mathematical Formulation**

    Given :math:`N_t` training trials :math:`\\mathbf{X}^{(h)} \\in \\mathbb{R}^{N_c \\times N_s}`
    for a stimulus frequency, TRCA finds the optimal spatial filter :math:`\\mathbf{w}` that
    maximizes the inter-trial covariance while constraining the variance:

    .. math::

        \\hat{\\mathbf{w}} = \\arg\\max_{\\mathbf{w}}
        \\frac{\\mathbf{w}^T \\mathbf{S} \\mathbf{w}}{\\mathbf{w}^T \\mathbf{Q} \\mathbf{w}}

    **Inter-trial Covariance Matrix S**

    The matrix :math:`\\mathbf{S}` captures the covariance between different trials:

    .. math::

        S_{j_1, j_2} = \\sum_{h_1=1}^{N_t} \\sum_{h_2=1, h_2 \\neq h_1}^{N_t}
        \\text{Cov}(x_{j_1}^{(h_1)}(t), x_{j_2}^{(h_2)}(t))

    where :math:`x_j^{(h)}(t)` is the signal from channel :math:`j` in trial :math:`h`.

    **Variance Constraint Matrix Q**

    The matrix :math:`\\mathbf{Q}` represents the pooled variance across all trials:

    .. math::

        \\mathbf{Q} = \\sum_{h=1}^{N_t} \\mathbf{X}^{(h)} (\\mathbf{X}^{(h)})^T

    **Generalized Eigenvalue Problem**

    The optimization is solved as a generalized eigenvalue problem:

    .. math::

        \\mathbf{S} \\mathbf{w} = \\lambda \\mathbf{Q} \\mathbf{w}

    The eigenvector corresponding to the largest eigenvalue gives the optimal
    spatial filter :math:`\\hat{\\mathbf{w}}`.

    **Template Construction**

    For each stimulus frequency :math:`f_n`, the template is the average of
    spatially filtered training trials:

    .. math::

        \\bar{\\mathbf{X}}_n = \\frac{1}{N_t} \\sum_{h=1}^{N_t} \\mathbf{X}_n^{(h)}

    **Ensemble TRCA**

    The ensemble method combines spatial filters from all stimulus frequencies
    into a filter bank :math:`\\mathbf{W} = [\\mathbf{w}_1, \\mathbf{w}_2, ..., \\mathbf{w}_{N_f}]`
    for improved robustness.

    **Filter Bank Approach**

    To capture harmonic components, EEG signals are decomposed into :math:`N_m`
    sub-bands using a filter bank. The correlation coefficient for sub-band :math:`m` is:

    .. math::

        r_n^{(m)} = \\rho\\left((\\mathbf{X}^{(m)})^T \\mathbf{W}^{(m)},
        (\\bar{\\mathbf{X}}_n^{(m)})^T \\mathbf{W}^{(m)}\\right)

    **Classification Rule**

    The final feature combines sub-band correlations with weights :math:`a^{(m)} = m^{-1.25} + 0.25`:

    .. math::

        \\rho_n = \\sum_{m=1}^{N_m} a^{(m)} \\cdot (r_n^{(m)})^2

    The predicted class is: :math:`\\hat{\\tau} = \\arg\\max_n \\rho_n`

    Parameters
    ----------
    n_fbands : int, default=5
        Number of sub-bands :math:`N_m` for the filter bank decomposition.
        Each sub-band captures different harmonic components of the SSVEP.

    is_ensemble : bool, default=True
        If True, use ensemble TRCA which combines spatial filters from all
        stimulus frequencies for improved robustness. If False, use only
        the class-specific spatial filter.

    method : str, default='original'
        Method for computing the inter-trial covariance matrix :math:`\\mathbf{S}`:

        - ``'original'``: Euclidean mean as in the original paper [1]_.
        - ``'riemann'``: Geodesic (Riemannian) mean, more robust to outliers
          but sensitive to ill-conditioned matrices.
        - ``'logeuclid'``: Log-Euclidean mean, a computationally stable
          alternative to the Riemannian mean.

    estimator : str, default='scm'
        Covariance estimator for regularization. Options include:

        - ``'scm'``: Sample covariance matrix (no regularization, as in [1]_).
        - ``'lwf'``: Ledoit-Wolf shrinkage estimator.
        - ``'oas'``: Oracle Approximating Shrinkage estimator.
        - ``'schaefer'``: Schäfer-Strimmer shrinkage estimator.

    Attributes
    ----------
    fb_coefs : list of float, length n_fbands
        Sub-band weights :math:`a^{(m)} = m^{-1.25} + 0.25` for filter bank fusion.
    classes_ : ndarray of shape (n_classes,)
        Encoded class labels extracted at fit time.
    n_classes : int
        Number of unique stimulus frequencies/classes.
    templates_ : ndarray of shape (n_classes, n_fbands, n_channels, n_samples)
        Average templates :math:`\\bar{\\mathbf{X}}_n^{(m)}` for each class and sub-band.
    weights_ : ndarray of shape (n_fbands, n_classes, n_channels)
        Spatial filter weights :math:`\\mathbf{w}_n^{(m)}` for each sub-band and class.
    freqs_ : list of str
        List of stimulus frequencies from training data.
    peaks_ : ndarray of shape (n_classes,)
        Numeric frequency values for filter bank design.
    sfreq_ : float
        Sampling frequency of the training data.

    References
    ----------
    .. [1] M. Nakanishi, Y. Wang, X. Chen, Y.-T. Wang, X. Gao, and T.-P. Jung,
           "Enhancing detection of SSVEPs for a high-speed brain speller using
           task-related component analysis",
           IEEE Trans. Biomed. Eng, 65(1):104-112, 2018.
           https://doi.org/10.1109/TBME.2017.2694818

    See Also
    --------
    SSVEP_CCA : CCA-based SSVEP classifier using sinusoidal references.
    SSVEP_MsetCCA : Multi-set CCA for learning optimal references from data.

    Notes
    -----
    Code based on the MATLAB implementation from the authors of [1]_:
    https://github.com/mnakanishi/TRCA-SSVEP

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
                X = np.concatenate((x1, x2), axis=0)

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
            log.warning("Try with lower n_fbands if there is an error.")

        # Initialize the final arrays
        self.templates_ = np.zeros((self.n_classes, self.n_fbands, n_channels, n_samples))
        self.weights_ = np.zeros((self.n_fbands, self.n_classes, n_channels))

        # for class_idx in self.classes_:
        for freq, k in self.one_hot_.items():
            X_cal = X[freq]  # Select data with a specific label

            # Filterbank approach
            for band_n in range(self.n_fbands):
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
        check_is_fitted(
            self,
            [
                "classes_",
                "n_classes",
                "peaks_",
                "one_hot_",
                "one_inv_",
                "freqs_",
                "le_",
                "sfreq_",
            ],
        )
        if self.n_classes is None:
            raise NotFittedError(
                "This SSVEP_TRCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        # Initialize pred array
        y_pred = []

        for x in X:
            # Initialize correlations array
            corr_array = np.zeros((self.n_fbands, self.n_classes))

            # Filter the data in the corresponding band
            for band_n in range(self.n_fbands):
                X_filter = filterbank(x, self.sfreq_, band_n, self.peaks_)

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
        check_is_fitted(
            self,
            [
                "classes_",
                "n_classes",
                "peaks_",
                "one_hot_",
                "one_inv_",
                "freqs_",
                "le_",
                "sfreq_",
            ],
        )
        if self.n_classes is None:
            raise NotFittedError(
                "This SSVEP_TRCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )
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
    """Multi-set Canonical Correlation Analysis (MsetCCA) for SSVEP detection [1]_.

    MsetCCA learns optimal reference signals from training data rather than using
    predefined sinusoidal references as in standard CCA. It maximizes the correlation
    among canonical variates from multiple sets of EEG trials recorded at the same
    stimulus frequency, extracting common SSVEP features.

    **Mathematical Formulation**

    Given :math:`N_t` training trials :math:`\\mathbf{X}_{n,h} \\in \\mathbb{R}^{N_c \\times N_s}`
    for stimulus frequency :math:`f_n`, MsetCCA finds spatial filters
    :math:`\\mathbf{w}_1, ..., \\mathbf{w}_{N_t}` that maximize inter-trial correlation.

    **MAXVAR Objective Function**

    The optimization problem maximizes the sum of pairwise covariances across trials
    subject to a variance constraint:

    .. math::

        \\tilde{\\mathbf{w}}_{n,1}, ..., \\tilde{\\mathbf{w}}_{n,N_t} =
        \\arg\\max_{\\mathbf{w}_1, ..., \\mathbf{w}_{N_t}}
        \\sum_{h_1 \\neq h_2}^{N_t} \\mathbf{w}_{h_1}^T \\mathbf{X}_{n,h_1}
        \\mathbf{X}_{n,h_2}^T \\mathbf{w}_{h_2}

    subject to:

    .. math::

        \\frac{1}{N_t} \\sum_{h=1}^{N_t} \\mathbf{w}_h^T \\mathbf{X}_{n,h}
        \\mathbf{X}_{n,h}^T \\mathbf{w}_h = 1

    **Generalized Eigenvalue Problem**

    The optimization transforms into a generalized eigenvalue problem. Let
    :math:`\\mathbf{Y}_n` be the concatenation of whitened trials stacked as
    :math:`[\\mathbf{X}_{n,1}; \\mathbf{X}_{n,2}; ...; \\mathbf{X}_{n,N_t}]`:

    .. math::

        (\\mathbf{R}_n - \\mathbf{S}_n) \\mathbf{w} = \\lambda \\mathbf{S}_n \\mathbf{w}

    where:

    - :math:`\\mathbf{R}_n = \\mathbf{Y}_n \\mathbf{Y}_n^T` is the total covariance matrix
    - :math:`\\mathbf{S}_n` is the block-diagonal matrix containing within-trial covariances

    The eigenvectors corresponding to the largest eigenvalues give the optimal
    spatial filters.

    **Whitening Preprocessing**

    Before solving the eigenvalue problem, each trial is whitened using:

    .. math::

        \\tilde{\\mathbf{X}} = \\mathbf{V} \\mathbf{X}, \\quad
        \\mathbf{V} = \\mathbf{\\Lambda}^{-1/2} \\mathbf{U}^T

    where :math:`\\mathbf{U} \\mathbf{\\Lambda} \\mathbf{U}^T` is the eigendecomposition
    of the covariance matrix of :math:`\\mathbf{X}`.

    **Reference Signal (Template) Construction**

    For each stimulus frequency :math:`f_n`, the optimized reference signal is
    the **average** of spatially filtered training trials:

    .. math::

        \\mathbf{Y}_n^{\\text{ref}} = \\frac{1}{N_t} \\sum_{h=1}^{N_t}
        \\mathbf{W}_h^T \\tilde{\\mathbf{X}}_{n,h}

    where :math:`\\mathbf{W}_h` contains the spatial filters for trial :math:`h`.

    **Classification Rule**

    For a test signal :math:`\\mathbf{X}`, CCA is computed between the test data
    and each reference signal :math:`\\mathbf{Y}_n^{\\text{ref}}`:

    .. math::

        \\rho_n = \\max_{\\mathbf{w}_x, \\mathbf{w}_y}
        \\text{corr}(\\mathbf{X}^T \\mathbf{w}_x,
        (\\mathbf{Y}_n^{\\text{ref}})^T \\mathbf{w}_y)

    The predicted class is: :math:`\\hat{f} = \\arg\\max_n \\rho_n`

    Parameters
    ----------
    n_filters : int, default=1
        Number of spatial filters (eigenvectors) to extract from the MAXVAR
        solution. Corresponds to the dimensionality of the learned reference signals.
        Higher values may capture more variance but risk overfitting.

    n_jobs : int, default=1
        Number of parallel jobs for whitening computation.
        Use ``-1`` to use all available cores.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Encoded class labels (0 to n_classes-1).
    freqs_ : list of str
        List of stimulus frequency labels from training data.
    one_hot_ : dict
        Mapping from frequency strings to encoded class labels.
    le_ : LabelEncoder
        Fitted label encoder for frequency strings.
    Ym : dict
        Dictionary mapping encoded class labels to optimized reference signals
        :math:`\\mathbf{Y}_n^{\\text{ref}}` of shape ``(n_filters, n_times)``.

    References
    ----------
    .. [1] Zhang, Y., Zhou, G., Jin, J., Wang, X., and Cichocki, A. (2014).
           Frequency recognition in SSVEP-based BCI using multiset canonical
           correlation analysis. International Journal of Neural Systems,
           24(04), 1450013.
           https://doi.org/10.1142/S0129065714500130

    See Also
    --------
    SSVEP_CCA : Standard CCA using sinusoidal references.
    SSVEP_TRCA : Task-related component analysis for SSVEP.

    Notes
    -----
    .. versionadded:: 0.5.0

    .. versionchanged:: 1.1.1
       Fixed label encoding to match paradigm output. Fixed template computation
       to use averaging instead of concatenation, matching the original algorithm.
    """

    def __init__(self, n_filters=1, n_jobs=1):
        self.n_jobs = n_jobs
        self.n_filters = n_filters
        self.cca = CCA(n_components=1)
        self.freqs_, self.le_, self.classes_ = [], None, None
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

        # Use unique labels from y for LabelEncoder to match the labels
        # passed by the paradigm, not the Epochs event_id keys
        self.freqs_ = list(np.unique(y))
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.classes_ = self.le_.transform(self.freqs_)
        for i, k in zip(self.freqs_, self.le_.transform(self.freqs_)):
            self.one_hot_[i] = k

        n_channels, n_times = X.info["nchan"], len(X.times)
        y_encoded = self.le_.transform(y)

        # Process each class separately according to MsetCCA algorithm
        # Reference: Zhang et al. 2014, "Frequency recognition in SSVEP-based BCI
        # using multiset canonical correlation analysis"
        for m_class in self.classes_:
            # Get trials for this class
            class_mask = y_encoded == m_class
            X_class = X[class_mask].get_data(copy=False)
            n_trials_class = X_class.shape[0]

            if n_trials_class < 2:
                raise ValueError(
                    f"Class {m_class} has only {n_trials_class} trial(s). "
                    "MsetCCA requires at least 2 trials per class."
                )

            # Whiten signals for this class
            if self.n_jobs == 1:
                X_white = np.array([_whitening(X_i) for X_i in X_class])
            else:
                X_white = np.array(
                    Parallel(n_jobs=self.n_jobs)(
                        delayed(_whitening)(X_i) for X_i in X_class
                    )
                )

            # Stack whitened trials: shape (n_channels * n_trials_class, n_times)
            Y = X_white.transpose(1, 0, 2).reshape(n_channels * n_trials_class, n_times)

            # Compute R (total covariance) and S (within-trial block diagonal)
            R = Y @ Y.T

            # Block diagonal mask for within-trial covariance
            mask = np.kron(np.eye(n_trials_class), np.ones((n_channels, n_channels)))
            S = R * mask

            # Solve generalized eigenvalue problem: (R - S) w = lambda * S * w
            # This finds spatial filters that maximize between-trial correlation
            try:
                _, tempW = linalg.eigh(
                    R - S,
                    S,
                    subset_by_index=[
                        R.shape[0] - self.n_filters,
                        R.shape[0] - 1,
                    ],
                )
            except linalg.LinAlgError:
                # Fall back to standard eigenvalue decomposition if generalized fails
                eigenvalues, tempW = linalg.eigh(R - S)
                tempW = tempW[:, -self.n_filters :]

            # Reshape to get per-trial filters: (n_trials_class, n_channels, n_filters)
            W = np.reshape(tempW, (n_trials_class, n_channels, self.n_filters))

            # Normalize filters
            W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-10)

            # Apply spatial filters to get filtered signals
            # Z shape: (n_trials_class, n_filters, n_times)
            Z = np.einsum("tcf,tcs->tfs", W, X_white)

            # Compute template as the MEAN of filtered trials (not concatenation)
            # This is the key difference from the previous implementation
            # Ym shape: (n_filters, n_times)
            self.Ym[m_class] = np.mean(Z, axis=0)

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
        check_is_fitted(self, ["classes_", "one_hot_", "Ym", "freqs_", "le_"])
        if self.classes_ is None:
            raise NotFittedError(
                "This SSVEP_MsetCCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        y = []
        for x in X:
            corr_f = {}
            # Whiten test data to match training preprocessing
            x_white = _whitening(x)
            for f in self.classes_:
                S_x, S_y = self.cca.fit_transform(x_white.T, self.Ym[f].T)
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
        check_is_fitted(self, ["classes_", "one_hot_", "Ym", "freqs_", "le_"])
        if self.classes_ is None:
            raise NotFittedError(
                "This SSVEP_MsetCCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        P = np.zeros(shape=(len(X), len(self.classes_)))
        for i, x in enumerate(X):
            # Whiten test data to match training preprocessing
            x_white = _whitening(x)
            for j, f in enumerate(self.classes_):
                S_x, S_y = self.cca.fit_transform(x_white.T, self.Ym[f].T)
                P[i, j] = np.corrcoef(S_x.T, S_y.T)[0, 1]
        return P / np.resize(P.sum(axis=1), P.T.shape).T
