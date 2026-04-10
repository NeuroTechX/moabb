import logging

import numpy as np
import scipy.linalg as linalg
from joblib import Parallel, delayed
from mne import BaseEpochs
from pyriemann.estimation import Covariances, Shrinkage
from pyriemann.utils.covariance import covariances, normalize
from pyriemann.utils.mean import mean_covariance
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import NotFittedError, check_is_fitted

from .utils import filterbank


log = logging.getLogger(__name__)


def _safe_corrcoef(a, b):
    """Return a finite correlation coefficient for 1D inputs."""
    corr = np.corrcoef(a, b)[0, 1]
    return corr if np.isfinite(corr) else -1.0


def _normalize_score_matrix(scores):
    """Convert per-class scores to robust row-wise probabilities."""
    scores = np.asarray(scores, dtype=float)
    n_trials, n_classes = scores.shape
    probas = np.zeros((n_trials, n_classes), dtype=float)
    eps = 1e-12

    scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    for i, row in enumerate(scores):
        finite_mask = np.isfinite(row)
        if not finite_mask.any():
            probas[i, :] = 1.0 / n_classes
            continue

        row = row.copy()
        min_finite = np.min(row[finite_mask])
        row[~finite_mask] = min_finite
        row = row - np.min(row) + eps
        denom = row.sum()

        if not np.isfinite(denom) or denom <= 0:
            probas[i, :] = 1.0 / n_classes
        else:
            probas[i, :] = row / denom

    return probas


def _infer_label_frequencies(X, y, classes, freq_map=None):
    """Infer numeric stimulus frequencies for each class label."""
    if freq_map is not None:
        missing = [cls for cls in classes if cls not in freq_map]
        if missing:
            raise ValueError(
                "freq_map is missing entries for class labels: "
                f"{missing}. Provide one frequency per class label."
            )
        inferred = {}
        for cls in classes:
            try:
                inferred[cls] = float(freq_map[cls])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Frequency for class label {cls!r} is not numeric: {freq_map[cls]!r}"
                ) from exc
        return inferred

    if len(y) != len(X):
        raise ValueError("X and y must have the same number of trials.")

    inv_event_id = {
        event_code: event_label for event_label, event_code in X.event_id.items()
    }
    label_to_codes = {}
    for label, event_code in zip(y, X.events[:, -1]):
        label_to_codes.setdefault(label, set()).add(int(event_code))

    def _to_float_or_none(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    class_numeric = {cls: _to_float_or_none(cls) for cls in classes}
    class_vals = [v for v in class_numeric.values() if v is not None]
    class_labels_look_ordinal = False
    if len(class_vals) == len(classes):
        class_vals = sorted(class_vals)
        n_classes = len(classes)
        class_labels_look_ordinal = class_vals == list(
            range(n_classes)
        ) or class_vals == list(range(1, n_classes + 1))

    inferred = {}
    label_event_code = {}
    for cls in classes:
        event_codes = label_to_codes.get(cls, set())
        if len(event_codes) != 1:
            raise ValueError(
                "Could not infer a unique event code for class label "
                f"{cls!r}. Got codes: {sorted(event_codes)}. "
                "Provide freq_map to disambiguate."
            )
        event_code = next(iter(event_codes))
        label_event_code[cls] = event_code
        event_label = inv_event_id.get(event_code)
        if event_label is None:
            raise ValueError(
                f"Event code {event_code} is missing from X.event_id. "
                "Cannot infer stimulus frequency."
            )

        cls_float = class_numeric[cls]
        event_float = _to_float_or_none(event_label)

        # Prefer numeric class labels when they are not ordinal class IDs.
        if cls_float is not None and not class_labels_look_ordinal:
            inferred[cls] = cls_float
            continue
        if event_float is not None:
            inferred[cls] = event_float
            continue
        if cls_float is not None:
            inferred[cls] = cls_float
            continue

        raise ValueError(
            "Could not infer numeric stimulus frequency for class label "
            f"{cls!r} from event label {event_label!r}. "
            "Use freq_map={class_label: frequency_hz}."
        )

    # Ordinal event labels such as 1,2,3 (or 0,1,2) are often event codes,
    # not physical stimulation frequencies. In this ambiguous case, fail fast.
    inferred_vals = sorted(inferred.values())
    n_classes = len(classes)
    looks_consecutive = inferred_vals == list(range(n_classes)) or inferred_vals == list(
        range(1, n_classes + 1)
    )
    matches_event_codes = all(
        np.isclose(inferred[cls], label_event_code[cls]) for cls in classes
    )
    if class_labels_look_ordinal and looks_consecutive and matches_event_codes:
        raise ValueError(
            "Could not infer physical stimulus frequencies from class labels/events. "
            "Detected ordinal event labels matching event codes. "
            "Pass freq_map={class_label: frequency_hz} or fit with non-encoded labels."
        )

    return inferred


def _build_sinusoidal_references(class_freqs, n_harmonics, signal_length, n_times):
    """Build harmonic sine/cosine reference matrices for each class label."""
    times = np.linspace(0, signal_length, n_times)
    references = {}
    for class_label, freq in class_freqs.items():
        harmonics = []
        for harmonic_idx in range(1, n_harmonics + 1):
            phase = 2 * np.pi * freq * harmonic_idx * times
            harmonics.extend([np.sin(phase), np.cos(phase)])
        references[class_label] = np.array(harmonics)
    return references


def _score_matrix_from_trials(X, classes, score_trial_fn):
    """Compute class-score matrix for all trials in ``X``."""
    scores = np.zeros((len(X), len(classes)))
    for trial_idx, trial in enumerate(X):
        scores[trial_idx, :] = score_trial_fn(trial)
    return scores


def _predict_labels_from_scores(scores, classes):
    """Return class labels of maximal score for each trial."""
    winners = np.argmax(scores, axis=1)
    return [classes[int(class_idx)] for class_idx in winners]


def _cca_trial_scores(cca, trial, references, classes):
    """Compute per-class CCA correlation scores for one trial."""
    scores = np.zeros(len(classes))
    for class_idx, class_label in enumerate(classes):
        S_x, S_y = cca.fit_transform(trial.T, references[class_label].T)
        scores[class_idx] = _safe_corrcoef(S_x.ravel(), S_y.ravel())
    return scores


def _signed_squared_fusion(correlations):
    """Fuse correlations with sign-preserving quadratic weighting."""
    return sum(np.sign(corr) * (corr**2) for corr in correlations)


def _ecca_trial_scores(trial, classes, templates, references, template_weights):
    """Compute eCCA fused scores for one trial across all classes."""
    scores = np.zeros(len(classes))
    for class_idx, class_label in enumerate(classes):
        template = templates[class_label]
        reference = references[class_label]

        cca_xy = CCA(n_components=1)
        S_x, S_y = cca_xy.fit_transform(trial.T, reference.T)
        r1 = _safe_corrcoef(S_x.ravel(), S_y.ravel())
        w_xy = cca_xy.x_weights_

        cca_xt = CCA(n_components=1)
        S_x2, S_y2 = cca_xt.fit_transform(trial.T, template.T)
        r2 = _safe_corrcoef(S_x2.ravel(), S_y2.ravel())

        r3 = _safe_corrcoef((trial.T @ w_xy).ravel(), (template.T @ w_xy).ravel())
        w_template = template_weights[class_label]
        r4 = _safe_corrcoef(
            (trial.T @ w_template).ravel(), (template.T @ w_template).ravel()
        )
        scores[class_idx] = _signed_squared_fusion([r1, r2, r3, r4])
    return scores


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
    n_harmonics : int
        Number of harmonics :math:`N_h` to include in the reference signal.
        Higher values capture more harmonic components of the SSVEP response.
        Defaults to ``3``.
    freq_map : dict or None
        Optional explicit mapping ``{class_label: stimulus_frequency_hz}``.
        If None, frequencies are inferred from ``X.event_id`` and event codes.
        Defaults to ``None``.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Class labels in the same label space as ``y``,
        of shape ``(n_classes,)``.
    freqs_ : list of str
        List of stimulus frequencies extracted from training data.
    one_hot_ : dict
        Mapping from class labels to class indices.
    slen_ : float
        Signal length in seconds.
    le_ : :class:`sklearn.preprocessing.LabelEncoder`
        Fitted label encoder for frequency strings.
    Yf : dict
        Dictionary mapping class labels to reference signals
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

    def __init__(self, n_harmonics=3, freq_map=None):
        self.Yf = {}
        self.cca = CCA(n_components=1)
        self.n_harmonics = n_harmonics
        self.freq_map = freq_map
        self.classes_ = []
        self.one_hot_ = {}
        self.class_freqs_ = {}
        self._le, self._slen, self._freqs = None, None, []

    def fit(self, X, y, sample_weight=None):
        """Compute reference sinusoid signal.

        These sinusoid are generated for each frequency in the dataset

        Parameters
        ----------
        X : :class:`mne.Epochs`
            The training data as MNE Epochs object.
        y : numpy.ndarray
            Label vector with frequency strings for each trial,
            of shape ``(n_trials,)``.
        sample_weight : Unused,
            Only for compatibility with scikit-learn

        Returns
        -------
        self : ``SSVEP_CCA``
            Instance of classifier.
        """
        if not isinstance(X, BaseEpochs):
            raise ValueError("X should be an MNE Epochs object.")

        y = np.asarray(y)
        self.slen_ = X.times[-1] - X.times[0]
        n_times = len(X.times)

        self.freqs_ = list(np.unique(y))
        self.classes_ = np.array(
            self.freqs_, dtype=y.dtype if y.dtype != object else object
        )
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.one_hot_ = {label: idx for idx, label in enumerate(self.classes_)}
        self.class_freqs_ = _infer_label_frequencies(X, y, self.classes_, self.freq_map)
        self.Yf = _build_sinusoidal_references(
            self.class_freqs_, self.n_harmonics, self.slen_, n_times
        )
        return self

    def predict(self, X):
        """Predict is made by taking the maximum correlation coefficient.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            The data to predict as MNE Epochs object.

        Returns
        -------
        y : list of int
            Predicted labels.
        """
        check_is_fitted(
            self, ["freqs_", "classes_", "one_hot_", "slen_", "le_", "class_freqs_"]
        )
        scores = _score_matrix_from_trials(
            X,
            self.classes_,
            lambda trial: _cca_trial_scores(self.cca, trial, self.Yf, self.classes_),
        )
        return _predict_labels_from_scores(scores, self.classes_)

    def predict_proba(self, X):
        """Probability could be computed from the correlation coefficient.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            The data to predict as MNE Epochs object.

        Returns
        -------
        proba : numpy.ndarray
            Probability of each class for each trial,
            of shape ``(n_trials, n_classes)``.
        """
        check_is_fitted(
            self, ["freqs_", "classes_", "one_hot_", "slen_", "le_", "class_freqs_"]
        )
        scores = _score_matrix_from_trials(
            X,
            self.classes_,
            lambda trial: _cca_trial_scores(self.cca, trial, self.Yf, self.classes_),
        )
        return _normalize_score_matrix(scores)


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
    n_fbands : int
        Number of sub-bands :math:`N_m` for the filter bank decomposition.
        Each sub-band captures different harmonic components of the SSVEP.
        Defaults to ``5``.

    is_ensemble : bool
        If True, use ensemble TRCA which combines spatial filters from all
        stimulus frequencies for improved robustness. If False, use only
        the class-specific spatial filter. Defaults to ``True``.

    method : str
        Method for computing the inter-trial covariance matrix :math:`\\mathbf{S}`.
        Defaults to ``'original'``.

        - ``'original'``: Euclidean mean as in the original paper [1]_.
        - ``'riemann'``: Geodesic (Riemannian) mean, more robust to outliers
          but sensitive to ill-conditioned matrices.
        - ``'logeuclid'``: Log-Euclidean mean, a computationally stable
          alternative to the Riemannian mean.

    estimator : str
        Covariance estimator for regularization. Defaults to ``'scm'``.
        Options include:

        - ``'scm'``: Sample covariance matrix (no regularization, as in [1]_).
        - ``'lwf'``: Ledoit-Wolf shrinkage estimator.
        - ``'oas'``: Oracle Approximating Shrinkage estimator.
        - ``'schaefer'``: Schäfer-Strimmer shrinkage estimator.

    Attributes
    ----------
    fb_coefs : list of float
        Sub-band weights :math:`a^{(m)} = m^{-1.25} + 0.25` for filter bank fusion,
        of length ``n_fbands``.
    classes_ : numpy.ndarray
        Encoded class labels extracted at fit time,
        of shape ``(n_classes,)``.
    n_classes : int
        Number of unique stimulus frequencies/classes.
    templates_ : numpy.ndarray
        Average templates :math:`\\bar{\\mathbf{X}}_n^{(m)}` for each class and sub-band,
        of shape ``(n_classes, n_fbands, n_channels, n_samples)``.
    weights_ : numpy.ndarray
        Spatial filter weights :math:`\\mathbf{w}_n^{(m)}` for each sub-band and class,
        of shape ``(n_fbands, n_classes, n_channels)``.
    freqs_ : list of str
        List of stimulus frequencies from training data.
    peaks_ : numpy.ndarray
        Numeric frequency values for filter bank design,
        of shape ``(n_classes,)``.
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

    def __init__(self, n_fbands=5, is_ensemble=True, method="original", estimator="scm"):
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

        # Symmetrize, shrink, and trace-normalize so that the Riemannian /
        # log-Euclidean mean operates on well-conditioned SPD matrices.
        S = (S + S.transpose(0, 2, 1)) / 2
        S = Shrinkage(shrinkage=0.01).fit_transform(S)
        S = normalize(S, "trace")

        S_mean = mean_covariance(S, metric=self.method)

        return S_mean, Q

    def _compute_trca(self, X):
        """Computation of TRCA spatial filters.

        Parameters
        ----------
        X : numpy.ndarray
            Training data, of shape ``(n_trials, n_channels, n_samples)``.

        Returns
        -------
        W : numpy.ndarray
            Weight coefficients for electrodes which can be used as
            a spatial filter, of shape ``(n_channels,)``.
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
        X : :class:`mne.Epochs`
            Training data. Trials are grouped by class, divided in n_fbands bands by
            the filterbank approach and then used to calculate weight vectors and
            templates for each class and band.

        y : numpy.ndarray
            Label vector in respect to X, of shape ``(n_trials,)``.

        Returns
        -------
        self : ``CCA``
            Instance of classifier.
        """
        if not isinstance(X, BaseEpochs):
            raise ValueError("X should be an MNE Epochs object.")

        y = np.array(y)
        n_channels, n_samples = X.info["nchan"], len(X.times)
        self.sfreq_ = X.info["sfreq"]
        self.freqs_ = list(np.unique(y))
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

        for freq, k in self.one_hot_.items():
            mask = y == freq
            X_cal = X[mask]  # Select data with boolean mask

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
        X : :class:`mne.Epochs`
            Testing data. This will be divided in self.n_fbands using the filterbank approach,
            then it will be transformed by the different spatial filters and compared to the
            previously fit templates according to the selected method for analysis (ensemble or
            not). Finally, correlation scores for all sub-bands of each class will be combined,
            resulting on a single correlation score per class, from which the maximal one is
            identified as the predicted class of the data.

        Returns
        -------
        y_pred : numpy.ndarray
            Prediction vector in respect to X, of shape ``(n_trials,)``.
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
                for _freq, k in self.one_hot_.items():
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
                        np.dot(X_filter.T, w).flatten(), np.dot(template.T, w).flatten()
                    )
                    corr_array[band_n, k] = r[0, 1]

            # Fusion for the filterbank analysis
            self.rho = np.dot(self.fb_coefs, corr_array)

            # Select the maximal value and append to predictions
            self.tau = np.argmax(self.rho)
            y_pred.append(self.one_inv_[self.tau])

        return y_pred

    def predict_proba(self, X):
        """Make predictions on unseen data with the associated probabilities.

        The new data observation X will be filtered
        with weights previously extracted and compared to the templates to assess
        similarity with each of them and select a class based on the maximal value.

        Parameters
        ----------
        X : numpy.ndarray
            Testing data, of shape ``(n_trials, n_channels, n_samples)``.
            This will be divided in self.n_fbands using the filter-bank approach,
            then it will be transformed by the different spatial filters and compared to the
            previously fit templates according to the selected method for analysis (ensemble or
            not). Finally, correlation scores for all sub-bands of each class will be combined,
            resulting on a single correlation score per class, from which the maximal one is
            identified as the predicted class of the data.

        Returns
        -------
        y_pred : numpy.ndarray
            Prediction vector in respect to X, of shape ``(n_trials,)``.
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
                for _freq, k in self.one_hot_.items():
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
                        np.dot(X_filter.T, w).flatten(), np.dot(template.T, w).flatten()
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
    X : numpy.ndarray
        EEG signal data, of shape ``(n_channels, n_samples)``.
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
    n_filters : int
        Number of spatial filters (eigenvectors) to extract from the MAXVAR
        solution. Corresponds to the dimensionality of the learned reference signals.
        Higher values may capture more variance but risk overfitting.
        Defaults to ``1``.

    n_jobs : int
        Number of parallel jobs for whitening computation.
        Use ``-1`` to use all available cores. Defaults to ``1``.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Class labels in the same label space as ``y``,
        of shape ``(n_classes,)``.
    freqs_ : list of str
        List of stimulus frequency labels from training data.
    one_hot_ : dict
        Mapping from class labels to class indices.
    le_ : :class:`sklearn.preprocessing.LabelEncoder`
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
        self.one_hot_, self.one_inv_, self.Ym = {}, {}, {}

    def fit(self, X, y, sample_weight=None):
        """Compute the optimized reference signal at each stimulus frequency.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            The training data as MNE Epochs object.

        y : numpy.ndarray
            The target labels for each trial, of shape ``(n_trials,)``.

        Returns
        -------
        self : ``SSVEP_MsetCCA``
            Instance of classifier.
        """
        if not isinstance(X, BaseEpochs):
            raise ValueError("X should be an MNE Epochs object.")

        # Use unique labels from y for LabelEncoder to match the labels
        # passed by the paradigm, not the Epochs event_id keys
        y = np.asarray(y)
        self.freqs_ = list(np.unique(y))
        self.classes_ = np.array(
            self.freqs_, dtype=y.dtype if y.dtype != object else object
        )
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.one_hot_, self.one_inv_, self.Ym = {}, {}, {}
        for class_idx, class_label in enumerate(self.classes_):
            self.one_hot_[class_label] = class_idx
            self.one_inv_[class_idx] = class_label

        n_channels, n_times = X.info["nchan"], len(X.times)

        # Process each class separately according to MsetCCA algorithm
        # Reference: Zhang et al. 2014, "Frequency recognition in SSVEP-based BCI
        # using multiset canonical correlation analysis"
        for class_label, class_idx in self.one_hot_.items():
            # Get trials for this class
            class_mask = y == class_label
            X_class = X[class_mask].get_data(copy=False)
            n_trials_class = X_class.shape[0]

            if n_trials_class < 2:
                raise ValueError(
                    f"Class {class_label!r} has only {n_trials_class} trial(s). "
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
                    subset_by_index=[R.shape[0] - self.n_filters, R.shape[0] - 1],
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
            self.Ym[class_idx] = np.mean(Z, axis=0)

        return self

    def predict(self, X):
        """Predict is made by taking the maximum correlation coefficient.

        Parameters
        ----------
        X : :class:`mne.Epochs`
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

        class_indices = np.arange(len(self.classes_))
        scores = _score_matrix_from_trials(
            X,
            self.classes_,
            lambda trial: _cca_trial_scores(
                self.cca, _whitening(trial), self.Ym, class_indices
            ),
        )
        return _predict_labels_from_scores(scores, self.classes_)

    def predict_proba(self, X):
        """Probability could be computed from the correlation coefficient.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            The data to predict as MNE Epochs object.

        Returns
        -------
        P : numpy.ndarray
            Probability of each class for each trial,
            of shape ``(n_trials, n_classes)``.
        """

        # Check is fit had been called
        check_is_fitted(self, ["classes_", "one_hot_", "Ym", "freqs_", "le_"])
        if self.classes_ is None:
            raise NotFittedError(
                "This SSVEP_MsetCCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        class_indices = np.arange(len(self.classes_))
        scores = _score_matrix_from_trials(
            X,
            self.classes_,
            lambda trial: _cca_trial_scores(
                self.cca, _whitening(trial), self.Ym, class_indices
            ),
        )
        return _normalize_score_matrix(scores)


class SSVEP_itCCA(BaseEstimator, ClassifierMixin):
    """Individual Template CCA (itCCA) for SSVEP detection [1]_.

    itCCA replaces the sinusoidal reference signals used in standard CCA with
    individual templates computed by averaging training trials for each stimulus
    frequency. This captures subject-specific SSVEP morphology that sinusoidal
    references cannot model.

    **Mathematical Formulation**

    For each stimulus frequency :math:`f_n`, the individual template is computed
    as the average across all :math:`N_t` training trials:

    .. math::

        \\bar{\\mathbf{X}}_n = \\frac{1}{N_t} \\sum_{h=1}^{N_t} \\mathbf{X}_n^{(h)}

    For a test signal :math:`\\mathbf{X}`, CCA is computed between the test data
    and each individual template :math:`\\bar{\\mathbf{X}}_n`:

    .. math::

        \\rho_n = \\max_{\\mathbf{w}_x, \\mathbf{w}_y}
        \\text{corr}(\\mathbf{X}^T \\mathbf{w}_x,
        \\bar{\\mathbf{X}}_n^T \\mathbf{w}_y)

    **Classification Rule**

    .. math::

        \\hat{f} = \\arg\\max_{n} \\rho_n

    Attributes
    ----------
    classes_ : numpy.ndarray
        Class labels in the same label space as ``y``,
        of shape ``(n_classes,)``.
    freqs_ : list of str
        List of stimulus frequencies from training data.
    one_hot_ : dict
        Mapping from class labels to class indices.
    le_ : :class:`sklearn.preprocessing.LabelEncoder`
        Fitted label encoder.
    templates_ : dict
        Dictionary mapping frequency strings to averaged templates,
        each of shape ``(n_channels, n_times)``.

    References
    ----------
    .. [1] Nakanishi, M., Wang, Y., Wang, Y.-T., Mitsukura, Y., & Jung, T.-P.
           (2014). A high-speed brain speller using steady-state visual evoked
           potentials. International Journal of Neural Systems, 24(06), 1450019.
           https://doi.org/10.1142/S0129065714500191

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    def __init__(self):
        self.cca = CCA(n_components=1)
        self.classes_ = []
        self.one_hot_ = {}
        self.templates_ = {}
        self._le, self._freqs = None, []

    def fit(self, X, y, sample_weight=None):
        """Compute individual templates by averaging trials per class.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            Training data as MNE Epochs object.
        y : numpy.ndarray
            Label vector with frequency strings for each trial,
            of shape ``(n_trials,)``.
        sample_weight : Unused
            Only for compatibility with scikit-learn.

        Returns
        -------
        self : SSVEP_itCCA
            Fitted instance.
        """
        if not isinstance(X, BaseEpochs):
            raise ValueError("X should be an MNE Epochs object.")

        y = np.array(y)
        self.freqs_ = list(np.unique(y))
        self.classes_ = np.array(
            self.freqs_, dtype=y.dtype if y.dtype != object else object
        )
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.one_hot_ = {label: idx for idx, label in enumerate(self.classes_)}
        self.templates_ = {}

        for class_label in self.classes_:
            mask = y == class_label
            X_f = X[mask].get_data(copy=False)
            self.templates_[class_label] = np.mean(X_f, axis=0)

        return self

    def predict(self, X):
        """Predict class labels for test data.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            Test data as MNE Epochs object.

        Returns
        -------
        y : list of int
            Predicted labels.
        """
        check_is_fitted(self, ["freqs_", "classes_", "one_hot_", "le_", "templates_"])
        scores = _score_matrix_from_trials(
            X,
            self.classes_,
            lambda trial: _cca_trial_scores(
                self.cca, trial, self.templates_, self.classes_
            ),
        )
        return _predict_labels_from_scores(scores, self.classes_)

    def predict_proba(self, X):
        """Predict class probabilities from correlation coefficients.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            Test data as MNE Epochs object.

        Returns
        -------
        P : numpy.ndarray
            Probability of each class for each trial,
            of shape ``(n_trials, n_classes)``.
        """
        check_is_fitted(self, ["freqs_", "classes_", "one_hot_", "le_", "templates_"])
        scores = _score_matrix_from_trials(
            X,
            self.classes_,
            lambda trial: _cca_trial_scores(
                self.cca, trial, self.templates_, self.classes_
            ),
        )
        return _normalize_score_matrix(scores)


class SSVEP_eCCA(BaseEstimator, ClassifierMixin):
    """Extended CCA (eCCA) for SSVEP detection [1]_.

    Extended CCA combines standard CCA with individual template CCA through a
    multi-feature fusion approach. It computes four correlation features that
    capture complementary information from both sinusoidal references and
    data-driven templates.

    **Mathematical Formulation**

    For each test signal :math:`\\mathbf{X}` and stimulus class :math:`n`,
    four correlation features are computed:

    .. math::

        r_1 &= \\rho(\\text{CCA}(\\mathbf{X}, \\mathbf{Y}_n)) \\\\
        r_2 &= \\rho(\\text{CCA}(\\mathbf{X}, \\bar{\\mathbf{X}}_n)) \\\\
        r_3 &= \\rho(\\mathbf{X}^T \\hat{\\mathbf{w}}_{xn},
               \\bar{\\mathbf{X}}_n^T \\hat{\\mathbf{w}}_{xn}) \\\\
        r_4 &= \\rho(\\mathbf{X}^T \\tilde{\\mathbf{w}}_{xn},
               \\bar{\\mathbf{X}}_n^T \\tilde{\\mathbf{w}}_{xn})

    where:

    - :math:`\\mathbf{Y}_n` is the sinusoidal reference for frequency :math:`f_n`
    - :math:`\\bar{\\mathbf{X}}_n` is the averaged individual template
    - :math:`\\hat{\\mathbf{w}}_{xn}` is the spatial filter from
      :math:`\\text{CCA}(\\mathbf{X}, \\mathbf{Y}_n)`
    - :math:`\\tilde{\\mathbf{w}}_{xn}` is the spatial filter from
      :math:`\\text{CCA}(\\bar{\\mathbf{X}}_n, \\mathbf{Y}_n)`

    **Classification Rule**

    Features are fused using signed squared correlation:

    .. math::

        \\rho_n = \\sum_{l=1}^{4} \\text{sign}(r_l) \\cdot r_l^2

    The predicted class is: :math:`\\hat{f} = \\arg\\max_n \\rho_n`

    Parameters
    ----------
    n_harmonics : int
        Number of harmonics for sinusoidal reference signal generation.
        Defaults to ``3``.
    freq_map : dict or None
        Optional explicit mapping ``{class_label: stimulus_frequency_hz}``.
        If None, frequencies are inferred from ``X.event_id`` and event codes.
        Defaults to ``None``.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Class labels in the same label space as ``y``,
        of shape ``(n_classes,)``.
    freqs_ : list of str
        Stimulus frequencies from training data.
    one_hot_ : dict
        Mapping from class labels to class indices.
    le_ : :class:`sklearn.preprocessing.LabelEncoder`
        Fitted label encoder.
    slen_ : float
        Signal length in seconds.
    templates_ : dict
        Averaged individual templates per frequency.
    Yf : dict
        Sinusoidal reference signals per frequency.
    w_template_ : dict
        Spatial filters from CCA(template, sinusoidal reference).

    References
    ----------
    .. [1] Chen, X., Wang, Y., Gao, S., Jung, T.-P., & Gao, X. (2015).
           Filter bank canonical correlation analysis for implementing a
           high-speed SSVEP-based brain-computer interface.
           PLOS ONE, 10(12), e0140703.
           https://doi.org/10.1371/journal.pone.0140703

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    def __init__(self, n_harmonics=3, freq_map=None):
        self.n_harmonics = n_harmonics
        self.freq_map = freq_map
        self.cca = CCA(n_components=1)
        self.classes_ = []
        self.one_hot_ = {}
        self.class_freqs_ = {}
        self.templates_ = {}
        self.Yf = {}
        self.w_template_ = {}

    def fit(self, X, y, sample_weight=None):
        """Compute individual templates, sinusoidal references, and spatial filters.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            Training data as MNE Epochs object.
        y : numpy.ndarray
            Label vector with frequency strings for each trial,
            of shape ``(n_trials,)``.
        sample_weight : Unused
            Only for compatibility with scikit-learn.

        Returns
        -------
        self : SSVEP_eCCA
            Fitted instance.
        """
        if not isinstance(X, BaseEpochs):
            raise ValueError("X should be an MNE Epochs object.")

        y = np.array(y)
        self.slen_ = X.times[-1] - X.times[0]
        n_times = len(X.times)
        self.freqs_ = list(np.unique(y))
        self.classes_ = np.array(
            self.freqs_, dtype=y.dtype if y.dtype != object else object
        )
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.one_hot_ = {label: idx for idx, label in enumerate(self.classes_)}
        self.class_freqs_ = _infer_label_frequencies(X, y, self.classes_, self.freq_map)
        self.templates_, self.Yf, self.w_template_ = {}, {}, {}

        for class_label in self.classes_:
            # Individual template
            mask = y == class_label
            X_f = X[mask].get_data(copy=False)
            self.templates_[class_label] = np.mean(X_f, axis=0)
        self.Yf = _build_sinusoidal_references(
            self.class_freqs_, self.n_harmonics, self.slen_, n_times
        )
        for class_label in self.classes_:
            # Spatial filter from CCA(template, sinusoidal reference)
            cca_tmp = CCA(n_components=1)
            cca_tmp.fit(self.templates_[class_label].T, self.Yf[class_label].T)
            self.w_template_[class_label] = cca_tmp.x_weights_

        return self

    def predict(self, X):
        """Predict class labels using 4-feature fusion.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            Test data as MNE Epochs object.

        Returns
        -------
        y : list of int
            Predicted labels.
        """
        check_is_fitted(
            self,
            [
                "freqs_",
                "classes_",
                "one_hot_",
                "le_",
                "slen_",
                "templates_",
                "Yf",
                "w_template_",
                "class_freqs_",
            ],
        )
        scores = _score_matrix_from_trials(
            X,
            self.classes_,
            lambda trial: _ecca_trial_scores(
                trial, self.classes_, self.templates_, self.Yf, self.w_template_
            ),
        )
        return _predict_labels_from_scores(scores, self.classes_)

    def predict_proba(self, X):
        """Predict class probabilities from fused correlation features.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            Test data as MNE Epochs object.

        Returns
        -------
        P : numpy.ndarray
            Probability of each class for each trial,
            of shape ``(n_trials, n_classes)``.
        """
        check_is_fitted(
            self,
            [
                "freqs_",
                "classes_",
                "one_hot_",
                "le_",
                "slen_",
                "templates_",
                "Yf",
                "w_template_",
                "class_freqs_",
            ],
        )
        scores = _score_matrix_from_trials(
            X,
            self.classes_,
            lambda trial: _ecca_trial_scores(
                trial, self.classes_, self.templates_, self.Yf, self.w_template_
            ),
        )
        return _normalize_score_matrix(scores)


class SSVEP_TRCA_R(BaseEstimator, ClassifierMixin):
    """Regularized TRCA (TRCA-R) for SSVEP detection [1]_.

    TRCA-R extends the standard TRCA method by incorporating reference signal
    projection as regularization. Training data is projected into the orthogonal
    complement of the sinusoidal reference signals before computing TRCA spatial
    filters. This removes the sinusoidal component and forces TRCA to learn
    filters from the residual, improving performance with limited training data.

    **Mathematical Formulation**

    For each stimulus frequency :math:`f_n`, sinusoidal references
    :math:`\\mathbf{Y}_n` are constructed. The projection matrix
    :math:`\\mathbf{P}_{\\perp}` removes the reference component:

    .. math::

        \\mathbf{P}_{\\perp} = \\mathbf{I} -
        \\mathbf{Y}_n (\\mathbf{Y}_n^T \\mathbf{Y}_n)^{-1} \\mathbf{Y}_n^T

    Projected training data: :math:`\\tilde{\\mathbf{X}} = \\mathbf{X} \\mathbf{P}_{\\perp}`

    TRCA spatial filters are then computed on the projected data as in standard TRCA.

    Parameters
    ----------
    n_fbands : int
        Number of sub-bands for the filter bank decomposition.
        Defaults to ``5``.
    n_harmonics : int
        Number of harmonics for sinusoidal reference generation.
        Defaults to ``3``.
    is_ensemble : bool
        If True, use ensemble TRCA with combined spatial filters.
        Defaults to ``True``.
    method : str
        Covariance estimation method: 'original', 'riemann', or 'logeuclid'.
        Defaults to ``'original'``.
    estimator : str
        Covariance estimator: 'scm', 'lwf', 'oas', or 'schaefer'.
        Defaults to ``'scm'``.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Encoded class labels, of shape ``(n_classes,)``.
    templates_ : numpy.ndarray
        Average templates for each class and sub-band,
        of shape ``(n_classes, n_fbands, n_channels, n_samples)``.
    weights_ : numpy.ndarray
        Spatial filter weights for each sub-band and class,
        of shape ``(n_fbands, n_classes, n_channels)``.

    References
    ----------
    .. [1] Wong, C. M., et al. (2020). Spatial filtering in SSVEP-based BCIs:
           unified framework and new improvements. IEEE Transactions on Biomedical
           Engineering, 67(11), 3057-3072.
           https://doi.org/10.1109/TBME.2020.2975552

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    def __init__(
        self,
        n_fbands=5,
        n_harmonics=3,
        is_ensemble=True,
        method="original",
        estimator="scm",
    ):
        self.n_fbands = n_fbands
        self.n_harmonics = n_harmonics
        self.is_ensemble = is_ensemble
        self.method = method
        self.estimator = estimator
        self.fb_coefs = [(x + 1) ** (-1.25) + 0.25 for x in range(self.n_fbands)]
        self.one_hot_, self.one_inv_ = {}, {}
        self.sfreq_, self.freqs_, self.peaks_ = None, None, None
        self.le_, self.classes_, self.n_classes = None, None, None
        self.templates_, self.weights_ = None, None

    def _build_reference(self, freq, n_samples, sfreq):
        """Build sinusoidal reference signal for a given frequency."""
        t = np.arange(n_samples) / sfreq
        yf = []
        for h in range(1, self.n_harmonics + 1):
            yf.append(np.sin(2 * np.pi * freq * h * t))
            yf.append(np.cos(2 * np.pi * freq * h * t))
        return np.array(yf)

    def _project_orthogonal(self, X, Y):
        """Project X into orthogonal complement of Y.

        Parameters
        ----------
        X : numpy.ndarray
            Input data, of shape ``(..., n_channels, n_samples)``.
        Y : ndarray
            Reference signals, of shape ``(2*n_harmonics, n_samples)``.

        Returns
        -------
        X_proj : numpy.ndarray
            Projected data, same shape as X.
        """
        YtY_inv = np.linalg.inv(Y @ Y.T)
        P = Y.T @ YtY_inv @ Y  # (n_samples, n_samples)
        return X - X @ P

    def _compute_trca(self, data):
        """Compute TRCA spatial filters."""
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_trials, n_channels, n_samples = data.shape

        S = np.zeros((n_channels, n_channels))
        for trial_i in range(n_trials - 1):
            x1 = np.squeeze(data[trial_i, :, :])
            x1 = x1 - np.mean(x1, axis=1, keepdims=True)
            for trial_j in range(trial_i + 1, n_trials):
                x2 = np.squeeze(data[trial_j, :, :])
                x2 = x2 - np.mean(x2, axis=1, keepdims=True)
                S = S + x1 @ x2.T + x2 @ x1.T

        UX = np.concatenate([data[t, :, :] for t in range(n_trials)], axis=1)
        UX = UX - np.mean(UX, axis=1, keepdims=True)
        Q = UX @ UX.T

        lambdas, W = linalg.eig(S, Q, left=True, right=False)
        arr1inds = lambdas.argsort()
        W = W[:, arr1inds[::-1]]
        return W[:, 0], W

    def fit(self, X, y):
        """Extract spatial filters and templates with reference projection.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            Training data as MNE Epochs object.
        y : numpy.ndarray
            Label vector with frequency strings for each trial,
            of shape ``(n_trials,)``.

        Returns
        -------
        self : SSVEP_TRCA_R
            Fitted instance.
        """
        if not isinstance(X, BaseEpochs):
            raise ValueError("X should be an MNE Epochs object.")

        y = np.array(y)
        n_channels, n_samples = X.info["nchan"], len(X.times)
        self.sfreq_ = X.info["sfreq"]
        self.freqs_ = list(np.unique(y))
        self.peaks_ = np.array([float(f) for f in self.freqs_])
        self.fb_coefs = [(x + 1) ** (-1.25) + 0.25 for x in range(self.n_fbands)]
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.classes_ = self.le_.transform(self.freqs_)
        self.n_classes = len(self.classes_)
        for i, k in zip(self.freqs_, self.classes_):
            self.one_hot_[i] = k
            self.one_inv_[k] = i

        self.templates_ = np.zeros((self.n_classes, self.n_fbands, n_channels, n_samples))
        self.weights_ = np.zeros((self.n_fbands, self.n_classes, n_channels))

        for freq, k in self.one_hot_.items():
            mask = y == freq
            X_cal = X[mask]
            freq_val = float(freq)

            for band_n in range(self.n_fbands):
                X_filter = filterbank(
                    X_cal.get_data(copy=False), self.sfreq_, band_n, self.peaks_
                )
                # Build reference and project
                Y_ref = self._build_reference(freq_val, n_samples, self.sfreq_)
                X_proj = self._project_orthogonal(X_filter, Y_ref)

                w_best, _ = self._compute_trca(X_proj)
                self.templates_[k, band_n, :, :] = np.mean(X_filter, axis=0)
                self.weights_[band_n, k, :] = w_best

        return self

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            Test data as MNE Epochs object.

        Returns
        -------
        y_pred : list of int
            Predicted labels.
        """
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
                "This SSVEP_TRCA_R instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        y_pred = []
        for x in X:
            corr_array = np.zeros((self.n_fbands, self.n_classes))
            for band_n in range(self.n_fbands):
                X_filter = filterbank(x, self.sfreq_, band_n, self.peaks_)
                for _freq, k in self.one_hot_.items():
                    template = np.squeeze(self.templates_[k, band_n, :, :])
                    if self.is_ensemble:
                        w = np.squeeze(self.weights_[band_n, :, :]).T
                    else:
                        w = np.squeeze(self.weights_[band_n, k, :]).T
                    r = np.corrcoef(
                        np.dot(X_filter.T, w).flatten(), np.dot(template.T, w).flatten()
                    )
                    corr_array[band_n, k] = r[0, 1]
            rho = np.dot(self.fb_coefs, corr_array)
            tau = np.argmax(rho)
            y_pred.append(self.one_inv_[tau])
        return y_pred

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            Test data as MNE Epochs object.

        Returns
        -------
        y_pred : numpy.ndarray
            Probabilities per class, of shape ``(n_trials, n_classes)``.
        """
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
                "This SSVEP_TRCA_R instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )
        n_trials = len(X)
        y_pred = np.zeros((n_trials, self.n_classes))

        for trial_n, X_test in enumerate(X):
            corr_array = np.zeros((self.n_fbands, self.n_classes))
            for band_n in range(self.n_fbands):
                X_filter = filterbank(X_test, self.sfreq_, band_n, self.peaks_)
                for _freq, k in self.one_hot_.items():
                    template = np.squeeze(self.templates_[k, band_n, :, :])
                    if self.is_ensemble:
                        w = np.squeeze(self.weights_[band_n, :, :]).T
                    else:
                        w = np.squeeze(self.weights_[band_n, k, :]).T
                    r = np.corrcoef(
                        np.dot(X_filter.T, w).flatten(), np.dot(template.T, w).flatten()
                    )
                    corr_array[band_n, k] = r[0, 1]
            normalized_coefs = self.fb_coefs / (np.sum(self.fb_coefs))
            rho = np.dot(normalized_coefs, corr_array)
            rho /= sum(rho)
            y_pred[trial_n] = rho
        return y_pred


class SSVEP_SSCOR(BaseEstimator, ClassifierMixin):
    """Sum of Squared Correlations (SSCOR) for SSVEP detection [1]_.

    SSCOR maximizes the sum of squared inter-trial correlations rather than
    raw covariances (as in TRCA). By normalizing each trial before computing
    the inter-trial covariance matrix, SSCOR directly optimizes the correlation
    metric used at prediction time.

    **Mathematical Formulation**

    For each stimulus frequency, SSCOR finds spatial filter :math:`\\mathbf{w}` that
    maximizes:

    .. math::

        \\hat{\\mathbf{w}} = \\arg\\max_{\\mathbf{w}}
        \\sum_{h_1 \\neq h_2} \\text{corr}(\\mathbf{w}^T \\mathbf{X}^{(h_1)},
        \\mathbf{w}^T \\mathbf{X}^{(h_2)})^2

    This is solved as a generalized eigenvalue problem with normalized
    (variance-unit) covariance matrices instead of raw covariance.

    Parameters
    ----------
    n_fbands : int
        Number of sub-bands for filter bank decomposition.
        Defaults to ``5``.
    is_ensemble : bool
        If True, use ensemble approach combining spatial filters from all classes.
        Defaults to ``True``.
    estimator : str
        Covariance estimator: 'scm', 'lwf', 'oas', or 'schaefer'.
        Defaults to ``'scm'``.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Encoded class labels, of shape ``(n_classes,)``.
    templates_ : numpy.ndarray
        Average templates,
        of shape ``(n_classes, n_fbands, n_channels, n_samples)``.
    weights_ : numpy.ndarray
        Spatial filter weights,
        of shape ``(n_fbands, n_classes, n_channels)``.

    References
    ----------
    .. [1] Kumar, G. K. & Reddy, M. R. (2019). Designing a sum of squared
           correlations framework for enhancing SSVEP-based BCIs. IEEE
           Transactions on Neural Systems and Rehabilitation Engineering,
           27(10), 2044-2050.
           https://doi.org/10.1109/TNSRE.2019.2940946

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    def __init__(self, n_fbands=5, is_ensemble=True, estimator="scm"):
        self.n_fbands = n_fbands
        self.is_ensemble = is_ensemble
        self.estimator = estimator
        self.fb_coefs = [(x + 1) ** (-1.25) + 0.25 for x in range(self.n_fbands)]
        self.one_hot_, self.one_inv_ = {}, {}
        self.sfreq_, self.freqs_, self.peaks_ = None, None, None
        self.le_, self.classes_, self.n_classes = None, None, None
        self.templates_, self.weights_ = None, None

    def _compute_sscor(self, data):
        """Compute SSCOR spatial filters.

        Uses normalized (correlation-based) covariance instead of raw covariance.
        Each trial is mean-centered and variance-normalized before computing S.

        Parameters
        ----------
        data : ndarray
            Input data, of shape ``(n_trials, n_channels, n_samples)``.

        Returns
        -------
        w_best : ndarray
            Best spatial filter, of shape ``(n_channels,)``.
        W : ndarray
            All spatial filters, of shape ``(n_channels, n_channels)``.
        """
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_trials, n_channels, n_samples = data.shape

        # Normalize each trial: mean-center and unit-variance per channel
        data_norm = np.copy(data)
        for t in range(n_trials):
            data_norm[t] = data_norm[t] - np.mean(data_norm[t], axis=1, keepdims=True)
            norms = np.linalg.norm(data_norm[t], axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            data_norm[t] = data_norm[t] / norms

        # S matrix: sum of cross-trial correlation matrices
        S = np.zeros((n_channels, n_channels))
        for trial_i in range(n_trials - 1):
            x1 = data_norm[trial_i]
            for trial_j in range(trial_i + 1, n_trials):
                x2 = data_norm[trial_j]
                S = S + x1 @ x2.T + x2 @ x1.T

        # Q matrix: pooled variance from normalized data
        UX = np.concatenate([data_norm[t] for t in range(n_trials)], axis=1)
        Q = UX @ UX.T

        lambdas, W = linalg.eig(S, Q, left=True, right=False)
        arr1inds = lambdas.argsort()
        W = W[:, arr1inds[::-1]]
        return W[:, 0], W

    def fit(self, X, y):
        """Extract spatial filters and templates.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            Training data as MNE Epochs object.
        y : numpy.ndarray
            Label vector with frequency strings for each trial,
            of shape ``(n_trials,)``.

        Returns
        -------
        self : SSVEP_SSCOR
            Fitted instance.
        """
        if not isinstance(X, BaseEpochs):
            raise ValueError("X should be an MNE Epochs object.")

        y = np.array(y)
        n_channels, n_samples = X.info["nchan"], len(X.times)
        self.sfreq_ = X.info["sfreq"]
        self.freqs_ = list(np.unique(y))
        self.peaks_ = np.array([float(f) for f in self.freqs_])
        self.fb_coefs = [(x + 1) ** (-1.25) + 0.25 for x in range(self.n_fbands)]
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.classes_ = self.le_.transform(self.freqs_)
        self.n_classes = len(self.classes_)
        for i, k in zip(self.freqs_, self.classes_):
            self.one_hot_[i] = k
            self.one_inv_[k] = i

        self.templates_ = np.zeros((self.n_classes, self.n_fbands, n_channels, n_samples))
        self.weights_ = np.zeros((self.n_fbands, self.n_classes, n_channels))

        for freq, k in self.one_hot_.items():
            mask = y == freq
            X_cal = X[mask]
            for band_n in range(self.n_fbands):
                X_filter = filterbank(
                    X_cal.get_data(copy=False), self.sfreq_, band_n, self.peaks_
                )
                w_best, _ = self._compute_sscor(X_filter)
                self.templates_[k, band_n, :, :] = np.mean(X_filter, axis=0)
                self.weights_[band_n, k, :] = w_best

        return self

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            Test data as MNE Epochs object.

        Returns
        -------
        y_pred : list of int
            Predicted labels.
        """
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
                "This SSVEP_SSCOR instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        y_pred = []
        for x in X:
            corr_array = np.zeros((self.n_fbands, self.n_classes))
            for band_n in range(self.n_fbands):
                X_filter = filterbank(x, self.sfreq_, band_n, self.peaks_)
                for _freq, k in self.one_hot_.items():
                    template = np.squeeze(self.templates_[k, band_n, :, :])
                    if self.is_ensemble:
                        w = np.squeeze(self.weights_[band_n, :, :]).T
                    else:
                        w = np.squeeze(self.weights_[band_n, k, :]).T
                    r = np.corrcoef(
                        np.dot(X_filter.T, w).flatten(), np.dot(template.T, w).flatten()
                    )
                    corr_array[band_n, k] = r[0, 1]
            rho = np.dot(self.fb_coefs, corr_array)
            tau = np.argmax(rho)
            y_pred.append(self.one_inv_[tau])
        return y_pred

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            Test data as MNE Epochs object.

        Returns
        -------
        y_pred : numpy.ndarray
            Probabilities per class, of shape ``(n_trials, n_classes)``.
        """
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
                "This SSVEP_SSCOR instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )
        n_trials = len(X)
        y_pred = np.zeros((n_trials, self.n_classes))

        for trial_n, X_test in enumerate(X):
            corr_array = np.zeros((self.n_fbands, self.n_classes))
            for band_n in range(self.n_fbands):
                X_filter = filterbank(X_test, self.sfreq_, band_n, self.peaks_)
                for _freq, k in self.one_hot_.items():
                    template = np.squeeze(self.templates_[k, band_n, :, :])
                    if self.is_ensemble:
                        w = np.squeeze(self.weights_[band_n, :, :]).T
                    else:
                        w = np.squeeze(self.weights_[band_n, k, :]).T
                    r = np.corrcoef(
                        np.dot(X_filter.T, w).flatten(), np.dot(template.T, w).flatten()
                    )
                    corr_array[band_n, k] = r[0, 1]
            normalized_coefs = self.fb_coefs / (np.sum(self.fb_coefs))
            rho = np.dot(normalized_coefs, corr_array)
            rho /= sum(rho)
            y_pred[trial_n] = rho
        return y_pred


class SSVEP_TDCA(BaseEstimator, ClassifierMixin):
    """Task Discriminant Component Analysis (TDCA) for SSVEP detection [1]_.

    TDCA is a spatio-temporal discriminant approach that learns common discriminative
    spatial filters across all stimulus classes using a Fisher-like criterion.
    Unlike TRCA which learns per-class filters, TDCA learns a shared projection
    that maximizes between-class scatter relative to within-class scatter.

    The method augments EEG data with delayed versions (temporal embedding) to
    capture temporal dynamics, then applies Fisher discriminant analysis in the
    augmented space.

    **Mathematical Formulation**

    1. **Temporal Embedding**: Each trial is augmented with delayed copies:

    .. math::

        \\tilde{\\mathbf{X}} = [\\mathbf{X}(t), \\mathbf{X}(t-1), ...,
        \\mathbf{X}(t-d+1)]

    where :math:`d` is the number of delays (``n_delay``).

    2. **Between-class scatter** :math:`\\mathbf{S}_b` and **within-class scatter**
    :math:`\\mathbf{S}_w` are computed across all classes.

    3. **Fisher criterion**: Solve :math:`\\mathbf{S}_b \\mathbf{w} = \\lambda \\mathbf{S}_w \\mathbf{w}`

    4. Top-k eigenvectors form the shared spatial filters.

    Parameters
    ----------
    n_fbands : int
        Number of sub-bands for filter bank decomposition.
        Defaults to ``5``.
    n_components : int
        Number of discriminant components to retain.
        Defaults to ``1``.
    n_delay : int
        Number of temporal delays for data augmentation.
        Defaults to ``6``.
    is_ensemble : bool
        If True, use combined filters from all sub-bands for prediction.
        Defaults to ``True``.

    Attributes
    ----------
    classes_ : numpy.ndarray
        Encoded class labels, of shape ``(n_classes,)``.
    templates_ : numpy.ndarray
        Templates in the augmented space,
        of shape ``(n_classes, n_fbands, n_channels * n_delay, n_samples_aug)``.
    weights_ : numpy.ndarray
        Shared spatial filters for each sub-band,
        of shape ``(n_fbands, n_components, n_channels * n_delay)``.

    References
    ----------
    .. [1] Liu, B., Chen, X., Li, X., Wang, Y., Gao, X., & Gao, S. (2021).
           Improving the performance of individually calibrated SSVEP-BCI by
           task-discriminant component analysis. IEEE Transactions on Neural
           Systems and Rehabilitation Engineering, 29, 1998-2007.
           https://doi.org/10.1109/TNSRE.2021.3114340

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    def __init__(self, n_fbands=5, n_components=1, n_delay=6, is_ensemble=True):
        self.n_fbands = n_fbands
        self.n_components = n_components
        self.n_delay = n_delay
        self.is_ensemble = is_ensemble
        self.fb_coefs = [(x + 1) ** (-1.25) + 0.25 for x in range(self.n_fbands)]
        self.one_hot_, self.one_inv_ = {}, {}
        self.sfreq_, self.freqs_, self.peaks_ = None, None, None
        self.le_, self.classes_, self.n_classes = None, None, None
        self.templates_, self.weights_ = None, None

    def _augment_data(self, X):
        """Augment data with temporal delays.

        Parameters
        ----------
        X : numpy.ndarray
            Input data, of shape ``(..., n_channels, n_samples)``.

        Returns
        -------
        X_aug : ndarray with channels expanded by n_delay factor, samples reduced.
        """
        if X.ndim == 2:
            n_channels, n_samples = X.shape
            n_samples_aug = n_samples - self.n_delay + 1
            X_aug = np.zeros((n_channels * self.n_delay, n_samples_aug))
            for d in range(self.n_delay):
                X_aug[d * n_channels : (d + 1) * n_channels, :] = X[
                    :, d : d + n_samples_aug
                ]
            return X_aug
        elif X.ndim == 3:
            n_trials, n_channels, n_samples = X.shape
            n_samples_aug = n_samples - self.n_delay + 1
            X_aug = np.zeros((n_trials, n_channels * self.n_delay, n_samples_aug))
            for d in range(self.n_delay):
                X_aug[:, d * n_channels : (d + 1) * n_channels, :] = X[
                    :, :, d : d + n_samples_aug
                ]
            return X_aug
        else:
            raise ValueError("X must be 2D or 3D.")

    def fit(self, X, y):
        """Learn discriminant spatial filters and class templates.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            Training data as MNE Epochs object.
        y : numpy.ndarray
            Label vector with frequency strings for each trial,
            of shape ``(n_trials,)``.

        Returns
        -------
        self : SSVEP_TDCA
            Fitted instance.
        """
        if not isinstance(X, BaseEpochs):
            raise ValueError("X should be an MNE Epochs object.")

        y = np.array(y)
        n_channels, n_samples = X.info["nchan"], len(X.times)
        self.sfreq_ = X.info["sfreq"]
        self.freqs_ = list(np.unique(y))
        self.peaks_ = np.array([float(f) for f in self.freqs_])
        self.fb_coefs = [(x + 1) ** (-1.25) + 0.25 for x in range(self.n_fbands)]
        self.le_ = LabelEncoder().fit(self.freqs_)
        self.classes_ = self.le_.transform(self.freqs_)
        self.n_classes = len(self.classes_)
        for i, k in zip(self.freqs_, self.classes_):
            self.one_hot_[i] = k
            self.one_inv_[k] = i

        n_aug_channels = n_channels * self.n_delay
        n_samples_aug = n_samples - self.n_delay + 1

        self.templates_ = np.zeros(
            (self.n_classes, self.n_fbands, n_aug_channels, n_samples_aug)
        )
        self.weights_ = np.zeros((self.n_fbands, self.n_components, n_aug_channels))

        for band_n in range(self.n_fbands):
            # Collect augmented data per class
            class_data = {}
            class_means = {}

            for freq, k in self.one_hot_.items():
                mask = y == freq
                X_cal = X[mask]
                X_filter = filterbank(
                    X_cal.get_data(copy=False), self.sfreq_, band_n, self.peaks_
                )
                X_aug = self._augment_data(X_filter)
                class_data[k] = X_aug
                class_means[k] = np.mean(X_aug, axis=0)

            # Global mean
            all_data = np.concatenate(list(class_data.values()), axis=0)
            global_mean = np.mean(all_data, axis=0)

            # Between-class scatter
            S_b = np.zeros((n_aug_channels, n_aug_channels))
            for k, mean_k in class_means.items():
                n_k = class_data[k].shape[0]
                diff = mean_k - global_mean
                S_b += n_k * (diff @ diff.T)

            # Within-class scatter
            S_w = np.zeros((n_aug_channels, n_aug_channels))
            for k, data_k in class_data.items():
                mean_k = class_means[k]
                for t in range(data_k.shape[0]):
                    diff = data_k[t] - mean_k
                    S_w += diff @ diff.T

            # Regularize S_w
            S_w += np.eye(n_aug_channels) * 1e-6

            # Solve generalized eigenvalue problem
            lambdas, W = linalg.eig(S_b, S_w, left=True, right=False)
            # Take real parts and sort
            lambdas = np.real(lambdas)
            W = np.real(W)
            arr1inds = lambdas.argsort()[::-1]
            W = W[:, arr1inds]

            self.weights_[band_n, :, :] = W[:, : self.n_components].T

            # Build templates in augmented space
            for k, mean_k in class_means.items():
                self.templates_[k, band_n, :, :] = mean_k

        return self

    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            Test data as MNE Epochs object.

        Returns
        -------
        y_pred : list of int
            Predicted labels.
        """
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
                "This SSVEP_TDCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

        y_pred = []
        for x in X:
            corr_array = np.zeros((self.n_fbands, self.n_classes))
            for band_n in range(self.n_fbands):
                X_filter = filterbank(x, self.sfreq_, band_n, self.peaks_)
                X_aug = self._augment_data(X_filter)

                w = self.weights_[band_n, :, :]  # (n_components, n_aug_channels)

                for _freq, k in self.one_hot_.items():
                    template = self.templates_[k, band_n, :, :]
                    r = np.corrcoef((w @ X_aug).flatten(), (w @ template).flatten())
                    corr_array[band_n, k] = r[0, 1]

            rho = np.dot(self.fb_coefs, corr_array)
            tau = np.argmax(rho)
            y_pred.append(self.one_inv_[tau])
        return y_pred

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : :class:`mne.Epochs`
            Test data as MNE Epochs object.

        Returns
        -------
        y_pred : numpy.ndarray
            Probabilities per class, of shape ``(n_trials, n_classes)``.
        """
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
                "This SSVEP_TDCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )
        n_trials = len(X)
        y_pred = np.zeros((n_trials, self.n_classes))

        for trial_n, X_test in enumerate(X):
            corr_array = np.zeros((self.n_fbands, self.n_classes))
            for band_n in range(self.n_fbands):
                X_filter = filterbank(X_test, self.sfreq_, band_n, self.peaks_)
                X_aug = self._augment_data(X_filter)

                w = self.weights_[band_n, :, :]

                for _freq, k in self.one_hot_.items():
                    template = self.templates_[k, band_n, :, :]
                    r = np.corrcoef((w @ X_aug).flatten(), (w @ template).flatten())
                    corr_array[band_n, k] = r[0, 1]

            normalized_coefs = self.fb_coefs / (np.sum(self.fb_coefs))
            rho = np.dot(normalized_coefs, corr_array)
            rho /= sum(rho)
            y_pred[trial_n] = rho
        return y_pred
