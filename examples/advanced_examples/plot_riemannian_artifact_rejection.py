"""
======================================================
Riemannian Artifact Rejection as a Pre-processing Step
======================================================

Electroencephalography (EEG) signal cleaning has long been a critical
challenge in the research community. The presence of artifacts can
significantly degrade EEG data quality, complicating analysis and
potentially leading to erroneous interpretations. These artifacts are
broadly categorized as either endogenous (biological: ocular, myogenic,
cardiac) or exogenous (environmental, instrumental). As noted in [3]_,
manual annotation of artifacts is time-consuming, subjective, and
impractical for large-scale EEG studies.

Riemannian geometry provides an elegant framework for automatic artifact
detection. The key feature of interest is the **covariance matrix** derived
from EEG epochs. Each epoch of :math:`N` channels and :math:`T` time
samples yields a covariance matrix :math:`\\Sigma = \\frac{1}{T-1} X X^\\top`
that lies on the manifold of Symmetric Positive Definite (SPD) matrices
:math:`\\mathcal{M}_N`. The affine-invariant Riemannian distance between
two SPD matrices is:

.. math::

    \\delta_R(\\Sigma_1, \\Sigma_2) = \\left\\| \\log\\left(
    \\Sigma_1^{-1/2} \\Sigma_2 \\Sigma_1^{-1/2} \\right)
    \\right\\|_F = \\sqrt{\\sum_{n=1}^{N} \\log^2(\\lambda_n)}

where :math:`\\lambda_n` are the eigenvalues of
:math:`\\Sigma_1^{-1} \\Sigma_2`.

This tutorial introduces three progressively more powerful Riemannian
artifact rejection methods and demonstrates how to integrate the first
two into a MOABB pre-processing pipeline using **pipeline surgery**:

1. **Riemannian Potato (RP)** [1]_ — a single-potato detector based on
   Riemannian distance to the geometric mean (barycenter).
2. **Riemannian Potato Field (RPF)** [2]_ — multiple potatoes, each
   targeting specific artifact types via channel subsets and frequency
   bands, combined using Fisher's method.
3. **Improved RPF (iRPF)** [3]_ — discussed conceptually at the end of
   this tutorial. Enhanced RPF with adaptive robust barycenter
   estimation, additional distance metrics, multiple p-value combination
   and meta-combination functions, and automatic adaptive thresholding
   via the Kneedle algorithm.

We apply these methods to the BNCI2014-009 P300 dataset and design a
potato field following the principles described in [3]_.

References
----------
.. [1] Barachant, A., Andreev, A., & Congedo, M. (2013). The Riemannian
       Potato: an automatic and adaptive artifact detection method for
       online experiments using Riemannian geometry. In TOBI Workshop IV
       (pp. 19-20).

.. [2] Barthelemy, Q., Mayaud, L., Ojeda, D., & Congedo, M. (2019).
       The Riemannian Potato Field: a tool for online signal quality index
       of EEG. IEEE Transactions on Neural Systems and Rehabilitation
       Engineering, 27(2), 244-255.

.. [3] Hajhassani, D., Barthelemy, Q., Mattout, J., & Congedo, M. (2026).
       Improved Riemannian potato field: an Automatic Artifact Rejection
       Method for EEG. Biomedical Signal Processing and Control, 112,
       108505. https://doi.org/10.1016/j.bspc.2025.108505
"""

# Authors: Davoud Hajhassani
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

##############################################################################
# Imports and Setup
# -----------------
#
# We import the necessary libraries. The key components are:
#
# - ``Potato`` from pyriemann for artifact detection
# - ``Covariances`` for estimating covariance matrices
# - ``FunctionTransformer`` to wrap our rejection functions into pipeline steps
# - ``StepType`` to insert steps at the right position in the MOABB pipeline

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyriemann.clustering import Potato
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.covariance import normalize
from scipy.stats import chi2, norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import moabb
from moabb.datasets import BNCI2014_009
from moabb.datasets.bids_interface import StepType
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300


# Suppress warnings from pyriemann's covariance estimation (RuntimeWarning
# for near-singular matrices) and sklearn deprecation notices (FutureWarning).
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pyriemann")

moabb.set_log_level("info")

##############################################################################
# Load Dataset and Visualize Data
# --------------------------------
#
# We use the BNCI2014-009 dataset, a 16-channel P300 speller recorded at
# 256 Hz from 10 subjects [4]_. The channels are:
# FC5, FC3, FC1, FCz, FC2, FC4, FC6, C3, C1, Cz, C2, C4, CP3, CPz, CP4, Pz.
#
# We load epochs using ``return_epochs=True`` to demonstrate the
# Riemannian artifact detection concepts visually before integrating them
# into the MOABB evaluation pipeline.
#
# .. [4] Hoffmann, U., Vesin, J.-M., Ebrahimi, T., & Diserens, K. (2008).
#    An efficient P300-based brain-computer interface for disabled subjects.
#    Journal of Neuroscience Methods, 167(1), 115-125.
#    https://doi.org/10.1016/j.jneumeth.2007.03.005

dataset = BNCI2014_009()
dataset.subject_list = dataset.subject_list[:1]

paradigm = P300(resample=128, scorer={"roc_auc": "roc_auc", "f1": "f1"})

# Load epochs for visualization
epochs, labels, meta = paradigm.get_data(dataset, return_epochs=True)
print(f"Loaded {len(epochs)} epochs, {len(epochs.ch_names)} channels")
print(f"Channels: {epochs.ch_names}")

##############################################################################
# The Riemannian Potato — Concept
# --------------------------------
#
# The **Riemannian Potato** (RP) [1]_ works as follows. For each EEG
# epoch :math:`i`, we compute the covariance matrix :math:`\\Sigma_i` and
# the Riemannian distance :math:`d_i` to the barycenter
# :math:`\\bar{\\Sigma}`:
#
# .. math::
#
#     \\bar{\\Sigma} = \\arg\\min_{\\Sigma \\in \\mathcal{M}_N}
#     \\sum_{i=1}^{I} \\delta_R^2(\\Sigma_i, \\Sigma)
#
# As noted in [3]_, Riemannian distances are empirically right-skewed
# and positive-only, so the **geometric z-score** is more appropriate
# than the arithmetic one:
#
# .. math::
#
#     \\mu = \\exp\\left( \\frac{1}{I} \\sum_{i=1}^{I}
#     \\log d_i \\right), \\quad
#     \\sigma = \\exp\\left( \\sqrt{\\frac{1}{I} \\sum_{i=1}^{I}
#     \\left( \\log \\frac{d_i}{\\mu} \\right)^2} \\right)
#
# The geometric z-score is then :math:`z_i = \\log(d_i / \\mu) /
# \\log(\\sigma)`, and the p-value is :math:`p_i = 1 - \\Phi(z_i)` where
# :math:`\\Phi` is the standard normal CDF. Epochs with z-scores above a
# threshold (e.g., 3) are flagged as artifacts.
#
# **Limitation:** As the number of channels increases, the distance
# becomes an average over many sensor contributions. Artifacts affecting
# only a few channels may not produce a large enough distance to be
# detected [3]_. This motivates the Potato Field.

##############################################################################
# Visualizing the Riemannian Potato
# ----------------------------------
#
# We compute covariance matrices and fit a Riemannian Potato to illustrate
# how z-scores separate clean from artifacted epochs. This visualization
# is inspired by Figure 1 and Figure 2 of [3]_.

data = epochs.get_data()
covs = Covariances(estimator="lwf").transform(data)

# Fit potato and get z-scores
potato = Potato(metric="riemann", threshold=3)
potato.fit(covs)
z_scores = potato.transform(covs)
is_clean = potato.predict(covs).astype(bool)

print(f"RP detected {(~is_clean).sum()}/{len(covs)} artifact epochs")

# --- Figure 1: Covariance matrix heatmaps (inspired by Fig. 2 of the paper)
# Show the barycenter, two clean epochs, and two outlier epochs
fig, axes = plt.subplots(1, 5, figsize=(15, 3), facecolor="white")

# Barycenter
barycenter = potato._mdm.covmeans_[0]
im = axes[0].imshow(barycenter, cmap="RdBu_r", aspect="equal")
axes[0].set_title("Barycenter\n(geometric mean)", fontsize=9)

# Two epochs closest to barycenter (most clean)
sorted_idx = np.argsort(np.abs(z_scores))
for j, idx in enumerate(sorted_idx[:2]):
    axes[1 + j].imshow(covs[idx], cmap="RdBu_r", aspect="equal")
    axes[1 + j].set_title(f"Clean epoch\nz = {z_scores[idx]:.2f}", fontsize=9)

# Two epochs farthest from barycenter (most artifacted)
for j, idx in enumerate(sorted_idx[-2:]):
    axes[3 + j].imshow(covs[idx], cmap="RdBu_r", aspect="equal")
    axes[3 + j].set_title(f"Outlier epoch\nz = {z_scores[idx]:.2f}", fontsize=9)

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle(
    "Covariance matrices: barycenter vs clean vs outlier epochs",
    fontsize=11,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

##############################################################################
# 2D Projection of the Riemannian Potato
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This visualization is inspired by Figure 1 of [3]_. We project the
# covariance matrices onto a 2D plane defined by two selected channels
# (FCz and Cz) and display the z-score isocontours. The characteristic
# "potato" shape arises from the non-linearity of the Riemannian
# manifold. Clean epochs cluster inside the potato, while artifacts
# fall outside.

# Select two central channels for 2D visualization
ch_idx_fcz = epochs.ch_names.index("FCz")
ch_idx_cz = epochs.ch_names.index("Cz")
covs_2d = covs[:, [ch_idx_fcz, ch_idx_cz], :][:, :, [ch_idx_fcz, ch_idx_cz]]

potato_2d = Potato(metric="riemann", threshold=3)
potato_2d.fit(covs_2d)
z_2d = potato_2d.transform(covs_2d)
is_clean_2d = potato_2d.predict(covs_2d).astype(bool)
barycenter_2d = potato_2d._mdm.covmeans_[0]

# Extract the (0,0) and (1,1) diagonal entries for plotting
x_vals = covs_2d[:, 0, 0]  # variance of FCz
y_vals = covs_2d[:, 1, 1]  # variance of Cz

fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")

# Create a grid and compute z-scores for isocontours
x_grid = np.linspace(x_vals.min() * 0.8, x_vals.max() * 1.1, 80)
y_grid = np.linspace(y_vals.min() * 0.8, y_vals.max() * 1.1, 80)
xx, yy = np.meshgrid(x_grid, y_grid)
# Build 2x2 SPD matrices from diagonal entries using the barycenter's
# off-diagonal as a reference
off_diag = barycenter_2d[0, 1]
grid_covs = np.zeros((len(x_grid) * len(y_grid), 2, 2))
grid_covs[:, 0, 0] = xx.ravel()
grid_covs[:, 1, 1] = yy.ravel()
grid_covs[:, 0, 1] = off_diag
grid_covs[:, 1, 0] = off_diag

# Only compute z-scores for valid SPD matrices (positive determinant)
det = grid_covs[:, 0, 0] * grid_covs[:, 1, 1] - off_diag**2
valid = det > 0
z_grid = np.full(len(grid_covs), np.nan)
if valid.sum() > 0:
    z_grid[valid] = potato_2d.transform(grid_covs[valid])

z_map = z_grid.reshape(xx.shape)
contour = ax.contourf(xx, yy, z_map, levels=20, cmap="coolwarm", alpha=0.4)
ax.contour(
    xx,
    yy,
    z_map,
    levels=[3],
    colors=["#C44E52"],
    linewidths=2,
    linestyles="--",
)
plt.colorbar(contour, ax=ax, label="z-score")

# Plot epochs
ax.scatter(
    x_vals[is_clean_2d],
    y_vals[is_clean_2d],
    c="#4C72B0",
    s=10,
    alpha=0.4,
    label="Clean",
)
ax.scatter(
    x_vals[~is_clean_2d],
    y_vals[~is_clean_2d],
    c="#C44E52",
    s=30,
    marker="x",
    label="Artifact",
    zorder=5,
)
ax.scatter(
    barycenter_2d[0, 0],
    barycenter_2d[1, 1],
    c="black",
    s=100,
    marker="+",
    linewidths=2,
    label="Barycenter",
    zorder=5,
)

ax.set_xlabel("Variance of FCz")
ax.set_ylabel("Variance of Cz")
ax.set_title(
    "2D projection of the Riemannian Potato\n"
    "(dashed red line = z=3 isocontour, the 'potato' boundary)"
)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

##############################################################################
# The z-score distribution shows how the potato separates clean data
# from outliers. Clean epochs cluster near z=0 while artifact epochs
# have high z-scores, beyond the threshold.

fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="white")

# Z-score histogram
axes[0].hist(z_scores[is_clean], bins=30, alpha=0.7, label="Clean", color="#4C72B0")
axes[0].hist(z_scores[~is_clean], bins=10, alpha=0.7, label="Artifact", color="#C44E52")
axes[0].axvline(3, color="k", linestyle="--", linewidth=1.5, label="Threshold (z=3)")
axes[0].set_xlabel("Geometric z-score")
axes[0].set_ylabel("Count")
axes[0].set_title("RP: Z-score distribution")
axes[0].legend()

# Sorted distances (inspired by Fig. 6 of the paper)
p_values = 1 - norm.cdf(z_scores)
sorted_p = np.sort(p_values)
axes[1].plot(sorted_p, ".-", markersize=2, color="#4C72B0")
axes[1].set_xlabel("Epoch index (sorted)")
axes[1].set_ylabel("p-value")
axes[1].set_title("RP: Sorted p-values (Signal Quality Index)")
axes[1].set_yscale("log")
axes[1].axhline(
    1 - norm.cdf(3),
    color="k",
    linestyle="--",
    linewidth=1.5,
    label=f"Threshold (p={1 - norm.cdf(3):.4f})",
)
axes[1].legend()

plt.tight_layout()
plt.show()

##############################################################################
# The Riemannian Potato Field — Concept
# ---------------------------------------
#
# The **Riemannian Potato Field** (RPF) [2]_ addresses the limitation of
# the single potato by using **multiple low-dimensional potatoes** in
# parallel, each tailored to detect specific types of artifacts that
# influence particular spatial regions within certain frequency bands.
#
# As described in [3]_, the output z-scores from all :math:`J` potatoes
# are converted to p-values and merged into a single **Signal Quality
# Index** (SQI) using **Fisher's combination function** [5]_:
#
# .. math::
#
#     q = -2 \\sum_{j=1}^{J} \\log(p_j)
#
# Under the null hypothesis (no artifact), :math:`q` follows a
# :math:`\\chi^2` distribution with :math:`2J` degrees of freedom, yielding
# a combined p-value: :math:`p = 1 - F_{\\chi^2(2J)}(q)`.
#
# The iRPF method [3]_ further introduces **Liptak's combination**:
#
# .. math::
#
#     q = \\frac{1}{\\sqrt{J}} \\sum_{j=1}^{J} \\Phi^{-1}(p_j)
#
# and a **Tippett meta-combination** of both Fisher and Liptak results:
# :math:`q = \\min(p_{\\text{Fisher}}, p_{\\text{Liptak}})`, providing
# a more precise determination of the rejection region.
#
# .. [5] Fisher, R. A. (1934). Statistical methods for research workers.
#    Oliver and Boyd, Edinburgh.
#
# Distance metrics for different artifact types
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A key insight from [3]_ is that different **distance metrics** are
# suited for detecting different artifact types:
#
# - **Riemannian distance** (affine-invariant): captures the full
#   structure of covariance matrices. Effective for artifacts with
#   co-variation across electrodes (e.g., blinks).
# - **Euclidean distance**: :math:`\\delta_E(\\Sigma_1, \\Sigma_2) =
#   \\|\\Sigma_1 - \\Sigma_2\\|_F`. More effective for artifacts like
#   vertical eye movements that produce large co-variation patterns.
# - **Diagonal Euclidean distance**: :math:`\\delta_{\\text{diag}(E)} =
#   \\|\\text{diag}(\\Sigma_1 - \\Sigma_2)\\|_F`. Focuses on the diagonal
#   elements (channel variances), effective for myogenic artifacts that
#   primarily impact individual channels without cross-electrode
#   co-variation.

##############################################################################
# Potato Field Design
# ---------------------
#
# As described in [3]_ (Section 2.5), the potato field must be customized
# per EEG headset. Each potato targets a specific artifact type by
# selecting appropriate channels, frequency band, and distance metric.
# The general design principles from [3]_ are:
#
# - **Ocular artifacts**: frontal channels (Fp, AF, F), low frequency
#   (0.1-7 Hz), Riemannian or Euclidean distance.
# - **Myogenic artifacts (EMG)**: peripheral channels (F7/F8, T7/T8,
#   P7/P8, O1/O2), high frequency (>20 Hz), diagonal Euclidean
#   distance — since EMG primarily impacts individual channel variances.
# - **General artifacts**: broader channel sets, wide frequency bands,
#   Riemannian distance.
#
# For the BNCI2014-009 dataset (16 channels, no EOG, no frontal-polar
# channels, bandpass 1-24 Hz applied by the P300 paradigm), we adapt
# the design principles from [3]_ (Tables 2-5):
#
# .. list-table:: Potato Field Configuration for BNCI2014-009
#    :header-rows: 1
#    :widths: 8 28 15 15 34
#
#    * - #
#      - Channels
#      - Metric
#      - Freq (Hz)
#      - Target Artifact
#    * - 1
#      - FC3, FCz, FC4
#      - euclid
#      - 1-7
#      - Ocular (frontal, low-freq)
#    * - 2
#      - FC3, FCz, FC4
#      - riemann
#      - 1-7
#      - Ocular (blinks, co-variation)
#    * - 3
#      - FC5, FC1, FC2, FC6
#      - riemann
#      - 16-24
#      - Myogenic lateral
#    * - 4
#      - C3, Cz, C4
#      - riemann
#      - 16-24
#      - Myogenic central
#    * - 5
#      - CP3, CPz, CP4, Pz
#      - riemann
#      - 1-24
#      - General parietal
#    * - 6
#      - All 16 channels
#      - riemann
#      - 1-24
#      - General (full headset)

# Use None for channels to indicate "all available channels". This is
# resolved dynamically in compute_potato_covariances, making the config
# reusable across datasets with different channel sets.
POTATO_FIELD_CONFIG = [
    {
        "channels": ["FC3", "FCz", "FC4"],
        "low_freq": 1.0,
        "high_freq": 7.0,
        "metric": "euclid",
        "normalization": None,
        "target": "Ocular (low-freq)",
    },
    {
        "channels": ["FC3", "FCz", "FC4"],
        "low_freq": 1.0,
        "high_freq": 7.0,
        "metric": "riemann",
        "normalization": None,
        "target": "Ocular (blinks)",
    },
    {
        "channels": ["FC5", "FC1", "FC2", "FC6"],
        "low_freq": 16.0,
        "high_freq": 24.0,
        "metric": "riemann",
        "normalization": "trace",
        "target": "Myogenic lateral",
    },
    {
        "channels": ["C3", "Cz", "C4"],
        "low_freq": 16.0,
        "high_freq": 24.0,
        "metric": "riemann",
        "normalization": "trace",
        "target": "Myogenic central",
    },
    {
        "channels": ["CP3", "CPz", "CP4", "Pz"],
        "low_freq": 1.0,
        "high_freq": 24.0,
        "metric": "riemann",
        "normalization": None,
        "target": "General parietal",
    },
    {
        "channels": None,
        "low_freq": 1.0,
        "high_freq": 24.0,
        "metric": "riemann",
        "normalization": None,
        "target": "General (all channels)",
    },
]

##############################################################################
# Potato Field — Visual Demonstration
# -------------------------------------
#
# We compute per-potato covariance matrices and fit individual potatoes
# to visualize how each potato targets different artifacts. This follows
# the approach described in [3]_ (Section 2.5) where each potato
# operates on a specific channel subset and frequency band.


def compute_potato_covariances(epochs, config):
    """Compute covariance matrices for each potato in the field.

    For each potato configuration, this function:
    1. Resolves channel selection (``None`` means all channels)
    2. Copies and picks the relevant channel subset
    3. Applies bandpass filtering for the target frequency band.
       The paradigm already applies a broadband filter (e.g., 1-24 Hz
       for P300); per-potato filtering further narrows within that band.
    4. Computes covariance matrices using Ledoit-Wolf shrinkage
    5. Optionally normalizes the covariances

    Parameters
    ----------
    epochs : mne.Epochs
        The input epochs.
    config : list of dict
        Potato field configuration. Each dict must contain:
        ``channels`` (list of str or None for all channels),
        ``low_freq``, ``high_freq``, ``metric``, ``normalization``,
        and ``target``.

    Returns
    -------
    cov_list : list of ndarray
        List of covariance matrices arrays, one per potato.
    """
    cov_list = []
    for potato_cfg in config:
        channels = potato_cfg["channels"]
        if channels is None:
            channels = epochs.ch_names
        ep = epochs.copy().pick(channels)
        # IIR filtering is used here for speed; the paradigm's broadband
        # filter has already been applied, so this narrows the band further.
        ep.filter(
            l_freq=potato_cfg["low_freq"],
            h_freq=potato_cfg["high_freq"],
            method="iir",
            verbose=False,
        )
        covs = Covariances(estimator="lwf").transform(ep.get_data())
        if potato_cfg.get("normalization") is not None:
            covs = normalize(covs, potato_cfg["normalization"])
        cov_list.append(covs)
    return cov_list


# Compute covariance matrices for each potato
cov_list = compute_potato_covariances(epochs, POTATO_FIELD_CONFIG)

# Fit individual potatoes and collect z-scores
potato_z_scores = []
potato_p_values = []
for i, (cov, cfg) in enumerate(zip(cov_list, POTATO_FIELD_CONFIG)):
    p = Potato(metric=cfg["metric"], threshold=3)
    p.fit(cov)
    z = p.transform(cov)
    pv = 1 - norm.cdf(z)
    potato_z_scores.append(z)
    potato_p_values.append(pv)
    n_outliers = (z > 3).sum()
    ch_list = cfg["channels"] if cfg["channels"] is not None else epochs.ch_names
    ch_display = f"ch={ch_list[:3]}{'...' if len(ch_list) > 3 else ''}"
    print(
        f"Potato {i + 1} ({cfg['target']}): "
        f"{n_outliers}/{len(z)} outliers, "
        f"{ch_display}, "
        f"freq={cfg['low_freq']}-{cfg['high_freq']} Hz"
    )

##############################################################################
# Per-potato z-score distributions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Each potato produces its own z-score distribution. Potatoes targeting
# different artifact types will flag different epochs, demonstrating why
# a field of potatoes provides better coverage than a single potato.

fig, axes = plt.subplots(2, 3, figsize=(14, 7), facecolor="white")
axes = axes.ravel()

for i, (z, cfg) in enumerate(zip(potato_z_scores, POTATO_FIELD_CONFIG)):
    ax = axes[i]
    ax.hist(z, bins=40, alpha=0.7, color="#4C72B0", edgecolor="white")
    ax.axvline(3, color="#C44E52", linestyle="--", linewidth=1.5, label="z=3")
    n_out = (z > 3).sum()
    ax.set_title(
        f"Potato {i + 1}: {cfg['target']}\n"
        f"({cfg['metric']}, {cfg['low_freq']:.0f}-{cfg['high_freq']:.0f} Hz, "
        f"{n_out} outliers)",
        fontsize=9,
    )
    ax.set_xlabel("z-score")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

fig.suptitle(
    "Per-potato z-score distributions across the Potato Field",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

##############################################################################
# Fisher's Combination — Combining p-values from all potatoes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Following [3]_, we combine the per-potato p-values into a single SQI
# using Fisher's combination method. The sorted combined p-values reveal
# the separation between clean and artifacted epochs. This plot is
# inspired by Figure 6 of [3]_, which shows how the Kneedle algorithm
# can automatically detect the optimal rejection threshold.

# Combine p-values using Fisher's method (as in pyriemann's PotatoField)
p_matrix = np.array(potato_p_values)  # shape: (n_potatoes, n_epochs)
p_matrix[p_matrix < 1e-10] = 1e-10
q = -2 * np.sum(np.log(p_matrix), axis=0)

n_potatoes = len(POTATO_FIELD_CONFIG)
combined_p = 1 - chi2.cdf(q, df=2 * n_potatoes)

sorted_p = np.sort(combined_p)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="white")

# Sorted SQI values (inspired by Fig. 6 of the paper)
axes[0].plot(sorted_p, ".-", markersize=2, color="#4C72B0")
axes[0].set_xlabel("Epoch index (sorted by SQI)")
axes[0].set_ylabel("Combined p-value (SQI)")
axes[0].set_title("RPF: Sorted Signal Quality Index\n(Fisher's combination)")
axes[0].set_yscale("log")
axes[0].axhline(0.01, color="#C44E52", linestyle="--", label="p = 0.01 threshold")
n_rejected = (combined_p < 0.01).sum()
axes[0].legend()
axes[0].annotate(
    f"{n_rejected} epochs\nbelow threshold",
    xy=(n_rejected // 2, 0.01),
    xytext=(n_rejected + 50, 0.001),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="gray"),
)

# Per-potato contribution heatmap
# Show which potatoes flag which epochs (sorted by combined SQI)
sorted_idx = np.argsort(combined_p)
p_sorted = p_matrix[:, sorted_idx]
im = axes[1].imshow(
    np.log10(p_sorted),
    aspect="auto",
    cmap="RdYlBu",
    interpolation="nearest",
)
axes[1].set_xlabel("Epoch index (sorted by SQI)")
axes[1].set_ylabel("Potato index")
axes[1].set_yticks(range(n_potatoes))
axes[1].set_yticklabels(
    [f"P{i + 1}: {cfg['target']}" for i, cfg in enumerate(POTATO_FIELD_CONFIG)],
    fontsize=8,
)
axes[1].set_title("Per-potato log10(p-values)\n(sorted by combined SQI)")
plt.colorbar(im, ax=axes[1], label="log10(p-value)")

plt.tight_layout()
plt.show()

##############################################################################
# Pipeline Surgery — Integrating Artifact Rejection into MOABB
# ---------------------------------------------------------------
#
# Now we integrate the RP and RPF methods into MOABB's pre-processing
# pipeline using **pipeline surgery**, as demonstrated in the
# ``plot_pre_processing_steps`` example. We create ``FunctionTransformer``
# wrappers and insert them as ``StepType.EPOCHS`` steps.


def riemannian_potato_rejection(epochs):
    """Reject artifacts using a single Riemannian Potato.

    Computes covariance matrices for all channels, fits a Potato,
    and drops epochs whose geometric z-score exceeds the threshold.

    Note: The Potato is fit and evaluated on the same data. In a rigorous
    benchmark you would fit only on training folds to avoid information
    leakage. Here the Potato is an unsupervised outlier detector, so the
    leakage is minimal and acceptable for tutorial purposes.
    """
    data = epochs.get_data()
    n_before = len(data)

    covs = Covariances(estimator="lwf").transform(data)
    potato = Potato(metric="riemann", threshold=3)
    potato.fit(covs)
    is_clean = potato.predict(covs).astype(bool)

    n_rejected = n_before - is_clean.sum()
    print(f"  RP: rejected {n_rejected}/{n_before} epochs")

    if is_clean.sum() == 0:
        raise ValueError(
            f"All {n_before} epochs rejected by Riemannian Potato — "
            "consider raising the threshold or checking data quality."
        )
    return epochs[is_clean]


def riemannian_potato_field_rejection(epochs):
    """Reject artifacts using a Riemannian Potato Field.

    For each potato configuration, computes covariance matrices on the
    relevant channel subset and frequency band. Each potato uses its own
    metric (e.g., Riemannian or Euclidean) as specified in the config.
    Per-potato p-values are combined into a single SQI using Fisher's
    method, and epochs below the significance threshold are rejected.

    Note: Each potato is fit and evaluated on the same data. In a rigorous
    benchmark you would fit only on training folds to avoid information
    leakage. Here the potatoes are unsupervised outlier detectors, so the
    leakage is minimal and acceptable for tutorial purposes.
    """
    n_before = len(epochs)
    available_chs = set(epochs.ch_names)

    # Filter config to only include potatoes whose channels are available.
    # Potatoes with channels=None use all available channels and are always valid.
    valid_config = [
        cfg
        for cfg in POTATO_FIELD_CONFIG
        if cfg["channels"] is None or all(ch in available_chs for ch in cfg["channels"])
    ]
    if not valid_config:
        return epochs

    cov_list = compute_potato_covariances(epochs, valid_config)

    # Fit individual potatoes with per-potato metrics and collect p-values
    p_values_list = []
    for cov, cfg in zip(cov_list, valid_config):
        p = Potato(metric=cfg["metric"], threshold=3)
        p.fit(cov)
        z = p.transform(cov)
        p_values_list.append(1 - norm.cdf(z))

    # Combine p-values using Fisher's method
    p_matrix = np.array(p_values_list)
    p_matrix = np.clip(p_matrix, 1e-10, 1.0)
    q = -2 * np.sum(np.log(p_matrix), axis=0)
    combined_p = 1 - chi2.cdf(q, df=2 * len(valid_config))

    is_clean = combined_p >= 0.01

    n_rejected = n_before - is_clean.sum()
    print(f"  RPF: rejected {n_rejected}/{n_before} epochs")

    if is_clean.sum() == 0:
        raise ValueError(
            f"All {n_before} epochs rejected by Riemannian Potato Field — "
            "consider raising the threshold or checking data quality."
        )
    return epochs[is_clean]


# Build pipelines with artifact rejection inserted
rp_pipeline = paradigm.make_process_pipelines(dataset)[0]
rp_pipeline.insert_step(
    StepType.EPOCHS,
    FunctionTransformer(riemannian_potato_rejection),
    after=StepType.EPOCHS,
)

rpf_pipeline = paradigm.make_process_pipelines(dataset)[0]
rpf_pipeline.insert_step(
    StepType.EPOCHS,
    FunctionTransformer(riemannian_potato_field_rejection),
    after=StepType.EPOCHS,
)

##############################################################################
# Evaluation — Compare With vs Without Artifact Rejection
# ---------------------------------------------------------
#
# We define a P300 classification pipeline (Xdawn covariances +
# Tangent Space + LDA) and run ``WithinSessionEvaluation`` three times:
# once with the default pipeline, once with RP, and once with RPF.
# This comparison follows the evaluation methodology used in [3]_,
# where different artifact rejection methods are compared on their
# impact on downstream classification performance.

pipelines = {}
pipelines["RG+LDA"] = make_pipeline(
    XdawnCovariances(
        nfilter=2,
        classes=[1],
        estimator="lwf",
        xdawn_estimator="scm",
    ),
    TangentSpace(),
    LDA(solver="lsqr", shrinkage="auto"),
)

evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=[dataset],
    suffix="rar_tutorial",
    overwrite=True,
)

# We call evaluation.evaluate() separately for each preprocessing variant
# because evaluation.process() only supports a single process_pipeline.
# This lets us compare different preprocessing strategies on the same data.

# 1. Default pipeline (no artifact rejection)
default_pipeline = paradigm.make_process_pipelines(dataset)[0]
results_default = list(
    evaluation.evaluate(
        dataset=dataset,
        pipelines=pipelines,
        param_grid=None,
        process_pipeline=default_pipeline,
    )
)
for r in results_default:
    r["preprocessing"] = "No rejection"

# 2. Riemannian Potato
results_rp = list(
    evaluation.evaluate(
        dataset=dataset,
        pipelines=pipelines,
        param_grid=None,
        process_pipeline=rp_pipeline,
    )
)
for r in results_rp:
    r["preprocessing"] = "RP"

# 3. Riemannian Potato Field
results_rpf = list(
    evaluation.evaluate(
        dataset=dataset,
        pipelines=pipelines,
        param_grid=None,
        process_pipeline=rpf_pipeline,
    )
)
for r in results_rpf:
    r["preprocessing"] = "RPF"

##############################################################################
# Results Visualization
# ----------------------
#
# We compare both ROC AUC and F1-score across the three preprocessing
# approaches. This comparison is inspired by Figure 7 of [3]_, which
# compares evaluation metrics (Recall, Specificity, Precision, F1-Score)
# across RP, RPF, iRPF, Isolation Forest, and Autoreject. Here we focus
# on the effect of artifact rejection on downstream P300 classification.
#
# Note that the BNCI2014-009 dataset was recorded in a controlled
# laboratory setting, so it is relatively clean. The improvement from
# artifact rejection is expected to be modest. For noisier recordings
# (e.g., ambulatory EEG, real-world BCI), the benefits of artifact
# rejection are typically much larger, as demonstrated in [3]_ where
# iRPF achieved gains of up to 24% in F1-score on artifact-heavy
# datasets.

all_results = results_default + results_rp + results_rpf
df = pd.DataFrame(all_results)

order = ["No rejection", "RP", "RPF"]
colors = ["#4C72B0", "#DD8452", "#55A868"]
metrics = {"score_roc_auc": "ROC AUC", "score_f1": "F1-Score"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

for ax, (metric_col, metric_name) in zip(axes, metrics.items()):
    grouped = df.groupby("preprocessing")[metric_col]
    means = grouped.mean().reindex(order)
    stds = grouped.std().reindex(order)

    bars = ax.bar(order, means, yerr=stds, capsize=5, color=colors, edgecolor="white")
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Preprocessing Method")
    ax.set_title(f"P300 Classification: {metric_name}")

    if metric_col == "score_roc_auc":
        ax.set_ylim(0.5, 1.0)
    else:
        ax.set_ylim(0, 1.0)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

fig.suptitle(
    "BNCI2014-009: Effect of Riemannian Artifact Rejection",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

##############################################################################
# Per-session breakdown
# ~~~~~~~~~~~~~~~~~~~~~~

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

sessions = sorted(df["session"].unique())
x = np.arange(len(sessions))
width = 0.25

for ax, (metric_col, metric_name) in zip(axes, metrics.items()):
    for i, (method, color) in enumerate(zip(order, colors)):
        method_df = df.loc[df["preprocessing"] == method]
        # Use groupby to handle potential duplicate or missing sessions
        session_scores = method_df.groupby("session")[metric_col].mean()
        scores = [session_scores.get(s, np.nan) for s in sessions]
        ax.bar(
            x + i * width,
            scores,
            width,
            label=method,
            color=color,
            edgecolor="white",
        )

    ax.set_xlabel("Session")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Per-session {metric_name}")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Session {s}" for s in sessions])
    ax.legend()

    if metric_col == "score_roc_auc":
        ax.set_ylim(0.5, 1.0)
    else:
        ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.show()

##############################################################################
# Notes on the Improved Riemannian Potato Field (iRPF)
# -------------------------------------------------------
#
# The methods shown above (RP and RPF) are the building blocks of the
# improved Riemannian Potato Field (iRPF) introduced in [3]_. The key
# innovations of iRPF over RPF are:
#
# **Automatic outlier rejection before training.** iRPF computes the
# field root mean square (FRMS) of each epoch and adaptively determines
# a rejection threshold, removing severely corrupted data before the
# barycenter estimation step.
#
# **Adaptive robust barycenter estimation.** Instead of using a fixed
# z-score threshold for iterative outlier removal during barycenter
# computation, iRPF applies the **Kneedle algorithm** (Satopaa et al.,
# 2011) to the sorted p-values. This automatically finds the "knee"
# point separating inliers from outliers at each iteration.
#
# **Additional distance metrics.** Beyond the affine-invariant
# Riemannian distance used in RP and RPF, iRPF introduces:
#
# - **Euclidean distance**: more effective for artifacts with
#   co-variation across electrodes (e.g., vertical eye movements)
# - **Diagonal Euclidean distance**: focuses on channel variances,
#   effective for myogenic artifacts that primarily impact individual
#   channels without cross-electrode co-variation
#
# **Multiple combination and meta-combination functions.** iRPF combines
# per-potato p-values using both Fisher's and Liptak's combination
# functions, then applies a Tippett meta-combination (minimum p-value)
# across both results. This provides a more precise rejection region.
#
# **Automatic adaptive thresholding.** The Kneedle algorithm is applied
# to the sorted SQI values to dynamically determine the rejection
# threshold, eliminating the need for manual threshold tuning.
#
# As reported in [3]_, iRPF achieves gains of up to 22% in recall,
# 102% in specificity, 54% in precision, and 24% in F1-score compared
# to Isolation Forest, Autoreject, RP, and RPF, while performing
# artifact cleaning in under 8 ms per epoch.
#
# The iRPF implementation is available in the
# `RAR package <https://github.com/Davoud-Hajhassani/Riemannian-Artifact-Rejection>`_.
# The RP and RPF methods demonstrated here are fully available in Python
# via `pyriemann <https://pyriemann.readthedocs.io/>`_ and can be
# readily integrated into any MOABB analysis pipeline using the pipeline
# surgery approach shown in this tutorial.
