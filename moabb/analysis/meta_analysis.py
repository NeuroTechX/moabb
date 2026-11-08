"""Meta-analysis functions for MOABB."""

import itertools
import logging

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.utils import check_random_state


log = logging.getLogger(__name__)


def _validate_finite_scores(data, *, name="score data"):
    """Return numeric score data after rejecting NaN and infinities."""
    values = np.asarray(data, dtype=float)
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


# Stouffer's method, used by combine_pvalues, turns a p-value of exactly 0 or 1
# into an infinite z-score. The permutation branch already keeps its p-values
# strictly inside (0, 1); these are the tightest bounds that do the same for the
# Wilcoxon branch without moving any p-value that is already valid.
_P_FLOOR = np.nextafter(0.0, 1.0)
_P_CEIL = np.nextafter(1.0, 0.0)


def collapse_session_scores(df):
    """Prepare results dataframe for computing statistics.

    Parameters
    ----------
    df: :class:`pandas.DataFrame`
        results from evaluation

    Returns
    -------
    df: :class:`pandas.DataFrame`
        Aggregated results, samples are index, columns are pipelines,
        and values are scores
    """
    return (
        df.groupby(["pipeline", "dataset", "subject"], sort=False, observed=True)
        .mean(numeric_only=True)
        .reset_index()
    )


def compute_lowest_subject_scores(df, reference_pipeline, percentile=20):
    """Score pipelines on subjects selected from a reference pipeline.

    For every dataset, rank subjects by the score of ``reference_pipeline``
    and keep only the ``percentile`` percent that score lowest. Every pipeline
    is then averaged over that same cohort.

    The supplied session scores are first macro-averaged per subject, so every
    subject weighs the same regardless of how many sessions it contributes.
    This helper does not recompute subject-level F1 scores from predictions.

    Parameters
    ----------
    df: :class:`pandas.DataFrame`
        results obtained by an evaluation, with at least the ``dataset``,
        ``pipeline``, ``subject`` and ``score`` columns
    reference_pipeline: str
        Pipeline whose per-subject scores define the lowest-performing cohort
        separately for each dataset. Every pipeline in a dataset must contain
        exactly the same subjects as this reference.
    percentile: float, default=20
        percentage of subjects to keep, in ``(0, 100]``. The number of
        retained subjects is rounded up, and is at least one, so a value
        small enough always falls back to the single worst subject.

    Returns
    -------
    scores: :class:`pandas.DataFrame`
        One row per (dataset, pipeline) pair, with the mean ``score`` over
        the retained subjects and the ``n_subjects`` that were retained.

    References
    ----------
    .. [1] Gnassounou, T., Collas, A., Flamary, R., and Gramfort, A. (2025).
           PSDNorm: Test-Time Temporal Normalization for Deep Learning in
           Sleep Staging.
           https://arxiv.org/abs/2503.04582
    """
    if not 0 < percentile <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")

    score_values = _validate_finite_scores(df["score"], name="raw scores")
    for column in ("dataset", "pipeline", "subject"):
        if df[column].isna().any():
            raise ValueError(f"{column} must not contain missing values")

    validated_df = df.copy()
    validated_df["score"] = score_values
    subject_scores = collapse_session_scores(validated_df)
    rows = []
    for dataset, dataset_scores in subject_scores.groupby(
        "dataset", sort=False, observed=True
    ):
        reference_scores = dataset_scores[
            dataset_scores["pipeline"] == reference_pipeline
        ]
        if reference_scores.empty:
            raise ValueError(
                f"reference pipeline {reference_pipeline!r} is missing from "
                f"dataset {dataset!r}"
            )

        reference_subjects = set(reference_scores["subject"])
        n_keep = max(1, int(np.ceil(len(reference_subjects) * percentile / 100)))
        selected_subjects = set(
            reference_scores.sort_values("score", kind="stable")
            .head(n_keep)["subject"]
            .tolist()
        )

        for pipeline, pipeline_scores in dataset_scores.groupby(
            "pipeline", sort=False, observed=True
        ):
            if set(pipeline_scores["subject"]) != reference_subjects:
                raise ValueError(
                    f"pipeline {pipeline!r} in dataset {dataset!r} must contain "
                    f"the same subjects as reference pipeline "
                    f"{reference_pipeline!r}"
                )
            selected_scores = pipeline_scores[
                pipeline_scores["subject"].map(selected_subjects.__contains__)
            ]
            rows.append(
                {
                    "dataset": dataset,
                    "pipeline": pipeline,
                    "score": selected_scores["score"].mean(),
                    "n_subjects": n_keep,
                }
            )
    return pd.DataFrame(rows, columns=["dataset", "pipeline", "score", "n_subjects"])


def compute_pvals_wilcoxon(df, order=None):
    """Compute Wilcoxon rank-sum test on aggregated results.

    Returns a square matrix of p-values computed via the Wilcoxon rank-sum test,
    order defines the order of rows and columns.

    Entry ``[i, j]`` is the one-sided p-value for the hypothesis that pipeline
    ``order[i]`` scores higher than pipeline ``order[j]`` (SciPy's
    ``alternative="greater"``); the opposite direction is tested at ``[j, i]``.
    The p-values are not corrected for multiple comparisons: with ``k``
    pipelines this function performs ``k * (k - 1)`` tests, so apply a
    correction (for example Bonferroni or Holm) before interpreting them.

    Parameters
    ----------
    df: :class:`pandas.DataFrame`
        Aggregated results, samples are index, columns are pipelines,
        and values are scores
    order: list of length (n_pipelines)
        Names corresponding to df columns

    Returns
    -------
    pvals: ndarray of shape (n_pipelines, n_pipelines)
        pvalues
    """
    _validate_finite_scores(df)
    if order is None:
        order = df.columns
    else:
        if set(order) != set(df.columns):  # was assert, now raises properly
            raise ValueError("provided order does not have all columns of dataframe")

    order_list = list(order)
    vals = df[order_list].to_numpy(dtype=float)
    n_pipes = len(order_list)
    out = np.zeros((n_pipes, n_pipes))
    diffs = vals[:, :, None] - vals[:, None, :]
    eye = np.eye(n_pipes, dtype=bool)
    zero_diff = (diffs == 0).all(axis=0) & ~eye
    out[zero_diff] = 0.5
    active = ~eye & ~zero_diff
    if active.any():
        d_act = diffs[:, active]
        has_zero = (d_act == 0).any(axis=0)
        if d_act.shape[0] <= 50 and d_act.shape[0] > 1:
            abs_sorted = np.sort(np.abs(d_act), axis=0)
            has_ties = (abs_sorted[1:] == abs_sorted[:-1]).any(axis=0)
        else:
            has_ties = np.zeros(d_act.shape[1], dtype=bool)
        batchable = ~has_zero & ~has_ties
        p_act = np.empty(d_act.shape[1])
        if batchable.any():
            p_act[batchable] = stats.wilcoxon(
                d_act[:, batchable], alternative="greater", axis=0
            ).pvalue
        if (~batchable).any():
            for idx in np.flatnonzero(~batchable):
                p_act[idx] = stats.wilcoxon(d_act[:, idx], alternative="greater")[1]
        out[active] = np.clip(p_act, _P_FLOOR, _P_CEIL)
    return out


def _pairedttest_exact(data):
    """Exact paired t-test.

    Returns p-values for exact t-test that runs through all possible
    permutations of the first dimension. Very bad idea for size greater than 12

    Parameters
    ----------
    data: ndarray of shape (n_subj, n_pipelines, n_pipelines)
        Differences between scores for each pair of pipelines per subject

    Returns
    -------
    pvals: ndarray of shape (n_pipelines, n_pipelines)
        pvalues
    """
    n_subj = data.shape[0]
    true = data.sum(axis=0)
    nperms = 2**n_subj
    perms = np.array(
        list(itertools.product([-1, 1], repeat=n_subj)), dtype=data.dtype
    )
    randperms = (perms @ data.reshape(n_subj, -1)).reshape(nperms, *data.shape[1:])
    out = (randperms >= true).sum(axis=0, dtype=np.int32)

    # Correct for p-values equal to 1
    # as they are invalid p-values for Stouffer's method.
    # Note: as this is an exact test,
    # one of the t-test is computed with the original statistic
    # So in practice out cannot contain zeros.

    out[out >= nperms] = nperms - 1

    return out / nperms


def _pairedttest_random(data, nperms, seed=None):
    """Randomized paired t-test.

    Returns p-values based on nperms permutations of a paired t-test.

    Parameters
    ----------
    data: ndarray of shape (n_subj, n_pipelines, n_pipelines)
        Differences between scores for each pair of pipelines per subject

    Returns
    -------
    pvals: ndarray of shape (n_pipelines, n_pipelines)
        pvalues
    """
    rng = check_random_state(seed)
    n_subj = data.shape[0]
    true = data.sum(axis=0)
    perms = np.array(
        [rng.choice([1, -1], size=(n_subj,), replace=True) for _ in range(nperms)],
        dtype=data.dtype,
    )
    randperms = (perms @ data.reshape(n_subj, -1)).reshape(nperms, *data.shape[1:])
    out = 1 + (randperms >= true).sum(axis=0, dtype=np.int32)

    # Correct p-values >= 1
    # as they are invalid p-values for Stouffer's method.
    # Note: as out is initialized with ones,
    # it cannot contain zeros.

    out[out >= nperms] = nperms - 1
    return out / nperms


def compute_pvals_perm(df, order=None, seed=None):
    """Compute permutation test on aggregated results.

    Returns a square matrix of p-values computed via permutation test,
    order defines the order of rows and columns

    Parameters
    ----------
    df: :class:`pandas.DataFrame`
        Aggregated results, samples are index, columns are pipelines,
        and values are scores
    order: list of length (n_pipelines)
        Names corresponding to df columns
    seed: int | None
        random seed for reproducibility

    Returns
    -------
    pvals: ndarray of shape (n_pipelines, n_pipelines)
        pvalues
    """
    _validate_finite_scores(df)
    if order is None:
        order = df.columns
    else:
        if set(order) != set(df.columns):  # was assert, now raises properly
            raise ValueError("provided order does not have all columns of dataframe")
    # reshape df into matrix (sub, k, k) of differences
    vals = df[list(order)].to_numpy(dtype=float)
    data = vals[:, :, None] - vals[:, None, :]
    if data.shape[0] > 13:
        p = _pairedttest_random(data, 10000, seed=seed)
    else:
        p = _pairedttest_exact(data)
    return p


def _corrected_resampled_ttest(data, n_train, n_test):
    r"""Nadeau & Bengio corrected resampled t-test.

    Computes one-tailed p-values for the paired differences in ``data``
    using the variance correction of Nadeau & Bengio, which accounts for
    the dependence between resampled train/test splits (e.g. the folds of
    a (repeated) k-fold cross-validation). The naive resampled t-test
    treats the per-fold scores as independent, which underestimates the
    variance and inflates type-I errors; the corrected statistic is

    .. math::

        t = \frac{\frac{1}{n} \sum_{j=1}^{n} x_j}
                 {\sqrt{(\frac{1}{n} + \frac{n_2}{n_1})\,\hat\sigma^2}}

    where :math:`x_j` is the score difference on resample :math:`j`,
    :math:`n` is the number of resamples, :math:`n_1` and :math:`n_2` are
    the number of training and testing examples of each split, and
    :math:`\hat\sigma^2` is the sample variance of the :math:`x_j`.
    Under the null hypothesis, :math:`t` follows a Student distribution
    with :math:`n - 1` degrees of freedom.

    Parameters
    ----------
    data: ndarray of shape (n_resamples, n_pipelines, n_pipelines)
        Differences between scores for each pair of pipelines per
        cross-validation resample.
    n_train: int
        Number of training examples in each resample.
    n_test: int
        Number of testing examples in each resample.

    Returns
    -------
    pvals: ndarray of shape (n_pipelines, n_pipelines)
        One-tailed p-values; ``pvals[i, j]`` is small when pipeline ``i``
        scores significantly higher than pipeline ``j``. The diagonal
        is 0, following :func:`compute_pvals_wilcoxon`.

    References
    ----------
    .. [1] Nadeau, C., & Bengio, Y. (1999). Inference for the
       generalization error. Advances in Neural Information Processing
       Systems 12; extended version in Machine Learning, 52, 239-281
       (2003). https://doi.org/10.1023/A:1024068626366
    .. [2] Bouckaert, R. R., & Frank, E. (2004). Evaluating the
       replicability of significance tests for comparing learning
       algorithms. PAKDD 2004.
       https://doi.org/10.1007/978-3-540-24775-3_3
    """
    _validate_finite_scores(data, name="corrected t-test score differences")
    n_train, n_test = _validate_finite_scores(
        [n_train, n_test], name="n_train and n_test"
    )
    n = data.shape[0]
    if n < 2:
        raise ValueError(
            f"The corrected resampled t-test needs at least 2 resamples, got {n}"
        )
    if n_train <= 0 or n_test <= 0:
        raise ValueError(
            f"n_train and n_test must be positive, got n_train={n_train}, n_test={n_test}"
        )
    mean = data.mean(axis=0)
    var = data.var(axis=0, ddof=1)
    denom = np.sqrt((1.0 / n + n_test / n_train) * var)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mean / denom
    # 0/0 (all differences identical and zero-mean) -> no evidence, t = 0
    t[np.isnan(t)] = 0.0
    pvals = stats.t.sf(t, df=n - 1)
    # Keep p-values strictly inside (0, 1): 0 and 1 are invalid inputs
    # for Stouffer's method used in combine_pvalues.
    eps = np.finfo(np.float64).eps
    pvals = np.clip(pvals, eps, 1 - eps)
    np.fill_diagonal(pvals, 0.0)
    return pvals


def compute_pvals_corrected_ttest(df, n_train, n_test, order=None):
    """Compute the Nadeau & Bengio corrected resampled t-test.

    Returns a kxk matrix of one-tailed p-values comparing each pair of
    pipelines with the corrected resampled t-test of Nadeau & Bengio
    [1]_. Use this test when the rows of ``df`` are scores of overlapping
    cross-validation resamples (e.g. the folds of a within-session
    (repeated) k-fold evaluation): the folds share training examples, so
    a standard paired t-test underestimates the variance and is
    overconfident. The correction inflates the variance by
    :math:`n_2 / n_1`, the test/train ratio of each split.

    For k-fold cross-validation repeated r times on ``m`` examples, use
    ``n_test = m // k`` and ``n_train = m - n_test`` and pass the
    ``n = r * k`` per-fold scores as rows.

    Parameters
    ----------
    df: :class:`pandas.DataFrame`
        Scores of each cross-validation resample; samples (resamples) are
        index, columns are pipelines, and values are scores.
    n_train: int
        Number of training examples in each resample.
    n_test: int
        Number of testing examples in each resample.
    order: list of length (n_pipelines)
        Names corresponding to df columns.

    Returns
    -------
    pvals: ndarray of shape (n_pipelines, n_pipelines)
        One-tailed p-values; ``pvals[i, j]`` is small when pipeline ``i``
        scores significantly higher than pipeline ``j``.

    References
    ----------
    .. [1] Nadeau, C., & Bengio, Y. (1999). Inference for the
       generalization error. Advances in Neural Information Processing
       Systems 12; extended version in Machine Learning, 52, 239-281
       (2003). https://doi.org/10.1023/A:1024068626366
    """
    if order is None:
        order = df.columns
    else:
        if set(order) != set(df.columns):
            raise ValueError("provided order does not have all columns of dataframe")
    # reshape df into matrix (n_resamples, k, k) of differences
    vals = _validate_finite_scores(
        df[list(order)], name="corrected t-test score differences"
    )
    data = vals[:, :, None] - vals[:, None, :]
    return _corrected_resampled_ttest(data, n_train, n_test)


def compute_effect(df, order=None):
    """Compute effect size across datasets.

    Returns kxk matrix of effect sizes, order defines the order of rows/columns

    Parameters
    ----------
    df: :class:`pandas.DataFrame`
        Aggregated results, samples are index, columns are pipelines, and values are
        scores
    order: list
        list of length (num algorithms) with names corresponding to df columns

    Returns
    -------
    effect: ndarray of shape (n_pipelines, n_pipelines)
        array of effect size
    """
    _validate_finite_scores(df)
    if order is None:
        order = df.columns
    else:
        if set(order) != set(df.columns):  # was assert, now raises properly
            raise ValueError("provided order does not have all columns of dataframe")

    order_list = list(order)
    vals = df[order_list].to_numpy(dtype=float)
    n_pipes = len(order_list)
    out = np.zeros((n_pipes, n_pipes))
    for i in range(n_pipes):
        col_i = vals[:, i]
        for j in range(n_pipes):
            if i != j:
                # for now it's just the standardized difference
                diffs = col_i - vals[:, j]
                mean = diffs.mean()
                # Keep the sample-std (ddof=1) semantics for nonconstant
                # samples, but treat a nonempty constant difference directly.
                if (diffs == diffs[0]).all():
                    # The paired differences have no spread, so the standardized
                    # difference is 0/0 when the two pipelines score identically
                    # and c/0 when they differ by a constant. Identical pipelines
                    # have no effect, so report 0 rather than the NaN (plus
                    # RuntimeWarning) that NumPy would produce. A constant offset
                    # really is an unbounded effect, so keep the sign and make
                    # the infinity deliberate rather than a division accident.
                    out[i, j] = 0.0 if mean == 0 else np.copysign(np.inf, mean)
                else:
                    std = diffs.std(ddof=1)
                    out[i, j] = mean / std
    return out


def compute_dataset_statistics(df, perm_cutoff=20):
    """Compute meta-analysis statistics from results dataframe.

    Parameters
    ----------
    df: :class:`pandas.DataFrame`
        results obtained by an evaluation
    perm_cutoff: int, default=20
        threshold value for using permutation or Wilcoxon tests

    Returns
    -------
    stats: :class:`pandas.DataFrame`
        Table of effect and p-values for each dataset and all pipelines
    """
    df = collapse_session_scores(df)
    out = {}
    for d, group in df.groupby("dataset", sort=False, observed=True):
        score_data = group.pivot(
            index="subject", values="score", columns="pipeline"
        )
        algs = score_data.columns.tolist()
        if score_data.shape[0] < perm_cutoff:
            p = compute_pvals_perm(score_data, algs)
        else:
            p = compute_pvals_wilcoxon(score_data, algs)
        t = compute_effect(score_data, algs)
        n_algs = len(algs)
        stats_df = pd.DataFrame(
            {
                "pipe1": np.tile(algs, n_algs),
                "pipe2": np.repeat(algs, n_algs),
                "p": p.ravel(order="F"),
                "smd": t.ravel(order="F"),
                "nsub": score_data.shape[0],
            }
        )
        out[d] = stats_df
    return pd.concat(out, axis=0, names=["dataset", "index"]).reset_index()


def combine_effects(effects, nsubs):
    """Combine effects for meta-analysis statistics.

    Function that takes effects from each experiments and number of subjects to
    return meta-analysis effect

    Parameters
    ----------
    effects: :class:`pandas.DataFrame`
        effects for 2 pipelines computed on different datasets
    nsubs: float
        average number of subject per datasets

    Returns
    -------
    effect: float
        Estimatation of the combined effects
    """
    W = np.sqrt(nsubs)
    W = W / W.sum()
    return (W * effects).sum()


def combine_pvalues(p, nsubs):
    """Combine p-values for meta-analysis statistics.

    Function that takes pvals from each experiments and number of subjects to
    return meta-analysis significance using Stouffer's method

    Parameters
    ----------
    p: :class:`pandas.DataFrame`
        p-values for 2 pipelines computed on different datasets
    nsubs: float
        average number of subject per datasets

    Returns
    -------
    pval: float
        Estimatation of the combined p-value
    """
    if len(p) == 1:
        return p.item()
    else:
        W = np.sqrt(nsubs)
        out = stats.combine_pvalues(np.array(p), weights=W, method="stouffer")[1]
        return out


def find_significant_differences(df, perm_cutoff=20):
    """Compute differences between pipelines across datasets.

    Compute matrices of p-values and effects for all algorithms over all datasets via
    combined p-values and combined effects methods

    Parameters
    ----------
    df: :class:`pandas.DataFrame`
        Table of effect and p-values for each dataset and all pipelines, returned by
        compute_dataset_statistics
    perm_cutoff: int, default=20
        threshold value  to stop using permutation tests, which can be very expensive
        computationally, using Wilcoxon rank-sum test instead

    Returns
    -------
    dfP: :class:`pandas.DataFrame` of shape (n_pipelines, n_pipelines)
        p-values per algorithm pairs
    dfT: :class:`pandas.DataFrame` of shape (n_pipelines, n_pipelines)
        signed standardized mean differences
    """
    dsets = df.dataset.unique()
    algs = df.pipe1.unique()
    nsubs = (
        df.groupby("dataset", sort=False, observed=True)["nsub"]
        .mean()
        .reindex(dsets)
        .to_numpy()
    )
    P_full = df.pivot_table(values="p", index=["dataset", "pipe1"], columns="pipe2")
    T_full = df.pivot_table(values="smd", index=["dataset", "pipe1"], columns="pipe2")
    P = np.full((len(algs), len(algs)), np.nan)
    T = np.full((len(algs), len(algs)), np.nan)
    full_grid = len(P_full) == len(dsets) * len(algs) and len(T_full) == len(
        dsets
    ) * len(algs)
    if full_grid:
        midx = pd.MultiIndex.from_product([P_full.index.levels[0], algs])
        p_cube = P_full.reindex(index=midx, columns=algs).to_numpy().reshape(
            len(dsets), len(algs), len(algs)
        )
        t_cube = T_full.reindex(index=midx, columns=algs).to_numpy().reshape(
            len(dsets), len(algs), len(algs)
        )
    for i in range(len(algs)):
        for j in range(len(algs)):
            if i != j:
                if full_grid:
                    p = p_cube[:, i, j]
                    t = t_cube[:, i, j]
                else:
                    p = P_full.loc[(slice(None), algs[i]), algs[j]]
                    t = T_full.loc[(slice(None), algs[i]), algs[j]]
                P[i, j] = combine_pvalues(p, nsubs)
                if np.isnan(P[i, j]):
                    log.info("NaN p-value found, turned to 1")
                    print("NaN")
                    # P[i, j] = 1.0
                T[i, j] = combine_effects(t, nsubs)
    dfP = pd.DataFrame(index=algs, columns=algs, data=P)
    dfT = pd.DataFrame(index=algs, columns=algs, data=T)
    return dfP, dfT
