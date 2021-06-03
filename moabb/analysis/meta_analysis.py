import itertools
import logging

import numpy as np
import pandas as pd
import scipy.stats as stats


log = logging.getLogger(__name__)


def collapse_session_scores(df):
    return df.groupby(["pipeline", "dataset", "subject"]).mean().reset_index()


def compute_pvals_wilcoxon(df, order=None):
    """Returns kxk matrix of p-values computed via the Wilcoxon rank-sum test,
    order defines the order of rows and columns

    df: DataFrame, samples are index, columns are pipelines, and values are
    scores

    order: list of length (num algorithms) with names corresponding to columns
    of df

    """
    if order is None:
        order = df.columns
    else:
        errormsg = "provided order does not have all columns of dataframe"
        assert set(order) == set(df.columns), errormsg

    out = np.zeros((len(df.columns), len(df.columns)))
    for i in range(len(order)):
        for j in range(len(order)):
            if i != j:
                pipe1 = order[i]
                pipe2 = order[j]
                p = stats.wilcoxon(df.loc[:, pipe1], df.loc[:, pipe2])[1]
                p /= 2
                # we want the one-tailed p-value
                diff = (df.loc[:, pipe1] - df.loc[:, pipe2]).mean()
                if diff < 0:
                    p = 1 - p  # was in the other side of the distribution
                out[i, j] = p
    return out


def _pairedttest_exhaustive(data):
    """Returns p-values for exhaustive ttest that runs through all possible
    permutations of the first dimension. Very bad idea for size greater than 12

    data is a (subj, alg, alg) matrix of differences between scores for each
    pair of algorithms per subject

    """
    out = np.ones((data.shape[1], data.shape[1]))
    true = data.sum(axis=0)
    nperms = 2 ** data.shape[0]
    for perm in itertools.product([-1, 1], repeat=data.shape[0]):
        # turn into numpy array
        perm = np.array(perm)
        # multiply permutation by subject dimension and sum over subjects
        randperm = (data * perm[:, None, None]).sum(axis=0)
        # compare to true difference (numpy autocasts bool to 0/1)
        out += randperm > true
    out = out / nperms
    # control for cases where pval is 1
    out[out == 1] = 1 - (1 / nperms)
    return out


def _pairedttest_random(data, nperms):
    """Returns p-values based on nperms permutations of a paired ttest

    data is a (subj, alg, alg) matrix of differences between scores for each
    pair of algorithms per subject
    """
    out = np.ones((data.shape[1], data.shape[1]))
    true = data.sum(axis=0)
    for _ in range(nperms):
        perm = np.random.randint(2, size=(data.shape[0],))
        perm[perm == 0] = -1
        # multiply permutation by subject dimension and sum over subjects
        randperm = (data * perm[:, None, None]).sum(axis=0)
        # compare to true difference (numpy autocasts bool to 0/1)
        out += randperm > true
    out[out == nperms] = nperms - 1
    return out / nperms


def compute_pvals_perm(df, order=None):
    """Returns kxk matrix of p-values computed via permutation test,
    order defines the order of rows and columns

    df: DataFrame, samples are index, columns are pipelines, and values are
    scores

    order: list of length (num algorithms) with names corresponding to columns
    of df

    """
    if order is None:
        order = df.columns
    else:
        errormsg = "provided order does not have all columns of dataframe, order = {}, dfcols = {}".format(
            set(order), set(df.columns)
        )
        assert set(order) == set(df.columns), errormsg
    # reshape df into matrix (sub, k, k) of differences
    data = np.zeros((df.shape[0], len(order), len(order)))
    for i in range(len(order) - 1):
        for j in range(i + 1, len(order)):
            pipe1 = order[i]
            pipe2 = order[j]
            data[:, i, j] = df.loc[:, pipe1] - df.loc[:, pipe2]
            data[:, j, i] = df.loc[:, pipe2] - df.loc[:, pipe1]
    if data.shape[0] > 13:
        p = _pairedttest_random(data, 10000)
    else:
        p = _pairedttest_exhaustive(data)
    return p


def compute_effect(df, order=None):
    """Returns kxk matrix of effect sizes, order defines the order of rows/columns

    df: DataFrame, samples are index, columns are pipelines, and values are
    scores

    order: list of length (num algorithms) with names corresponding to
    columns of df

    """
    if order is None:
        order = df.columns
    else:
        errormsg = "provided order does not have all columns of dataframe"
        assert set(order) == set(df.columns), errormsg

    out = np.zeros((len(df.columns), len(df.columns)))
    for i, pipe1 in enumerate(order):
        for j, pipe2 in enumerate(order):
            if i != j:
                # for now it's just the standardized difference
                diffs = df.loc[:, pipe1] - df.loc[:, pipe2]
                diffs = diffs.mean() / diffs.std()
                out[i, j] = diffs
    return out


def compute_dataset_statistics(df, perm_cutoff=20):
    """
    Returns dict of datasets to DataFrames with stats

    """
    df = collapse_session_scores(df)
    algs = df.pipeline.unique()
    dsets = df.dataset.unique()
    out = {}
    for d in dsets:
        score_data = df[df.dataset == d].pivot(
            index="subject", values="score", columns="pipeline"
        )
        if score_data.shape[0] < perm_cutoff:
            p = compute_pvals_perm(score_data, algs)
        else:
            p = compute_pvals_wilcoxon(score_data, algs)
        t = compute_effect(score_data, algs)
        P = pd.DataFrame(index=pd.Index(algs, name="pipe1"), columns=algs, data=p)
        T = pd.DataFrame(index=pd.Index(algs, name="pipe1"), columns=algs, data=t)
        D1 = pd.melt(P.reset_index(), id_vars="pipe1", var_name="pipe2", value_name="p")
        D2 = pd.melt(T.reset_index(), id_vars="pipe1", var_name="pipe2", value_name="smd")
        stats_df = D1.merge(D2)
        stats_df["nsub"] = score_data.shape[0]
        out[d] = stats_df
    return pd.concat(out, axis=0, names=["dataset", "index"]).reset_index()


def combine_effects(effects, nsubs):
    """Function that takes effects from each experiments and number of subjects to
    return meta-analysis effect

    """
    W = np.sqrt(nsubs)
    W = W / W.sum()
    return (W * effects).sum()


def combine_pvalues(p, nsubs):
    """Function that takes pvals from each experiments and number of subjects to
    return meta-analysis significance

    """
    if len(p) == 1:
        return p.item()
    else:
        W = np.sqrt(nsubs)
        out = stats.combine_pvalues(np.array(p), weights=W, method="stouffer")[1]
        return out


def find_significant_differences(df, perm_cutoff=20):
    """Compute matrix of p-values for all algorithms over all datasets via
    combined p-values method

    df: DataFrame, output of compute_dataset_statistics

    perm_cutoff: int, opt -- cutoff at which to stop using permutation tests,
                 which can be very expensive computationally

    Out:

    P: matrix (k,k) of p-values per algorithm pair

    T: matrix (k,k) of signed standardized mean difference

    """
    dsets = df.dataset.unique()
    algs = df.pipe1.unique()
    nsubs = np.array([df.loc[df.dataset == d, "nsub"].mean() for d in dsets])
    P_full = df.pivot_table(values="p", index=["dataset", "pipe1"], columns="pipe2")
    T_full = df.pivot_table(values="smd", index=["dataset", "pipe1"], columns="pipe2")
    P = np.full((len(algs), len(algs)), np.NaN)
    T = np.full((len(algs), len(algs)), np.NaN)
    for i in range(len(algs)):
        for j in range(len(algs)):
            if i != j:
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
