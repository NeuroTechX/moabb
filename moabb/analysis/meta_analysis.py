import numpy as np
import pandas as pd
import itertools
import scipy.stats as stats


def collapse_session_scores(df):
    return df.groupby(['pipeline', 'dataset', 'subject']).mean().reset_index()


def compute_pvals_wilcoxon(df, order=None):
    '''Returns kxk matrix of p-values computed via the Wilcoxon rank-sum test,
    order defines the order of rows and columns

    df: DataFrame, samples are index, columns are pipelines, and values are
    scores

    order: list of length (num algorithms) with names corresponding to columns
    of df

    '''
    if order is None:
        order = df.columns
    else:
        errormsg = 'provided order does not have all columns of dataframe'
        assert set(order) == set(df.columns), errormsg

    out = np.zeros((len(df.columns), len(df.columns)))
    for i in range(len(order)-1):
        for j in range(i+1, len(order)):
            pipe1 = order[i]
            pipe2 = order[j]
            p = stats.wilcoxon(df.loc[:, pipe1], df.loc[:, pipe2])[1]
            p /= 2
            # we want the one-tailed p-value
            diff = (df.loc[:, pipe1] - df.loc[:, pipe2]).mean()
            if diff < 0:
                p = 1 - p # was in the other side of the distribution
            out[i, j] = p
    return out


def _pairedttest_exhaustive(data):
    '''Returns p-values for exhaustive ttest that runs through all possible
    permutations of the first dimension. Very bad idea for size greater than 12

    data is a (subj, alg, alg) matrix of differences between scores for each
    pair of algorithms per subject

    '''
    out = np.zeros((data.shape[1], data.shape[1]))
    true = data.sum(axis=0)
    for perm in itertools.product([-1, 1], repeat=data.shape[0]):
        # turn into numpy array
        perm = np.array(perm)
        # multiply permutation by subject dimension and sum over subjects
        randperm = (data * perm[:, None, None]).sum(axis=0)
        # compare to true difference (numpy autocasts bool to 0/1)
        out += (randperm > true)
    out[out==0] = 1e-10
    return out / (2**data.shape[0])


def _pairedttest_random(data, nperms):
    '''Returns p-values based on nperms permutations of a paired ttest

    data is a (subj, alg, alg) matrix of differences between scores for each
    pair of algorithms per subject
    '''
    out = np.zeros((data.shape[1], data.shape[1]))
    true = data.sum(axis=0)
    for i in range(nperms):
        perm = np.random.randint(2, size=(data.shape[0],))
        perm[perm == 0] = -1
        # multiply permutation by subject dimension and sum over subjects
        randperm = (data * perm[:, None, None]).sum(axis=0)
        # compare to true difference (numpy autocasts bool to 0/1)
        out += (randperm > true)
    out[out==0] = 1e-10
    return out / nperms


def compute_pvals_perm(df, order=None):
    '''Returns kxk matrix of p-values computed via permutation test,
    order defines the order of rows and columns

    df: DataFrame, samples are index, columns are pipelines, and values are
    scores

    order: list of length (num algorithms) with names corresponding to columns
    of df

    '''
    if order is None:
        order = df.columns
    else:
        errormsg = 'provided order does not have all columns of dataframe'
        assert set(order) == set(df.columns), errormsg
    # reshape df into matrix (sub, k, k) of differences
    data = np.zeros((df.shape[0], len(order), len(order)))
    for i in range(len(order)-1):
        for j in range(i+1, len(order)):
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
    '''Returns kxk matrix of effect sizes, order defines the order of rows/columns

    df: DataFrame, samples are index, columns are pipelines, and values are
    scores

    order: list of length (num algorithms) with names corresponding to
    columns of df

    '''
    if order is None:
        order = df.columns
    else:
        errormsg = 'provided order does not have all columns of dataframe'
        assert set(order) == set(df.columns), errormsg

    out = np.zeros((len(df.columns), len(df.columns)))
    for i, pipe1 in enumerate(order[:-1]):
        for j in range(i+1, len(order)):
            pipe2 = order[j]
            # for now it's just the standardized difference
            diffs = (df.loc[:, pipe1] - df.loc[:, pipe2])
            diffs = diffs.mean() / diffs.std()
            out[i, j] = diffs
    return out


def find_significant_differences(df, perm_cutoff=20):
    '''Compute matrix of p-values for all algorithms over all datasets via
    combined p-values method

    df: DataFrame, long format

    perm_cutoff: int, opt -- cutoff at which to stop using permutation tests,
                 which can be very expensive computationally

    Out:

    P: matrix (k,k) of p-values per algorithm pair

    T: matrix (k,k) of signed standardized mean difference

    '''
    df = collapse_session_scores(df)
    algs = df.pipeline.unique()
    dsets = df.dataset.unique()
    P_full = np.zeros((len(algs), len(algs), len(dsets)))
    T_full = np.zeros((len(algs), len(algs), len(dsets)))
    W = np.zeros((len(dsets),))
    for ind, d in enumerate(dsets):
        score_data = df[df.dataset == d].pivot(index='subject',
                                               values='score',
                                               columns='pipeline')
        if score_data.shape[0] < perm_cutoff:
            P_full[..., ind] = compute_pvals_perm(score_data, algs)
        else:
            P_full[..., ind] = compute_pvals_wilcoxon(score_data, algs)
        T_full[..., ind] = compute_effect(score_data, algs)
        W[ind] = np.sqrt(score_data.shape[0])
    W_norm = W / W.sum()
    P = np.full((len(algs), len(algs)), np.NaN)
    T = np.full((len(algs), len(algs)), np.NaN)
    for i in range(len(algs)-1):
        for j in range(i+1, len(algs)):
            # print("p vals: {}\n".format(P_full[i,j,:]))
            P[i, j] = stats.combine_pvalues(P_full[i, j, :], weights=W)[1]
            # print("effect vals: {}\n".format(T_full[i,j,:]))
            effect_signs = np.sign(T_full[i,j,:])
            if abs(effect_signs.sum()) == T_full.shape[-1]:
                print("all effects same direction")
                print("{} - {} \n {} \n".format(algs[i], algs[j], T_full[i, j, :]))
            T[i, j] = (W_norm * T_full[i, j, :]).mean()
    dfP = pd.DataFrame(index=algs, columns=algs, data=P)
    dfT = pd.DataFrame(index=algs, columns=algs, data=T)
    return dfP, dfT
