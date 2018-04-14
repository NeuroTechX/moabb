import numpy as np
import scipy.stats as stats


def rmANOVA(df):
    '''
    My attempt at a repeated-measures ANOVA
    In:
        data: dataframe

    Out:
        x: symmetric matrix of f-statistics
        **coming soon** p: p-values for each element of x
    '''

    stats_dict = dict()
    for dset in df['dataset'].unique():
        alg_list = []
        for alg in df['pipeline'].unique():
            ix = np.logical_and(df['dataset'] == dset, df['pipeline'] == alg)
            alg_list.append(df[ix]['score'].as_matrix())

        # some datasets and algorithms may not exist?
        alg_list = [a for a in alg_list if len(a) > 0]
        M = np.stack(alg_list).T
        stats_dict[dset] = _rmanova(M)
    return stats_dict


def _rmanova(matrix):
    mean_subj = matrix.mean(axis=1)
    mean_algo = matrix.mean(axis=0)
    grand_mean = matrix[:].mean()

    # SS: sum of squared difference
    SS_algo = len(mean_subj) * np.sum((mean_algo - grand_mean)**2)
    SS_within_subj = np.sum((matrix - mean_algo[np.newaxis, :])**2)
    SS_subject = len(mean_algo) * np.sum((mean_subj - grand_mean)**2)
    SS_error = SS_within_subj - SS_subject

    # MS: Mean of squared difference
    MS_algo = SS_algo / (len(mean_algo) - 1)
    MS_error = SS_error / ((len(mean_algo) - 1)*(len(mean_subj) - 1))

    # F-statistics
    f = MS_algo/MS_error
    n, k = matrix.shape
    df1 = k-1
    df2 = (k-1)*(n-1)  # calculated as one-way repeated-measures ANOVA
    p = stats.f.sf(f, df1, df2)
    return f, p
