import pandas as pd
import seaborn as sea
import numpy as np
import os
import scipy.stats as stats


def violinplot(data):
    '''
    Input:
        data: dataframe

    Out:
        ax: pyplot Axes reference
    '''

    ax = sea.violinplot(data=data, y="score", x="dataset", hue="pipeline", inner="stick",cut=0)
    ax.set_ylim([0.5,1])
    
    return ax


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
            alg_list.append(df[np.logical_and(
                df['dataset'] == dset, df['pipeline'] == alg)]['score'].as_matrix())
        alg_list = [a for a in alg_list if len(a) > 0] #some datasets and algorithms may not exist?
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
