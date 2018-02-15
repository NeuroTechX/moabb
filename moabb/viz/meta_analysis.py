import pandas as pd
import seaborn as sea
import numpy as np
import os
import scipy.stats as stats


def violinplot(data):
    '''
    Function that takes a dict of results with the following columns:
    id, time, score, dataset, n_samples
    and keys correspond to pipelines
    and gives a violin plot for each dataset and pipeline over score
    '''
    df = pd.concat(data.values(), keys=data.keys()).reset_index()
    df = df.rename(columns={'level_0':'pipeline'})

    ax = sea.violinplot(data=df, x="score",y="dataset",hue="pipeline")
    return ax

def rmANOVA(data):
    '''
    My attempt at a repeated-measures ANOVA 
    In:
        data: dict of pipeline:DataFrame

    Out:
        x: symmetric matrix of f-statistics
        **coming soon** p: p-values for each element of x
    '''
    
    df = pd.concat(data.values(), keys=data.keys()).reset_index()
    df = df.rename(columns={'level_0':'pipeline'})

    def compute_anova(matrix):
        mean_subj = matrix.mean(axis=1)
        mean_algo = matrix.mean(axis=0)
        grand_mean = matrix[:].mean()

        # SS: sum of squared difference
        SS_algo = len(mean_subj) * np.sum((mean_algo - grand_mean)**2)
        SS_within_subj = np.sum((matrix - mean_algo[np.newaxis,:])**2)
        SS_subject = len(mean_algo) * np.sum((mean_subj - grand_mean)**2)
        SS_error = SS_within_subj - SS_subject

        # MS: Mean of squared difference
        MS_algo = SS_algo / (len(mean_algo) - 1)
        MS_error = SS_error / ((len(mean_algo) - 1)*(len(mean_subj) - 1))

        # F-statistics
        f = MS_algo/MS_error
        n,k = matrix.shape
        df1 = (n-1)*(k-1)
        df2 = n*k - n - k - 1 # calculated as one-way repeated-measures ANOVA
        p = stats.f.pdf(f, df1, df2)
        return f, p

    stats_dict = dict()
    for dset in df['dataset'].unique():
        tmp = df[df['dataset'] == dset]
        alg_list = []
        for alg in df['pipeline'].unique():
            alg_list.append(tmp[tmp['pipeline'] == alg]['score'].as_matrix())
        M = np.stack(alg_list).T
        print('{}\n{}'.format(dset,M))
        stats_dict[dset] = compute_anova(M)
    return stats_dict
