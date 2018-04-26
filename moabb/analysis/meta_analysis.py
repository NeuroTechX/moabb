import numpy as np
import scipy.stats as stats



def collapse_session_scores(df):
    return df.groupby(['pipeline', 'dataset', 'subject']).mean().reset_index()

def compute_pvals_Wilcoxon(df, order=None):
    '''Returns kxk matrix of p-values computed via the Wilcoxon rank-sum test,
    order defines the order of rows and columns 
    
    df: DataFrame, samples are index, columns are pipelines, and values are scores

    order: list of length (num algorithms) with names corresponding to columns of df

    '''
    if order is None:
        order = df.columns
    else:
        errormsg = 'provided order does not have all columns of dataframe'
        assert set(order) == set(df.columns), errormsg
    
    out = np.zeros((len(df.columns), len(df.columns)))
    for i, pipe1 in enumerate(order[:-1]):
        for j, pipe2 in enumerate(order[i+1:]):
            p = stats.wilcoxon(df[:, pipe1], df[:, pipe2])
            # we want the one-tailed p-value
            p /= 2
            out[i,j] = p
            out[j,i] = p
    return out
    

def pairedttest_perm(df, order=None):
    '''Returns kxk matrix of p-values computed via permutation test,
    order defines the order of rows and columns 
    
    df: DataFrame, samples are index, columns are pipelines, and values are scores

    order: list of length (num algorithms) with names corresponding to columns of df

    '''
    if order is None:
        order = df.columns
    else:
        errormsg = 'provided order does not have all columns of dataframe'
        assert set(order) == set(df.columns), errormsg
    # reshape df into matrix (sub, k, k) of differences
    data = np.zeros((df.shape[0], len(order), len(order)))
    

def compute_effect(df, order=None):
    '''Returns kxk matrix of effect sizes, order defines the order of rows/columns 
    
    df: DataFrame, samples are index, columns are pipelines, and values are scores

    order: list of length (num algorithms) with names corresponding to columns of df

    '''
    pass

def find_significant_differences(df, perm_cutoff=20):
    '''
    Compute matrix of p-values for all algorithms over all datasets via
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
    for ind, d in enumerate(dsets):
        score_data = df[df.dataset == d].pivot(index='subject',
                                               values='score',
                                               columns='pipeline')
        if score_data.shape[0] < 20:
            P_full[..., ind] = compute_pvals_perm(score_data, algs)
        else:
            P_full[..., ind] = compute_pvals_Wilcoxon(score_data, algs)
        T_full[...,ind] = compute_effect(score_data)




    
