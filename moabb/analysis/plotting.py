import logging
import matplotlib.pyplot as plt
import seaborn as sea
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from scipy.stats import t

from moabb.analysis.meta_analysis import collapse_session_scores
from moabb.analysis.meta_analysis import combine_effects, combine_pvalues


PIPELINE_PALETTE = sea.color_palette("husl", 6)
sea.set_palette(PIPELINE_PALETTE)
sea.set(font='serif', style='whitegrid')

log = logging.getLogger()


def _simplify_names(x):
    if len(x) > 10:
        return x.split(' ')[0]
    else:
        return x


def score_plot(data, pipelines=None):
    '''
    In:
        data: output of Results.to_dataframe()
        pipelines: list of string|None, pipelines to include in this plot
    Out:
        ax: pyplot Axes reference
    '''
    data = collapse_session_scores(data)
    data['dataset'] = data['dataset'].apply(_simplify_names)
    if pipelines is not None:
        data = data[data.pipeline.isin(pipelines)]
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_subplot(111)
    sea.stripplot(data=data, y="dataset", x="score", jitter=True, dodge=True,
                  hue="pipeline", zorder=1, alpha=0.7, ax=ax)
    # sea.pointplot(data=data, y="score", x="dataset",
    #               hue="pipeline", zorder=1, ax=ax)
    # sometimes the score is lower than 0.5 (...not sure how to deal with that)
    ax.set_xlim([0, 1])
    ax.set_title('Scores per dataset and algorithm')
    handles, labels = ax.get_legend_handles_labels()
    color_dict = {l: h.get_facecolor()[0] for l, h in zip(labels, handles)}
    return fig, color_dict


def paired_plot(data, alg1, alg2):
    '''
    returns figure with an axis that has a paired plot on it
    Data: dataframe from Results
    alg1: name of a member of column data.pipeline
    alg2: name of a member of column data.pipeline

    '''
    data = collapse_session_scores(data)
    data = data[data.pipeline.isin([alg1, alg2])]
    data = data.pivot_table(values='score', columns='pipeline',
                            index=['subject', 'dataset'])
    data = data.reset_index()
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111)
    data.plot.scatter(alg1, alg2, ax=ax)
    ax.plot([0, 1], [0, 1], ls='--', c='k')
    ax.set_xlim([0.5, 1])
    ax.set_ylim([0.5, 1])
    return fig


def ordering_heatmap(sig_df, effect_df, p_threshold=0.05):
    '''Visualize significances as a heatmap with green/grey/red for significantly
    higher/significantly lower.
    sig_df is a DataFrame of pipeline x pipeline where each value is a p-value,
    effect_df is a DF where each value is an effect size

    '''
    effect_df.columns = effect_df.columns.map(_simplify_names)
    sig_df.columns = sig_df.columns.map(_simplify_names)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sea.heatmap(data=effect_df, center=0, annot=True,
                mask=(sig_df > p_threshold),
                fmt="2.2f", cbar_kws={'label': 'Meta-effect'},
                cmap=sea.light_palette("green"))
    for l in ax.get_xticklabels():
        l.set_rotation(45)
    ax.tick_params(axis='y', rotation=0.9)
    plt.tight_layout()
    return fig


def meta_analysis_plot(stats_df, alg1, alg2):
    '''A meta-analysis style plot that shows the standardized effect with
    confidence intervals over all datasets for two algorithms.
    Hypothesis is that alg1 is larger than alg2'''
    assert (alg1 in stats_df.pipe1.unique())
    assert (alg2 in stats_df.pipe1.unique())
    df_fw = stats_df.loc[(stats_df.pipe1 == alg1) & (stats_df.pipe2 == alg2)]
    df_fw = df_fw.sort_values(by='pipe1')
    df_bk = stats_df.loc[(stats_df.pipe1 == alg2) & (stats_df.pipe2 == alg1)]
    df_bk = df_bk.sort_values(by='pipe1')
    dsets = df_fw.dataset.unique()
    ci = []
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 5)
    sig_ind = []
    pvals = []
    ax = fig.add_subplot(gs[0, :-1])
    ax.set_yticks(np.arange(len(dsets) + 1))
    ax.set_yticklabels(['Meta-effect'] + [_simplify_names(d) for d in dsets])
    pval_ax = fig.add_subplot(gs[0, -1], sharey=ax)
    plt.setp(pval_ax.get_yticklabels(), visible=False)
    _min = 0
    _max = 0
    for ind, d in enumerate(dsets):
        nsub = float(df_fw.loc[df_fw.dataset == d, 'nsub'])
        t_dof = nsub - 1
        ci.append(t.ppf(0.95, t_dof)/np.sqrt(nsub))
        v = float(df_fw.loc[df_fw.dataset == d, 'smd'])
        if v > 0:
            p = df_fw.loc[df_fw.dataset == d, 'p'].item()
            if p < 0.05:
                sig_ind.append(ind)
                pvals.append(p)
        else:
            p = df_bk.loc[df_bk.dataset == d, 'p'].item()
            if p < 0.05:
                sig_ind.append(ind)
                pvals.append(p)
        _min = _min if (_min < (v-ci[-1])) else (v-ci[-1])
        _max = _max if (_max > (v+ci[-1])) else (v+ci[-1])
        ax.plot(np.array([v - ci[-1], v + ci[-1]]),
                np.ones((2,)) * (ind + 1), c='tab:grey')
    _range = max(abs(_min), abs(_max))
    ax.set_xlim((0-_range, 0+_range))
    final_effect = combine_effects(df_fw['smd'], df_fw['nsub'])
    ax.scatter(pd.concat([pd.Series([final_effect]), df_fw['smd']]),
               np.arange(len(dsets) + 1),
               s=np.array([50] + [30]*len(dsets)),
               marker='D',
               c=['k'] + ['tab:grey']*len(dsets))
    sig_ind = np.array(sig_ind)
    ax.scatter(df_fw['smd'].iloc[sig_ind],
               sig_ind + 1.4, s=20,
               marker='*', c='r')
    # pvalues axis stuf
    pval_ax.set_xlim([-0.1, 0.1])
    pval_ax.grid(False)
    pval_ax.set_title('p-value')
    pval_ax.set_xticks([])
    for spine in pval_ax.spines.values():
        spine.set_visible(False)
    for ind, p in zip(sig_ind, pvals):
        pval_ax.text(0, ind+1, horizontalalignment='center',
                     verticalalignment='center',
                     s='{:.2e}'.format(p), fontsize=8)
    if final_effect > 0:
        p = combine_pvalues(df_fw['p'], df_fw['nsub'])
        if p < 0.05:
            ax.scatter([final_effect], [-0.4], s=20, marker='*', c='r')
            pval_ax.text(0, 0, horizontalalignment='center',
                         verticalalignment='center',
                         s='{:.2e}'.format(p), fontsize=8)
    else:
        p = combine_pvalues(df_bk['p'], df_bk['nsub'])
        if p < 0.05:
            ax.scatter([final_effect], [-0.4], s=20, marker='*', c='r')
            pval_ax.text(0, 0, horizontalalignment='center',
                         verticalalignment='center',
                         s='{:.2e}'.format(p), fontsize=8)

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(0, linestyle='--', c='k')
    ax.axhline(0.5, linestyle='-', linewidth=3, c='k')
    ax.set_title('{} vs {}'.format(alg2, alg1))
    ax.set_xlabel('Standardized Mean Difference')
    fig.tight_layout()

    return fig
