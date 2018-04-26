import os
import platform
from datetime import datetime
import pandas as pd
from moabb.analysis import plotting as plt  # flake8: noqa
from moabb.analysis.results import Results  # flake8: noqa
from moabb.analysis.meta_analysis import find_significant_differences


def analyze(results, out_path, name='analysis', plot=False):
    '''Analyze results.

    Given a results dataframe, generates a folder with
    results and a dataframe of the exact data used to generate those results,
    aswell as introspection to return information on the computer

    parameters
    ----------
    out_path: location to store analysis folder

    results: Dataframe generated from Results object

    path: string/None

    plot: whether to plot results

    Either path or results is necessary

    '''
    # input checks #
    if type(out_path) is not str:
        raise ValueError('Given out_path argument is not string')
    elif not os.path.isdir(out_path):
        raise IOError('Given directory does not exist')
    else:
        analysis_path = os.path.join(out_path, name)

    os.makedirs(analysis_path, exist_ok=True)
    # TODO: no good cross-platform way of recording CPU info?
    with open(os.path.join(analysis_path, 'info.txt'), 'a') as f:
        dt = datetime.now()
        f.write(
            'Date: {:%Y-%m-%d}\n Time: {:%H:%M}\n'.format(dt,
                                                          dt))
        f.write('System: {}\n'.format(platform.system()))
        f.write('CPU: {}\n'.format(platform.processor()))

    results.to_csv(os.path.join(analysis_path, 'data.csv'))
    sig_df, effect_df = find_significant_differences(results)
    sig_df.index = sig_df.index.rename('Pipe1')
    effect_df.index = effect_df.index.rename('Pipe1')
    D1 = pd.melt(sig_df.reset_index(), id_vars='Pipe1',
                 var_name='Pipe2', value_name='p-value')
    D2 = pd.melt(effect_df.reset_index(), id_vars='Pipe1',
                 var_name='Pipe2', value_name='effect size')
    D1.merge(D2).to_csv(os.path.join(analysis_path, 'stats.csv'))

    if plot:
        fig, color_dict = plt.score_plot(results)
        fig.savefig(os.path.join(analysis_path, 'scores.pdf'))
        fig = plt.ordering_heatmap(sig_df, effect_df)
        fig.savefig(os.path.join(analysis_path, 'ordering.pdf'))
        
