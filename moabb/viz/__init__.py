import os
import platform
from datetime import datetime

from moabb.viz import plotting as plt
from moabb.viz.results import Results


def analyze(results, out_path, name='analysis', suffix=''):
    '''Given a results object (or the location for one), generates a folder with
    results and a dataframe of the exact data used to generate those results, as
    well as introspection to return information on the computer

    In:
    out_path: location to store analysis folder

    results: Obj/tuple;

    path: string/None

    Either path or results is necessary

    '''
    # input checks #
    if type(results) is not Results:
        res = Results(*results, suffix=suffix)
    if type(out_path) is not str:
        raise ValueError('Given out_path argument is not string')
    elif not os.path.isdir(out_path):
        raise IOError('Given directory does not exist')
    else:
        analysis_path = os.path.join(out_path, name)

    os.makedirs(analysis_path, exist_ok=True)
    # TODO: no good cross-platform way of recording CPU info?
    with open(os.path.join(analysis_path, 'info.txt'), 'a') as f:
        f.write(
            'Date: {:%Y-%m-%d}\n Time: {:%H:%M}\n'.format(datetime.now(), datetime.now()))
        f.write('System: {}\n'.format(platform.system()))
        f.write('CPU: {}\n'.format(platform.processor()))

    res = results

    data = res.to_dataframe()
    data.to_csv(os.path.join(analysis_path, 'data.csv'))

    plt.score_plot(data).savefig(os.path.join(analysis_path, 'scores.pdf'))
    plt.time_line_plot(data).savefig(os.path.join(analysis_path, 'time2d.pdf'))
