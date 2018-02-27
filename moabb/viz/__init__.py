import pandas as pd
import h5py
import numpy as np
import os, platform
from datetime import datetime
from . import plotting as plt

class Results:
    '''Class to hold results from the evaluation.evaluate method. Appropriate test
    would be to ensure the result of 'evaluate' is consistent and can be
    accepted by 'results.add'

    Saves dataframe per pipeline and can query to see if particular subject has
    already been run

    '''

    def __init__(self, evaluation=None, path=None):
        """
        class that will abstract result storage
        """
        if path is None:
            path = os.path.join(os.path.dirname(__file__), 'results.hd5')
        self.filepath = path
        if not os.path.isfile(path):
            if evaluation is None:
                raise ValueError(
                    'If no path is provided, must provide an evaluation')
            with h5py.File(path, 'w') as f:
                f.attrs['eval'] = np.string_(type(evaluation).__name__)
        else:
            with h5py.File(path, 'r') as f:
                if f.attrs['eval'] != np.string_(type(evaluation).__name__):
                    raise ValueError('Given results file has different evaluation to current: {} vs {}'.format(
                        f.attrs['eval'], type(evaluation).__name__))

    def add(self, pipeline_dict):
        def to_list(d):
            if type(d) is dict:
                return [d]
            elif type(d) is not list:
                raise ValueError('Results are given as neither dict nor list but {}'.format(
                    type(d).__name__))
            else:
                return d
        with h5py.File(self.filepath, 'r+') as f:
            for name, data_dict in pipeline_dict.items():
                if name not in f.keys():
                    # create pipeline main group if nonexistant
                    f.create_group(name)
                ppline_grp = f[name]
                dlist = to_list(data_dict)
                d1 = dlist[0]
                dname = d1['dataset'].code
                if dname not in ppline_grp.keys():
                    # create dataset subgroup if nonexistant
                    dset = ppline_grp.create_group(dname)
                    dset.attrs['n_subj'] = len(d1['dataset'].subject_list)
                    dset.attrs['n_sessions'] = d1['dataset'].n_sessions
                    dt = h5py.special_dtype(vlen=str)
                    dset.create_dataset('id', (0,), dtype=dt, maxshape=(None,))
                    dset.create_dataset('data', (0, 3), maxshape=(None, 3))
                    dset.attrs['channels'] = d1['n_channels']
                    dset.attrs.create(
                        'columns', ['score', 'time', 'samples'], dtype=dt)
                dset = ppline_grp[dname]
                for d in dlist:
                    # add id and scores to group
                    length = len(dset['id']) + 1
                    dset['id'].resize(length, 0)
                    dset['data'].resize(length, 0)
                    dset['id'][-1] = str(d['id'])
                    dset['data'][-1, :] = np.asarray([d['score'], d['time'], d['n_samples']])

    def to_dataframe(self):
        df_list = []
        with h5py.File(self.filepath, 'r') as f:
            for name, p_group in f.items():
                for dname, dset in p_group.items():
                    array = np.array(dset['data'])
                    ids = np.array(dset['id'])
                    df = pd.DataFrame(array, columns=dset.attrs['columns'])
                    df['id'] = ids
                    df['channels'] = dset.attrs['channels']
                    df['n_sessions'] = dset.attrs['n_sessions']
                    df['dataset'] = dname
                    df['pipeline'] = name
                    df_list.append(df)
        return pd.concat(df_list, ignore_index=True)
                

    def not_yet_computed(self, pipeline_dict, dataset, subj):
        def already_computed(p, d, s):
            with h5py.File(self.filepath, 'r') as f:
                if p not in f.keys():
                    return False
                else:
                    pipe_grp = f[p]
                    if d.code not in pipe_grp.keys():
                        return False
                    else:
                        dset = pipe_grp[d.code]
                        return (str(s) in dset['id'])
        return {k: pipeline_dict[k] for k in pipeline_dict.keys() if not already_computed(k, dataset, subj)}


def analyze(out_path, results=None, path=None, name='analysis'):
    '''Given a results object (or the location for one), generates a folder with
    results and a dataframe of the exact data used to generate those results, as
    well as introspection to return information on the computer
    
    In:
    out_path: location to store analysis folder

    results: Obj/None; 

    path: string/None

    Either path or results is necessary

    '''
    ### input checks ###
    if results is not None and type(results) is not Results:
        raise ValueError('Given results argument is not of type moabb.viz.Results')
    if path is not None:
        if type(path) is not str:
            raise ValueError('Given path argument is not string')
        elif not os.path.isfile(path):
            raise IOError('Given results file does not exist')

    if not (bool(results is None) ^ bool(path is None)):
        raise ValueError('Either results or path must be given, but not both')

    if type(out_path) is not str:
        raise ValueError('Given out_path argument is not string')
    elif not os.path.isdir(out_path):
        raise IOError('Given directory does not exist')
    else:
        analysis_path = os.path.join(out_path,name)
        if os.path.isdir(analysis_path):
            raise IOError("Analysis directory {} already exists".format(analysis_path))

    os.mkdir(analysis_path)
    # TODO: no good cross-platform way of recording CPU info?
    with open(os.path.join(analysis_path,'info.txt'),'a') as f:
        f.write('Date: {:%Y-%m-%d}\n Time: {:%H:%M}\n'.format(datetime.now(), datetime.now()))
        f.write('System: {}\n'.format(platform.system()))
        f.write('CPU: {}\n'.format(platform.processor()))

    if results is None:
        res = Results(path=path)
    else:
        res = results

    data = res.to_dataframe()
    data.to_csv(os.path.join(analysis_path,'data.csv'))

    plt.score_plot(data).savefig(os.path.join(analysis_path,'scores.pdf'))
    plt.time_line_plot(data).savefig(os.path.join(analysis_path,'time2d.pdf'))


