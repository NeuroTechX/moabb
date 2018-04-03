import os
import h5py
import re
import hashlib
import numpy as np
import pandas as pd
import inspect
from sklearn.base import BaseEstimator

from datetime import datetime


def get_digest(obj):
    """Return hash of an object repr.

    If there are memory addresses, wipes them
    """
    if issubclass(type(obj), BaseEstimator):
        str_repr = repr(obj.get_params())
    else:
        str_repr = repr(obj)
    str_no_addresses = re.sub('0x[a-z0-9]*', '0x__', str_repr).encode('utf8')
    return hashlib.md5(str_no_addresses).hexdigest()


class Results:
    '''Class to hold results from the evaluation.evaluate method.

    Appropriate test would be to ensure the result of 'evaluate' is
    consistent and can be accepted by 'results.add'

    Saves dataframe per pipeline and can query to see if particular
    subject has already been run

    '''

    def __init__(self, evaluation_class, paradigm_class, suffix='',
                 overwrite=False):
        """
        class that will abstract result storage
        """
        import moabb
        from moabb.paradigms.base import BaseParadigm
        from moabb.evaluations.base import BaseEvaluation
        assert issubclass(evaluation_class, BaseEvaluation)
        assert issubclass(paradigm_class, BaseParadigm)

        self.mod_dir = os.path.dirname(
            os.path.abspath(inspect.getsourcefile(moabb)))
        self.filepath = os.path.join(self.mod_dir, 'results',
                                     paradigm_class.__name__,
                                     evaluation_class.__name__,
                                     'results{}.hdf5'.format('_'+suffix))

        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.filepath = self.filepath

        if overwrite and os.path.isfile(self.filepath):
            os.remove(self.filepath)

        if not os.path.isfile(self.filepath):
            with h5py.File(self.filepath, 'w') as f:
                f.attrs['create_time'] = np.string_(
                    '{:%Y-%m-%d, %H:%M}'.format(datetime.now()))

    def add(self, results, pipelines):
        """add results"""
        def to_list(res):
            if type(res) is dict:
                return [res]
            elif type(res) is not list:
                raise ValueError("Results are given as neither dict nor"
                                 "list but {}".format(type(res).__name__))
            else:
                return res

        with h5py.File(self.filepath, 'r+') as f:
            for name, data_dict in results.items():
                digest = get_digest(pipelines[name])
                if digest not in f.keys():
                    # create pipeline main group if nonexistant
                    f.create_group(digest)

                ppline_grp = f[digest]
                ppline_grp.attrs['name'] = name
                ppline_grp.attrs['repr'] = repr(pipelines[name])

                dlist = to_list(data_dict)
                d1 = dlist[0]  # FIXME: handle multiple session ?
                dname = d1['dataset'].code
                if dname not in ppline_grp.keys():
                    # create dataset subgroup if nonexistant
                    dset = ppline_grp.create_group(dname)
                    dset.attrs['n_subj'] = len(d1['dataset'].subject_list)
                    dset.attrs['n_sessions'] = d1['dataset'].n_sessions
                    dt = h5py.special_dtype(vlen=str)
                    dset.create_dataset('id', (0, 2), dtype=dt,
                                        maxshape=(None, 2))
                    dset.create_dataset('data', (0, 3),
                                        maxshape=(None, 3))
                    dset.attrs['channels'] = d1['n_channels']
                    dset.attrs.create('columns',
                                      ['score', 'time', 'samples'],
                                      dtype=dt)
                dset = ppline_grp[dname]
                for d in dlist:
                    # add id and scores to group
                    length = len(dset['id']) + 1
                    dset['id'].resize(length, 0)
                    dset['data'].resize(length, 0)
                    dset['id'][-1, :] = np.asarray([str(d['subject']),
                                                    str(d['session'])])
                    dset['data'][-1, :] = np.asarray([d['score'],
                                                      d['time'],
                                                      d['n_samples']])

    def to_dataframe(self):
        df_list = []
        with h5py.File(self.filepath, 'r') as f:
            for _, p_group in f.items():
                name = p_group.attrs['name']
                for dname, dset in p_group.items():
                    array = np.array(dset['data'])
                    ids = np.array(dset['id'])
                    df = pd.DataFrame(array,
                                      columns=dset.attrs['columns'])
                    df['subject'] = ids[:, 0]
                    df['session'] = ids[:, 1]
                    df['channels'] = dset.attrs['channels']
                    df['n_sessions'] = dset.attrs['n_sessions']
                    df['dataset'] = dname
                    df['pipeline'] = name
                    df_list.append(df)
        return pd.concat(df_list, ignore_index=True)

    def not_yet_computed(self, pipelines, dataset, subj):
        """Check if a results has already been computed."""
        ret = {k: pipelines[k] for k in pipelines.keys()
               if not self._already_computed(pipelines[k], dataset, subj)}
        return ret

    def _already_computed(self, pipeline, dataset, subject, session=None):
        """Check if we have results for a current combination of pipeline
        / dataset / subject.
        """
        with h5py.File(self.filepath, 'r') as f:
            # get the digest from repr
            digest = get_digest(pipeline)

            # check if digest present
            if digest not in f.keys():
                return False
            else:
                pipe_grp = f[digest]
                # if present, check for dataset code
                if dataset.code not in pipe_grp.keys():
                    return False
                else:
                    # if dataset, check for subject
                    dset = pipe_grp[dataset.code]
                    return (str(subject) in dset['id'][:, 0])
