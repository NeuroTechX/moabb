import os
import h5py
import hashlib
import sqlite3
import numpy as np
import pandas as pd
import inspect
import logging

log = logging.getLogger()

from datetime import datetime


def get_digest(obj):
    """Return hash of an object repr."""
    return hashlib.md5(repr(obj).encode('utf8')).hexdigest()

class Results:
    '''Class to hold results from the evaluation.evaluate method. Appropriate test
    would be to ensure the result of 'evaluate' is consistent and can be
    accepted by 'results.add'

    Saves dataframe per pipeline and can query to see if particular subject has
    already been run

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
        self.mod_dir = os.path.dirname(os.path.abspath(inspect.getsourcefile(moabb)))
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
                raise ValueError("Results are given as neither dict nor list"
                                 "but {}".format(type(res).__name__))
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
                    dset['data'][-1, :] = np.asarray([d['score'],
                                                      d['time'],
                                                      d['n_samples']])

    def to_dataframe(self):
        df_list = []
        with h5py.File(self.filepath, 'r') as f:
            for _, p_group in f.items():
                name = p_group['name']
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

    def not_yet_computed(self, pipelines, dataset, subj):
        """Check if a results has already been computed."""
        ret = {k: pipelines[k] for k in pipelines.keys()
               if not self._already_computed(pipelines[k], dataset, subj)}
        return ret

    def _already_computed(self, pipeline, dataset, subject):
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
                    pipe_grp = f[p]
                    if d.code not in pipe_grp.keys():
                        return False
                    else:
                        dset = pipe_grp[d.code]
                        return (str(s) in dset['id'])
        ret = {k: pipeline_dict[k] for k in pipeline_dict.keys()
               if not already_computed(k, dataset, subj)}
        return ret


class ResultsDB:
    '''Class to interface with results database. Can add new data and also return
    dataframes based on queries '''

    def __init__(self, write=False, evaluation=None, paradigm=None):
        """
        Initialize class. If database does not exist, create it.
        write: bool, do you want to write values (if yes, then need eval and paradigm)
        evaluation: BaseEvaluation child
        paradigm: BaseParadigm child
        """
        # first ensure that the database exists
        import moabb
        self.mod_dir = os.path.dirname(
            os.path.abspath(inspect.getsourcefile(moabb)))
        self.filepath = os.path.join(self.mod_dir, 'results', 'results.db')
        if not os.path.isfile(self.filepath):
            os.makedirs(os.path.join(self.mod_dir, 'results'), exist_ok=True)
            self._setup()
        self.conn = sqlite3.connect(self.filepath)

        if write:
            if evaluation is None or paradigm is None:
                raise ValueError('If writing results, evaluation and paradigm must be specified')
        self.write = write
        self.evaluation = '{0!r}'.format(evaluation)
        self.paradigm = '{0!r}'.format(paradigm)
        
            

    def _setup(self):
        '''
        Set up initial table structure 
        '''
        self.conn = sqlite3.connect(self.filepath)
        with self.conn as c:
            c.execute('''
            CREATE TABLE context(id INTEGER PRIMARY KEY,
                                 eval TEXT,
                                 pprocess_hash TEXT,
                                 paradigm_hash TEXT)''')
            c.execute('''
            CREATE TABLE datasets(code TEXT PRIMARY KEY,
                                  sr REAL,
                                  subjects INTEGER)''')
        self.conn.close()

    def add(self, pipeline_dict):
        '''
        Add data appropriately to database. Fail if parameters not already there
        pipeline_dict: dict of (pipeline hash, dict of information)
        '''
        pass

    def check_dataset(self, dataset):
        '''Check if dataset is already in dset table and add if not.'''
        with self.conn as c:
            reslist = c.execute(
                "SELECT 1 FROM datasets WHERE code=?", (dataset.code,)).fetchall()
            log.debug(reslist)
            if len(reslist) == 0:
                # add dataset
                log.info('Adding dataset {} to database...'.format(dataset.code))
                raw = dataset.get_data([1], False)[0][0][0]
                sr = raw.info['sfreq']
                c.execute('INSERT INTO datasets VALUES(?,?,?)', (dataset.code, sr, len(dataset.subject_list)))

            elif len(reslist) != 1:
                raise ValueError(
                    "Multiple entries for dataset {} in database??".format(dataset.code))

    def not_yet_computed(self, pipeline_dict, dataset, subj):
        '''
        Confirm that subject, dataset, pipeline combos are not yet in database.
        Returns pipeline dict with only new pipelines'''
        pass

    def to_dataframe(self):
        '''
        Given search criteria (TBD) return dataframe of results
        '''
        pass
