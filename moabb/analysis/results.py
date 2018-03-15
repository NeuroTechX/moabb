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
        self.mod_dir = os.path.dirname(
            os.path.abspath(inspect.getsourcefile(moabb)))
        self.filepath = os.path.join(self.mod_dir, 'results',
                                     paradigm_class.__name__,
                                     evaluation_class.__name__,
                                     'results{}.hdf5'.format('_' + suffix))

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
    dataframes based on queries 

    The integrity of this structure depends on the pipelines and the paradigms
    having unambiguous __repr__ methods.

    '''

    def __init__(self, write=False, evaluation=None, _debug=False):
        """
        Initialize class. If database does not exist, create it.
        write: bool, do you want to write values (if yes, then need eval and paradigm)
        evaluation: BaseEvaluation child
        """
        # first ensure that the database exists
        import moabb
        self.mod_dir = os.path.dirname(
            os.path.abspath(inspect.getsourcefile(moabb)))
        if not _debug:
            self.filepath = os.path.join(self.mod_dir, 'results', 'results.db')
        else:
            self.filepath = os.path.join(
                self.mod_dir, 'results', 'results_test.db')
        if not os.path.isfile(self.filepath):
            os.makedirs(os.path.join(self.mod_dir, 'results'), exist_ok=True)
            self._setup()
        self.conn = sqlite3.connect(self.filepath)

        self.write = write
        if write:
            if evaluation is None:
                raise ValueError(
                    'If writing results, evaluation must be specified')
            self.evaluation = get_digest(evaluation)
            self.paradigm = get_digest(evaluation.paradigm)
            self.human_paradigm = evaluation.paradigm.human_paradigm
            self.context_id = self.check_context(
                self.evaluation, self.paradigm, self.human_paradigm)

    def check_context(self, eval_hash, par_hash, human_par_hash):
        '''
        Check if evaluation, paradigm, and human paradigm combination exists already.
        If so, return the generated ID. If not, insert and return the appropriate ID
        '''
        with self.conn as c:
            id_list = c.execute("SELECT id FROM context WHERE eval = ? AND pprocess_hash = ? AND paradigm_hash = ?", (
                eval_hash, par_hash, human_par_hash)).fetchall()
            log.debug(id_list)
            context_id = None
            if len(id_list) == 0:
                log.info('Adding new context to database')
                cur = c.cursor()
                cur.execute("INSERT INTO context(eval, pprocess_hash, paradigm_hash) VALUES(?,?,?)",
                            (eval_hash, par_hash, human_par_hash))
                context_id = cur.lastrowid
            else:
                context_id = id_list[0][0]
            return context_id

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
            c.execute('''
            CREATE TABLE scores(score FLOAT,
                                subj INT,
                                dataset TEXT,
                                time FLOAT,
                                n_samples INT,
                                n_channels INT, 
                                pipeline TEXT,
                                context INT)
            ''')
        self.conn.close()

    def in_table(self, pipe, code, subj):
        with self.conn as c:
            results = c.execute("SELECT * FROM scores WHERE pipeline = ? AND dataset = ? AND context = ? AND subj = ?",
                                (pipe, code, subj, self.context_id)).fetchall()
            return (len(results) > 0)
    
    def add(self, pipeline_dict):
        '''
        Add data appropriately to database. Fail if parameters not already there
        pipeline_dict: dict of (pipeline hash, dict of information)
        '''
        assert self.write, "Writing not enabled for this ResultsDB object"

        def _insert(pipe, res_list):
            with self.conn as c:
                c.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?,?,?)",
                              [(r['score'],
                                r['id'],
                                r['dataset'].code,
                                r['time'],
                                r['n_samples'],
                                r['n_channels'],
                                pipe,
                                self.context_id) for r in res_list])

        def _update(pipe, res_list):
            with self.conn as c:
                c.execute("DELETE FROM scores WHERE pipeline = ? AND dataset = ? AND context = ?",
                          (pipe, res_list[0]['dataset'].code, self.context_id))
            _insert(pipe, res_list)

        def to_list(res):
            if type(res) is dict:
                return [res]
            elif type(res) is not list:
                raise ValueError("Results are given as neither dict nor list"
                                 "but {}".format(type(res).__name__))
            else:
                return res

        for pipe, res_list in pipeline_dict.items():
            res_list = to_list(res_list)
            if self.in_table(pipe, res_list[0]['dataset'].code, res_list[0]['id']):
                _update(pipe, res_list)
            else:
                _insert(pipe, res_list)

    def check_dataset(self, dataset):
        '''Check if dataset is already in dset table and add if not.'''
        with self.conn as c:
            reslist = c.execute(
                "SELECT * FROM datasets WHERE code=?", (dataset.code,)).fetchall()
            log.debug(reslist)
            if len(reslist) == 0:
                # add dataset
                log.info('Adding dataset {} to database...'.format(dataset.code))
                raw = dataset.get_data([1], False)[0][0][0]
                sr = raw.info['sfreq']
                c.execute('INSERT INTO datasets VALUES(?,?,?)',
                          (dataset.code, sr, len(dataset.subject_list)))

            elif len(reslist) != 1:
                raise ValueError(
                    "Multiple entries for dataset {} in database??".format(dataset.code))

    def not_yet_computed(self, pipeline_dict, dataset, subj):
        '''
        Confirm that subject, dataset, pipeline combos are not yet in database.
        Returns pipeline dict with only new pipelines'''
        assert self.write, "Writing not enabled for this ResultsDB object"
        out = {}
        with self.conn as c:
            for pipe in pipeline_dict.keys():
                if not self.in_table(pipe, dataset.code, subj):
                    out[pipe] = pipeline_dict[pipe]
        return out
                

    def to_dataframe(self):
        '''
        Given search criteria (TBD) return dataframe of results
        '''
        if not self.write:
            raise NotImplementedError('Cross-preprocessing/cross-whatever search not yet implemented')
        return pd.read_sql_query('SELECT * FROM scores WHERE context = {:d}'.format(self.context_id),
                                 self.conn)
