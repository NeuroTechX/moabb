"""
BNCI 2014-001 Motor imagery dataset.
"""

from moabb.datasets.base import BaseDataset
from moabb.datasets import download as dl

from mne import create_info, concatenate_raws
from mne.io import RawArray
from mne.channels import read_montage
from mne.utils import verbose
import numpy as np
import os

BNCI_URL = 'http://bnci-horizon-2020.eu/database/data-sets/'
BBCI_URL = 'http://doc.ml.tu-berlin.de/bbci/'


def data_path(url, path=None, force_update=False, update_path=None, verbose=None):
    return [dl.data_path(url, 'BNCI', path, force_update, update_path, verbose)]


@verbose
def load_data(subject, dataset='001-2014', path=None, force_update=False,

              update_path=None, base_url=BNCI_URL,
              verbose=None):  # noqa: D301
    """Get paths to local copies of a BNCI dataset files.

    This will fetch data for a given BNCI dataset. Report to the bnci website
    for a complete description of the experimental setup of each dataset.

    Parameters
    ----------
    subject : int
        The subject to load.
    dataset : string
        The bnci dataset name.
    path : None | str
        Location of where to look for the BNCI data storing location.
        If None, the environment variable or config parameter
        ``MNE_DATASETS_BNCI_PATH`` is used. If it doesn't exist, the
        "~/mne_data" directory is used. If the BNCI dataset
        is not found under the given path, the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the dataset even if a local copy exists.
    update_path : bool | None
        If True, set the MNE_DATASETS_BNCI_PATH in mne-python
        config to the given path. If None, the user is prompted.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raws : list
        List of raw instances for each non consecutive recording. Depending
        on the dataset it could be a BCI run or a different recording session.
    event_id: dict
        dictonary containing events and their code.
    """
    dataset_list = {'001-2014': _load_data_001_2014,
                    '002-2014': _load_data_002_2014,
                    '004-2014': _load_data_004_2014,
                    '008-2014': _load_data_008_2014,
                    '009-2014': _load_data_009_2014,
                    '001-2015': _load_data_001_2015,
                    '003-2015': _load_data_003_2015,
                    '004-2015': _load_data_004_2015,
                    '009-2015': _load_data_009_2015,
                    '010-2015': _load_data_010_2015,
                    '012-2015': _load_data_012_2015,
                    '013-2015': _load_data_013_2015}

    baseurl_list = {'001-2014': BNCI_URL,
                    '002-2014': BNCI_URL,
                    '001-2015': BNCI_URL,
                    '004-2014': BNCI_URL,
                    '008-2014': BNCI_URL,
                    '009-2014': BNCI_URL,
                    '003-2015': BNCI_URL,
                    '004-2015': BNCI_URL,
                    '009-2015': BBCI_URL,
                    '010-2015': BBCI_URL,
                    '012-2015': BBCI_URL,
                    '013-2015': BNCI_URL}

    if dataset not in dataset_list.keys():
        raise ValueError("Dataset '%s' is not a valid BNCI dataset ID. "
                         "Valid dataset are %s."
                         % (dataset, ", ".join(dataset_list.keys())))

    return dataset_list[dataset](subject, path, force_update, update_path,
                                 baseurl_list[dataset], verbose)


@verbose
def _load_data_001_2014(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 001-2014 dataset."""
    if (subject < 1) or (subject > 9):
        raise ValueError("Subject must be between 1 and 9. Got %d." % subject)

    ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3',
                'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
                'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz', 'EOG1', 'EOG2', 'EOG3']
    ch_types = ['eeg'] * 22 + ['eog'] * 3

    data_paths = []
    for r in ['T', 'E']:
        url = '{u}001-2014/A{s:02d}{r}.mat'.format(u=base_url, s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))

    return _convert_mi(data_paths, ch_names, ch_types)


@verbose
def _load_data_002_2014(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 002-2014 dataset."""
    if (subject < 1) or (subject > 14):
        raise ValueError("Subject must be between 1 and 14. Got %d." % subject)

    data_paths = []
    for r in ['T', 'E']:
        url = '{u}002-2014/S{s:02d}{r}.mat'.format(u=base_url, s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))

    return _convert_mi(data_paths, None, None)

@verbose
def _load_data_004_2014(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 004-2014 dataset."""
    if (subject < 1) or (subject > 9):
        raise ValueError("Subject must be between 1 and 9. Got %d." % subject)

    ch_names = ['C3', 'Cz', 'C4', 'EOG1', 'EOG2', 'EOG3']
    ch_types = ['eeg'] * 3 + ['eog'] * 3

    data_paths = []
    for r in ['T', 'E']:
        url = '{u}004-2014/B{s:02d}{r}.mat'.format(u=base_url, s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))


    return _convert_mi(data_paths, ch_names, ch_types)


@verbose
def _load_data_008_2014(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 008-2014 dataset."""
    if (subject < 1) or (subject > 8):
        raise ValueError("Subject must be between 1 and 8. Got %d." % subject)

    url = '{u}008-2014/A{s:02d}.mat'.format(u=base_url, s=subject)
    filename = data_path(url, path, force_update, update_path)[0]
    from scipy.io import loadmat

    run = loadmat(filename, struct_as_record=False, squeeze_me=True)['data']
    raw, event_id = _convert_run_p300_sl(run, verbose=verbose)

    return [raw], event_id


@verbose
def _load_data_009_2014(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 009-2014 dataset."""
    if (subject < 1) or (subject > 10):
        raise ValueError("Subject must be between 1 and 10. Got %d." % subject)

    # FIXME there is two type of speller, grid speller and geo-speller.
    # we load only grid speller data
    url = '{u}009-2014/A{s:02d}S.mat'.format(u=base_url, s=subject)
    filename = data_path(url, path, force_update, update_path)[0]
    from scipy.io import loadmat

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)['data']
    raws = []
    event_id = {}
    for run in data:
        raw, ev = _convert_run_p300_sl(run, verbose=verbose)
        raws.append(raw)
        event_id.update(ev)

    return raws, event_id


@verbose
def _load_data_001_2015(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 001-2015 dataset."""
    if (subject < 1) or (subject > 12):
        raise ValueError("Subject must be between 1 and 12. Got %d." % subject)

    if subject in [8, 9, 10, 11]:
        sessions = ['A', 'B', 'C']  # 3 sessions for those subjects
    else:
        sessions = ['A', 'B']

    data_paths = []
    for r in sessions:
        url = '{u}001-2015/S{s:02d}{r}.mat'.format(u=base_url, s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))

    ch_names = ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2',
                'C4', 'C6', 'CP3', 'CPz', 'CP4']
    ch_types = ['eeg'] * 13


    return _convert_mi(data_paths, ch_names, ch_types)

@verbose
def _load_data_003_2015(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 003-2015 dataset."""
    if (subject < 1) or (subject > 10):
        raise ValueError("Subject must be between 1 and 12. Got %d." % subject)

    url = '{u}003-2015/s{s:d}.mat'.format(u=base_url, s=subject)
    filename = data_path(url, path, force_update, update_path)[0]

    raws = list()
    event_id = {'Target': 2, 'Non-Target': 1}
    from scipy.io import loadmat

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    data = data['s%d' % subject]
    sfreq = 256.

    ch_names = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'Oz', 'PO8',
                'Target', 'Flash']

    ch_types = ['eeg'] * 8 + ['stim'] * 2
    montage = read_montage('standard_1005')

    info = create_info(ch_names=ch_names, ch_types=ch_types,
                       sfreq=sfreq, montage=montage)

    for run in [data.train, data.test]:
        # flash events on the channel 9
        flashs = run[9:10]
        ix_flash = flashs[0] > 0
        flashs[0, ix_flash] += 2  # add 2 to avoid overlapp on event id
        flash_code = np.unique(flashs[0, ix_flash])

        if len(flash_code) == 36:
            # char mode
            evd = {'Char%d' % ii: (ii + 2) for ii in range(1, 37)}
        else:
            # row / column mode
            evd = {'Col%d' % ii: (ii + 2) for ii in range(1, 7)}
            evd.update({'Row%d' % ii: (ii + 8) for ii in range(1, 7)})

        # target events are on channel 10
        targets = np.zeros_like(flashs)
        targets[0, ix_flash] = run[10, ix_flash] + 1

        eeg_data = np.r_[run[1:-2] * 1e-6, targets, flashs]
        raw = RawArray(data=eeg_data, info=info, verbose=verbose)
        raws.append(raw)
        event_id.update(evd)
    return raws, event_id


@verbose
def _load_data_004_2015(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 004-2015 dataset."""
    if (subject < 1) or (subject > 9):
        raise ValueError("Subject must be between 1 and 9. Got %d." % subject)

    subjects = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'L']

    url = '{u}004-2015/{s}.mat'.format(u=base_url, s=subjects[subject - 1])
    filename = data_path(url, path, force_update, update_path)[0]

    ch_names = ['AFz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz',
                'FC4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'CP3', 'CPz',
                'CP4', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4',
                'P6', 'P8', 'PO3', 'PO4', 'O1', 'O2']
    ch_types = ['eeg'] * 30

    return _convert_mi([filename], ch_names, ch_types)


@verbose
def _load_data_009_2015(subject, path=None, force_update=False,
                        update_path=None, base_url=BBCI_URL,
                        verbose=None):
    """Load data for 009-2015 dataset."""
    if (subject < 1) or (subject > 21):
        raise ValueError("Subject must be between 1 and 21. Got %d." % subject)

    subjects = ['fce', 'kw', 'faz', 'fcj', 'fcg', 'far', 'faw', 'fax', 'fcc',
                'fcm', 'fas', 'fch', 'fcd', 'fca', 'fcb', 'fau', 'fci', 'fav',
                'fat', 'fcl', 'fck']
    s = subjects[subject - 1]
    url = '{u}BNCIHorizon2020-AMUSE/AMUSE_VP{s}.mat'.format(u=base_url, s=s)
    filename = data_path(url, path, force_update, update_path)[0]

    ch_types = ['eeg'] * 60 + ['eog'] * 2

    return _convert_bbci(filename, ch_types, verbose=None)


@verbose
def _load_data_010_2015(subject, path=None, force_update=False,
                        update_path=None, base_url=BBCI_URL,
                        verbose=None):
    """Load data for 010-2015 dataset."""
    if (subject < 1) or (subject > 12):
        raise ValueError("Subject must be between 1 and 12. Got %d." % subject)

    subjects = ['fat', 'gcb', 'gcc', 'gcd', 'gce', 'gcf', 'gcg', 'gch', 'iay',
                'icn', 'icr', 'pia']

    s = subjects[subject - 1]
    url = '{u}BNCIHorizon2020-RSVP/RSVP_VP{s}.mat'.format(u=base_url, s=s)
    filename = data_path(url, path, force_update, update_path)[0]

    ch_types = ['eeg'] * 63

    return _convert_bbci(filename, ch_types, verbose=None)


@verbose
def _load_data_012_2015(subject, path=None, force_update=False,
                        update_path=None, base_url=BBCI_URL,
                        verbose=None):
    """Load data for 012-2015 dataset."""
    if (subject < 1) or (subject > 12):
        raise ValueError("Subject must be between 1 and 12. Got %d." % subject)

    subjects = ['nv', 'nw', 'nx', 'ny', 'nz', 'mg', 'oa', 'ob', 'oc', 'od',
                'ja', 'oe']

    s = subjects[subject - 1]
    url = '{u}BNCIHorizon2020-PASS2D/PASS2D_VP{s}.mat'.format(u=base_url, s=s)
    filename = data_path(url, path, force_update, update_path)[0]

    ch_types = ['eeg'] * 63

    return _convert_bbci(filename, ch_types, verbose=None)


@verbose
def _load_data_013_2015(subject, path=None, force_update=False,
                        update_path=None, base_url=BNCI_URL,
                        verbose=None):
    """Load data for 013-2015 dataset."""
    if (subject < 1) or (subject > 6):
        raise ValueError("Subject must be between 1 and 6. Got %d." % subject)

    data_paths = []
    for r in ['s1', 's2']:
        url = '{u}013-2015/Subject{s:02d}_{r}.mat'.format(u=base_url,
                                                          s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))

    raws = []
    event_id = {}
    from scipy.io import loadmat

    for filename in data_paths:
        data = loadmat(filename, struct_as_record=False, squeeze_me=True)
        for run in data['run']:
            raw, evd = _convert_run_epfl(run, verbose=verbose)
            raws.append(raw)
            event_id.update(evd)
    return raws, event_id

def _convert_mi(data_paths, ch_names, ch_types):
    '''
    Processes (Graz) motor imagery data from MAT files, returns list of recording sessions
    (each session corresponds to a separate placement of electrodes)
    '''
    from scipy.io import loadmat
    sessions = []
    event_id = {}
    for filename in data_paths:
        data = loadmat(filename, struct_as_record=False, squeeze_me=True)
        if type(data['data']) is np.ndarray:
            run_array = data['data']
        else:
            run_array = [data['data']]
        for run in run_array:
            raw, evd = _convert_run(run, ch_names, ch_types, None)
            raws.append(raw)
            event_id.update(evd)
        sessions.append(runs)
    # change labels to match rest
    standardize_keys(event_id)
    return sessions, event_id

def standardize_keys(d):
    master_list = [['both feet', 'feet'],
                   ['left hand', 'left_hand'],
                   ['right hand', 'right_hand'],
                   ['FEET', 'feet'],
                   ['HAND', 'right_hand'],
                   ['NAV', 'navigation'],
                   ['SUB', 'subtraction'],
                   ['WORD', 'word_ass']]
    for old, new in master_list:
        if old in d.keys():
            d[new] = d.pop(old)


@verbose
def _convert_run(run, ch_names=None, ch_types=None, verbose=None):
    """Convert one run to raw."""
    # parse eeg data
    event_id = {}
    n_chan = run.X.shape[1]
    montage = read_montage('standard_1005')
    eeg_data = 1e-6 * run.X
    sfreq = run.fs

    if not ch_names:
        ch_names = ['EEG%d' % ch for ch in range(1, n_chan + 1)]
        montage = None  # no montage

    if not ch_types:
        ch_types = ['eeg'] * n_chan

    trigger = np.zeros((len(eeg_data), 1))
    # some runs does not contains trials i.e baseline runs
    if len(run.trial) > 0:
        trigger[run.trial - 1, 0] = run.y

    eeg_data = np.c_[eeg_data, trigger]
    ch_names = ch_names + ['stim']
    ch_types = ch_types + ['stim']
    event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}
    info = create_info(ch_names=ch_names, ch_types=ch_types,
                       sfreq=sfreq, montage=montage)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    return raw, event_id


@verbose
def _convert_run_p300_sl(run, verbose=None):
    """Convert one p300 run from santa lucia file format."""
    montage = read_montage('standard_1005')
    eeg_data = 1e-6 * run.X
    sfreq = 256
    ch_names = list(run.channels) + ['Target stim', 'Flash stim']
    ch_types = ['eeg'] * len(run.channels) + ['stim'] * 2

    flash_stim = run.y_stim
    flash_stim[flash_stim > 0] += 2
    eeg_data = np.c_[eeg_data, run.y, flash_stim]
    event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}
    event_id.update({ev: (ii + 3) for ii, ev in enumerate(run.classes_stim)})
    info = create_info(ch_names=ch_names, ch_types=ch_types,
                       sfreq=sfreq, montage=montage)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    return raw, event_id


@verbose
def _convert_bbci(filename, ch_types, verbose=None):
    """Convert one file in bbci format."""
    raws = []
    event_id = {}
    from scipy.io import loadmat

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    for run in data['data']:
        raw, evd = _convert_run_bbci(run, ch_types, verbose)
        raws.append(raw)
        event_id.update(evd)

    return raws, event_id


@verbose
def _convert_run_bbci(run, ch_types, verbose=None):
    """Convert one run to raw."""
    # parse eeg data
    montage = read_montage('standard_1005')
    eeg_data = 1e-6 * run.X
    sfreq = run.fs

    ch_names = list(run.channels)

    trigger = np.zeros((len(eeg_data), 1))
    trigger[run.trial - 1, 0] = run.y
    event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}

    flash = np.zeros((len(eeg_data), 1))
    flash[run.trial - 1, 0] = run.y_stim + 2
    ev_fl = {'Stim%d' % (stim): (stim + 2) for stim in np.unique(run.y_stim)}
    event_id.update(ev_fl)

    eeg_data = np.c_[eeg_data, trigger, flash]
    ch_names = ch_names + ['Target', 'Flash']
    ch_types = ch_types + ['stim'] * 2

    info = create_info(ch_names=ch_names, ch_types=ch_types,
                       sfreq=sfreq, montage=montage)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    return raw, event_id


@verbose
def _convert_run_epfl(run, verbose=None):
    """Convert one run to raw."""
    # parse eeg data
    event_id = {}

    montage = read_montage('standard_1005')
    eeg_data = 1e-6 * run.eeg
    sfreq = run.header.SampleRate

    ch_names = list(run.header.Label[:-1])
    ch_types = ['eeg'] * len(ch_names)

    trigger = np.zeros((len(eeg_data), 1))

    for ii, typ in enumerate(run.header.EVENT.TYP):
        if typ in [6, 9]:  # Error
            trigger[run.header.EVENT.POS[ii] - 1, 0] = 2
        elif typ in [5, 10]:  # correct
            trigger[run.header.EVENT.POS[ii] - 1, 0] = 1

    eeg_data = np.c_[eeg_data, trigger]
    ch_names = ch_names + ['stim']
    ch_types = ch_types + ['stim']
    event_id = {'correct': 1, 'error': 2}

    info = create_info(ch_names=ch_names, ch_types=ch_types,
                       sfreq=sfreq, montage=montage)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    return raw, event_id


class MNEBNCI(BaseDataset):
    """Base BNCI dataset"""

    def _get_single_subject_data(self, subject, stack_sessions):
        """return data for a single subject"""
<<<<<<< HEAD
        raw_files, event_id = load_data(subject=subject, dataset=self.code,
                                        verbose=False)
        print(event_id)
        return raw_files
=======
        sessions = load_data(subject=subject, dataset=self.code,
                                        verbose=False)[0]
        if stack_sessions:
            new_sessions = [[run for session in sessions for run in session]]
        else:
            new_sessions = [sessions]
        return new_sessions
>>>>>>> master


class BNCI2014001(MNEBNCI):
    """BNCI 2014-001 Motor Imagery dataset"""

    def __init__(self, tmin=3.5, tmax=5.5):
        super().__init__(subjects=range(1,10),
                         sessions_per_subject=2,
                         events=dict(zip(['left_hand','right_hand','feet','tongue'],
                                  [1,2,3,4])),
                         code='001-2014',
                         interval=[tmin, tmax],
                         paradigm='imagery'
        )


class BNCI2014002(MNEBNCI):
    """BNCI 2014-002 Motor Imagery dataset"""

    def __init__(self, tmin=3.5, tmax=5.5):
        super().__init__(range(1,15),
                         1,
                         dict(zip(['right_hand','feet'],
                                  [1,2])),
                         '002-2014',
                         [tmin, tmax],
                         'imagery'
        )


class BNCI2014004(MNEBNCI):
    """BNCI 2014-004 Motor Imagery dataset"""

    def __init__(self, tmin=4.5, tmax=6.5):
        super().__init__(range(1,10),
                         2,
                         dict(zip(['left_hand','right_hand'],
                                  [1,2])),
                         '004-2014',
                         [tmin, tmax],
                         'imagery'
        )

class BNCI2015001(MNEBNCI):
    """BNCI 2015-001 Motor Imagery dataset"""

    def __init__(self, tmin=4, tmax=7.5):
        super().__init__(range(1,13),
                         2,
                         dict(zip(['right_hand','feet'],
                                  [1,2])),
                         '001-2015',
                         [tmin, tmax],
                         'imagery'
        )

class BNCI2015004(MNEBNCI):
    """BNCI 2015-004 Motor Imagery dataset"""

    def __init__(self, tmin=4.25, tmax=10):
        super().__init__(range(1,10),
                         2,
                         dict(right_hand=4, feet=5,
                                 navigation=3, subtraction=2, word_ass=1),
                         '004-2015',
                         [tmin, tmax],
                         'imagery'
        )
