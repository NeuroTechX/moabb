"""
BNCI 2014-001 Motor imagery dataset.
"""

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.utils import verbose
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


BNCI_URL = "http://bnci-horizon-2020.eu/database/data-sets/"
BBCI_URL = "http://doc.ml.tu-berlin.de/bbci/"


def data_path(url, path=None, force_update=False, update_path=None, verbose=None):
    return [dl.data_dl(url, "BNCI", path, force_update, verbose)]


@verbose
def load_data(
    subject,
    dataset="001-2014",
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    verbose=None,
):  # noqa: D301
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
    dataset_list = {
        "001-2014": _load_data_001_2014,
        "002-2014": _load_data_002_2014,
        "004-2014": _load_data_004_2014,
        "008-2014": _load_data_008_2014,
        "009-2014": _load_data_009_2014,
        "001-2015": _load_data_001_2015,
        "003-2015": _load_data_003_2015,
        "004-2015": _load_data_004_2015,
        "009-2015": _load_data_009_2015,
        "010-2015": _load_data_010_2015,
        "012-2015": _load_data_012_2015,
        "013-2015": _load_data_013_2015,
    }

    baseurl_list = {
        "001-2014": BNCI_URL,
        "002-2014": BNCI_URL,
        "001-2015": BNCI_URL,
        "004-2014": BNCI_URL,
        "008-2014": BNCI_URL,
        "009-2014": BNCI_URL,
        "003-2015": BNCI_URL,
        "004-2015": BNCI_URL,
        "009-2015": BBCI_URL,
        "010-2015": BBCI_URL,
        "012-2015": BBCI_URL,
        "013-2015": BNCI_URL,
    }

    if dataset not in dataset_list.keys():
        raise ValueError(
            "Dataset '%s' is not a valid BNCI dataset ID. "
            "Valid dataset are %s." % (dataset, ", ".join(dataset_list.keys()))
        )

    return dataset_list[dataset](
        subject, path, force_update, update_path, baseurl_list[dataset], verbose
    )


@verbose
def _load_data_001_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    verbose=None,
):
    """Load data for 001-2014 dataset."""
    if (subject < 1) or (subject > 9):
        raise ValueError("Subject must be between 1 and 9. Got %d." % subject)

    # fmt: off
    ch_names = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
        "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
        "EOG1", "EOG2", "EOG3",
    ]
    # fmt: on
    ch_types = ["eeg"] * 22 + ["eog"] * 3

    sessions = {}
    for r in ["T", "E"]:
        url = "{u}001-2014/A{s:02d}{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)
        runs, ev = _convert_mi(filename[0], ch_names, ch_types)
        # FIXME: deal with run with no event (1:3) and name them
        sessions["session_%s" % r] = {"run_%d" % ii: run for ii, run in enumerate(runs)}
    return sessions


@verbose
def _load_data_002_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    verbose=None,
):
    """Load data for 002-2014 dataset."""
    if (subject < 1) or (subject > 14):
        raise ValueError("Subject must be between 1 and 14. Got %d." % subject)

    runs = []
    for r in ["T", "E"]:
        url = "{u}002-2014/S{s:02d}{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)[0]

        # FIXME: electrode position and name are not provided directly.
        raws, _ = _convert_mi(filename, None, ["eeg"] * 15)
        runs.extend(raws)

    runs = {"run_%d" % ii: run for ii, run in enumerate(runs)}
    return {"session_0": runs}


@verbose
def _load_data_004_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    verbose=None,
):
    """Load data for 004-2014 dataset."""
    if (subject < 1) or (subject > 9):
        raise ValueError("Subject must be between 1 and 9. Got %d." % subject)

    ch_names = ["C3", "Cz", "C4", "EOG1", "EOG2", "EOG3"]
    ch_types = ["eeg"] * 3 + ["eog"] * 3

    sessions = []
    for r in ["T", "E"]:
        url = "{u}004-2014/B{s:02d}{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)[0]
        raws, _ = _convert_mi(filename, ch_names, ch_types)
        sessions.extend(raws)

    sessions = {"session_%d" % ii: {"run_0": run} for ii, run in enumerate(sessions)}
    return sessions


@verbose
def _load_data_008_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    verbose=None,
):
    """Load data for 008-2014 dataset."""
    if (subject < 1) or (subject > 8):
        raise ValueError("Subject must be between 1 and 8. Got %d." % subject)

    url = "{u}008-2014/A{s:02d}.mat".format(u=base_url, s=subject)
    filename = data_path(url, path, force_update, update_path)[0]

    run = loadmat(filename, struct_as_record=False, squeeze_me=True)["data"]
    raw, event_id = _convert_run_p300_sl(run, verbose=verbose)

    sessions = {"session_0": {"run_0": raw}}

    return sessions


@verbose
def _load_data_009_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    verbose=None,
):
    """Load data for 009-2014 dataset."""
    if (subject < 1) or (subject > 10):
        raise ValueError("Subject must be between 1 and 10. Got %d." % subject)

    # FIXME there is two type of speller, grid speller and geo-speller.
    # we load only grid speller data
    url = "{u}009-2014/A{s:02d}S.mat".format(u=base_url, s=subject)
    filename = data_path(url, path, force_update, update_path)[0]

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)["data"]
    sess = []
    event_id = {}
    for run in data:
        raw, ev = _convert_run_p300_sl(run, verbose=verbose)
        sess.append(raw)
        event_id.update(ev)

    sessions = {}
    for i, sessi in enumerate(sess):
        sessions["session_" + str(i)] = {"run_0": sessi}

    return sessions


@verbose
def _load_data_001_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    verbose=None,
):
    """Load data for 001-2015 dataset."""
    if (subject < 1) or (subject > 12):
        raise ValueError("Subject must be between 1 and 12. Got %d." % subject)

    if subject in [8, 9, 10, 11]:
        ses = ["A", "B", "C"]  # 3 sessions for those subjects
    else:
        ses = ["A", "B"]

    # fmt: off
    ch_names = [
        "FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz",
        "C2", "C4", "C6", "CP3", "CPz", "CP4",
    ]
    # fmt: on
    ch_types = ["eeg"] * 13

    sessions = {}
    for r in ses:
        url = "{u}001-2015/S{s:02d}{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)
        runs, ev = _convert_mi(filename[0], ch_names, ch_types)
        sessions["session_%s" % r] = {"run_%d" % ii: run for ii, run in enumerate(runs)}
    return sessions


@verbose
def _load_data_003_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    verbose=None,
):
    """Load data for 003-2015 dataset."""
    if (subject < 1) or (subject > 10):
        raise ValueError("Subject must be between 1 and 12. Got %d." % subject)

    url = "{u}003-2015/s{s:d}.mat".format(u=base_url, s=subject)
    filename = data_path(url, path, force_update, update_path)[0]

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    data = data["s%d" % subject]
    sfreq = 256.0

    ch_names = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "Oz", "PO8", "Target", "Flash"]

    ch_types = ["eeg"] * 8 + ["stim"] * 2
    montage = make_standard_montage("standard_1005")

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    sessions = {}
    sessions["session_0"] = {}
    for ri, run in enumerate([data.train, data.test]):
        # flash events on the channel 9
        flashs = run[9:10]
        ix_flash = flashs[0] > 0
        flashs[0, ix_flash] += 2  # add 2 to avoid overlapp on event id
        flash_code = np.unique(flashs[0, ix_flash])

        if len(flash_code) == 36:
            # char mode
            evd = {"Char%d" % ii: (ii + 2) for ii in range(1, 37)}
        else:
            # row / column mode
            evd = {"Col%d" % ii: (ii + 2) for ii in range(1, 7)}
            evd.update({"Row%d" % ii: (ii + 8) for ii in range(1, 7)})

        # target events are on channel 10
        targets = np.zeros_like(flashs)
        targets[0, ix_flash] = run[10, ix_flash] + 1

        eeg_data = np.r_[run[1:-2] * 1e-6, targets, flashs]
        raw = RawArray(data=eeg_data, info=info, verbose=verbose)
        raw.set_montage(montage)
        sessions["session_0"]["run_" + str(ri)] = raw

    return sessions


@verbose
def _load_data_004_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    verbose=None,
):
    """Load data for 004-2015 dataset."""
    if (subject < 1) or (subject > 9):
        raise ValueError("Subject must be between 1 and 9. Got %d." % subject)

    subjects = ["A", "C", "D", "E", "F", "G", "H", "J", "L"]

    url = "{u}004-2015/{s}.mat".format(u=base_url, s=subjects[subject - 1])
    filename = data_path(url, path, force_update, update_path)[0]

    # fmt: off
    ch_names = [
        "AFz", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T3", "C3",
        "Cz", "C4", "T4", "CP3", "CPz", "CP4", "P7", "P5", "P3", "P1", "Pz",
        "P2", "P4", "P6", "P8", "PO3", "PO4", "O1", "O2",
    ]
    # fmt: on
    ch_types = ["eeg"] * 30
    raws, ev = _convert_mi(filename, ch_names, ch_types)
    sessions = {"session_%d" % ii: {"run_0": run} for ii, run in enumerate(raws)}
    return sessions


@verbose
def _load_data_009_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    verbose=None,
):
    """Load data for 009-2015 dataset."""
    if (subject < 1) or (subject > 21):
        raise ValueError("Subject must be between 1 and 21. Got %d." % subject)

    # fmt: off
    subjects = [
        "fce", "kw", "faz", "fcj", "fcg", "far", "faw", "fax", "fcc", "fcm", "fas",
        "fch", "fcd", "fca", "fcb", "fau", "fci", "fav", "fat", "fcl", "fck",
    ]
    # fmt: on
    s = subjects[subject - 1]
    url = "{u}BNCIHorizon2020-AMUSE/AMUSE_VP{s}.mat".format(u=base_url, s=s)
    filename = data_path(url, path, force_update, update_path)[0]

    ch_types = ["eeg"] * 60 + ["eog"] * 2

    return _convert_bbci(filename, ch_types, verbose=None)


@verbose
def _load_data_010_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    verbose=None,
):
    """Load data for 010-2015 dataset."""
    if (subject < 1) or (subject > 12):
        raise ValueError("Subject must be between 1 and 12. Got %d." % subject)

    # fmt: off
    subjects = [
        "fat", "gcb", "gcc", "gcd", "gce", "gcf",
        "gcg", "gch", "iay", "icn", "icr", "pia",
    ]
    # fmt: on

    s = subjects[subject - 1]
    url = "{u}BNCIHorizon2020-RSVP/RSVP_VP{s}.mat".format(u=base_url, s=s)
    filename = data_path(url, path, force_update, update_path)[0]

    ch_types = ["eeg"] * 63

    return _convert_bbci(filename, ch_types, verbose=None)


@verbose
def _load_data_012_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    verbose=None,
):
    """Load data for 012-2015 dataset."""
    if (subject < 1) or (subject > 12):
        raise ValueError("Subject must be between 1 and 12. Got %d." % subject)

    subjects = ["nv", "nw", "nx", "ny", "nz", "mg", "oa", "ob", "oc", "od", "ja", "oe"]

    s = subjects[subject - 1]
    url = "{u}BNCIHorizon2020-PASS2D/PASS2D_VP{s}.mat".format(u=base_url, s=s)
    filename = data_path(url, path, force_update, update_path)[0]

    ch_types = ["eeg"] * 63

    return _convert_bbci(filename, ch_types, verbose=None)


@verbose
def _load_data_013_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    verbose=None,
):
    """Load data for 013-2015 dataset."""
    if (subject < 1) or (subject > 6):
        raise ValueError("Subject must be between 1 and 6. Got %d." % subject)

    data_paths = []
    for r in ["s1", "s2"]:
        url = "{u}013-2015/Subject{s:02d}_{r}.mat".format(u=base_url, s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))

    raws = []
    event_id = {}

    for filename in data_paths:
        data = loadmat(filename, struct_as_record=False, squeeze_me=True)
        for run in data["run"]:
            raw, evd = _convert_run_epfl(run, verbose=verbose)
            raws.append(raw)
            event_id.update(evd)
    return raws, event_id


def _convert_mi(filename, ch_names, ch_types):
    """
    Processes (Graz) motor imagery data from MAT files, returns list of
    recording runs.
    """
    runs = []
    event_id = {}
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)

    if isinstance(data["data"], np.ndarray):
        run_array = data["data"]
    else:
        run_array = [data["data"]]

    for run in run_array:
        raw, evd = _convert_run(run, ch_names, ch_types, None)
        if raw is None:
            continue
        runs.append(raw)
        event_id.update(evd)
    # change labels to match rest
    standardize_keys(event_id)
    return runs, event_id


def standardize_keys(d):
    master_list = [
        ["both feet", "feet"],
        ["left hand", "left_hand"],
        ["right hand", "right_hand"],
        ["FEET", "feet"],
        ["HAND", "right_hand"],
        ["NAV", "navigation"],
        ["SUB", "subtraction"],
        ["WORD", "word_ass"],
    ]
    for old, new in master_list:
        if old in d.keys():
            d[new] = d.pop(old)


@verbose
def _convert_run(run, ch_names=None, ch_types=None, verbose=None):
    """Convert one run to raw."""
    # parse eeg data
    event_id = {}
    n_chan = run.X.shape[1]
    montage = make_standard_montage("standard_1005")
    eeg_data = 1e-6 * run.X
    sfreq = run.fs

    if not ch_names:
        ch_names = ["EEG%d" % ch for ch in range(1, n_chan + 1)]
        montage = None  # no montage

    if not ch_types:
        ch_types = ["eeg"] * n_chan

    trigger = np.zeros((len(eeg_data), 1))
    # some runs does not contains trials i.e baseline runs
    if len(run.trial) > 0:
        trigger[run.trial - 1, 0] = run.y
    else:
        return None, None

    eeg_data = np.c_[eeg_data, trigger]
    ch_names = ch_names + ["stim"]
    ch_types = ch_types + ["stim"]
    event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    raw.set_montage(montage)
    return raw, event_id


@verbose
def _convert_run_p300_sl(run, verbose=None):
    """Convert one p300 run from santa lucia file format."""
    montage = make_standard_montage("standard_1005")
    eeg_data = 1e-6 * run.X
    sfreq = 256
    ch_names = list(run.channels) + ["Target stim", "Flash stim"]
    ch_types = ["eeg"] * len(run.channels) + ["stim"] * 2

    flash_stim = run.y_stim
    flash_stim[flash_stim > 0] += 2
    eeg_data = np.c_[eeg_data, run.y, flash_stim]
    event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}
    event_id.update({ev: (ii + 3) for ii, ev in enumerate(run.classes_stim)})
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    raw.set_montage(montage)
    return raw, event_id


@verbose
def _convert_bbci(filename, ch_types, verbose=None):
    """Convert one file in bbci format."""
    raws = []
    event_id = {}

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    for run in data["data"]:
        raw, evd = _convert_run_bbci(run, ch_types, verbose)
        raws.append(raw)
        event_id.update(evd)

    return raws, event_id


@verbose
def _convert_run_bbci(run, ch_types, verbose=None):
    """Convert one run to raw."""
    # parse eeg data
    montage = make_standard_montage("standard_1005")
    eeg_data = 1e-6 * run.X
    sfreq = run.fs

    ch_names = list(run.channels)

    trigger = np.zeros((len(eeg_data), 1))
    trigger[run.trial - 1, 0] = run.y
    event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}

    flash = np.zeros((len(eeg_data), 1))
    flash[run.trial - 1, 0] = run.y_stim + 2
    ev_fl = {"Stim%d" % (stim): (stim + 2) for stim in np.unique(run.y_stim)}
    event_id.update(ev_fl)

    eeg_data = np.c_[eeg_data, trigger, flash]
    ch_names = ch_names + ["Target", "Flash"]
    ch_types = ch_types + ["stim"] * 2

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    raw.set_montage(montage)
    return raw, event_id


@verbose
def _convert_run_epfl(run, verbose=None):
    """Convert one run to raw."""
    # parse eeg data
    event_id = {}

    montage = make_standard_montage("standard_1005")
    eeg_data = 1e-6 * run.eeg
    sfreq = run.header.SampleRate

    ch_names = list(run.header.Label[:-1])
    ch_types = ["eeg"] * len(ch_names)

    trigger = np.zeros((len(eeg_data), 1))

    for ii, typ in enumerate(run.header.EVENT.TYP):
        if typ in [6, 9]:  # Error
            trigger[run.header.EVENT.POS[ii] - 1, 0] = 2
        elif typ in [5, 10]:  # correct
            trigger[run.header.EVENT.POS[ii] - 1, 0] = 1

    eeg_data = np.c_[eeg_data, trigger]
    ch_names = ch_names + ["stim"]
    ch_types = ch_types + ["stim"]
    event_id = {"correct": 1, "error": 2}

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    raw.set_montage(montage)
    return raw, event_id


class MNEBNCI(BaseDataset):
    """Base BNCI dataset"""

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        sessions = load_data(subject=subject, dataset=self.code, verbose=False)
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return load_data(
            subject=subject,
            dataset=self.code,
            verbose=verbose,
            update_path=update_path,
            path=path,
            force_update=force_update,
        )


class BNCI2014001(MNEBNCI):
    """BNCI 2014-001 Motor Imagery dataset.

    Dataset IIa from BCI Competition 4 [1]_.

    **Dataset Description**

    This data set consists of EEG data from 9 subjects.  The cue-based BCI
    paradigm consisted of four different motor imagery tasks, namely the imag-
    ination of movement of the left hand (class 1), right hand (class 2), both
    feet (class 3), and tongue (class 4).  Two sessions on different days were
    recorded for each subject.  Each session is comprised of 6 runs separated
    by short breaks.  One run consists of 48 trials (12 for each of the four
    possible classes), yielding a total of 288 trials per session.

    The subjects were sitting in a comfortable armchair in front of a computer
    screen.  At the beginning of a trial ( t = 0 s), a fixation cross appeared
    on the black screen.  In addition, a short acoustic warning tone was
    presented.  After two seconds ( t = 2 s), a cue in the form of an arrow
    pointing either to the left, right, down or up (corresponding to one of the
    four classes left hand, right hand, foot or tongue) appeared and stayed on
    the screen for 1.25 s.  This prompted the subjects to perform the desired
    motor imagery task.  No feedback was provided.  The subjects were ask to
    carry out the motor imagery task until the fixation cross disappeared from
    the screen at t = 6 s.

    Twenty-two Ag/AgCl electrodes (with inter-electrode distances of 3.5 cm)
    were used to record the EEG; the montage is shown in Figure 3 left.  All
    signals were recorded monopolarly with the left mastoid serving as
    reference and the right mastoid as ground. The signals were sampled with.
    250 Hz and bandpass-filtered between 0.5 Hz and 100 Hz. The sensitivity of
    the amplifier was set to 100 μV . An additional 50 Hz notch filter was
    enabled to suppress line noise

    References
    ----------

    .. [1] Tangermann, M., Müller, K.R., Aertsen, A., Birbaumer, N., Braun, C.,
           Brunner, C., Leeb, R., Mehring, C., Miller, K.J., Mueller-Putz, G.
           and Nolte, G., 2012. Review of the BCI competition IV.
           Frontiers in neuroscience, 6, p.55.
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=2,
            events={"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4},
            code="001-2014",
            interval=[2, 6],
            paradigm="imagery",
            doi="10.3389/fnins.2012.00055",
        )


class BNCI2014002(MNEBNCI):
    """BNCI 2014-002 Motor Imagery dataset.

    Motor Imagery Dataset from [1]_.

    **Dataset description**

    The session consisted of eight runs, five of them for training and three
    with feedback for validation.  One run was composed of 20 trials.  Taken
    together, we recorded 50 trials per class for training and 30 trials per
    class for validation.  Participants had the task of performing sustained (5
    seconds) kinaesthetic motor imagery (MI) of the right hand and of the feet
    each as instructed by the cue. At 0 s, a white colored cross appeared on
    screen, 2 s later a beep sounded to catch the participant’s attention. The
    cue was displayed from 3 s to 4 s. Participants were instructed to start
    with MI as soon as they recognized the cue and to perform the indicated MI
    until the cross disappeared at 8 s. A rest period with a random length
    between 2 s and 3 s was presented between trials. Participants did not
    receive feedback during training.  Feedback was presented in form of a
    white
    coloured bar-graph.  The length of the bar-graph reflected the amount of
    correct classifications over the last second.  EEG was measured with a
    biosignal amplifier and active Ag/AgCl electrodes (g.USBamp, g.LADYbird,
    Guger Technologies OG, Schiedlberg, Austria) at a sampling rate of 512 Hz.
    The electrodes placement was designed for obtaining three Laplacian
    derivations.  Center electrodes at positions C3, Cz, and C4 and four
    additional electrodes around each center electrode with a distance of 2.5
    cm, 15 electrodes total.  The reference electrode was mounted on the left
    mastoid and the ground electrode on the right mastoid.  The 13 participants
    were aged between 20 and 30 years, 8 naive to the task, and had no known
    medical or neurological diseases.

    References
    -----------

    .. [1] Steyrl, D., Scherer, R., Faller, J. and Müller-Putz, G.R., 2016.
           Random forests in non-invasive sensorimotor rhythm brain-computer
           interfaces: a practical and convenient non-linear classifier.
           Biomedical Engineering/Biomedizinische Technik, 61(1), pp.77-86.

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 15)),
            sessions_per_subject=1,
            events={"right_hand": 1, "feet": 2},
            code="002-2014",
            interval=[3, 8],
            paradigm="imagery",
            doi="10.1515/bmt-2014-0117",
        )


class BNCI2014004(MNEBNCI):
    """BNCI 2014-004 Motor Imagery dataset.

    Dataset B from BCI Competition 2008.

    **Dataset description**

    This data set consists of EEG data from 9 subjects of a study published in
    [1]_. The subjects were right-handed, had normal or corrected-to-normal
    vision and were paid for participating in the experiments.
    All volunteers were sitting in an armchair, watching a flat screen monitor
    placed approximately 1 m away at eye level. For each subject 5 sessions
    are provided, whereby the first two sessions contain training data without
    feedback (screening), and the last three sessions were recorded with
    feedback.

    Three bipolar recordings (C3, Cz, and C4) were recorded with a sampling
    frequency of 250 Hz.They were bandpass- filtered between 0.5 Hz and 100 Hz,
    and a notch filter at 50 Hz was enabled.  The placement of the three
    bipolar recordings (large or small distances, more anterior or posterior)
    were slightly different for each subject (for more details see [1]).
    The electrode position Fz served as EEG ground. In addition to the EEG
    channels, the electrooculogram (EOG) was recorded with three monopolar
    electrodes.

    The cue-based screening paradigm consisted of two classes,
    namely the motor imagery (MI) of left hand (class 1) and right hand
    (class 2).
    Each subject participated in two screening sessions without feedback
    recorded on two different days within two weeks.
    Each session consisted of six runs with ten trials each and two classes of
    imagery.  This resulted in 20 trials per run and 120 trials per session.
    Data of 120 repetitions of each MI class were available for each person in
    total.  Prior to the first motor im- agery training the subject executed
    and imagined different movements for each body part and selected the one
    which they could imagine best (e. g., squeezing a ball or pulling a brake).

    Each trial started with a fixation cross and an additional short acoustic
    warning tone (1 kHz, 70 ms).  Some seconds later a visual cue was presented
    for 1.25 seconds.  Afterwards the subjects had to imagine the corresponding
    hand movement over a period of 4 seconds.  Each trial was followed by a
    short break of at least 1.5 seconds.  A randomized time of up to 1 second
    was added to the break to avoid adaptation

    For the three online feedback sessions four runs with smiley feedback
    were recorded, whereby each run consisted of twenty trials for each type of
    motor imagery.  At the beginning of each trial (second 0) the feedback (a
    gray smiley) was centered on the screen.  At second 2, a short warning beep
    (1 kHz, 70 ms) was given. The cue was presented from second 3 to 7.5. At
    second 7.5 the screen went blank and a random interval between 1.0 and 2.0
    seconds was added to the trial.

    References
    ----------

    .. [1] R. Leeb, F. Lee, C. Keinrath, R. Scherer, H. Bischof,
           G. Pfurtscheller. Brain-computer communication: motivation, aim,
           and impact of exploring a virtual apartment. IEEE Transactions on
           Neural Systems and Rehabilitation Engineering 15, 473–482, 2007

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=5,
            events={"left_hand": 1, "right_hand": 2},
            code="004-2014",
            interval=[3, 7.5],
            paradigm="imagery",
            doi="10.1109/TNSRE.2007.906956",
        )


class BNCI2014008(MNEBNCI):
    """BNCI 2014-008 P300 dataset

    Dataset from [1]_.

    **Dataset description**

    This dataset represents a complete record of P300 evoked potentials
    using a paradigm originally described by Farwell and Donchin [2]_.
    In these sessions, 8 users with amyotrophic lateral sclerosis (ALS)
    focused on one out of 36 different characters. The objective in this
    contest is to predict the correct character in each of the provided
    character selection epochs.

    We included in the study a total of eight volunteers, all naïve to BCI
    training. Scalp EEG signals were recorded (g.MOBILAB, g.tec, Austria)
    from eight channels according to 10–10 standard (Fz, Cz, Pz, Oz, P3, P4,
    PO7 and PO8) using active electrodes (g.Ladybird, g.tec, Austria).
    All channels were referenced to the right earlobe and grounded to the left
    mastoid. The EEG signal was digitized at 256 Hz and band-pass filtered
    between 0.1 and 30 Hz.

    Participants were required to copy spell seven predefined words of five
    characters each (runs), by controlling a P300 matrix speller. Rows and
    columns on the interface were randomly intensified for 125ms, with an
    inter stimulus interval (ISI) of 125ms, yielding a 250 ms lag between the
    appearance of two stimuli (stimulus onset asynchrony, SOA).

    In the first three runs (15 trials in total) EEG data was stored to
    perform a calibration of the BCI classifier. Thus no feedback was provided
    to the participant up to this point. A stepwise linear discriminant
    analysis (SWLDA) was applied to the data from the three calibration runs
    (i.e., runs 1–3) to determine the classifier weights (i.e., classifier
    coefficients). These weights were then applied during the subsequent four
    testing runs (i.e., runs 4–7) when participants were provided with
    feedback.

    References
    ----------
    .. [1] A. Riccio, L. Simione, F. Schettini, A. Pizzimenti, M. Inghilleri,
           M. O. Belardinelli, D. Mattia, and F. Cincotti (2013). Attention
           and P300-based BCI performance in people with amyotrophic lateral
           sclerosis. Front. Hum. Neurosci., vol. 7:, pag. 732.
    .. [2] L. A. Farwell and E. Donchin, Talking off the top of your head:
           toward a mental prosthesis utilizing eventrelated
           brain potentials, Electroencephalogr. Clin. Neurophysiol.,
           vol. 70, n. 6, pagg. 510–523, 1988.

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 9)),
            sessions_per_subject=1,
            events={"Target": 2, "NonTarget": 1},
            code="008-2014",
            interval=[0, 1.0],
            paradigm="p300",
            doi="10.3389/fnhum.2013.00732",
        )


class BNCI2014009(MNEBNCI):
    """BNCI 2014-009 P300 dataset.

    Dataset from [1]_.

    **Dataset description**

    This dataset presents a complete record of P300 evoked potentials
    using two different paradigms: a paradigm based on the P300 Speller in
    overt attention condition and a paradigm based used in covert attention
    condition. In these sessions, 10 healthy subjects focused on one out of 36
    different characters. The objective was to predict the correct character
    in each of the provided character selection epochs.
    (Note: right now only the overt attention data is available via MOABB)

    In the first interface, cues are organized in a 6×6 matrix and each
    character is always visible on the screen and spatially separated from the
    others. By design, no fixation cue is provided, as the subject is expected
    to gaze at the target character. Stimulation consists in the
    intensification of whole lines (rows or columns) of six characters.

    Ten healthy subjects (10 female, mean age = 26.8 ± 5.6, table I) with
    previous experience with P300-based BCIs attended 3 recording sessions.
    Scalp EEG potentials were measured using 16 Ag/AgCl electrodes that
    covered the left, right and central scalp (Fz, FCz, Cz, CPz, Pz, Oz, F3,
    F4, C3, C4, CP3, CP4, P3, P4, PO7, PO8) per the 10-10 standard. Each
    electrode was referenced to the linked earlobes and grounded to the
    right mastoid. The EEG was acquired at 256 Hz, high pass- and low
    pass-filtered with cutoff frequencies of 0.1 Hz and 20 Hz, respectively.
    Each subject attended 4 recording sessions. During each session,
    the subject performed three runs with each of the stimulation interfaces.

    References
    ----------

    .. [1] P Aricò, F Aloise, F Schettini, S Salinari, D Mattia and F Cincotti
           (2013). Influence of P300 latency jitter on event related potential-
           based brain–computer interface performance. Journal of Neural
           Engineering, vol. 11, number 3.

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=3,
            events={"Target": 2, "NonTarget": 1},
            code="009-2014",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.1088/1741-2560/11/3/035008",
        )


class BNCI2015001(MNEBNCI):
    """BNCI 2015-001 Motor Imagery dataset.

    Dataset from [1]_.

    **Dataset description**

    We acquired the EEG from three Laplacian derivations, 3.5 cm (center-to-
    center) around the electrode positions (according to International 10-20
    System of Electrode Placement) C3 (FC3, C5, CP3 and C1), Cz (FCz, C1, CPz
    and C2) and C4 (FC4, C2, CP4 and C6).  The acquisition hardware was a
    g.GAMMAsys active electrode system along with a g.USBamp amplifier (g.tec,
    Guger Tech- nologies OEG, Graz, Austria).  The system sampled at 512 Hz,
    with a bandpass filter between 0.5 and 100 Hz and a notch filter at 50 Hz.
    The order of the channels in the data is FC3, FCz, FC4, C5, C3, C1, Cz, C2,
    C4, C6, CP3, CPz, CP4.

    The task for the user was to perform sustained right hand versus both feet
    movement imagery starting from the cue (second 3) to the end of the cross
    period (sec- ond 8).  A trial started with 3 s of reference period,
    followed by a brisk audible cue and a visual cue (arrow right for right
    hand, arrow down for both feet) from second 3 to 4.25.
    The activity period, where the users received feedback, lasted from
    second 4 to 8. There was a random 2 to 3 s pause between the trials.

    References
    ----------

    .. [1] J. Faller, C. Vidaurre, T. Solis-Escalante, C. Neuper and R.
           Scherer (2012). Autocalibration and recurrent adaptation: Towards a
           plug and play online ERD- BCI.  IEEE Transactions on Neural Systems
           and Rehabilitation Engineering, 20(3), 313-319.

    """

    def __init__(self):
        # FIXME: some participant have 3 sessions
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=2,
            events={"right_hand": 1, "feet": 2},
            code="001-2015",
            interval=[0, 5],
            paradigm="imagery",
            doi="10.1109/tnsre.2012.2189584",
        )


class BNCI2015003(MNEBNCI):
    """BNCI 2015-003 P300 dataset.

    Dataset from [1]_.

    **Dataset description**

    This dataset contains recordings from 10 subjects performing a visual P300
    task for spelling. Results were published in [1]_. Sampling frequency was
    256 Hz and there were 8 electrodes ('Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7',
    'Oz', 'PO8') which were referenced to the right earlobe. Each subject
    participated in only one session. For more information, see [1]_.

    References
    ----------

    .. [1]  C. Guger, S. Daban, E. Sellers, C. Holzner, G. Krausz,
            R. Carabalona, F. Gramatica, and G. Edlinger (2009). How many
            people are able to control a P300-based brain-computer interface
            (BCI)?. Neuroscience Letters, vol. 462, pp. 94–98.

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events={"Target": 2, "NonTarget": 1},
            code="003-2015",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.1016/j.neulet.2009.06.045",
        )


class BNCI2015004(MNEBNCI):
    """BNCI 2015-004 Motor Imagery dataset.

    Dataset from [1]_.

    **Dataset description**

    We provide EEG data recorded from nine users with disability (spinal cord
    injury and stroke) on two different days (sessions).  Users performed,
    follow- ing a cue-guided experimental paradigm, five distinct mental tasks
    (MT).  MTs include mental word association (condition WORD), mental
    subtraction (SUB), spatial navigation (NAV), right hand motor imagery
    (HAND) and
    feet motor imagery (FEET). Details on the experimental paradigm are
    summarized in Figure 1.  The session for a single subject consisted of 8
    runs resulting in 40 trials of each class for each day.  One single
    experimental run consisted of 25 cues, with 5 of each mental task.  Cues
    were presented in random order.

    EEG was recorded from 30 electrode channels placed on the scalp according
    to the international 10-20 system.  Electrode positions included channels
    AFz, F7, F3, Fz, F4, F8, FC3, FCz, FC4, T3, C3, Cz, C4, T4, CP3, CPz,CP4,
    P7, P5, P3, P1, Pz, P2, P4, P6, P8, PO3, PO4, O1, and O2.  Reference and
    ground were placed at the left and right mastoid, respectively.  The g.tec
    GAMMAsys system with g.LADYbird active electrodes and two g.USBamp
    biosignal
    amplifiers (Guger Technolgies, Graz, Austria) was used for recording.  EEG
    was band pass filtered 0.5-100 Hz (notch filter at 50 Hz) and sampled at a
    rate of 256 Hz.

    The duration of a single imagery trials is 10 s.  At t = 0 s, a cross was
    presented in the middle of the screen.  Participants were asked to relax
    and
    fixate the cross to avoid eye movements.  At t = 3 s, a beep was sounded to
    get the participant’s attention.  The cue indicating the requested imagery
    task, one out of five graphical symbols, was presented from t = 3 s to t =
    4.25 s.  At t = 10 s, a second beep was sounded and the fixation-cross
    disappeared, which indicated the end of the trial.  A variable break
    (inter-trial-interval, ITI) lasting between 2.5 s and 3.5 s occurred
    before
    the start of the next trial.  Participants were asked to avoid movements
    during the imagery period, and to move and blink during the
    ITI. Experimental runs began and ended with a blank screen (duration 4 s)

    References
    ----------

    .. [1] Scherer R, Faller J, Friedrich EVC, Opisso E, Costa U, Kübler A, et
           al. (2015) Individually Adapted Imagery Improves Brain-Computer
           Interface Performance in End-Users with Disability. PLoS ONE 10(5).
           https://doi.org/10.1371/journal.pone.0123727

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=2,
            events=dict(right_hand=4, feet=5, navigation=3, subtraction=2, word_ass=1),
            code="004-2015",
            interval=[3, 10],
            paradigm="imagery",
            doi="10.1371/journal.pone.0123727",
        )
