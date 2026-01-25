"""BNCI 2014-001 Motor imagery dataset."""

import io
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from mne import Annotations, create_info
from mne.channels import make_standard_montage
from mne.io import RawArray, read_raw_gdf
from mne.utils import verbose
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.utils import depreciated_alias


BNCI_URL = "http://bnci-horizon-2020.eu/database/data-sets/"
BBCI_URL = "http://doc.ml.tu-berlin.de/bbci/"
BNCI_URL_001_2019 = "http://bnci-horizon-2020.eu/database/data-sets/001-2019/"


def data_path(url, path=None, force_update=False, update_path=None, verbose=None):
    return [dl.data_dl(url, "BNCI", path, force_update, verbose)]


_map = {"T": "train", "E": "test"}


@verbose
def load_data(
    subject,
    dataset="BNCI2014-001",
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
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
    only_filenames : bool
        If True, return only the local path of the files without
        loading the data.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raws : list
        List of raw instances for each non consecutive recording. Depending
        on the dataset it could be a BCI run or a different recording session.
    event_id: dict
        dictionary containing events and their code.
    """
    dataset_list = {
        "BNCI2003-004": _load_data_iva_2003,
        "BNCI2014-001": _load_data_001_2014,
        "BNCI2014-002": _load_data_002_2014,
        "BNCI2014-004": _load_data_004_2014,
        "BNCI2014-008": _load_data_008_2014,
        "BNCI2014-009": _load_data_009_2014,
        "BNCI2015-001": _load_data_001_2015,
        "BNCI2015-003": _load_data_003_2015,
        "BNCI2015-004": _load_data_004_2015,
        "BNCI2015-006": _load_data_006_2015,
        "BNCI2015-007": _load_data_007_2015,
        "BNCI2015-008": _load_data_008_2015,
        "BNCI2015-009": _load_data_009_2015,
        "BNCI2015-010": _load_data_010_2015,
        "BNCI2015-012": _load_data_012_2015,
        "BNCI2015-013": _load_data_013_2015,
    }

    baseurl_list = {
        "BNCI2003-004": "https://www.bbci.de/competition/",
        "BNCI2014-001": BNCI_URL,
        "BNCI2014-002": BNCI_URL,
        "BNCI2015-001": BNCI_URL,
        "BNCI2014-004": BNCI_URL,
        "BNCI2014-008": BNCI_URL,
        "BNCI2014-009": BNCI_URL,
        "BNCI2015-003": BNCI_URL,
        "BNCI2015-004": BNCI_URL,
        "BNCI2015-006": BBCI_URL,
        "BNCI2015-007": BBCI_URL,
        "BNCI2015-008": BBCI_URL,
        "BNCI2015-009": BBCI_URL,
        "BNCI2015-010": BBCI_URL,
        "BNCI2015-012": BBCI_URL,
        "BNCI2015-013": BNCI_URL,
    }

    if dataset not in dataset_list.keys():
        raise ValueError(
            "Dataset '%s' is not a valid BNCI dataset ID. "
            "Valid dataset are %s." % (dataset, ", ".join(dataset_list.keys()))
        )

    return dataset_list[dataset](
        subject,
        path,
        force_update,
        update_path,
        baseurl_list[dataset],
        only_filenames,
        verbose,
    )


@verbose
def _load_data_iva_2003(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=None,
    only_filenames=False,
    verbose=None,
):
    """Loads data for the BNCI2003-IVa dataset."""
    # Raises ValueError is subject is not between 1 and 5
    if (subject < 1) or (subject > 5):
        raise ValueError(f"Subject must be between 1 and 5. Got {subject}")

    subject_names = ["aa", "al", "av", "aw", "ay"]

    # fmt: off
    ch_names = ['Fp1', 'AFp1', 'Fpz', 'AFp2', 'Fp2', 'AF7', 'AF3',
                'AF4', 'AF8', 'FAF5', 'FAF1', 'FAF2', 'FAF6', 'F7',
                'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FFC7',
                'FFC5', 'FFC3', 'FFC1', 'FFC2', 'FFC4', 'FFC6', 'FFC8',
                'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                'FC6', 'FT8', 'FT10', 'CFC7', 'CFC5', 'CFC3', 'CFC1',
                'CFC2', 'CFC4', 'CFC6', 'CFC8', 'T7', 'C5', 'C3', 'C1',
                'Cz','C2', 'C4', 'C6', 'T8', 'CCP7', 'CCP5', 'CCP3', 'CCP1',
                'CCP2', 'CCP4', 'CCP6', 'CCP8', 'TP9', 'TP7', 'CP5', 'CP3',
                'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'PCP7',
                'PCP5', 'PCP3', 'PCP1', 'PCP2', 'PCP4', 'PCP6', 'PCP8', 'P9',
                'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10',
                'PPO7', 'PPO5', 'PPO1', 'PPO2', 'PPO6', 'PPO8', 'PO7', 'PO3',
                'PO1', 'POz', 'PO2', 'PO4', 'PO8', 'OPO1', 'OPO2', 'O1',
                'Oz', 'O2', 'OI1', 'OI2', 'I1', 'I2']
    # fmt: on
    ch_type = ["eeg"] * 118

    url = "{u}download/competition_iii/berlin/100Hz/data_set_IVa_{r}_mat.zip".format(
        u=base_url, r=subject_names[subject - 1]
    )

    filename = data_path(url, path, force_update, update_path)

    if only_filenames:
        return filename

    runs, ev = _convert_bbci2003(filename[0], ch_names, ch_type)
    _finalize_raw(runs, "BNCI2003-004", subject)

    session = {"0train": {"0": runs}}
    return session


@verbose
def _load_data_001_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
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
    filenames = []
    for session_idx, r in enumerate(["T", "E"]):
        url = "{u}001-2014/A{s:02d}{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)
        filenames += filename
        if only_filenames:
            continue
        runs, ev = _convert_mi(
            filename[0],
            ch_names,
            ch_types,
            dataset_code="BNCI2014-001",
            subject_id=subject,
        )
        # FIXME: deal with run with no event (1:3) and name them
        sessions[f"{session_idx}{_map[r]}"] = {
            str(ii): run for ii, run in enumerate(runs)
        }
    if only_filenames:
        return filenames
    return sessions


@verbose
def _load_data_002_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 002-2014 dataset."""
    if (subject < 1) or (subject > 14):
        raise ValueError("Subject must be between 1 and 14. Got %d." % subject)

    runs = []
    filenames = []
    for r in ["T", "E"]:
        url = "{u}002-2014/S{s:02d}{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)[0]
        filenames.append(filename)
        if only_filenames:
            continue
        # FIXME: electrode position and name are not provided directly.
        raws, _ = _convert_mi(
            filename, None, ["eeg"] * 15, dataset_code="BNCI2014-002", subject_id=subject
        )
        runs.extend(zip([r] * len(raws), raws))
    if only_filenames:
        return filenames
    runs = {f"{ii}{_map[r]}": run for ii, (r, run) in enumerate(runs)}
    return {"0": runs}


@verbose
def _load_data_004_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 004-2014 dataset."""
    if (subject < 1) or (subject > 9):
        raise ValueError("Subject must be between 1 and 9. Got %d." % subject)

    ch_names = ["C3", "Cz", "C4", "EOG1", "EOG2", "EOG3"]
    ch_types = ["eeg"] * 3 + ["eog"] * 3

    sessions = []
    filenames = []
    for r in ["T", "E"]:
        url = "{u}004-2014/B{s:02d}{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)[0]
        filenames.append(filename)
        if only_filenames:
            continue
        raws, _ = _convert_mi(
            filename, ch_names, ch_types, dataset_code="BNCI2014-004", subject_id=subject
        )
        sessions.extend(zip([r] * len(raws), raws))

    if only_filenames:
        return filenames
    sessions = {f"{ii}{_map[r]}": {"0": run} for ii, (r, run) in enumerate(sessions)}
    return sessions


@verbose
def _load_data_008_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 008-2014 dataset."""
    if (subject < 1) or (subject > 8):
        raise ValueError("Subject must be between 1 and 8. Got %d." % subject)

    url = "{u}008-2014/A{s:02d}.mat".format(u=base_url, s=subject)
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]
    run = loadmat(filename, struct_as_record=False, squeeze_me=True)["data"]
    raw, event_id = _convert_run_p300_sl(run, verbose=verbose)

    # Enrich with BNCI2014-008 specific metadata (age, gender, ALSfrs, onsetALS)
    _enrich_run_with_metadata(raw, run, "BNCI2014-008", subject)

    sessions = {"0": {"0": raw}}

    return sessions


@verbose
def _load_data_009_2014(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 009-2014 dataset."""
    if (subject < 1) or (subject > 10):
        raise ValueError("Subject must be between 1 and 10. Got %d." % subject)

    # FIXME there is two type of speller, grid speller and geo-speller.
    # we load only grid speller data
    url = "{u}009-2014/A{s:02d}S.mat".format(u=base_url, s=subject)
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)["data"]
    sess = []
    event_id = {}
    for run in data:
        raw, ev = _convert_run_p300_sl(run, verbose=verbose)
        _finalize_raw(raw, "BNCI2014-009", subject)
        # Raw EEG data are scaled by a factor 10.
        # See https://github.com/NeuroTechX/moabb/issues/275
        raw._data[:16, :] /= 10.0
        sess.append(raw)
        event_id.update(ev)

    sessions = {}
    for i, sessi in enumerate(sess):
        sessions[str(i)] = {"0": sessi}

    return sessions


@verbose
def _load_data_001_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 001-2015 dataset."""
    if (subject < 1) or (subject > 12):
        raise ValueError("Subject must be between 1 and 12. Got %d." % subject)

    if subject in [8, 9, 10, 11]:
        ses = [(0, "A"), (1, "B"), (2, "C")]  # 3 sessions for those subjects
    else:
        ses = [(0, "A"), (1, "B")]

    # fmt: off
    ch_names = [
        "FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz",
        "C2", "C4", "C6", "CP3", "CPz", "CP4",
    ]
    # fmt: on
    ch_types = ["eeg"] * 13

    sessions = {}
    filenames = []
    for session_idx, r in ses:
        url = "{u}001-2015/S{s:02d}{r}.mat".format(u=base_url, s=subject, r=r)
        filename = data_path(url, path, force_update, update_path)
        filenames += filename
        if only_filenames:
            continue
        runs, ev = _convert_mi(filename[0], ch_names, ch_types, subject_id=subject)
        sessions[f"{session_idx}{r}"] = {str(ii): run for ii, run in enumerate(runs)}
    if only_filenames:
        return filenames
    return sessions


@verbose
def _load_data_003_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 003-2015 dataset."""
    if (subject < 1) or (subject > 10):
        raise ValueError("Subject must be between 1 and 12. Got %d." % subject)

    url = "{u}003-2015/s{s:d}.mat".format(u=base_url, s=subject)
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    data = data["s%d" % subject]
    sfreq = 256.0

    ch_names = ["Fz", "Cz", "P3", "Pz", "P4", "PO7", "Oz", "PO8", "Target", "Flash"]

    ch_types = ["eeg"] * 8 + ["stim"] * 2

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    sessions = {}
    sessions["0"] = {}
    for r_name, run in [("0train", data.train), ("1test", data.test)]:
        # flash events on the channel 9
        flashs = run[9:10]
        ix_flash = flashs[0] > 0
        flashs[0, ix_flash] += 2  # add 2 to avoid overlap on event id
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
        # Enrich raw object with additional metadata
        raw.info["line_freq"] = 50.0
        _finalize_raw(raw, "BNCI2015-003", subject)

        sessions["0"][r_name] = raw

    return sessions


@verbose
def _load_data_004_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 004-2015 dataset."""
    if (subject < 1) or (subject > 9):
        raise ValueError("Subject must be between 1 and 9. Got %d." % subject)

    subjects = ["A", "C", "D", "E", "F", "G", "H", "J", "L"]

    url = "{u}004-2015/{s}.mat".format(u=base_url, s=subjects[subject - 1])
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]

    # fmt: off
    ch_names = [
        "AFz", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T3", "C3",
        "Cz", "C4", "T4", "CP3", "CPz", "CP4", "P7", "P5", "P3", "P1", "Pz",
        "P2", "P4", "P6", "P8", "PO3", "PO4", "O1", "O2",
    ]
    # fmt: on
    ch_types = ["eeg"] * 30
    raws, ev = _convert_mi(
        filename, ch_names, ch_types, dataset_code="BNCI2015-004", subject_id=subject
    )
    sessions = {str(ii): {"0": run} for ii, run in enumerate(raws)}
    return sessions


@verbose
def _load_data_007_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 007-2015 dataset (Motion VEP Speller).

    This dataset contains motion-onset visual evoked potentials (mVEPs)
    for gaze-independent BCI communication. Uses BBCI data format.
    """
    if (subject < 1) or (subject > 16):
        raise ValueError("Subject must be between 1 and 16. Got %d." % subject)

    # Subject codes for the 16 subjects
    # fmt: off
    subjects = [
        "fat", "gdf", "gdg", "iac", "iba", "ibe", "ibq", "ibs",
        "ibt", "ibu", "ibv", "ibw", "ibx", "iby", "ice", "icv",
    ]
    # fmt: on

    s = subjects[subject - 1]
    url = "{u}BNCIHorizon2020-MVEP/MVEP_VP{s}.mat".format(u=base_url, s=s)
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]

    ch_types = ["eeg"] * 63

    raws, event_id = _convert_bbci(filename, ch_types, verbose=None)
    for raw in raws:
        _finalize_raw(raw, "BNCI2015-007", subject)
    return raws, event_id


@verbose
def _load_data_008_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 008-2015 dataset (Center Speller).

    This dataset contains P300 evoked potentials recorded during a gaze-independent
    two-stage visual speller paradigm called the "Center Speller".
    """
    if (subject < 1) or (subject > 13):
        raise ValueError("Subject must be between 1 and 13. Got %d." % subject)

    # fmt: off
    subjects = [
        "iac", "iba", "ibb", "ibc", "ibd", "ibe", "ibf",
        "ibg", "ibh", "ibi", "ibj", "ica", "saf",
    ]
    # fmt: on

    s = subjects[subject - 1]
    url = "{u}BNCIHorizon2020-CenterSpeller/CenterSpeller_VP{s}.mat".format(
        u=base_url, s=s
    )
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]

    ch_types = ["eeg"] * 63

    raws, event_id = _convert_bbci(filename, ch_types, verbose=None)
    for raw in raws:
        _finalize_raw(raw, "BNCI2015-008", subject)
    return raws, event_id


@verbose
def _load_data_006_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    only_filenames=False,
    verbose=None,
):
    if (subject < 1) or (subject > 11):
        raise ValueError("Subject must be between 1 and 11. Got %d." % subject)
    subjects = [
        "vp1",
        "vp2",
        "vp3",
        "vp4",
        "vp5",
        "vp6",
        "vp7",
        "vp8",
        "vp9",
        "vp10",
        "vp11",
    ]
    s = subjects[subject - 1]
    url = "{u}BNCIHorizon2020-MusicBCI/MusicBCI_{s}.mat".format(u=base_url, s=s)
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]
    mat_data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    data = mat_data["data"]
    eeg_data = data.X
    sfreq = float(data.fs)
    ch_names = [str(ch).strip() for ch in data.clab]
    ch_types = ["eeg"] * len(ch_names)
    trigger = np.zeros((len(eeg_data), 1))
    if hasattr(data, "trial") and len(data.trial) > 0:
        trial_indices = np.array(data.trial).flatten() - 1
        trial_labels = np.array(data.y).flatten()
        valid_mask = (trial_indices >= 0) & (trial_indices < len(eeg_data))
        trigger[trial_indices[valid_mask], 0] = trial_labels[valid_mask]
    eeg_data = np.c_[eeg_data, trigger]
    ch_names = ch_names + ["STI"]
    ch_types = ch_types + ["stim"]
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")
    raw.info["line_freq"] = 50.0
    _finalize_raw(raw, "BNCI2015-006", subject)
    sessions = {"0": {"0": raw}}
    return sessions


@verbose
def _load_data_009_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    only_filenames=False,
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
    if only_filenames:
        return [filename]

    ch_types = ["eeg"] * 60 + ["eog"] * 2

    raws, event_id = _convert_bbci(filename, ch_types, verbose=None)
    for raw in raws:
        _finalize_raw(raw, "BNCI2015-009", subject)
    return raws, event_id


@verbose
def _load_data_010_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    only_filenames=False,
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
    if only_filenames:
        return [filename]

    ch_types = ["eeg"] * 63

    raws, event_id = _convert_bbci(filename, ch_types, verbose=None)
    for raw in raws:
        _finalize_raw(raw, "BNCI2015-010", subject)
    return raws, event_id


@verbose
def _load_data_012_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BBCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 012-2015 dataset."""
    if (subject < 1) or (subject > 12):
        raise ValueError("Subject must be between 1 and 12. Got %d." % subject)

    subjects = ["nv", "nw", "nx", "ny", "nz", "mg", "oa", "ob", "oc", "od", "ja", "oe"]

    s = subjects[subject - 1]
    url = "{u}BNCIHorizon2020-PASS2D/PASS2D_VP{s}.mat".format(u=base_url, s=s)
    filename = data_path(url, path, force_update, update_path)[0]
    if only_filenames:
        return [filename]

    ch_types = ["eeg"] * 63

    raws, event_id = _convert_bbci(filename, ch_types, verbose=None)
    for raw in raws:
        _finalize_raw(raw, "BNCI2015-012", subject)
    return raws, event_id


@verbose
def _load_data_013_2015(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BNCI_URL,
    only_filenames=False,
    verbose=None,
):
    """Load data for 013-2015 dataset."""
    if (subject < 1) or (subject > 6):
        raise ValueError("Subject must be between 1 and 6. Got %d." % subject)

    data_paths = []
    for r in ["s1", "s2"]:
        url = "{u}013-2015/Subject{s:02d}_{r}.mat".format(u=base_url, s=subject, r=r)
        data_paths.extend(data_path(url, path, force_update, update_path))
    if only_filenames:
        return data_paths

    raws = []
    event_id = {}

    for filename in data_paths:
        data = loadmat(filename, struct_as_record=False, squeeze_me=True)
        for run in data["run"]:
            raw, evd = _convert_run_epfl(run, verbose=verbose)
            _finalize_raw(raw, "BNCI2015-013", subject)
            raws.append(raw)
            event_id.update(evd)
    return raws, event_id


def _finalize_raw(raw, dataset_code, subject_id):
    """Finalize raw object with montage, measurement date, and subject ID.

    This function should be called by each conversion function after creating
    the raw object to ensure all required metadata is set for BIDS compliance.

    Parameters
    ----------
    raw : instance of RawArray
        Raw object to finalize.
    dataset_code : str
        Dataset code (e.g., 'BNCI2014-001').
    subject_id : int
        Subject number.
    """
    # Set montage if not already set and we have standard EEG channels
    if raw.get_montage() is None:
        eeg_picks = [
            ch for ch, typ in zip(raw.ch_names, raw.get_channel_types()) if typ == "eeg"
        ]

        if eeg_picks:
            montage = make_standard_montage("standard_1005")
            if any(ch in montage.ch_names for ch in eeg_picks):
                raw.set_montage(montage, on_missing="ignore")

    # Set measurement date if not already set (required for BIDS)
    if raw.info["meas_date"] is None:
        year = MNEBNCI._dataset_years.get(dataset_code, 2010)
        raw.set_meas_date(datetime(year, 1, 1, tzinfo=timezone.utc))

    # Ensure subject_info has an ID (required for BIDS)
    subject_info = raw.info.get("subject_info")
    if subject_info is not None:
        if "his_id" not in str(subject_info):
            subject_info["his_id"] = f"sub-{subject_id:02d}"


def _enrich_run_with_metadata(raw, run, dataset_code, subject_id):
    """Extract metadata from run object and enrich raw object.

    This function extracts subject-specific metadata (age, gender, etc.) from
    MAT file run objects and enriches the raw object. It also sets the
    measurement date based on when the data was collected. After metadata
    extraction, it calls _finalize_raw() to ensure BIDS compliance.

    Parameters
    ----------
    raw : instance of RawArray
        Raw object to enrich.
    run : MAT file run object
        Run object containing metadata.
    dataset_code : str
        Dataset code to determine which fields to extract.
    subject_id : int
        Subject number for BIDS subject_info.
    """
    # BNCI2014-001 and BNCI2014-004: have age, gender, artifacts
    if dataset_code in ["BNCI2014-001", "BNCI2014-004"]:
        subject_info = {}

        # Set measurement date (BCI Competition IV 2008)
        rec_year = 2008
        raw.set_meas_date(datetime(rec_year, 1, 1, tzinfo=timezone.utc))

        # Extract age
        age = int(run.age)
        birth_year = rec_year - age
        subject_info["birthday"] = date(birth_year, 1, 1)

        # Extract gender
        gender_str = str(run.gender).lower()
        if gender_str in ["male", "m"]:
            subject_info["sex"] = 1
        elif gender_str in ["female", "f"]:
            subject_info["sex"] = 2
        else:
            subject_info["sex"] = 0  # Unknown

        # Set handedness (right-handed based on dataset description)
        subject_info["hand"] = 1

        raw.info["subject_info"] = subject_info

        # Extract artifacts information
        artifacts = run.artifacts
        if len(artifacts) > 0:
            n_artifacts = len(np.nonzero(artifacts)[0])
            if n_artifacts > 0:
                current_desc = raw.info.get("description") or ""
                raw.info["description"] = (
                    current_desc + f"Artifacts: {n_artifacts}/{len(artifacts)} trials; "
                )

    # BNCI2014-008: has age (string!), gender, ALSfrs, onsetALS
    elif dataset_code == "BNCI2014-008":
        subject_info = {}

        # Set measurement date (recorded ~2012)
        rec_year = 2012
        raw.set_meas_date(datetime(rec_year, 1, 1, tzinfo=timezone.utc))

        # Extract age (note: age is stored as string in this dataset!)
        age = int(run.age)
        birth_year = rec_year - age
        subject_info["birthday"] = date(birth_year, 1, 1)

        # Extract gender
        gender_str = str(run.gender).lower()
        if gender_str in ["male", "m"]:
            subject_info["sex"] = 1
        elif gender_str in ["female", "f"]:
            subject_info["sex"] = 2
        else:
            subject_info["sex"] = 0  # Unknown

        # Set handedness (unknown for P300 datasets)
        subject_info["hand"] = 0  # Unknown

        raw.info["subject_info"] = subject_info

        # Extract ALS-specific information
        alsfrs = str(run.ALSfrs)
        onset = str(run.onsetALS)
        current_desc = raw.info.get("description") or ""
        # ALS = Amyotrophic Lateral Sclerosis (medical condition, not a typo)
        raw.info["description"] = current_desc + f"ALSfrs: {alsfrs}; ALS onset: {onset}; "

    # Finalize raw object (montage, measurement date fallback, subject ID)
    _finalize_raw(raw, dataset_code, subject_id)


def _convert_mi(filename, ch_names, ch_types, dataset_code=None, subject_id=None):
    """Process (Graz) motor imagery data from MAT files.

    Parameters
    ----------
    filename : str
        Path to the MAT file.
    ch_names : list of str
        List of channel names.
    ch_types : list of str
        List of channel types.
    dataset_code : str, optional
        Dataset code for metadata extraction.
    subject_id : int, optional
        Subject number for BIDS compliance.

    Returns
    -------
    raw : instance of RawArray
        returns list of recording runs."""
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

        # Enrich with metadata if dataset code is provided
        if dataset_code and subject_id:
            _enrich_run_with_metadata(raw, run, dataset_code, subject_id)
        elif subject_id:
            # No dataset-specific metadata, but still need BIDS finalization
            _finalize_raw(raw, dataset_code or "unknown", subject_id)

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
    # Use 'STI' instead of 'stim' to avoid ambiguity with channel type
    ch_names = ch_names + ["STI"]
    ch_types = ch_types + ["stim"]
    event_id = {ev: (ii + 1) for ii, ev in enumerate(run.classes)}
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    raw.set_montage(montage)

    # Set line frequency (50 Hz for European datasets)
    raw.info["line_freq"] = 50.0

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
    # Montage is now set in _get_single_subject_data() method
    raw.set_montage(montage)

    # Set line frequency (50 Hz for European datasets)
    raw.info["line_freq"] = 50.0

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

    # Set line frequency (50 Hz for European datasets)
    raw.info["line_freq"] = 50.0

    return raw, event_id


def _convert_bbci2003(filename, ch_names, ch_type):
    """
    Process motor imagery data from MAT files.

     Parameters
     ----------
        filename (str):
            Path to the MAT file.
        ch_names (list of str):
            List of channel names.
        ch_type (list of str):
            List of channel types.

    Returns
    -------
        raw (instance of RawArray):
            returns MNE Raw object.
    """
    zip_path = Path(filename)

    with zipfile.ZipFile(zip_path, "r") as z:
        mat_files = [f for f in z.namelist() if f.endswith(".mat")]

        if not mat_files:
            raise FileNotFoundError("No .mat file found in zip archive.")

        with z.open(mat_files[0]) as f:
            data = loadmat(io.BytesIO(f.read()))

        run = data
        raw, ev = _convert_run_bbci2003(run, ch_names, ch_type)
        return raw, ev


@verbose
def _convert_run_bbci2003(run, ch_names, ch_types, verbose=None):
    """
    Converts one run to a raw MNE object.

    Parameters
    ----------
        run (ndarray):
            The continuous EEG signal.
        ch_names (list of str):
            List of channel names.
        ch_types (list of str):
            List of channel types.
        verbose (bool, str, int, or None):
            If not None, override default verbose level (see :func:`mne.verbose`
            and :ref:`Logging documentation <tut_logging>` for more).

    Returns:
        raw (instance of RawArray):
            MNE Raw object.
        event_id (dict):
            Dictionary containing class names.
    """
    class_map = {
        "right": "right_hand",
        "foot": "feet",
    }

    raw_labels = run["mrk"]["y"][0, 0][0]
    labels_mask = ~np.isnan(raw_labels)
    valid_labels = raw_labels[labels_mask]
    labels = valid_labels.astype(int) - 1

    raw_positions = run["mrk"][0][0]["pos"][0]
    positions = raw_positions[labels_mask]

    sfreq = float(run["nfo"][0, 0]["fs"][0, 0])
    eeg_data = run["cnt"]
    raw_classes = run["mrk"]["className"]

    while isinstance(raw_classes, (list, np.ndarray)) and len(raw_classes) == 1:
        raw_classes = raw_classes[0]
    class_names = [cls[0] for cls in raw_classes]

    for i, word in enumerate(class_names):
        if word in class_map:
            class_names[i] = class_map[word]

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    onset = positions / sfreq
    duration = 0
    description = [class_names[i] for i in labels]
    annotations = Annotations(onset=onset, duration=duration, description=description)

    event_id = {name: i for i, name in enumerate(class_names)}
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)
    raw.set_annotations(annotations)

    # Enrich raw object with additional metadata
    raw.info["line_freq"] = 50.0

    return raw, event_id


@verbose
def _convert_run_epfl(run, verbose=None):
    """Convert one run to raw."""
    # parse eeg data
    event_id = {}

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
    # Use 'STI' instead of 'stim' to avoid ambiguity with channel type
    ch_names = ch_names + ["STI"]
    ch_types = ch_types + ["stim"]
    event_id = {"correct": 1, "error": 2}

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data=eeg_data.T, info=info, verbose=verbose)

    # Enrich raw object with additional metadata
    raw.info["line_freq"] = 50.0

    return raw, event_id


class MNEBNCI(BaseDataset):
    """Base class for BNCI Horizon 2020 datasets.

    The BNCI Horizon 2020 project (Brain/Neural Computer Interaction Horizon 2020)
    was an EU-funded initiative to advance brain-computer interface research. This
    base class provides common functionality for loading and processing datasets
    from the BNCI database.

    **Data Sources**

    - Primary: http://bnci-horizon-2020.eu/database/data-sets/
    - Secondary: http://doc.ml.tu-berlin.de/bbci/ (BBCI archive for some datasets)

    **Common Features**

    All BNCI datasets share these characteristics:

    - Automatic electrode montage setting (standard_1005)
    - BIDS-compliant metadata (measurement date, subject info)
    - Consistent event naming conventions (Target/NonTarget for P300, class names for MI)
    - Line frequency set to 50 Hz (European recordings)
    - Data stored in MAT format (MATLAB)

    **Supported Paradigms**

    BNCI datasets cover multiple BCI paradigms:

    - Motor Imagery (MI): BNCI2003_004, BNCI2014_001, BNCI2014_002, BNCI2014_004,
      BNCI2015_001, BNCI2015_004, BNCI2019_001
    - P300/ERP: BNCI2014_008, BNCI2014_009, BNCI2015_003, BNCI2015_006,
      BNCI2015_007, BNCI2015_008, BNCI2015_009, BNCI2015_010, BNCI2015_012
    - Error-Related Potentials: BNCI2015_013

    Attributes
    ----------
    _dataset_years : dict
        Mapping from dataset codes to recording years, used for setting
        measurement dates for BIDS compliance.

    See Also
    --------
    BNCI2014_001 : 4-class motor imagery (BCI Competition IV Dataset 2a)
    BNCI2014_008 : P300 speller for ALS patients
    BNCI2015_004 : 5-class mental tasks for users with disability
    BNCI2015_009 : AMUSE auditory spatial P300 paradigm
    BNCI2015_010 : RSVP gaze-independent visual P300
    """

    # Dataset collection/publication years for measurement dates
    _dataset_years = {
        "BNCI2003-004": 2003,
        "BNCI2014-001": 2008,
        "BNCI2014-002": 2008,
        "BNCI2014-004": 2008,
        "BNCI2014-008": 2012,
        "BNCI2014-009": 2012,
        "BNCI2015-001": 2010,
        "BNCI2015-003": 2009,
        "BNCI2015-004": 2013,
        "BNCI2015-007": 2012,  # Motion VEP Speller dataset (Schaeff2012)
        "BNCI2015-009": 2015,  # AMUSE dataset from BNCI Horizon 2020
        "BNCI2015-010": 2015,  # RSVP dataset from BNCI Horizon 2020
        "BNCI2015-012": 2015,  # PASS2D dataset from BNCI Horizon 2020
        "BNCI2015-013": 2015,  # Error-related potentials dataset
    }

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
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
            only_filenames=True,
        )


class BNCI2003_004(MNEBNCI):
    """
    BNCI2003_IVa Motor Imagery dataset.

    Dataset IVa from BCI Competition III [1]_.

    **Dataset Description**

    This data set was recorded from five healthy subjects. Subjects sat in
    a comfortable chair with arms resting on armrests. This data set
    contains only data from the 4 initial sessions without feedback.
    Visual cues indicated for 3.5 s which of the following 3 motor
    imageries the subject should perform: (L) left hand, (R) right hand,
    (F) right foot. The presentation of target cues were intermitted by
    periods of random length, 1.75 to 2.25 s, in which the subject could
    relax.

    There were two types of visual stimulation: (1) where targets were
    indicated by letters appearing behind a fixation cross (which might
    nevertheless induce little target-correlated eye movements), and (2)
    where a randomly moving object indicated targets (inducing target-
    uncorrelated eye movements). From subjects al and aw 2 sessions of
    both types were recorded, while from the other subjects 3 sessions
    of type (2) and 1 session of type (1) were recorded.

    References
    ----------
    .. [1] Guido Dornhege, Benjamin Blankertz, Gabriel Curio, and Klaus-Robert
           Müller. Boosting bit rates in non-invasive EEG single-trial
           classifications by feature combination and multi-class paradigms.
           IEEE Trans. Biomed. Eng., 51(6):993-1002, June 2004.

    Notes
    -----
    .. versionadded:: 0.4.0

    This is one of the earliest and most influential motor imagery BCI datasets,
    used extensively for benchmarking classification algorithms. The dataset
    was part of BCI Competition III and has been cited in hundreds of papers.

    See Also
    --------
    BNCI2014_001 : BCI Competition IV 4-class motor imagery dataset
    BNCI2014_004 : BCI Competition 2008 2-class motor imagery dataset
    """

    _participant_demographics = {
        "n_subjects": 5,
        "health_status": "healthy subjects",
        "handedness": "not specified",
        "bci_experience": "not specified",
        "location": "Berlin Institute of Technology, Germany",
    }

    ARTICLE_METADATA = {
        "n_subjects": 5,
        "sessions_per_subject": 1,
        "sampling_rate": 100,
        "n_channels": 118,
        "channel_types": {"eeg": 118},
        "paradigm": "imagery",
        "events": {"right_hand": 0, "feet": 1},
        "doi": "10.1109/TBME.2004.827088",
        "license": "CC BY 4.0",
        "montage": "standard_1005",
        "data_url": "http://bnci-horizon-2020.eu/database/data-sets/004-2003/",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 6)),
            sessions_per_subject=1,
            events={"right_hand": 0, "feet": 1},
            code="BNCI2003-004",
            interval=[0, 3.5],
            paradigm="imagery",
            doi="10.1109/TBME.2004.827088",
        )


@depreciated_alias("BNCI2014001", "1.1")
class BNCI2014_001(MNEBNCI):
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

    Notes
    -----
    .. versionadded:: 0.4.0

    This is one of the most widely used motor imagery datasets in BCI research,
    commonly referred to as "BCI Competition IV Dataset 2a". It serves as a
    standard benchmark for 4-class motor imagery classification algorithms.

    The dataset is particularly useful for:

    - Multi-class motor imagery classification (4 classes)
    - Transfer learning studies (9 subjects, 2 sessions each)
    - Cross-session variability analysis

    See Also
    --------
    BNCI2014_004 : BCI Competition 2008 2-class motor imagery (Dataset B)
    BNCI2003_004 : BCI Competition III 2-class motor imagery

    Examples
    --------
    >>> from moabb.datasets import BNCI2014_001
    >>> dataset = BNCI2014_001()
    >>> dataset.subject_list
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    _participant_demographics = {
        "n_subjects": 9,
        "health_status": "healthy subjects",
        "handedness": "not specified",
        "bci_experience": "not specified",
        "location": "Graz University of Technology, Austria",
    }

    ARTICLE_METADATA = {
        "n_subjects": 9,
        "sessions_per_subject": 2,
        "sampling_rate": 250,
        "n_channels": 22,
        "channel_types": {"eeg": 22},
        "reference": "left mastoid",
        "ground": "right mastoid",
        "paradigm": "imagery",
        "events": {"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4},
        "doi": "10.3389/fnins.2012.00055",
        "license": "CC BY 4.0",
        "montage": "standard_1005",
        "equipment": "Ag/AgCl electrodes",
        "data_url": "http://bnci-horizon-2020.eu/database/data-sets/001-2014/",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=2,
            events={"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4},
            code="BNCI2014-001",
            interval=[2, 6],
            paradigm="imagery",
            doi="10.3389/fnins.2012.00055",
        )


@depreciated_alias("BNCI2014002", "1.1")
class BNCI2014_002(MNEBNCI):
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

    **Participant Demographics**

    - 14 healthy volunteers (note: documentation mentions both 13 and 14)
    - Age range: 20-30 years
    - BCI experience: 8 out of 14 naive to the task
    - Health status: No known medical or neurological diseases
    - Location: Graz University of Technology, Austria

    References
    ----------
    .. [1] Steyrl, D., Scherer, R., Faller, J. and Müller-Putz, G.R., 2016.
           Random forests in non-invasive sensorimotor rhythm brain-computer
           interfaces: a practical and convenient non-linear classifier.
           Biomedical Engineering/Biomedizinische Technique, 61(1), pp.77-86.
    """

    # Participant demographics from paper
    _participant_demographics = {
        "n_subjects": 14,
        "age_range": (20, 30),
        "health_status": "healthy volunteers",
        "bci_experience": "8 out of 14 naive to the task",
        "location": "Graz University of Technology, Austria",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 15)),
            sessions_per_subject=1,
            events={"right_hand": 1, "feet": 2},
            code="BNCI2014-002",
            interval=[3, 8],
            paradigm="imagery",
            doi="10.1515/bmt-2014-0117",
        )


@depreciated_alias("BNCI2014004", "1.1")
class BNCI2014_004(MNEBNCI):
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

    Notes
    -----
    .. versionadded:: 0.4.0

    This dataset is notable for:

    - Including both screening (no feedback) and online feedback sessions
    - Minimal electrode setup (3 bipolar channels: C3, Cz, C4)
    - 5 sessions per subject enabling longitudinal/learning studies
    - BCI Competition 2008 Dataset B benchmark

    The dataset is particularly useful for studying:

    - BCI learning effects across sessions
    - Performance with minimal electrode configurations
    - Transition from calibration to online feedback

    See Also
    --------
    BNCI2014_001 : BCI Competition IV 4-class motor imagery (Dataset 2a)
    BNCI2003_004 : BCI Competition III motor imagery dataset

    Examples
    --------
    >>> from moabb.datasets import BNCI2014_004
    >>> dataset = BNCI2014_004()
    >>> dataset.subject_list
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    """

    _participant_demographics = {
        "n_subjects": 9,
        "health_status": "healthy subjects",
        "handedness": "right-handed",
        "vision": "normal or corrected-to-normal",
        "bci_experience": "not specified",
        "location": "Graz University of Technology, Austria",
    }

    ARTICLE_METADATA = {
        "n_subjects": 9,
        "sessions_per_subject": 5,
        "sampling_rate": 250,
        "n_channels": 6,
        "channel_types": {"eeg": 3, "eog": 3},
        "reference": "bipolar",
        "ground": "Fz",
        "paradigm": "imagery",
        "events": {"left_hand": 1, "right_hand": 2},
        "doi": "10.1109/TNSRE.2007.906956",
        "license": "CC BY 4.0",
        "montage": "standard_1005",
        "data_url": "http://bnci-horizon-2020.eu/database/data-sets/004-2014/",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=5,
            events={"left_hand": 1, "right_hand": 2},
            code="BNCI2014-004",
            interval=[3, 7.5],
            paradigm="imagery",
            doi="10.1109/TNSRE.2007.906956",
        )


@depreciated_alias("BNCI2014008", "1.1")
class BNCI2014_008(MNEBNCI):
    """BNCI 2014-008 P300 dataset.

    Dataset from [1]_.

    **Dataset description**

    This dataset represents a complete record of P300 evoked potentials
    using a paradigm originally described by Farwell and Donchin [2]_.
    In these sessions, 8 users with amyotrophic lateral sclerosis (ALSO)
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

    Notes
    -----
    .. versionadded:: 0.4.0

    This is a clinically important dataset featuring participants with
    amyotrophic lateral sclerosis (ALS). ALS is a progressive neurodegenerative
    disease that affects motor neurons, eventually leading to loss of voluntary
    muscle control. P300-based BCIs offer a communication pathway for individuals
    with ALS who may lose the ability to speak or use conventional interfaces.

    Key aspects of this dataset:

    - Clinical population: All participants have ALS diagnosis
    - BCI-naive users: No prior BCI training or experience
    - Copy spelling task: 7 predefined words (35 characters total)
    - Calibration + online: First 3 runs calibration, last 4 runs with feedback

    The study investigated the relationship between attention and P300 BCI
    performance in ALS patients, finding that attentional capabilities
    significantly influence BCI accuracy.

    See Also
    --------
    BNCI2014_009 : P300 speller with healthy controls
    BNCI2015_003 : P300 speller with healthy naive users

    Examples
    --------
    >>> from moabb.datasets import BNCI2014_008
    >>> dataset = BNCI2014_008()
    >>> dataset.subject_list
    [1, 2, 3, 4, 5, 6, 7, 8]
    """

    _participant_demographics = {
        "n_subjects": 8,
        "health_status": "ALS patients (amyotrophic lateral sclerosis)",
        "bci_experience": "naive (no prior BCI training)",
        "location": "Fondazione Santa Lucia, Rome, Italy",
        "clinical_note": "All participants diagnosed with ALS",
    }

    ARTICLE_METADATA = {
        "n_subjects": 8,
        "sessions_per_subject": 1,
        "sampling_rate": 256,
        "n_channels": 8,
        "channel_types": {"eeg": 8},
        "reference": "right earlobe",
        "ground": "left mastoid",
        "paradigm": "p300",
        "events": {"Target": 2, "NonTarget": 1},
        "doi": "10.3389/fnhum.2013.00732",
        "license": "CC BY 4.0",
        "montage": "standard_1005",
        "equipment": "g.MOBILAB, g.tec, Austria; g.Ladybird active electrodes",
        "data_url": "http://bnci-horizon-2020.eu/database/data-sets/008-2014/",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 9)),
            sessions_per_subject=1,
            events={"Target": 2, "NonTarget": 1},
            code="BNCI2014-008",
            interval=[0, 1.0],
            paradigm="p300",
            doi="10.3389/fnhum.2013.00732",
        )


@depreciated_alias("BNCI2014009", "1.1")
class BNCI2014_009(MNEBNCI):
    """BNCI 2014-009 P300 dataset.

    Dataset from [1]_.

    **Dataset description**

    This dataset presents a complete record of P300 evoked potentials
    using two different paradigms: a paradigm based on the P300 Speller in
    overt attention condition and a paradigm based used in convert attention
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

    **Participant Demographics**

    - 10 healthy subjects
    - Gender: All female (10/10)
    - Age: Mean = 26.8 ± 5.6 years
    - BCI experience: Previous experience with P300-based BCIs
    - Location: Neuroelectrical Imaging and BCI Laboratory, IRCCS Fondazione
      Santa Lucia, Rome, Italy

    References
    ----------
    .. [1] P Aricò, F Aloise, F Schettini, S Salinari, D Mattia and F Cincotti
           (2013). Influence of P300 latency jitter on event related potential-
           based brain–computer interface performance. Journal of Neural
           Engineering, vol. 11, number 3.
    """

    # Participant demographics from paper
    _participant_demographics = {
        "n_subjects": 10,
        "gender": {"female": 10, "male": 0},
        "age_mean": 26.8,
        "age_std": 5.6,
        "health_status": "healthy subjects",
        "bci_experience": "previous experience with P300-based BCIs",
        "location": "IRCCS Fondazione Santa Lucia, Rome, Italy",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=3,
            events={"Target": 2, "NonTarget": 1},
            code="BNCI2014-009",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.1088/1741-2560/11/3/035008",
        )


@depreciated_alias("BNCI2015001", "1.1")
class BNCI2015_001(MNEBNCI):
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
    period (sec- and 8).  A trial started with 3 s of reference period,
    followed by a brisk audible cue and a visual cue (arrow right for right
    hand, arrow down for both feet) from second 3 to 4.25.
    The activity period, where the users received feedback, lasted from
    second 4 to 8. There was a random 2 to 3 s pause between the trials.

    **Participant Demographics**

    - 12 novice users
    - BCI experience: Novice to motor imagery BCIs
    - Performance: 10 out of 12 participants reached >70% accuracy within
      1-3 sessions (10-80 min online time), with a median accuracy of
      80.2 ± 11.3% in the last session
    - Location: Institute for Knowledge Discovery, Graz University of
      Technology, Austria

    References
    ----------
    .. [1] J. Faller, C. Vidaurre, T. Solis-Escalante, C. Neuper and R.
           Scherer (2012). Autocalibration and recurrent adaptation: Towards a
           plug and play online ERD- BCI.  IEEE Transactions on Neural Systems
           and Rehabilitation Engineering, 20(3), 313-319.
    """

    # Participant demographics from paper
    _participant_demographics = {
        "n_subjects": 12,
        "bci_experience": "novice users",
        "performance_note": "10/12 reached >70% accuracy, median 80.2±11.3%",
        "location": "Graz University of Technology, Austria",
    }

    def __init__(self):
        # FIXME: some participant have 3 sessions
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=2,
            events={"right_hand": 1, "feet": 2},
            code="BNCI2015-001",
            interval=[0, 5],
            paradigm="imagery",
            doi="10.1109/tnsre.2012.2189584",
        )


@depreciated_alias("BNCI2015003", "1.1")
class BNCI2015_003(MNEBNCI):
    """BNCI 2015-003 P300 Speller dataset.

    .. admonition:: Dataset summary

        ============= ======= ======= =================== =============== =============== ============
        Name          #Subj   #Chan   #Trials/class       Trials length   Sampling Rate   #Sessions
        ============= ======= ======= =================== =============== =============== ============
        BNCI2015_003  10      8       Variable T/NT       0.8s            256Hz           1
        ============= ======= ======= =================== =============== =============== ============

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG recordings from 10 subjects performing a visual
    P300 spelling task using the g.tec intendix system. The study investigated
    P300-based BCI performance across a large population (100 subjects tested),
    with this dataset containing a subset of 10 participants.

    The paradigm uses a standard 6×6 visual P300 matrix speller (row-column
    paradigm). Subjects focus on the target character while rows and columns
    of the matrix flash in random order. The attended character elicits a P300
    response when its row or column is highlighted, enabling character selection
    through EEG classification.

    **Participants**

    - 10 subjects (from a larger study of 100 participants)
    - Original study tested 100 subjects to determine BCI usability rates
    - Age range: Not specified in original publication
    - BCI experience: Naive users (first-time P300 BCI users)
    - Location: g.tec medical engineering, Graz, Austria

    **Recording Details**

    - Equipment: g.tec intendix P300 BCI system
    - Channels: 8 EEG electrodes
    - Electrode positions: Fz, Cz, P3, Pz, P4, PO7, Oz, PO8
    - Reference: Right earlobe
    - Ground: Not specified
    - Sampling rate: 256 Hz
    - Electrode type: g.LADYbird active electrodes

    **Experimental Procedure**

    The P300 matrix speller paradigm:

    - 6×6 character matrix displayed on screen
    - Rows and columns flash in pseudo-random sequence
    - Subject attends to target character
    - Target row/column flashes elicit P300 response
    - Multiple repetitions per character for averaging
    - Epoch length: 0.8 seconds post-stimulus

    Trial structure:

    - Copy spelling task: Subjects spell predefined words
    - Training phase: WATER (5 characters)
    - Testing phase: LUCAS (5 characters)
    - Online feedback provided during testing

    **Event Codes**

    - Target (2): Attended stimulus (character's row or column flashing)
    - NonTarget (1): Non-attended stimuli (other rows/columns flashing)

    **Data Organization**

    - 1 session per subject
    - Data stored in MAT format
    - Single run per session

    **Performance Results**

    From the original study of 100 subjects:

    - 72.8% achieved >70% accuracy (high performers)
    - 18.6% achieved 30-70% accuracy (intermediate)
    - 8.6% achieved <30% accuracy (low performers)
    - The study demonstrated that the majority of naive users can successfully
      operate a P300-based BCI without prior training.

    References
    ----------
    .. [1] Guger, C., Daban, S., Sellers, E., Holzner, C., Krausz, G.,
           Carabalona, R., Gramatica, F., & Edlinger, G. (2009). How many
           people are able to control a P300-based brain-computer interface
           (BCI)?. Neuroscience Letters, 462(1), 94-98.
           https://doi.org/10.1016/j.neulet.2009.06.045

    Notes
    -----
    .. versionadded:: 0.4.0

    This dataset is particularly useful for:

    - Studying P300 BCI performance in naive users
    - Comparing classification algorithms on a standard P300 task
    - Investigating individual differences in BCI performance
    - Baseline comparisons with minimal electrode setups (8 channels)

    The g.tec intendix system used in this study is a commercial BCI system
    designed for communication applications, particularly for patients with
    severe motor impairments (e.g., ALS, locked-in syndrome).

    See Also
    --------
    BNCI2014_008 : P300 speller with ALS patients and healthy controls
    BNCI2015_010 : RSVP gaze-independent P300 paradigm
    BNCI2015_012 : PASS2D auditory P300 paradigm

    Examples
    --------
    >>> from moabb.datasets import BNCI2015_003
    >>> dataset = BNCI2015_003()
    >>> dataset.subject_list
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """

    _participant_demographics = {
        "n_subjects": 10,
        "original_study_subjects": 100,
        "health_status": "healthy subjects",
        "bci_experience": "naive users",
        "location": "g.tec medical engineering, Graz, Austria",
    }

    ARTICLE_METADATA = {
        "n_subjects": 10,
        "sessions_per_subject": 1,
        "sampling_rate": 256,
        "n_channels": 8,
        "channel_types": {"eeg": 8},
        "reference": "right earlobe",
        "paradigm": "p300",
        "events": {"Target": 2, "NonTarget": 1},
        "doi": "10.1016/j.neulet.2009.06.045",
        "license": "CC BY 4.0",
        "montage": "standard_1005",
        "equipment": "g.tec intendix P300 BCI system",
        "data_url": "http://bnci-horizon-2020.eu/database/data-sets/003-2015/",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events={"Target": 2, "NonTarget": 1},
            code="BNCI2015-003",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.1016/j.neulet.2009.06.045",
        )


@depreciated_alias("BNCI2015004", "1.1")
class BNCI2015_004(MNEBNCI):
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
    amplifiers (Guger Technologies, Graz, Austria) was used for recording.  EEG
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

    **Participant Demographics**

    - 9 individuals with severe motor disabilities
    - Gender: 7 female, 2 male
    - Age range: 20-57 years (median = 38 years, SD = 10)
    - Disabilities: Spinal cord injury (SCI) or stroke, with varying time
      since onset
    - Functional status: Some participants had tetraplegia
    - BCI experience: Naive to the task (no previous BCI training)
    - Recruitment: Attended rehabilitation program at Guttmann Institute,
      Barcelona, Spain
    - Location: Institute for Knowledge Discovery, Graz University of
      Technology, Austria (data collection site: Guttmann Institute, Spain)

    Notes
    -----
    This dataset is unique in MOABB as it includes participants with severe
    motor disabilities rather than healthy volunteers, making it valuable for
    evaluating BCI performance in the target end-user population. Individual
    participant details including months since injury/stroke and specific
    disability types are documented in Table 1 of the reference paper.

    References
    ----------
    .. [1] Scherer R, Faller J, Friedrich EVC, Opisso E, Costa U, Kübler A, et
           al. (2015) Individually Adapted Imagery Improves Brain-Computer
           Interface Performance in End-Users with Disability. PLoS ONE 10(5).
           https://doi.org/10.1371/journal.pone.0123727
    """

    # Participant demographics from paper
    _participant_demographics = {
        "n_subjects": 9,
        "gender": {"female": 7, "male": 2},
        "age_range": (20, 57),
        "age_median": 38,
        "age_std": 10,
        "health_status": "individuals with severe motor disabilities",
        "disability_types": ["spinal cord injury", "stroke"],
        "functional_status": "some with tetraplegia",
        "bci_experience": "naive (no previous BCI training)",
        "location": "Guttmann Institute, Barcelona, Spain (rehabilitation center)",
        "data_provider": "Graz University of Technology, Austria",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=2,
            events=dict(right_hand=4, feet=5, navigation=3, subtraction=2, word_ass=1),
            code="BNCI2015-004",
            interval=[3, 10],
            paradigm="imagery",
            doi="10.1371/journal.pone.0123727",
        )


class BNCI2015_006(MNEBNCI):
    """BNCI 2015-006 Music BCI (Auditory Attention) dataset.

    .. admonition:: Dataset summary

        ============= ======= ======= ================= =============== =============== ============
        Name          #Subj   #Chan   #Trials/class     Trials length   Sampling Rate   #Sessions
        ============= ======= ======= ================= =============== =============== ============
        BNCI2015_006  11      64      variable          variable        100Hz           1
        ============= ======= ======= ================= =============== =============== ============

    Dataset from [1]_.

    **Dataset Description**

    This dataset investigates the suitability of musical stimuli for use in a P300
    paradigm. 11 subjects listened to polyphonic music clips featuring three
    instruments playing together. A multi-streamed oddball paradigm was used.

    References
    ----------
    .. [1] Treder, M. S., Purwins, H., Miklody, D., Sturm, I., & Blankertz, B.
           (2014). Decoding auditory attention to instruments in polyphonic music
           using single-trial EEG classification. Journal of Neural Engineering,
           11(2), 026009. https://doi.org/10.1088/1741-2560/11/2/026009

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    _participant_demographics = {
        "n_subjects": 11,
        "health_status": "healthy subjects",
        "location": "Technische Universitat Berlin, Germany",
    }

    ARTICLE_METADATA = {
        "n_subjects": 11,
        "sessions_per_subject": 1,
        "sampling_rate": 100,
        "n_channels": 64,
        "paradigm": "p300",
        "events": {"attended_deviant": 1, "unattended_deviant": 2},
        "doi": "10.1088/1741-2560/11/2/026009",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 12)),
            sessions_per_subject=1,
            events={"attended_deviant": 1, "unattended_deviant": 2},
            code="BNCI2015-006",
            interval=[0, 1.0],
            paradigm="p300",
            doi="10.1088/1741-2560/11/2/026009",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        sessions = load_data(subject=subject, dataset=self.code, verbose=False)
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the data paths of the dataset."""
        return load_data(
            subject=subject,
            dataset=self.code,
            verbose=verbose,
            update_path=update_path,
            path=path,
            force_update=force_update,
            only_filenames=True,
        )


class BNCI2015_007(MNEBNCI):
    """BNCI 2015-007 Motion VEP (mVEP) Speller dataset.

    .. admonition:: Dataset summary

        ============= ======= ======= =================== =============== =============== ============
        Name          #Subj   #Chan   #Trials/class       Trials length   Sampling Rate   #Sessions
        ============= ======= ======= =================== =============== =============== ============
        BNCI2015_007  16      63      ~1800 NT / ~360 T   0.7s            100Hz           1
        ============= ======= ======= =================== =============== =============== ============

    Dataset from [1]_.

    **Dataset Description**

    This dataset implements a motion-onset visual evoked potential (mVEP) based
    brain-computer interface for gaze-independent spelling. Unlike conventional
    flash-based P300 spellers that use luminance changes, this paradigm uses
    motion onset (moving bar stimuli) to elicit visual evoked potentials,
    specifically the N200 component. This approach has advantages including
    lower visual fatigue, reduced luminance and contrast requirements, and
    potential for use in bright environments.

    The motion VEP (mVEP) speller operates by presenting moving bar stimuli at
    different positions in a matrix layout. When the user attends to a target
    position, the motion onset at that location elicits a characteristic N200
    response that can be detected to determine the user's intended selection.

    **Participants**

    - 16 healthy subjects
    - Gender: Not specified in metadata
    - Age: Not specified in metadata
    - BCI experience: Not specified
    - Health status: Healthy volunteers
    - Location: Neurotechnology Group, Technische Universitat Berlin, Germany

    **Recording Details**

    - Equipment: BrainProducts actiCap active electrode system
    - Channels: 63 EEG electrodes (standard 10-10 system)
    - Sampling rate: 100 Hz (downsampled from original recording)
    - Reference: Nose reference
    - Montage: standard_1005
    - Filters: Bandpass filtered during preprocessing
    - Units: uV (converted to V during loading)

    **Experimental Procedure**

    - 6x6 matrix speller layout (36 possible targets)
    - Motion onset stimulation (moving bars)
    - 6 stimulus positions per row/column
    - Overt attention paradigm (gaze-dependent) and covert attention modes
    - One recording session per subject with multiple runs (typically 2)
    - Each run contains multiple spelling sequences

    **Data Organization**

    Each subject has one MAT file containing 2 recording runs. File naming
    convention: ``MVEP_VP{subject_code}.mat`` (e.g., MVEP_VPfat.mat).
    Subject codes (16 subjects): fat, gdf, gdg, iac, iba, ibe, ibq, ibs,
    ibt, ibu, ibv, ibw, ibx, iby, ice, icv.

    **Event Codes**

    - Target (1): Target stimulus attended by user (elicits N200 response)
    - Non-target (2): Non-target stimulus (ignore)
    - Stim1-Stim6 (3-8): Additional stimulus position codes

    **Clinical Relevance**

    The mVEP paradigm offers an alternative to flash-based P300 spellers that
    may be beneficial for users with photosensitivity or in environments where
    bright flashes are undesirable. The reduced luminance and contrast
    requirements make this paradigm more comfortable for extended use.

    References
    ----------
    .. [1] Schaeff, S., Treder, M. S., Venthur, B., & Blankertz, B. (2012).
           Exploring motion VEPs for gaze-independent communication.
           Journal of Neural Engineering, 9(4), 045006.
           https://doi.org/10.1088/1741-2560/9/4/045006

    Notes
    -----
    .. versionadded:: 1.2.0

    The N200 component is a negative deflection occurring approximately
    200 ms after motion onset. It is modulated by attention and provides
    the basis for target detection in this paradigm.

    This dataset uses BBCI format and is loaded via the ``_convert_bbci``
    function, similar to AMUSE, RSVP, and PASS2D datasets.

    See Also
    --------
    BNCI2015_009 : AMUSE auditory speller (alternative attention-based paradigm)
    BNCI2015_010 : RSVP visual speller (gaze-independent visual paradigm)
    """

    _participant_demographics = {
        "n_subjects": 16,
        "health_status": "healthy subjects",
        "bci_experience": "not specified",
        "location": "Neurotechnology Group, TU Berlin, Germany",
        "subject_codes": [
            "fat",
            "gdf",
            "gdg",
            "iac",
            "iba",
            "ibe",
            "ibq",
            "ibs",
            "ibt",
            "ibu",
            "ibv",
            "ibw",
            "ibx",
            "iby",
            "ice",
            "icv",
        ],
    }

    # Technical metadata for programmatic access
    ARTICLE_METADATA = {
        "n_subjects": 16,
        "sessions_per_subject": 1,
        "sampling_rate": 100,
        "n_channels": 63,
        "channel_types": {"eeg": 63},
        "reference": "nose",
        "montage": "standard_1005",
        "file_format": "MAT",
        "paradigm": "p300",  # Uses oddball-like Target/Non-target detection
        "data_url": "http://doc.ml.tu-berlin.de/bbci/BNCIHorizon2020-MVEP/",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 17)),
            sessions_per_subject=1,
            events={"Target": 1, "Non-target": 2},
            code="BNCI2015-007",
            interval=[0, 0.7],
            paradigm="p300",  # Oddball-like paradigm with Target/Non-target
            doi="10.1088/1741-2560/9/4/045006",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject.

        This dataset returns raws and event_id from load_data, so we need
        to convert to the standard sessions format.
        """
        raws, event_id = load_data(subject=subject, dataset=self.code, verbose=False)
        # Convert list of raws to sessions format
        # Single session with multiple runs
        sessions = {"0": {str(ii): raw for ii, raw in enumerate(raws)}}
        return sessions


class BNCI2015_008(MNEBNCI):
    """BNCI 2015-008 Center Speller P300 dataset.

    .. admonition:: Dataset summary

        ============= ======= ======= =================== =============== =============== ============
        Name          #Subj   #Chan   #Trials/class       Trials length   Sampling Rate   #Sessions
        ============= ======= ======= =================== =============== =============== ============
        BNCI2015_008  13      63      ~1180 T / ~5900 NT  1.0s            250Hz           2
        ============= ======= ======= =================== =============== =============== ============

    Dataset from [1]_, also known as Treder2011.

    **Dataset Description**

    This dataset contains P300 evoked potentials recorded during a gaze-independent
    two-stage visual speller paradigm called the "Center Speller". Unlike traditional
    matrix spellers that require gaze fixation on target cells, the Center Speller
    allows users to focus on the screen center while covertly attending to peripheral
    stimuli.

    The paradigm uses a two-stage selection process where users first select a group
    of characters, then select individual characters within that group. This design
    enables efficient spelling without requiring eye movements, making it suitable
    for users with severe motor disabilities affecting eye control.

    **Participants**

    - 13 healthy subjects
    - BCI experience: Previous experience with P300-based BCIs
    - Location: Machine Learning Laboratory, TU Berlin, Germany

    **Recording Details**

    - Channels: 63 EEG electrodes (standard 10-10 system)
    - Sampling rate: 250 Hz
    - Reference: Nose reference

    **Data Organization**

    - Subject codes: iac, iba, ibb, ibc, ibd, ibe, ibf, ibg, ibh, ibi, ibj, ica, saf
    - Two runs per subject (calibration + online)
    - Data URL: http://doc.ml.tu-berlin.de/bbci/BNCIHorizon2020-CenterSpeller/

    **Event Codes**

    - Target (1): Target stimulus presented (attended)
    - NonTarget (2): Non-target stimulus presented (not attended)

    References
    ----------
    .. [1] Treder, M. S., Schmidt, N. M., & Blankertz, B. (2011). Gaze-independent
           brain-computer interfaces based on covert attention and feature attention.
           Journal of Neural Engineering, 8(6), 066003.
           https://doi.org/10.1088/1741-2560/8/6/066003

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    _participant_demographics = {
        "n_subjects": 13,
        "health_status": "healthy subjects",
        "bci_experience": "previous experience with P300-based BCIs",
        "location": "Machine Learning Laboratory, TU Berlin, Germany",
        "subject_codes": [
            "iac",
            "iba",
            "ibb",
            "ibc",
            "ibd",
            "ibe",
            "ibf",
            "ibg",
            "ibh",
            "ibi",
            "ibj",
            "ica",
            "saf",
        ],
    }

    # Technical metadata for programmatic access
    ARTICLE_METADATA = {
        "n_subjects": 13,
        "sessions_per_subject": 2,
        "sampling_rate": 250,
        "n_channels": 63,
        "channel_types": {"eeg": 63},
        "reference": "nose",
        "montage": "standard_1005",
        "file_format": "MAT",
        "paradigm": "p300",
        "data_url": "http://doc.ml.tu-berlin.de/bbci/BNCIHorizon2020-CenterSpeller/",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 14)),
            sessions_per_subject=2,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-008",
            interval=[0, 1.0],
            paradigm="p300",
            doi="10.1088/1741-2560/8/6/066003",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raws, event_id = load_data(subject=subject, dataset=self.code, verbose=False)
        sessions = {"0": {str(ii): raw for ii, raw in enumerate(raws)}}
        return sessions


class BNCI2015_009(MNEBNCI):
    """BNCI 2015-009 AMUSE (Auditory Multi-class Spatial ERP) dataset.

    .. admonition:: Dataset summary

        ============= ======= ======= =================== =============== =============== ============
        Name          #Subj   #Chan   #Trials/class       Trials length   Sampling Rate   #Sessions
        ============= ======= ======= =================== =============== =============== ============
        BNCI2015_009  21      62      Variable T/NT       0.8s            1000Hz          varies
        ============= ======= ======= =================== =============== =============== ============

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG recordings from 21 subjects performing an
    auditory spatial attention task for brain-computer interface (BCI) control.
    The AMUSE (Auditory Multi-class Spatial ERP) paradigm uses auditory stimuli
    from different spatial locations to elicit P300-like event-related potentials.

    Subjects were presented with auditory stimuli (75 ms bandpass filtered white
    noise, 150-8000 Hz) from 8 loudspeakers arranged at ear height in a circle
    around the subject, with 45 degree spacing at approximately 1 meter distance.
    By attending to stimuli from a specific spatial location, subjects could
    select one of multiple targets, enabling multi-class BCI control without
    relying on visual stimulation.

    **Participants**

    - 21 healthy subjects
    - Location: Berlin Institute of Technology, Germany

    **Recording Details**

    - Equipment: 128-channel Brain Products amplifier
    - Channels: 60 EEG + 2 EOG (62 total)
    - Electrode type: Ag/AgCl electrodes
    - Sampling rate: 1000 Hz (downsampled to 100 Hz for analysis in original paper)
    - Auditory stimuli: 75 ms bandpass filtered white noise (150-8000 Hz), 58 dB
    - Speaker setup: 8 speakers at ear height, 45 degree spacing, ~1 meter distance

    **Experimental Procedure**

    - Auditory oddball paradigm with spatial cues
    - 8 possible target locations (loudspeakers)
    - Subjects attend to stimuli from a designated target location
    - Target stimuli (attended location) elicit P300-like responses
    - Non-target stimuli (non-attended locations) serve as controls
    - Variable number of sessions per subject

    **Event Codes**

    - Target (2): Attended auditory stimulus (from target location)
    - NonTarget (1): Non-attended auditory stimulus (from other locations)

    References
    ----------
    .. [1] Schreuder, M., Blankertz, B., & Tangermann, M. (2010). A New Auditory
           Multi-Class Brain-Computer Interface Paradigm: Spatial Hearing as an
           Informative Cue. PLoS ONE, 5(4), e9813.
           https://doi.org/10.1371/journal.pone.0009813

    Notes
    -----
    .. versionadded:: 1.3.0

    This dataset uses an auditory spatial attention paradigm, which provides
    an alternative to visual P300 paradigms for users with visual impairments.

    See Also
    --------
    BNCI2015_010 : RSVP visual P300 paradigm
    BNCI2015_012 : PASS2D auditory P300 paradigm
    """

    _participant_demographics = {
        "n_subjects": 21,
        "health_status": "healthy subjects",
        "location": "Berlin Institute of Technology, Germany",
    }

    ARTICLE_METADATA = {
        "n_subjects": 21,
        "sessions_per_subject": "varies",
        "sampling_rate": 1000,
        "n_channels": 62,
        "channel_types": {"eeg": 60, "eog": 2},
        "paradigm": "p300",
        "events": {"NonTarget": 1, "Target": 2},
        "doi": "10.1371/journal.pone.0009813",
        "data_url": "http://doc.ml.tu-berlin.de/bbci/BNCIHorizon2020-AMUSE/",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 22)),
            sessions_per_subject=1,
            events={"NonTarget": 1, "Target": 2},
            code="BNCI2015-009",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.1371/journal.pone.0009813",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raws, _ = load_data(subject=subject, dataset=self.code, verbose=False)
        sessions = {}
        for ii, raw in enumerate(raws):
            sessions[str(ii)] = {"0": raw}
        return sessions


class BNCI2015_010(MNEBNCI):
    """BNCI 2015-010 RSVP (Rapid Serial Visual Presentation) P300 dataset.

    .. admonition:: Dataset summary

        ============= ======= ======= =================== =============== =============== ============
        Name          #Subj   #Chan   #Trials/class       Trials length   Sampling Rate   #Sessions
        ============= ======= ======= =================== =============== =============== ============
        BNCI2015_010  12      63      ~180 T / ~5220 NT   0.8s            1000Hz          1
        ============= ======= ======= =================== =============== =============== ============

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG recordings from 12 healthy subjects performing a
    gaze-independent P300-based spelling task using Rapid Serial Visual
    Presentation (RSVP). Unlike traditional matrix spellers that require gaze
    fixation on target cells, the RSVP paradigm presents symbols one-by-one at
    a central location, allowing users to spell without eye movements.

    The RSVP approach is particularly valuable for users with limited eye
    movement control, such as those with advanced amyotrophic lateral sclerosis
    (ALS) or other conditions affecting oculomotor function.

    **Paradigm Details**

    - Gaze-independent visual P300 speller
    - 30 symbols presented one-by-one in pseudo-random sequence at screen center
    - Subjects attend to target symbols while ignoring non-targets
    - Mean spelling rate: 1.43 symbols/minute
    - Mean online accuracy: 94.8%

    **Participants**

    - 12 healthy subjects
    - BCI experience: Previous experience with P300-based BCIs
    - Location: Machine Learning Laboratory, TU Berlin, Germany

    **Recording Details**

    - Equipment: BrainAmp amplifier (Brain Products GmbH)
    - Channels: 63 EEG electrodes (standard 10-10 system)
    - Sampling rate: 1000 Hz
    - Reference: Nose reference

    **Event Codes**

    - Target (1): Target stimulus presented (attended symbol)
    - NonTarget (2): Non-target stimulus presented (ignored symbol)

    References
    ----------
    .. [1] Acqualagna, L., & Blankertz, B. (2013). Gaze-independent BCI-spelling
           using rapid serial visual presentation (RSVP). Clinical Neurophysiology,
           124(5), 901-908. https://doi.org/10.1016/j.clinph.2012.12.050

    Notes
    -----
    .. versionadded:: 1.3.0

    The RSVP paradigm eliminates the need for saccadic eye movements, making it
    suitable for users who cannot control gaze direction.

    See Also
    --------
    BNCI2015_008 : Center Speller gaze-independent P300 paradigm
    BNCI2015_009 : AMUSE auditory spatial P300 paradigm
    """

    _participant_demographics = {
        "n_subjects": 12,
        "health_status": "healthy subjects",
        "bci_experience": "previous experience with P300-based BCIs",
        "location": "Machine Learning Laboratory, TU Berlin, Germany",
        "subject_codes": [
            "fat",
            "gcb",
            "gcc",
            "gcd",
            "gce",
            "gcf",
            "gcg",
            "gch",
            "iay",
            "icn",
            "icr",
            "pia",
        ],
    }

    ARTICLE_METADATA = {
        "n_subjects": 12,
        "sessions_per_subject": 1,
        "sampling_rate": 1000,
        "n_channels": 63,
        "channel_types": {"eeg": 63},
        "paradigm": "p300",
        "events": {"Target": 1, "NonTarget": 2},
        "doi": "10.1016/j.clinph.2012.12.050",
        "data_url": "http://doc.ml.tu-berlin.de/bbci/BNCIHorizon2020-RSVP/",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-010",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.1016/j.clinph.2012.12.050",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raws, _ = load_data(subject=subject, dataset=self.code, verbose=False)
        sessions = {}
        for ii, raw in enumerate(raws):
            sessions[str(ii)] = {"0": raw}
        return sessions


class BNCI2015_012(MNEBNCI):
    """BNCI 2015-012 PASS2D (Predictive Auditory Spatial Speller with 2D stimuli) dataset.

    .. admonition:: Dataset summary

        ============= ======= ======= =================== =============== =============== ============
        Name          #Subj   #Chan   #Trials/class       Trials length   Sampling Rate   #Sessions
        ============= ======= ======= =================== =============== =============== ============
        BNCI2015_012  12      63      Variable T/NT       0.8s            1000Hz          1
        ============= ======= ======= =================== =============== =============== ============

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG recordings from 12 subjects using the PASS2D
    (Predictive Auditory Spatial Speller with 2D stimuli) paradigm. The system
    employs a 2D auditory oddball paradigm where stimuli vary along two dimensions:
    pitch (low, medium, high) and spatial direction (left, middle, right), creating
    a 3x3 matrix of 9 unique auditory stimuli.

    The PASS2D paradigm is designed for gaze-independent brain-computer interface
    (BCI) communication, achieving spelling speeds of 0.8+ characters per minute
    with an information transfer rate of approximately 3.4 bits/min.

    **Participants**

    - 12 healthy subjects (9 male, 3 female)
    - Age range: 21-34 years (mean 25.1 years)
    - All non-smokers with normal hearing
    - Location: Berlin Institute of Technology (TU Berlin), Germany

    **Recording Details**

    - Equipment: Fast'n Easy Cap with 63 wet Ag/AgCl electrodes
    - Amplifiers: Two 32-channel Brain Products amplifiers
    - Headphones: Sennheiser PMX 200 neckband headphones
    - Channels: 63 EEG
    - Reference: Nose
    - Sampling rate: 1000 Hz

    **Experimental Procedure**

    The paradigm uses a 2D auditory stimulus matrix:

    - Pitch dimension: Low (~370 Hz), Medium (~494 Hz), High (~659 Hz)
    - Spatial dimension: Left, Middle (center), Right
    - Total: 9 unique stimulus combinations (3 pitch x 3 direction)

    **Event Codes**

    - Target (1): Attended stimulus (elicits P300 response)
    - NonTarget (2): Non-attended stimuli

    References
    ----------
    .. [1] Hohne, J., Schreuder, M., Blankertz, B., & Tangermann, M. (2011).
           A Novel 9-Class Auditory ERP Paradigm Driving a Predictive Text Entry
           System. Frontiers in Neuroscience, 5, 99.
           https://doi.org/10.3389/fnins.2011.00099

    Notes
    -----
    .. versionadded:: 1.3.0

    The PASS2D paradigm extends traditional auditory oddball from 1D to 2D,
    increasing the number of possible targets from 2-3 to 9.

    See Also
    --------
    BNCI2015_009 : AMUSE auditory spatial P300 paradigm (1D spatial)
    BNCI2015_010 : RSVP visual P300 paradigm
    """

    _participant_demographics = {
        "n_subjects": 12,
        "gender": {"male": 9, "female": 3},
        "age_range": (21, 34),
        "age_mean": 25.1,
        "health_status": "healthy non-smokers with normal hearing",
        "location": "Berlin Institute of Technology (TU Berlin), Germany",
        "subject_codes": [
            "nv",
            "nw",
            "nx",
            "ny",
            "nz",
            "mg",
            "oa",
            "ob",
            "oc",
            "od",
            "ja",
            "oe",
        ],
    }

    ARTICLE_METADATA = {
        "n_subjects": 12,
        "sessions_per_subject": 1,
        "sampling_rate": 1000,
        "n_channels": 63,
        "channel_types": {"eeg": 63},
        "paradigm": "p300",
        "events": {"Target": 1, "NonTarget": 2},
        "doi": "10.3389/fnins.2011.00099",
        "data_url": "http://doc.ml.tu-berlin.de/bbci/BNCIHorizon2020-PASS2D/",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 2},
            code="BNCI2015-012",
            interval=[0, 0.8],
            paradigm="p300",
            doi="10.3389/fnins.2011.00099",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raws, _ = load_data(subject=subject, dataset=self.code, verbose=False)
        sessions = {}
        for ii, raw in enumerate(raws):
            sessions[str(ii)] = {"0": raw}
        return sessions


class BNCI2015_013(MNEBNCI):
    """BNCI 2015-013 Error-Related Potentials dataset.

    .. admonition:: Dataset summary

        ============= ======= ======= =================== =============== =============== ============
        Name          #Subj   #Chan   #Trials/class       Trials length   Sampling Rate   #Sessions
        ============= ======= ======= =================== =============== =============== ============
        BNCI2015_013  6       64      Variable            0.6s            512Hz           2
        ============= ======= ======= =================== =============== =============== ============

    Dataset from [1]_.

    **Dataset Description**

    This dataset contains EEG recordings from 6 subjects during an error-related
    potential (ErrP) monitoring task. The paradigm investigates how a
    brain-computer interface can learn from user-detected errors to improve its
    behavior over time, enabling adaptive human-robot interaction.

    Subjects passively observed cursor movements on a screen without providing
    any physical or mental input to control the cursor. The cursor moved
    automatically (controlled by the system), and users evaluated whether each
    movement was toward or away from a target location.

    **Participants**

    - 6 healthy subjects
    - Location: EPFL, CNBI Laboratory, Lausanne, Switzerland

    **Recording Details**

    - Channels: 64 EEG electrodes
    - Key electrode: FCz (grand-average ERP measurements)
    - Digital filtering: Bandpass 1-10 Hz (4th-order Butterworth filter)
    - Paradigm: Cursor control monitoring (passive observation/evaluation)

    **Experimental Procedure**

    - Cursor control monitoring task
    - System-controlled cursor movements (no user physical/mental control input)
    - User indicates whether cursor moves toward or away from target
    - Error trials: cursor moves away from target (user detects error)
    - Correct trials: cursor moves toward target

    **Event Codes**

    - correct (1): Correct cursor movement toward target
    - error (2): Erroneous cursor movement away from target

    References
    ----------
    .. [1] Chavarriaga, R., & Millan, J. d. R. (2010). Learning from EEG
           error-related potentials in noninvasive brain-computer interfaces.
           IEEE Transactions on Neural Systems and Rehabilitation Engineering,
           18(4), 381-388. https://doi.org/10.1109/TNSRE.2010.2053387

    Notes
    -----
    .. versionadded:: 1.3.0

    This dataset uses an error-related potential paradigm where subjects
    passively observe system actions and the BCI learns from detected errors.

    See Also
    --------
    BNCI2015_003 : P300 visual speller paradigm
    BNCI2015_009 : AMUSE auditory spatial P300 paradigm
    """

    _participant_demographics = {
        "n_subjects": 6,
        "health_status": "healthy subjects",
        "location": "EPFL, CNBI Laboratory, Lausanne, Switzerland",
    }

    ARTICLE_METADATA = {
        "n_subjects": 6,
        "sessions_per_subject": 2,
        "n_channels": 64,
        "channel_types": {"eeg": 64},
        "paradigm": "erp",
        "events": {"correct": 1, "error": 2},
        "doi": "10.1109/TNSRE.2010.2053387",
        "data_url": "http://bnci-horizon-2020.eu/database/data-sets/013-2015/",
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 7)),
            sessions_per_subject=2,
            events={"correct": 1, "error": 2},
            code="BNCI2015-013",
            interval=[0, 0.6],
            paradigm="erp",
            doi="10.1109/TNSRE.2010.2053387",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        raws, _ = load_data(subject=subject, dataset=self.code, verbose=False)
        sessions = {}
        for ii, raw in enumerate(raws):
            sessions[str(ii)] = {"0": raw}
        return sessions


class BNCI2019_001(BaseDataset):
    """BNCI 2019-001 Motor Imagery dataset for Spinal Cord Injury patients.

    .. admonition:: Dataset summary

        ============= ======= ======= ================= =============== =============== ============
        Name          #Subj   #Chan   #Trials/class     Trials length   Sampling Rate   #Sessions
        ============= ======= ======= ================= =============== =============== ============
        BNCI2019_001  10      61+3EOG 72 per class      3s              256Hz           1
        ============= ======= ======= ================= =============== =============== ============

    Dataset from [1]_.

    **Dataset Description**

    This dataset consists of EEG recordings from 10 participants with cervical
    spinal cord injury (SCI) performing attempted hand and arm movements.

    Participants attempted five movement types: supination, pronation, hand open,
    palmar grasp, and lateral grasp.

    **Participants**

    - 10 participants with cervical spinal cord injury
    - Age range: 20-69 years
    - Gender: 9 male, 1 female

    **Recording Details**

    - Channels: 61 EEG + 3 EOG electrodes
    - Sampling rate: 256 Hz
    - Reference: Left earlobe

    **Motor Imagery Classes**

    - supination (776): Forearm supination
    - pronation (777): Forearm pronation
    - hand_open (779): Hand opening movement
    - palmar_grasp (925): Palmar (power) grasp
    - lateral_grasp (926): Lateral (key) grasp

    References
    ----------
    .. [1] Ofner, P. et al. (2019). Attempted arm and hand movements can be
           decoded from low-frequency EEG from persons with spinal cord injury.
           Scientific Reports, 9(1), 7134.
           https://doi.org/10.1038/s41598-019-43594-9

    Notes
    -----
    .. versionadded:: 1.2.0
    """

    _EVENTS = {
        "supination": 776,
        "pronation": 777,
        "hand_open": 779,
        "palmar_grasp": 925,
        "lateral_grasp": 926,
    }

    _MOVEMENT_RUNS = [3, 4, 5, 6, 7, 10, 11, 12, 13]

    _participant_demographics = {
        "n_subjects": 10,
        "gender": {"male": 9, "female": 1},
        "age_range": (20, 69),
        "health_status": "cervical spinal cord injury patients",
    }

    ARTICLE_METADATA = {
        "n_subjects": 10,
        "sessions_per_subject": 1,
        "sampling_rate": 256,
        "n_channels": 64,
        "paradigm": "imagery",
        "doi": "10.1038/s41598-019-43594-9",
        "license": "CC BY 4.0",
        "data_url": BNCI_URL_001_2019,
    }

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events=self._EVENTS,
            code="BNCI2019-001",
            interval=[2, 5],
            paradigm="imagery",
            doi="10.1038/s41598-019-43594-9",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        file_paths = self.data_path(subject)
        montage = make_standard_montage("standard_1005")
        eog_channels = ["eog-l", "eog-m", "eog-r"]
        data = {}
        for run_idx, path in enumerate(file_paths):
            raw = read_raw_gdf(path, eog=eog_channels, preload=True, verbose="ERROR")
            raw.set_montage(montage, on_missing="ignore")
            raw._data[np.isnan(raw._data)] = 0
            raw._data[:-3] *= 1e-6
            stim = raw.annotations.description.astype(np.dtype("<21U"))
            stim[stim == "776"] = "supination"
            stim[stim == "777"] = "pronation"
            stim[stim == "779"] = "hand_open"
            stim[stim == "925"] = "palmar_grasp"
            stim[stim == "926"] = "lateral_grasp"
            raw.annotations.description = stim
            raw.info["line_freq"] = 50.0
            raw.set_meas_date(datetime(2019, 1, 1, tzinfo=timezone.utc))
            data[str(run_idx)] = raw
        return {"0": data}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return paths to data files for a given subject."""
        if subject not in self.subject_list:
            raise ValueError(
                f"Invalid subject number {subject}. Valid: {self.subject_list}"
            )
        url = f"{BNCI_URL_001_2019}P{subject:02d}.zip"
        zip_path = dl.data_dl(url, "BNCI", path, force_update, verbose)
        zip_dir = Path(zip_path).parent
        if not (zip_dir / f"P{subject:02d} Run 3.gdf").exists():
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(zip_dir)
        paths = []
        for run in self._MOVEMENT_RUNS:
            gdf_path = zip_dir / f"P{subject:02d} Run {run}.gdf"
            if gdf_path.exists():
                paths.append(str(gdf_path))
        if not paths:
            raise FileNotFoundError(f"No GDF files found for subject {subject}")
        return paths
