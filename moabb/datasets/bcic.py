import io
import zipfile
from pathlib import Path

import numpy as np
from mne import Annotations, create_info
from mne.io import RawArray
from mne.utils import verbose
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


BCIC_URL = "https://www.bbci.de/competition/"


class MNEBCIC(BaseDataset):
    """Base BCIC Dataset"""

    def _get_single_subject_data(self, subject):
        session = load_data(subject=subject, dataset=self.code, verbose=False)
        return session

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


class BCICIII_IVa(MNEBCIC):
    """
    BCICIII-IVa Motor Imagery dataset.

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
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 6)),
            sessions_per_subject=1,
            events={"right_hand": 0, "feet": 1},
            code="BCICIII-IVa",
            interval=[0, 3.5],
            paradigm="imagery",
            doi="",
        )


def data_path(url, path=None, force_update=False, update_path=None, verbose=None):
    return [dl.data_dl(url, "BNCI", path, force_update, verbose)]


@verbose
def load_data(
    subject,
    dataset="BCICIII-IVa",
    path=None,
    force_update=False,
    update_path=None,
    base_url=BCIC_URL,
    only_filenames=False,
    verbose=None,
):
    """
    Gets paths to local copies of a dataset file.

    This will fetch data for a given dataset.

    Parameters
    ----------
        subject (int):
            The subject to load.
        dataset (str):
            The dataset name.
        path (str):
            Location of where to look for the BCIC data storing
            location. If None, the environment variable or config parameter
            ``MNE_DATASETS_BNCI_PATH`` is used. If it doesn't exist, the
            "~/mne_data" directory is used. If the BCIC dataset is not
            found under the given path, the data will be automatically
            downloaded to the specified folder.
        force_update (bool):
            Force update of the dataset even if a local copy exists.
        update_path (bool):
            If True, set the MNE_DATASETS_BNCI_PATH in mne-python
            config to the given path. If None, the user is prompted.
        only_filenames (bool):
            If True, return only the local path of the files without
            loading the data.
        verbose (bool):
            If not None, override default verbose level
            (see :func:`mne.verbose` and :ref:`Logging documentation
            <tut_logging>` for more).

    Returns
    -------
        raws (list):
            List of raw instances for each non-consecutive recording.
            Depending on the dataset it could be a BCI run or a different
            recording session.
        event_id (dict):
            Dictionary containing events and their code.
    """

    dataset_list = {"BCICIII-IVa": _load_data_iva_III}

    baseurl_list = {"BCICIII-IVa": BCIC_URL}

    # Raises ValueError if dataset is not found
    if dataset not in dataset_list.keys():
        raise ValueError(
            f"Dataset {dataset} is not a valid BCIC dataset ID. "
            f"Valid datasets are: {', '.join(dataset_list.keys())}."
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
def _load_data_iva_III(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=BCIC_URL,
    only_filenames=False,
    verbose=None,
):
    """Loads data for the BCICIII-IVa dataset."""
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

    runs, ev = _convert_mi(filename[0], ch_names, ch_type)

    session = {"0train": {"0": runs}}
    return session


def _convert_mi(filename, ch_names, ch_type):
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
        raw, ev = _convert_run(run, ch_names, ch_type)
        return raw, ev


@verbose
def _convert_run(run, ch_names, ch_types, verbose=None):
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

    return raw, event_id
