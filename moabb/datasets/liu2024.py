import mne
import numpy as np
import os
import zipfile as z
from pathlib import Path
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.utils import add_stim_channel_epoch, add_stim_channel_trial


_LIU2024_URL = "https://figshare.com/articles/dataset/EEG_datasets_of_stroke_patients/21679035/5"

class Liu2024(BaseDataset):
    """

    Dataset [1]_ from the study on burst-VEP [2]_. 

    .. admonition:: Dataset summary


        ============  =======  =======  ==========  =================  ============  ===============  ===========
        Name           #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        ============  =======  =======  ==========  =================  ============  ===============  ===========
        BNCI2014_001       9       22           4                144  4s            250Hz                      2
        ============  =======  =======  ==========  =================  ============  ===============  ===========


    **Dataset description**

    References
    ----------

    .. [1] Liu, Haijie; Lv, Xiaodong (2022). EEG datasets of stroke patients. figshare. Dataset.
           DOI: https://doi.org/10.6084/m9.figshare.21679035.v5

    .. [2] Liu, H., Wei, P., Wang, H. et al. An EEG motor imagery dataset for brain computer interface in acute stroke
           patients. Sci Data 11, 131 (2024).
           DOI: https://doi.org/10.1038/s41597-023-02787-8
           
    Notes
    -----

    .. versionadded:: 1.0.0

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 9 + 1)),
            sessions_per_subject=1,
            events={"right_hand": 1, "left_hand": 2},
            code="Liu2024",
            interval=(0, 4),
            paradigm="imagery",
            doi="10.6084/m9.figshare.21679035.v5",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the data paths of a single subject."""
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        subject_paths = []

        url = "https://figshare.com/ndownloader/files/38516654"
        path_zip = dl.data_dl(url, self.code) 
        #sub = f"sub-{subject:02d}"
        path_zip = Path(path_zip)
        path_folder = path_zip.parent

        if not (path_folder / "edffile").is_dir():
            zip_ref = z.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)
        sub = f"sub-{subject:02d}"

        subject_path = path_folder / "edffile" / sub / "eeg" / f"{sub}_task-motor-imagery_eeg.edf"
        subject_paths.append(str(subject_path))

        return subject_paths

  
    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""
        file_path_list = self.data_path(subject)

        # Channels
        montage = mne.channels.read_custom_montage(file_path_list[-1])

        # There is only one session, each of 3 runs
        sessions = {"0": {}}
        for i_b in range(NR_RUNS):
            # EEG
            raw = mne.io.read_raw_gdf(
                file_path_list[2 * i_b],
                stim_channel="status",
                preload=True,
                verbose=False,
            )

            # Drop redundant ANA and EXG channels
            ana = [f"ANA{1 + i}" for i in range(32)]
            exg = [f"EXG{1 + i}" for i in range(8)]
            raw.drop_channels(ana + exg)

            # Set electrode positions
            raw.set_montage(montage)

            # Read info file
            tmp = loadmat(file_path_list[2 * i_b + 1])

            # Labels at trial level (i.e., symbols)
            trial_labels = tmp["labels"].astype("uint8").flatten() - 1

            # Codes (select optimized subset and layout, and repeat to trial length)
            subset = (
                tmp["subset"].astype("uint8").flatten() - 1
            )  # the optimized subset of 36 codes from a set of 65
            layout = (
                tmp["layout"].astype("uint8").flatten() - 1
            )  # the optimized position of the 36 codes in the grid
            codes = tmp["codes"][:, subset[layout]]
            codes = np.tile(codes, (NR_CYCLES_PER_TRIAL, 1))

            # Find onsets of trials
            events = mne.find_events(raw, verbose=False)
            trial_onsets = events[:, 0]

            # Create stim channel with trial information (i.e., symbols)
            # Specifically: 200 = symbol-0, 201 = symbol-1, 202 = symbol-2, etc.
            raw = add_stim_channel_trial(raw, trial_onsets, trial_labels, offset=200)

            # Create stim channel with epoch information (i.e., 1 / 0, or on / off)
            # Specifically: 100 = "0", 101 = "1"
            raw = add_stim_channel_epoch(
                raw, trial_onsets, trial_labels, codes, PRESENTATION_RATE, offset=100
            )

            # Add data as a new run
            run_name = str(i_b)
            sessions["0"][run_name] = raw

        return sessions


    