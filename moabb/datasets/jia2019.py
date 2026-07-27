"""Jia2019 motor-imagery EEG dataset for stroke patients."""

from pathlib import Path

import numpy as np
import scipy.io as sio
from mne import Annotations, create_info
from mne.io import RawArray

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)


# Figshare article "EEG data of motor imagery for stroke" (Tianyu Jia, 2019),
# data DOI 10.6084/m9.figshare.7636301. Each of the 15 subjects has two files,
# ``exp1-S<n>-left.mat`` and ``exp1-S<n>-right.mat``, holding left-hand and
# right-hand motor-imagery trials respectively. Files are resolved by name from
# the Figshare files API so the loader tracks any future article version.
JIA2019_ARTICLE_ID = "7636301"

# Stable Figshare file ids for the 30 subject/class files in article 7636301.
# These let an already downloaded dataset load without querying the Figshare
# API. The API is still used when any requested file is missing, so a future
# article revision remains discoverable.
_FILE_IDS = {
    1: {"left": "14185838", "right": "14185841"},
    2: {"left": "14185844", "right": "14185847"},
    3: {"left": "14185850", "right": "14185940"},
    4: {"left": "14185943", "right": "14185859"},
    5: {"left": "14185958", "right": "14185961"},
    6: {"left": "14185964", "right": "14185967"},
    7: {"left": "14185874", "right": "14185955"},
    8: {"left": "14185979", "right": "14185883"},
    9: {"left": "14185886", "right": "14185889"},
    10: {"left": "14185892", "right": "14185895"},
    11: {"left": "14185898", "right": "14185901"},
    12: {"left": "14185904", "right": "14185907"},
    13: {"left": "14185910", "right": "14185949"},
    14: {"left": "14185916", "right": "14185946"},
    15: {"left": "14185934", "right": "14185937"},
}

# The distributed .mat files carry only the numeric EEG arrays (no channel
# labels). The source paper reports a 63-channel 10-10 montage at 512 Hz, but
# the exact electrode order is not published, so channels are named generically.
JIA2019_SFREQ = 512.0
JIA2019_N_CHANNELS = 63

# Class code per source file (the movement class is encoded by the file name).
_EVENTS = {"left_hand": 1, "right_hand": 2}
# Matlab variables inside each file: two acquisition blocks of 20 trials each.
_BLOCK_KEYS = ("DATA1", "DATA2")


class Jia2019(BaseDataset):
    """Motor-imagery EEG dataset for stroke patients (Jia 2019) [1]_.

    .. admonition:: Dataset summary

        =======  =======  =======  ==========  =================  ============  ===============  ===========
        Name       #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        =======  =======  =======  ==========  =================  ============  ===============  ===========
        Jia2019       15       63           2                 40         6.8 s           512 Hz            1
        =======  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    EEG recorded from 15 stroke patients performing cue-based left-hand versus
    right-hand motor imagery. For each patient one hand is paretic (affected) and
    the other is unaffected; both left- and right-hand imagination were acquired.
    Signals were recorded with 63 channels under the international 10-10 system at
    512 Hz (as reported in [2]_, which decodes this dataset).

    Each subject contributes two MATLAB files, ``exp1-S<n>-left.mat`` and
    ``exp1-S<n>-right.mat``, corresponding to left-hand and right-hand motor
    imagery. Every file holds two variables, ``DATA1`` and ``DATA2``, each an
    array of 20 epoched trials of shape ``(63 channels, n_samples)``. The movement
    class is therefore data-borne: it is fixed by which file a trial comes from
    (``*-left.mat`` -> left hand, ``*-right.mat`` -> right hand), giving 40 trials
    per class per subject (80 trials total).

    This loader exposes a single session with two runs, one per acquisition block
    (``DATA1`` -> run ``"0"``, ``DATA2`` -> run ``"1"``). Each run concatenates the
    20 left-hand trials of that block with the 20 right-hand trials of the same
    block into a continuous :class:`mne.io.RawArray`, and an MNE annotation marks
    the onset of every trial with its class label (``left_hand`` / ``right_hand``).
    Amplitudes are converted from microvolts to volts.

    The stored epoch length varies across subjects (3500 samples for some,
    4000 for others; roughly 6.8 to 7.8 s at 512 Hz); the default analysis
    interval ``[0, 6.8]`` s stays within the shortest epoch. The distributed
    files contain no channel labels, so the 63 EEG channels are named generically
    (``Ch1`` .. ``Ch63``); no montage positions are attached.

    References
    ----------

    .. [1] Jia, T. (2019). EEG data of motor imagery for stroke. Figshare.
       DOI: https://doi.org/10.6084/m9.figshare.7636301

    .. [2] Wang, K., Zhao, Y., He, D., Xia, Q., Li, G., Wang, N., Peng, N., &
       Jiang, B. (2026). PA-TCNet: Pathology-Aware Temporal Calibration with
       Physiology-Guided Target Refinement for Cross-Subject Motor Imagery EEG
       Decoding in Stroke Patients. arXiv:2604.16554.

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=JIA2019_SFREQ,
            n_channels=JIA2019_N_CHANNELS,
            channel_types={"eeg": JIA2019_N_CHANNELS},
            montage=None,
            reference=None,
            ground=None,
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=15,
            health_status="stroke",
            clinical_population="stroke",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trials_per_class={"left_hand": 40, "right_hand": 40},
            study_design=(
                "Cue-based left-hand vs right-hand motor imagery in stroke "
                "patients; for each patient one hand is paretic and the other "
                "unaffected, and both were imagined."
            ),
            synchronicity="cue-based",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.6084/m9.figshare.7636301",
            description=(
                "EEG from 15 stroke patients performing left- vs right-hand motor "
                "imagery, 63 channels (10-10) at 512 Hz, 40 trials per class."
            ),
            investigators=["Tianyu Jia"],
            country="CN",
            data_url="https://doi.org/10.6084/m9.figshare.7636301",
            publication_year=2019,
            license="CC-BY-4.0",
            repository="Figshare",
            keywords=[
                "motor imagery",
                "BCI",
                "brain-computer interface",
                "EEG",
                "stroke",
                "neurorehabilitation",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=2,
        tags=Tags(pathology=["Stroke"], modality=["Motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="epoched", preprocessing_applied=True
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery", imagery_tasks=["left_hand", "right_hand"]
        ),
        cross_validation=CrossValidationMetadata(evaluation_type=["within_subject"]),
        file_format="MAT",
        data_processed=True,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Jia2019",
            interval=[0, 6.8],
            paradigm="imagery",
            doi="10.6084/m9.figshare.7636301",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local paths of a subject's left- and right-hand files.

        Parameters
        ----------
        subject : int
            Subject number (1-15).
        path : None | str
            Storage location override.
        force_update : bool
            Re-download even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            Verbosity level.

        Returns
        -------
        list of str
            ``[left_hand_file, right_hand_file]`` local paths.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        dataset_root = (
            Path(dl.get_dataset_path(self.code, path))
            / f"MNE-{self.code.lower()}-data"
            / "files"
        )
        local_paths = [
            dataset_root / _FILE_IDS[subject][side] for side in ("left", "right")
        ]
        if not force_update and all(local_path.is_file() for local_path in local_paths):
            return [str(local_path) for local_path in local_paths]

        filelist = dl.fs_get_file_list(JIA2019_ARTICLE_ID)
        name_to_id = dl.fs_get_file_id(filelist)

        paths = []
        for side in ("left", "right"):
            fname = f"exp1-S{subject}-{side}.mat"
            if fname not in name_to_id:
                raise ValueError(
                    f"{fname} not found in Figshare article {JIA2019_ARTICLE_ID}"
                )
            url = f"https://ndownloader.figshare.com/files/{name_to_id[fname]}"
            paths.append(dl.data_dl(url, self.code, path, force_update, verbose))
        return paths

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as ``{"0": {run: Raw}}``."""
        left_path, right_path = self.data_path(subject)
        left = sio.loadmat(left_path)
        right = sio.loadmat(right_path)

        runs = {}
        for run_idx, key in enumerate(_BLOCK_KEYS):
            left_trials = self._trials(left[key])
            right_trials = self._trials(right[key])
            runs[str(run_idx)] = self._build_raw(left_trials, right_trials)
        return {"0": runs}

    @staticmethod
    def _trials(block):
        """Return the list of ``(n_channels, n_samples)`` trials in a block.

        ``block`` is the ``(1, n_trials)`` MATLAB cell array stored under
        ``DATA1`` / ``DATA2``.
        """
        block = np.asarray(block)
        return [np.asarray(block[0, i], dtype=float) for i in range(block.shape[1])]

    @staticmethod
    def _build_raw(left_trials, right_trials):
        """Build a continuous run from left- and right-hand epoched trials.

        Trials are concatenated along time; an MNE annotation marks each trial
        onset with its class label. Amplitudes are converted from microvolts to
        volts. Events are carried as annotations rather than a stim channel so
        that a trial starting at sample 0 is not dropped by
        :func:`mne.find_events` (which requires a rising edge).
        """
        labelled = [(t, "left_hand") for t in left_trials] + [
            (t, "right_hand") for t in right_trials
        ]
        n_channels = labelled[0][0].shape[0]

        cont = np.concatenate([t for t, _ in labelled], axis=1) * 1e-6
        ch_names = [f"Ch{i + 1}" for i in range(n_channels)]
        info = create_info(ch_names=ch_names, sfreq=JIA2019_SFREQ, ch_types="eeg")
        raw = RawArray(data=cont, info=info, verbose=False)

        onsets = []
        descriptions = []
        pos = 0
        for trial, label in labelled:
            onsets.append(pos / JIA2019_SFREQ)
            descriptions.append(label)
            pos += trial.shape[1]
        raw.set_annotations(
            Annotations(onset=onsets, duration=0.0, description=descriptions)
        )
        return raw
