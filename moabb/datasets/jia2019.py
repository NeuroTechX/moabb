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

    **Dataset description**

    15 stroke patients performed cue-based left- vs right-hand motor imagery
    (63 channels, 512 Hz, as reported in [2]_). The class is given by the file
    (``exp1-S<n>-left.mat`` / ``-right.mat``); each file holds two blocks
    (``DATA1``, ``DATA2``) of 20 epoched trials. The loader exposes one session
    with one run per block, concatenating that block's left- and right-hand trials
    and annotating every trial onset; amplitudes are converted from µV to V.

    Epochs are 3500 or 4000 samples depending on the subject; the ``[0, 6.8]`` s
    interval fits the shortest. The files carry no channel labels, so channels
    are named ``Ch1`` .. ``Ch63`` and no montage is attached.

    References
    ----------

    .. [1] Jia, T. (2019). EEG data of motor imagery for stroke. Figshare.
       DOI: https://doi.org/10.6084/m9.figshare.7636301

    .. [2] Wang, X., Zhao, Y., He, D., Xia, Q., Li, G., Wang, N., Peng, N., &
       Jiang, B. (2026). PA-TCNet: Pathology-Aware Temporal Calibration with
       Physiology-Guided Target Refinement for Cross-Subject Motor Imagery EEG
       Decoding in Stroke Patients. arXiv:2604.16554.
       DOI: https://doi.org/10.48550/arXiv.2604.16554

    Notes
    -----

    .. versionadded:: 1.8

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=JIA2019_SFREQ,
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
            related_paper_dois=["10.48550/arXiv.2604.16554"],
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

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Jia2019",
            interval=[0, 6.8],
            paradigm="imagery",
            selected_subjects=subjects,
            selected_sessions=sessions,
            doi="10.6084/m9.figshare.7636301",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return ``[left_hand_file, right_hand_file]`` local paths."""
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
        mats = [sio.loadmat(p) for p in self.data_path(subject)]  # [left, right]
        runs = {}
        for run_idx, key in enumerate(_BLOCK_KEYS):
            # Each block is a (1, n_trials) MATLAB cell array of (63, n_samples).
            left, right = ([np.asarray(t, dtype=float) for t in m[key][0]] for m in mats)
            runs[str(run_idx)] = self._build_raw(left, right)
        return {"0": runs}

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
        if not labelled or any(
            t.ndim != 2
            or t.shape[0] != JIA2019_N_CHANNELS
            or t.shape[1] < int(round(6.8 * JIA2019_SFREQ)) + 1
            for t, _ in labelled
        ):
            raise ValueError("Expected 63-channel trials covering the analysis interval")

        cont = np.concatenate([t for t, _ in labelled], axis=1) * 1e-6
        ch_names = [f"Ch{i + 1}" for i in range(JIA2019_N_CHANNELS)]
        info = create_info(ch_names=ch_names, sfreq=JIA2019_SFREQ, ch_types="eeg")
        raw = RawArray(data=cont, info=info, verbose=False)

        lengths = [t.shape[1] for t, _ in labelled]
        onsets = list(np.cumsum([0, *lengths[:-1]]) / JIA2019_SFREQ)
        raw.set_annotations(
            Annotations(
                onset=onsets, duration=0.0, description=[lab for _, lab in labelled]
            )
        )
        raw.set_annotations(
            raw.annotations + Annotations(onsets[1:], 0.0, "EDGE boundary")
        )
        return raw
