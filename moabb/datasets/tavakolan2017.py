"""Classifying three imaginary states of the same upper extremity.

Tavakolan, Frehlick, Yong, and Menon (2017), PLOS ONE.
DOI: 10.1371/journal.pone.0174161
"""

import logging
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)
from .utils import build_raw_from_epochs, safe_extract_zip


log = logging.getLogger(__name__)

DRYAD_BASE_URL = "https://datadryad.org/stash/downloads/file_stream/"

# Mapping of (subject, session) -> Dryad file_stream ID.
# Obtained from the Dryad API v2 for doi:10.5061/dryad.6qs86.
# fmt: off
_FILE_IDS = {
    (1, 1): 13762, (1, 2): 13764, (1, 3): 13766, (1, 4): 13768,
    (2, 1): 13770, (2, 2): 13792, (2, 3): 13814, (2, 4): 13836,
    (3, 1): 13772, (3, 2): 13794, (3, 3): 13816, (3, 4): 13838,
    (4, 1): 13774, (4, 2): 13796, (4, 3): 13818, (4, 4): 13840,
    (5, 1): 13776, (5, 2): 13798, (5, 3): 13820, (5, 4): 13842,
    (6, 1): 13778, (6, 2): 13800, (6, 3): 13822, (6, 4): 13844,
    (7, 1): 13780, (7, 2): 13802, (7, 3): 13824, (7, 4): 13846,
    (8, 1): 13782, (8, 2): 13804, (8, 3): 13826, (8, 4): 13848,
    (9, 1): 13784, (9, 2): 13806, (9, 3): 13828, (9, 4): 13850,
    (10, 1): 13786, (10, 2): 13808, (10, 3): 13830, (10, 4): 13852,
    (11, 1): 13788, (11, 2): 13810, (11, 3): 13832, (11, 4): 13854,
    (12, 1): 13790, (12, 2): 13812, (12, 3): 13834, (12, 4): 13856,
}
# fmt: on


class Tavakolan2017(BaseDataset):
    """Motor imagery dataset for three imaginary states of the same upper extremity.

    Dataset from [1]_.

    This dataset contains 32-channel EEG recordings from 12 healthy subjects
    performing motor imagery of the right upper extremity.  Subjects imagined
    three tasks: rest, grasping (opening/closing fingers to grab an object),
    and elbow flexion/extension (moving the forearm up and down).

    EEG was recorded at 1000 Hz using a 32-channel EGI Geodesic Sensor Net
    (GES 400 series amplifier) with Cz as the online reference.  Each subject
    completed 4 sessions on separate days, with 20 trials per class per session
    (60 trials total per session).

    Each trial consisted of a 3 s visual cue (during which the subject
    performed the imagery) followed by a 5-7 s rest interval.  The imagery
    interval [0, 3] s after cue onset is used for analysis.

    The data is stored on the Dryad Digital Repository [2]_ as ZIP archives
    (one per subject-session) containing MATLAB ``.mat`` files.

    Notes
    -----
    The original channel labels follow the EGI HydroCel Geodesic Sensor Net
    naming convention (E1-E32 plus Cz reference).  The ``GSN-HydroCel-32``
    montage from MNE is applied.

    The three classes map to:

    - ``rest``: relaxation without movement or imagery
    - ``right_hand``: MI-GRASP -- imagining opening and closing fingers
    - ``right_elbow_flexion``: MI-ELBOW -- imagining forearm up/down movement

    References
    ----------
    .. [1] M. Tavakolan, Z. Frehlick, X. Yong, and C. Menon,
       "Classifying three imaginary states of the same upper extremity
       using time-domain features," PLoS ONE, vol. 12, no. 3, e0174161,
       2017. DOI: 10.1371/journal.pone.0174161

    .. [2] M. Tavakolan, Z. Frehlick, X. Yong, and C. Menon,
       "Data from: Classifying three imaginary states of the same upper
       extremity using time-domain features," Dryad, 2017.
       DOI: 10.5061/dryad.6qs86
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="GSN-HydroCel-32",
            hardware="EGI Geodesic Net Amps 400 series",
            sensor_type="Ag/AgCl sponge",
            reference="Cz",
            impedance_threshold_kohm=50,
            filters={"bandpass": [0.1, 100]},
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="healthy",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={
                "rest": 1,
                "right_hand": 2,
                "right_elbow_flexion": 3,
            },
            paradigm="imagery",
            n_classes=3,
            class_labels=["rest", "right_hand", "right_elbow_flexion"],
            trial_duration=3.0,
            study_design=(
                "Three-class motor imagery of the same upper extremity: "
                "rest, grasping (MI-GRASP), and elbow flexion (MI-ELBOW). "
                "20 trials per class per session, 4 sessions per subject."
            ),
            feedback_type="none",
            stimulus_type="visual cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            instructions=(
                "REST: relax without movement. "
                "MI-GRASP: imagine opening and closing all fingers to grab "
                "an object. MI-ELBOW: imagine moving the forearm up and down."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0174161",
            investigators=[
                "Mojgan Tavakolan",
                "Zack Frehlick",
                "Xinyi Yong",
                "Carlo Menon",
            ],
            senior_author="Carlo Menon",
            institution="Simon Fraser University",
            institution_department="MENRVA Research Group, School of Mechatronic Systems Engineering",
            country="CA",
            data_url="https://datadryad.org/stash/dataset/doi:10.5061/dryad.6qs86",
            repository="Dryad",
            license="CC0-1.0",
            publication_year=2017,
            ethics_approval=["Simon Fraser University Office of Research Ethics"],
            keywords=[
                "motor imagery",
                "EEG",
                "upper extremity",
                "same limb",
                "time-domain features",
                "SVM",
                "BCI",
            ],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="epoched",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["rest", "right_hand", "right_elbow_flexion"],
            cue_duration_s=3.0,
            imagery_duration_s=3.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=2880,
            trials_context=("12 subjects x 4 sessions x 60 trials (20 per class)"),
            n_trials_per_class={"rest": 20, "right_hand": 20, "right_elbow_flexion": 20},
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control", "rehabilitation"],
            environment="laboratory",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Research"],
        ),
        sessions_per_subject=4,
        runs_per_session=1,
        file_format="MAT",
    )

    _events = {
        "rest": 1,
        "right_hand": 2,
        "right_elbow_flexion": 3,
    }

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 13)),
            sessions_per_subject=4,
            events=self._events,
            code="Tavakolan2017",
            interval=[0, 3],
            paradigm="imagery",
            doi="10.1371/journal.pone.0174161",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject across all sessions.

        Each session is stored as a separate ZIP/MAT file on Dryad.  The .mat
        file contains three matrices named ``rest``, ``grasp``, and ``elbow``,
        each with shape ``(n_channels, n_samples, n_trials)`` where
        ``n_channels`` is 32 and ``n_trials`` is 20 per class.

        If the variable names differ from this expectation, the loader falls
        back to iterating over available keys and using shape heuristics to
        identify the data.
        """
        sessions = {}
        for ses_idx in range(1, 5):
            mat_path = self.data_path(subject, session=ses_idx)
            mat = loadmat(
                mat_path,
                squeeze_me=True,
                struct_as_record=False,
                verify_compressed_data_integrity=False,
            )

            # Try known variable names first, then fall back to heuristic
            epoch_data, event_ids = self._extract_trials(mat, subject, ses_idx)

            if epoch_data is None:
                raise ValueError(
                    f"Could not parse .mat file for subject {subject}, "
                    f"session {ses_idx}. Available keys: "
                    f"{[k for k in mat.keys() if not k.startswith('__')]}"
                )

            raw = build_raw_from_epochs(
                epoch_data,
                self._get_ch_names(epoch_data.shape[1]),
                1000,
                event_ids,
                "GSN-HydroCel-32",
            )
            sessions[str(ses_idx - 1)] = {"0": raw}

        return sessions

    def _extract_trials(self, mat, subject, session):
        """Extract trial data and event labels from the loaded .mat dict.

        Returns
        -------
        data : ndarray, shape (n_trials, n_channels, n_samples) or None
        event_ids : ndarray, shape (n_trials,) or None
        """
        # Strategy 1: named variables "rest", "grasp"/"mi_grasp", "elbow"/"mi_elbow"
        rest_keys = ["rest", "Rest", "REST", "mi_rest"]
        grasp_keys = ["grasp", "Grasp", "GRASP", "mi_grasp", "MI_Grasp", "miGrasp"]
        elbow_keys = ["elbow", "Elbow", "ELBOW", "mi_elbow", "MI_Elbow", "miElbow"]

        rest_data = self._find_var(mat, rest_keys)
        grasp_data = self._find_var(mat, grasp_keys)
        elbow_data = self._find_var(mat, elbow_keys)

        if rest_data is not None and grasp_data is not None and elbow_data is not None:
            return self._combine_classes(rest_data, grasp_data, elbow_data)

        # Strategy 2: look for a struct with fields containing the data
        for key in mat:
            if key.startswith("__"):
                continue
            val = mat[key]
            if hasattr(val, "dtype") and val.dtype.names is not None:
                # Structured array -- try to find rest/grasp/elbow fields
                rest_data = self._find_struct_field(val, rest_keys)
                grasp_data = self._find_struct_field(val, grasp_keys)
                elbow_data = self._find_struct_field(val, elbow_keys)
                if (
                    rest_data is not None
                    and grasp_data is not None
                    and elbow_data is not None
                ):
                    return self._combine_classes(rest_data, grasp_data, elbow_data)

        # Strategy 3: look for a single 4D array (n_channels, n_samples, n_trials, n_classes)
        # or (n_classes, n_channels, n_samples, n_trials)
        for key in mat:
            if key.startswith("__"):
                continue
            val = mat[key]
            if isinstance(val, np.ndarray) and val.ndim == 4:
                # Try to identify the class dimension
                shape = val.shape
                # If one dimension is exactly 3 (n_classes)
                if shape[-1] == 3:
                    # (n_channels, n_samples, n_trials, 3)
                    rest_data = val[:, :, :, 0]
                    grasp_data = val[:, :, :, 1]
                    elbow_data = val[:, :, :, 2]
                    return self._combine_classes(rest_data, grasp_data, elbow_data)
                elif shape[0] == 3:
                    # (3, n_channels, n_samples, n_trials)
                    rest_data = val[0, :, :, :]
                    grasp_data = val[1, :, :, :]
                    elbow_data = val[2, :, :, :]
                    return self._combine_classes(rest_data, grasp_data, elbow_data)

        # Strategy 4: look for a "data" key with a "label"/"labels" key
        data_key = self._find_key(mat, ["data", "Data", "DATA", "eeg", "EEG"])
        label_key = self._find_key(
            mat,
            [
                "label",
                "labels",
                "Label",
                "Labels",
                "LABEL",
                "LABELS",
                "class",
                "Class",
                "CLASS",
                "classes",
                "Classes",
            ],
        )
        if data_key is not None and label_key is not None:
            data_val = mat[data_key]
            labels = mat[label_key].ravel()
            if isinstance(data_val, np.ndarray) and data_val.ndim == 3:
                # Could be (n_trials, n_channels, n_samples) or
                # (n_channels, n_samples, n_trials)
                if data_val.shape[0] == len(labels):
                    # (n_trials, n_channels, n_samples)
                    epoch_data = data_val
                elif data_val.shape[2] == len(labels):
                    # (n_channels, n_samples, n_trials) -> transpose
                    epoch_data = data_val.transpose(2, 0, 1)
                else:
                    return None, None

                # Map labels to our event codes
                unique_labels = np.unique(labels)
                if len(unique_labels) == 3:
                    label_map = {
                        unique_labels[0]: 1,  # rest
                        unique_labels[1]: 2,  # grasp
                        unique_labels[2]: 3,  # elbow
                    }
                    event_ids = np.array([label_map[lab] for lab in labels])
                    return epoch_data, event_ids

        return None, None

    @staticmethod
    def _find_var(mat, candidate_keys):
        """Return the first matching ndarray from mat, or None."""
        for k in candidate_keys:
            if k in mat and isinstance(mat[k], np.ndarray):
                return mat[k]
        return None

    @staticmethod
    def _find_struct_field(struct_arr, candidate_keys):
        """Return the first matching field from a structured array."""
        if struct_arr.dtype.names is None:
            return None
        for k in candidate_keys:
            if k in struct_arr.dtype.names:
                val = struct_arr[k]
                if hasattr(val, "item"):
                    val = val.item()
                if isinstance(val, np.ndarray):
                    return val
        return None

    @staticmethod
    def _find_key(mat, candidate_keys):
        """Return the first matching key from mat, or None."""
        for k in candidate_keys:
            if k in mat:
                return k
        return None

    @staticmethod
    def _combine_classes(rest_data, grasp_data, elbow_data):
        """Combine three class arrays into (n_trials, n_channels, n_samples).

        Each input array can be:
        - (n_channels, n_samples, n_trials) -- standard EGI export
        - (n_trials, n_channels, n_samples) -- already in epoch format
        """
        arrays = []
        event_ids = []
        for class_data, event_code in [
            (rest_data, 1),
            (grasp_data, 2),
            (elbow_data, 3),
        ]:
            if class_data.ndim == 3:
                # Determine orientation: if dim0 < dim2, likely
                # (n_channels, n_samples, n_trials) -> transpose to
                # (n_trials, n_channels, n_samples)
                if class_data.shape[0] < class_data.shape[2]:
                    # Ambiguous if n_channels == n_trials, use n_channels heuristic
                    class_data = class_data.transpose(2, 0, 1)
                elif class_data.shape[0] > class_data.shape[2]:
                    # Already (n_trials, n_channels, n_samples) -- keep as is
                    pass
                else:
                    # Same first and last dim -- assume (ch, samples, trials)
                    class_data = class_data.transpose(2, 0, 1)

                n_trials = class_data.shape[0]
                arrays.append(class_data)
                event_ids.extend([event_code] * n_trials)
            elif class_data.ndim == 2:
                # Single trial: (n_channels, n_samples)
                arrays.append(class_data[np.newaxis, :, :])
                event_ids.append(event_code)

        epoch_data = np.concatenate(arrays, axis=0)
        return epoch_data, np.array(event_ids)

    @staticmethod
    def _get_ch_names(n_channels):
        """Return channel names for the EGI GSN-HydroCel net.

        The GSN-HydroCel-32 montage in MNE uses channel names E1-E32 plus
        Cz (the reference).  Since Cz is the online reference and typically
        not included in the data, we use E1 through E{n_channels}.
        """
        return [f"E{i}" for i in range(1, n_channels + 1)]

    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
        session=None,
    ):
        """Return local path to the .mat file for a given subject and session.

        Parameters
        ----------
        subject : int
            Subject number (1-12).
        path : str | None
            Custom download location.
        force_update : bool
            Force re-download.
        update_path : None
            Unused, kept for API compatibility.
        verbose : bool | None
            Verbosity level.
        session : int | None
            Session number (1-4).  If None, downloads session 1.

        Returns
        -------
        mat_path : str
            Path to the extracted .mat file.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        if session is None:
            session = 1
        if session not in (1, 2, 3, 4):
            raise ValueError(f"Invalid session number: {session}")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"
        subj_dir = data_dir / f"P{subject:02d}"
        subj_dir.mkdir(parents=True, exist_ok=True)

        # Check if a .mat file already exists for this session
        mat_files = list(subj_dir.glob(f"*Se{session:02d}*.mat")) + list(
            subj_dir.glob(f"*Session{session:02d}*.mat")
        )
        if mat_files and not force_update:
            return str(mat_files[0])

        # Download the ZIP from Dryad
        file_id = _FILE_IDS[(subject, session)]
        url = f"{DRYAD_BASE_URL}{file_id}"
        zip_path = dl.data_dl(url, sign, path, force_update, verbose)

        # Extract .mat files from the ZIP
        with zipfile.ZipFile(zip_path, "r") as zf:
            mat_members = [
                m
                for m in zf.infolist()
                if m.filename.endswith(".mat") and not m.filename.startswith("__MACOSX")
            ]
            safe_extract_zip(zf, subj_dir, members=mat_members)

        # Find the extracted .mat file
        mat_files = list(subj_dir.glob("**/*.mat"))
        if not mat_files:
            raise FileNotFoundError(
                f"No .mat file found after extracting ZIP for subject "
                f"{subject}, session {session}"
            )

        return str(mat_files[0]) if len(mat_files) == 1 else str(sorted(mat_files)[0])
