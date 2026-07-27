"""Sensory-Guided Joint Learning Motor Imagery dataset (Wang et al., 2026)."""

import logging
import warnings
import zipfile as z
from pathlib import Path

import mne
import numpy as np

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)


log = logging.getLogger(__name__)

# Figshare/KiltHub (CMU) article 32293995. One zip per experimental group.
# Each zip holds GroupName/SXXX/SXXX_run0.mat and SXXX_sessYY_runZZ[UD].mat,
# with a single MATLAB struct ``runData`` per file.
BASE_URL = "https://ndownloader.figshare.com/files/"
GROUP_FILES = {
    "BCI2000Control": "64710750",
    "EEGNetControl": "64710990",
    "JointLearning": "64710999",
    "TactileControl": "64710993",
}

# Subject ids are local to each group archive, so ``S001`` in two different
# groups denotes two different people. The main experiment contains 31
# participants (15 JointLearning, 8 BCI2000Control, 8 TactileControl), and the
# independent EEGNetControl experiment adds 8 more.
_SUBJECT_GROUP_PAIRS = (
    [("JointLearning", subject) for subject in range(1, 16)]
    + [("BCI2000Control", subject) for subject in range(1, 9)]
    + [("TactileControl", subject) for subject in range(1, 9)]
    + [("EEGNetControl", subject) for subject in range(1, 9)]
)
SUBJECT_MAP = dict(enumerate(_SUBJECT_GROUP_PAIRS, start=1))

# MOABB event dictionary. Target labels are normalised to this 4-class scheme.
EVENTS = {"left_hand": 1, "right_hand": 2, "up": 3, "down": 4}
_EVENT_TO_LABEL = {code: label for label, code in EVENTS.items()}

# Per-sample class codes stored in runData.trialTargetClass (EEGNet-style runs).
_EEGNET_CLASS_TO_EVENT = {0: 1, 1: 2, 2: 3, 3: 4}  # 0 left, 1 right, 2 up, 3 down

# Free-text target_label strings (runData.trialInfo.target_label / meta).
_LABEL_STR_TO_EVENT = {
    "left": 1,
    "right": 2,
    "up": 3,
    "down": 4,
    "left_hand": 1,
    "right_hand": 2,
}

# BCI2000-style runs store per-sample BCI2000 ``TargetCode`` values in
# ``runData.trialTargetCode`` instead of ``trialTargetClass``. These mappings
# follow the codebook in the public release's README.
_BCI2000_CODE_TO_EVENT = {
    "LR": {1: 2, 2: 1},  # 1 -> right_hand, 2 -> left_hand
    "UD": {1: 3, 2: 4},  # 1 -> up, 2 -> down
    "2D": {1: 2, 2: 1, 3: 3, 4: 4},
}


class SensoryGuidedMI2026(BaseDataset):
    """Motor imagery BCI dataset with sensory-guided joint learning [1]_.

    De-identified EEG from 39 participants recorded at Carnegie Mellon
    University while performing closed-loop motor-imagery cursor-control BCI
    tasks (kinesthetic imagery of left/right hand movement, plus up/down and
    2D control). The main experiment includes 31 participants and an
    independent EEGNet control experiment adds 8 participants. The released
    participants are split across four experimental groups:

    * ``JointLearning`` -- sensory-guided human-machine joint learning
      (tactile guidance + adaptive machine learning).
    * ``TactileControl`` -- tactile-stimulation control without the full
      joint-learning framework.
    * ``EEGNetControl`` -- online training with EEGNet feedback.
    * ``BCI2000Control`` -- BCI2000 cursor-control runs plus, for a subset of
      subjects, later EEGNet-based long-term runs.

    EEG signals are released segmented into trials, with no additional
    filtering or downsampling. 62 EEG channels of a 64-channel montage are
    retained. Each run-level ``.mat`` file contains one MATLAB struct
    ``runData`` whose ``trialSignal`` is either a fixed-length numeric array
    (``samples x channels x trials``, EEGNet-style) or a variable-length cell
    array (one ``trialLength x channels`` cell per trial, BCI2000-style).

    The run files are MATLAB v7.3 (HDF5) and are read with
    :func:`pymatreader.read_mat`, which returns ``runData`` as a nested dict.

    Target labels are normalised to a 4-class scheme
    (``left_hand``, ``right_hand``, ``up``, ``down``). For EEGNet-style runs
    the per-sample codes in ``trialTargetClass`` map 0->left, 1->right,
    2->up, 3->down; for BCI2000-style runs the per-sample BCI2000 ``TargetCode``
    in ``trialTargetCode`` is mapped by the run's control axis: left/right uses
    1->right and 2->left; up/down uses 1->up and 2->down; 2D additionally uses
    3->up and 4->down.

    References
    ----------
    .. [1] Wang, H., Zhang, Y., Karrenbach, M., Ding, Y., & He, B. (2026).
       Sensory-guided human-machine joint learning accelerates the acquisition
       of motor imagery brain computer interface control. Nature
       Communications, 17, 6177.
       DOI: https://doi.org/10.1038/s41467-026-75435-5
       Data: https://doi.org/10.1184/r1/32293995

    Notes
    -----
    .. note::
       Group archives reuse local subject ids (for example, each group has an
       ``S001`` belonging to a different participant). This adapter maps those
       ids to stable MOABB subject numbers: 1--15 are ``JointLearning``,
       16--23 ``BCI2000Control``, 24--31 ``TactileControl``, and 32--39
       ``EEGNetControl``.

    .. versionadded:: 1.1.1
    """

    nemar_id = "EXEMPT"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=62,
            channel_types={"eeg": 62},
            montage="standard_1005",
            hardware="EEG (64-channel montage, 62 EEG channels retained)",
            reference=None,
            ground=None,
            software="BCI2000; EEGNet online decoder",
            sensors=None,
            line_freq=60.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(has_eog=False),
        ),
        participants=ParticipantMetadata(
            n_subjects=39, health_status="healthy", species="homo sapiens"
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=4,
            class_labels=["left_hand", "right_hand", "up", "down"],
            trial_duration=None,
            study_design=(
                "Closed-loop motor-imagery BCI cursor control across four "
                "groups (JointLearning, TactileControl, EEGNetControl, "
                "BCI2000Control). Kinesthetic imagery of left/right hand "
                "movement drives 1D/2D cursor control with online feedback."
            ),
            feedback_type="online",
            stimulus_type="cursor control",
            stimulus_modalities=["visual", "tactile"],
            synchronicity="cue-based",
            mode="online",
            events=EVENTS,
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41467-026-75435-5",
            description=(
                "De-identified EEG and behavioural recordings from 39 "
                "participants across the main three-group study and an "
                "independent EEGNet control experiment evaluating a "
                "sensory-guided human-machine joint-learning framework for "
                "motor-imagery BCI. 62 EEG channels; trial-segmented signals "
                "stored as per-run runData MATLAB structs."
            ),
            investigators=[
                "Hanwen Wang",
                "Yisha Zhang",
                "Maxim Karrenbach",
                "Yidan Ding",
                "Bin He",
            ],
            senior_author="Bin He",
            contact_info=["bhe1@andrew.cmu.edu"],
            institution="Carnegie Mellon University",
            institution_department="Department of Biomedical Engineering",
            country="US",
            data_url="https://doi.org/10.1184/r1/32293995",
            publication_year=2026,
            license="CC-BY-NC-4.0",
            repository="KiltHub (Carnegie Mellon University) / Figshare",
            keywords=[
                "motor imagery",
                "brain-computer interface",
                "BCI",
                "EEG",
                "joint learning",
                "sensory feedback",
                "cursor control",
                "EEGNet",
                "BCI2000",
            ],
        ),
        sessions_per_subject=7,
        runs_per_session=6,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            preprocessing_steps=["trial segmentation only"],
            notes=(
                "No additional filtering or downsampling applied to the "
                "released trial-segmented EEG signals."
            ),
        ),
        cross_validation=CrossValidationMetadata(evaluation_type=["within_subject"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand", "up", "down"],
        ),
        data_structure=DataStructureMetadata(
            trials_context=(
                "Trial-segmented per run. Fixed-length EEGNet-style runs: "
                "samples x channels x trials (typically 5000 samples). "
                "Variable-length BCI2000-style runs: cell array, one "
                "trialLength x channels cell per trial."
            )
        ),
        file_format="MAT",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(SUBJECT_MAP),
            sessions_per_subject=7,
            events=EVENTS,
            code="SensoryGuidedMI2026",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1038/s41467-026-75435-5",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    # ------------------------------------------------------------------ #
    # Download / path resolution
    # ------------------------------------------------------------------ #
    def _group_dir(self, group, path=None, force_update=False, verbose=None):
        """Download and extract one group's zip; return its extracted folder."""
        dataset_root = (
            Path(dl.get_dataset_path(self.code, path)) / f"MNE-{self.code.lower()}-data"
        )
        for target in (dataset_root / "files" / group, dataset_root / group):
            if target.is_dir() and not force_update:
                return target

        url = BASE_URL + GROUP_FILES[group]
        path_zip = Path(
            dl.data_dl(
                url, self.code, path=path, force_update=force_update, verbose=verbose
            )
        )
        target = path_zip.parent / group
        if force_update or not target.is_dir():
            with z.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(path_zip.parent)
        return target if target.is_dir() else path_zip.parent

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the list of ``.mat`` run files for a single subject.

        MOABB subject ids are mapped to the release's group-local ``SXXX`` ids
        with :data:`SUBJECT_MAP`.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        group, group_subject = SUBJECT_MAP[subject]
        sub = f"S{group_subject:03d}"
        group_dir = self._group_dir(
            group, path=path, force_update=force_update, verbose=verbose
        )
        subject_paths = set((group_dir / sub).glob("*.mat"))
        subject_paths.update(group_dir.glob(f"**/{sub}/*.mat"))
        return [str(path) for path in sorted(subject_paths)]

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @staticmethod
    def _run_keys(fname):
        """Map a run filename to (session_key, run_key), both int-leading."""
        stem = Path(fname).stem  # e.g. S001_sess03_run01UD or S001_run0
        parts = stem.split("_")
        sess_key, run_key = "0", "0"
        for part in parts:
            if part.startswith("sess"):
                sess_key = str(int(part[4:]))
            elif part.startswith("run"):
                rest = part[3:]  # e.g. "01UD" or "0"
                digits = ""
                suffix = ""
                for ch in rest:
                    if ch.isdigit() and not suffix:
                        digits += ch
                    else:
                        suffix += ch
                run_key = (str(int(digits)) if digits else "0") + suffix
        return sess_key, run_key

    @staticmethod
    def _as_str_list(value):
        """Coerce a MATLAB channel-name field into a list of strings."""
        if value is None:
            return None
        arr = np.atleast_1d(np.asarray(value, dtype=object).ravel())
        names = []
        for item in arr:
            if isinstance(item, np.ndarray):
                item = item.item() if item.size == 1 else " ".join(map(str, item))
            names.append(str(item).strip())
        return names or None

    def _trial_label_events(self, rd, n_trials, fname):
        """Return an int event code per trial, or None if unresolved.

        ``rd`` is the ``runData`` dict returned by :func:`pymatreader.read_mat`.
        """
        # Preferred: explicit per-trial target strings. In the v7.3 release
        # ``trialInfo`` is an opaque MATLAB table that pymatreader returns as a
        # numeric handle, so this path only fires if it ever decodes to a dict.
        info = rd.get("trialInfo")
        if isinstance(info, dict):
            names = self._as_str_list(info.get("target_label"))
            if names and len(names) == n_trials:
                codes = [_LABEL_STR_TO_EVENT.get(str(n).lower()) for n in names]
                if all(c is not None for c in codes):
                    return np.asarray(codes, dtype=int)

        # EEGNet-style: per-sample class codes (0..3) -> per-trial mode.
        tclass = rd.get("trialTargetClass")
        if tclass is not None:
            tclass = np.asarray(tclass)
            if tclass.ndim == 2:  # samples x trials
                codes = []
                for col in range(tclass.shape[1]):
                    vals = tclass[:, col]
                    vals = vals[~np.isnan(vals)] if vals.dtype.kind == "f" else vals
                    if vals.size == 0:
                        codes.append(None)
                        continue
                    cls = int(np.bincount(vals.astype(int)).argmax())
                    codes.append(_EEGNET_CLASS_TO_EVENT.get(cls))
                if len(codes) == n_trials and all(c is not None for c in codes):
                    return np.asarray(codes, dtype=int)

        # BCI2000-style: reduce the per-sample TargetCode to one code per trial.
        # ``UD`` filenames are up/down. A non-UD run containing codes 3/4 is
        # 2D; otherwise it is left/right.
        tcode = rd.get("trialTargetCode")
        if tcode is not None:
            if isinstance(tcode, np.ndarray) and tcode.ndim == 2:
                cells = [tcode[:, c] for c in range(tcode.shape[1])]
            elif isinstance(tcode, (list, tuple)):
                cells = list(tcode)
            else:
                cells = [tcode]
            target_codes = []
            for cell in cells:
                vals = np.asarray(cell, dtype=float).ravel()
                vals = vals[~np.isnan(vals)]
                vals = vals[vals > 0]  # ignore inter-trial 0 codes
                if vals.size == 0:
                    target_codes.append(None)
                    continue
                code = int(np.bincount(vals.astype(int)).argmax())
                target_codes.append(code)

            if "UD" in Path(fname).stem.upper():
                axis = "UD"
            elif any(code in (3, 4) for code in target_codes if code is not None):
                axis = "2D"
            else:
                axis = "LR"
            mapping = _BCI2000_CODE_TO_EVENT[axis]
            codes = [
                mapping.get(code) if code is not None else None for code in target_codes
            ]
            if len(codes) == n_trials and all(c is not None for c in codes):
                return np.asarray(codes, dtype=int)
        return None

    @staticmethod
    def _read_mat(fname):
        """Read a MATLAB v7.3 (HDF5) run file into a ``runData`` dict.

        The released ``.mat`` files are MATLAB v7.3, which :func:`scipy.io.loadmat`
        cannot read; :func:`pymatreader.read_mat` handles both v7.2 and v7.3 and
        returns nested dicts. The opaque ``trialInfo`` table triggers harmless
        best-effort warnings that are silenced here.
        """
        try:
            from pymatreader import read_mat
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "SensoryGuidedMI2026 requires 'pymatreader' to read its "
                "MATLAB v7.3 (HDF5) run files. Install it with "
                "`pip install pymatreader`."
            ) from exc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            mat = read_mat(fname)
        return mat.get("runData")

    def _read_run(self, fname):
        """Build a continuous :class:`mne.io.RawArray` for one run file."""
        rd = self._read_mat(fname)
        if not isinstance(rd, dict):
            return None
        meta = rd.get("meta") or {}

        sfreq = float(meta.get("sampling_rate_hz") or 1000.0)
        ch_names = self._as_str_list(meta.get("selected_channels"))

        # Normalise trialSignal into a list of (n_samples, n_channels) arrays.
        # pymatreader returns a BCI2000-style cell array as a Python list of
        # variable-length (samples x channels) arrays, and an EEGNet-style fixed
        # array as a dense (samples x channels x trials) ndarray.
        signal = rd.get("trialSignal")
        trials = []
        if isinstance(signal, (list, tuple)):  # cell array: one cell per trial
            for cell in signal:
                arr = np.asarray(cell, dtype=float)
                if arr.ndim == 2 and arr.size:
                    trials.append(arr)
        elif isinstance(signal, np.ndarray):  # numeric samples x channels x trials
            dense = np.asarray(signal, dtype=float)
            if dense.ndim == 3:
                for t in range(dense.shape[2]):
                    trials.append(dense[:, :, t])
            elif dense.ndim == 2:
                trials.append(dense)

        if not trials:
            return None

        n_channels = trials[0].shape[1]
        if not ch_names or len(ch_names) != n_channels:
            ch_names = [f"EEG{i + 1}" for i in range(n_channels)]

        codes = self._trial_label_events(rd, len(trials), fname)
        if codes is None:
            return None

        # Concatenate trials and annotate every onset. Annotations preserve the
        # first event at sample zero, which mne.find_events would otherwise
        # discard when using a synthetic stim channel.
        segments = []
        onsets = []
        durations = []
        cursor = 0
        for trial in trials:
            seg = np.asarray(trial, dtype=float).T  # channels x samples
            if seg.shape[0] != n_channels:
                raise ValueError(
                    f"Inconsistent channel count in {fname}: expected "
                    f"{n_channels}, got {seg.shape[0]}"
                )
            segments.append(seg)
            onsets.append(cursor / sfreq)
            durations.append(seg.shape[1] / sfreq)
            cursor += seg.shape[1]

        data = 1e-6 * np.concatenate(segments, axis=1)
        info = mne.create_info(ch_names=list(ch_names), ch_types="eeg", sfreq=sfreq)
        raw = mne.io.RawArray(data=data, info=info, verbose=False)
        raw.set_annotations(
            mne.Annotations(
                onset=onsets,
                duration=durations,
                description=[_EVENT_TO_LABEL[int(code)] for code in codes],
            )
        )
        try:
            montage = mne.channels.make_standard_montage("standard_1005")
            raw.set_montage(montage, on_missing="ignore", verbose=False)
        except Exception:  # pragma: no cover - montage is best-effort
            pass
        return raw

    def _get_single_subject_data(self, subject):
        """Return ``{session: {run: raw}}`` for one subject."""
        sessions = {}
        for fname in self.data_path(subject):
            raw = self._read_run(fname)
            if raw is None:
                continue
            sess_key, run_key = self._run_keys(fname)
            runs = sessions.setdefault(sess_key, {})
            if run_key in runs:
                raise ValueError(
                    f"Duplicate session/run key ({sess_key}, {run_key}) for {fname}"
                )
            runs[run_key] = raw
        return sessions
