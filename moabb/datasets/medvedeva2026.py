"""Medvedeva2026 post-stroke motor recovery dataset (EEG modality)."""

import re
import warnings
import zipfile as z
from pathlib import Path
from zipfile import BadZipFile

import mne

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    Tags,
)


# EEG_FIF.zip on figshare (article 28904942, DOI 10.6084/m9.figshare.28904942)
MEDVEDEVA2026_URL = "https://ndownloader.figshare.com/files/60210938"

# File name pattern, e.g. "L_01_session2_left1.fif":
# {paretic_side}_{subject:02d}_session{n}_{target_hand}{run}.fif
_FNAME_RE = re.compile(
    r"^([RL])_(\d+)_session(\d+)_(left|right)(\d+)\.fif$", re.IGNORECASE
)


class Medvedeva2026(BaseDataset):
    """Post-stroke motor-recovery dataset [1]_ (EEG modality).

    **Dataset description**

    Multisession hybrid fNIRS-EEG recordings acquired during a neurorehabilitation
    programme in 16 post-stroke patients performing hand movements with the intact
    and the paretic hand. Only the EEG modality is exposed by this loader.

    Each patient was recorded across up to six sessions (session 1 = first day of
    rehabilitation, session 6 = immediately before withdrawal); the number of
    available sessions and runs per session varies from patient to patient. Within
    a run the patient responds to a lateralised flashing visual stimulus: the
    annotation ``"flash_l"`` marks a left-side stimulus and ``"flash_r"`` a
    right-side stimulus, at roughly 11 s inter-stimulus spacing.

    EEG was recorded at 500 Hz over eight electrodes placed above the sensorimotor
    cortex (FCC3, FCC4, CCP3, CCP4, FCC5, FCC6) with two earlobe channels (A1, A2),
    plus two hand-EMG channels (RH, LH) and one ECG channel (EKG). By default only
    the eight EEG channels are returned; pass ``return_all_modalities=True`` to keep
    the EMG and ECG channels as well.

    File names encode the side of the paretic limb, the subject identifier, the
    session number and the target hand of the run.

    References
    ----------

    .. [1] Medvedeva, A., Syrov, N., Yakovlev, L., Alieva, Y.,
       Berkmush-Antipova, A., Ivanova, G., & Kaplan, A. (2025). Multisession
       fNIRS-EEG data of Post-Stroke Motor Recovery: Recordings During Intact
       and Paretic Hand Movements. figshare. Dataset.
       DOI: https://doi.org/10.6084/m9.figshare.28904942

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=8,
            channel_types={"eeg": 8, "emg": 2, "ecg": 1},
            montage="10-05",
            reference=None,
            ground=None,
            line_freq=50.0,
            sensors=["FCC3", "FCC4", "CCP3", "CCP4", "FCC5", "A1", "FCC6", "A2"],
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_emg=True, emg_channels=2, other_physiological=["ecg"]
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=16,
            health_status="post-stroke patients",
            clinical_population="post-stroke patients undergoing motor rehabilitation",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["flash_l", "flash_r"],
            trial_duration=4.0,
            study_design="Post-stroke patients responded to lateralised flashing "
            "visual stimuli with intact and paretic hand movements across up to six "
            "rehabilitation sessions. Events are labelled by the side of the "
            "flashing stimulus: 'flash_l' (left) and 'flash_r' (right).",
            feedback_type="none",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="offline",
            events={"flash_l": 1, "flash_r": 2},
        ),
        documentation=DocumentationMetadata(
            doi="10.6084/m9.figshare.28904942",
            description="Multisession hybrid fNIRS-EEG dataset of 16 post-stroke "
            "patients recorded during intact and paretic hand movements over up to "
            "six rehabilitation sessions. This loader exposes the EEG modality.",
            investigators=[
                "Aleksandra Medvedeva",
                "Nikolay Syrov",
                "Lev Yakovlev",
                "Yana Alieva",
                "Artemiy Berkmush-Antipova",
                "Galina Ivanova",
                "Alexander Kaplan",
            ],
            senior_author="Alexander Kaplan",
            country="RU",
            data_url="https://doi.org/10.6084/m9.figshare.28904942",
            publication_year=2025,
            license="CC-BY-4.0",
            repository="Figshare",
            keywords=[
                "motor recovery",
                "stroke",
                "rehabilitation",
                "fNIRS",
                "EEG",
                "brain-computer interface",
            ],
        ),
        sessions_per_subject=6,
        tags=Tags(pathology=["Stroke"], modality=["Motor"], type=["Motor Imagery"]),
        file_format="EDF and FIF",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 16 + 1)),
            sessions_per_subject=6,
            events={"flash_l": 1, "flash_r": 2},
            code="Medvedeva2026",
            interval=(0, 4),
            paradigm="imagery",
            doi="10.6084/m9.figshare.28904942",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _download_and_extract(self):
        """Download the EEG FIF archive once and return its extraction folder."""
        path_zip = Path(dl.data_dl(MEDVEDEVA2026_URL, self.code))
        extract_dir = path_zip.parent / "EEG_FIF"

        if not extract_dir.is_dir() or not any(extract_dir.glob("*.fif")):
            extract_dir.mkdir(exist_ok=True)
            try:
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
            except BadZipFile:
                warnings.warn(
                    "Corrupted zip file detected, re-downloading...", stacklevel=2
                )
                path_zip.unlink(missing_ok=True)
                path_zip = Path(
                    dl.data_dl(MEDVEDEVA2026_URL, self.code, force_update=True)
                )
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)

        return extract_dir

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the list of EEG FIF file paths for a single subject."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        extract_dir = self._download_and_extract()

        subject_paths = []
        for fif_path in sorted(extract_dir.rglob("*.fif")):
            m = _FNAME_RE.match(fif_path.name)
            if m and int(m.group(2)) == subject:
                subject_paths.append(str(fif_path))

        return subject_paths

    def _get_single_subject_data(self, subject):
        """Return the EEG data of a single subject as {session: {run: raw}}."""
        file_paths = self.data_path(subject)

        sessions = {}
        for fpath in file_paths:
            m = _FNAME_RE.match(Path(fpath).name)
            paretic_side = m.group(1).upper()
            session_str = m.group(3)
            hand = m.group(4).lower()
            run_idx = m.group(5)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.read_raw_fif(fpath, preload=True, verbose=False)

            if not self.return_all_modalities:
                raw = raw.pick("eeg")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    montage = mne.channels.make_standard_montage("standard_1005")
                    raw = raw.set_montage(
                        montage, on_missing="ignore", match_case=False, verbose=False
                    )

            session = sessions.setdefault(session_str, {})
            # Use the paretic side from the file name to flag whether the hand
            # moved on this run is the paretic or the intact limb.
            limb = "paretic" if hand[0].upper() == paretic_side else "intact"
            # MOABB requires the run key to start with an integer index followed
            # by a letters+digits description only (no separators like "_").
            run_str = f"{len(session)}{limb}{hand}{run_idx}"
            session[run_str] = raw

        return sessions
