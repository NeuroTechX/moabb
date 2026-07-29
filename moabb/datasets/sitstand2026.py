"""Sit-to-stand / stand-to-sit transition motor imagery dataset (Leelakittisin 2026)."""

import zipfile as z
from pathlib import Path

import mne
import numpy as np

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


# Zenodo raw record (per-subject zips, each holding two session .mat files).
SITSTAND_BASE_URL = "https://zenodo.org/records/20348444/files/"

# The 60 EEG channels in acquisition order (readme.pdf, channel index 1-60),
# followed by the two EOG channels and the trigger channel (index 61-63).
EEG_CHANNELS = [
    "Fp1",
    "Fp2",
    "AF7",
    "AF8",
    "F7",
    "F8",
    "FT7",
    "FT8",
    "AF3",
    "AF4",
    "AFz",
    "Fz",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "FCz",
    "Cz",
    "FC1",
    "FC2",
    "FC3",
    "FC4",
    "FC5",
    "FC6",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "CPz",
    "Pz",
    "CP1",
    "CP2",
    "CP3",
    "CP4",
    "CP5",
    "CP6",
    "TP7",
    "TP8",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P7",
    "P8",
    "POz",
    "Oz",
    "PO3",
    "PO4",
    "PO7",
    "PO8",
    "PO9",
    "PO10",
    "O1",
    "O2",
]
EOG_CHANNELS = ["hEOG", "vEOG"]
TRIGGER_CHANNEL = "trigger"

# The raw ``.mat`` stores EEG/EOG signals in microvolts; MNE expects volts.
MICROVOLTS_TO_VOLTS = 1e-6


class SitStand2026(BaseDataset):
    """Sit-to-stand / stand-to-sit transition motor imagery dataset [1]_.

    **Dataset description**

    This is the first publicly accessible EEG dataset explicitly targeting the
    transitions between sitting and standing during both motor execution (ME)
    and motor imagery (MI) tasks. Twenty-two healthy participants (aged 22-28
    years) performed sit-to-stand and stand-to-sit transitions while 60-channel
    EEG, 2 electrooculography (EOG) and 6 electromyography (EMG) signals were
    recorded synchronously. Twenty-three subjects were recorded; subject S05 was
    excluded for poor signal quality, leaving 22 subjects. Each subject has two
    recording sessions (``S<ID>_S1.mat`` and ``S<ID>_S2.mat``).

    EEG and EOG were sampled at 1200 Hz; EMG at 2000 Hz (EMG is not loaded by
    this MOABB loader, which returns only the 1200 Hz EEG/EOG montage). The hEOG
    electrode is at the right temple, vEOG at the right infra-orbital position.

    The raw ``.mat`` file for each session stores an ``eeg`` matrix of shape
    (n_channels x n_timepoints) with 63 rows: 60 EEG channels, hEOG, vEOG, and a
    trigger channel (the 63rd row) carrying the following event codes:

    - 1  : eyes closed, resting
    - 2  : eyes opened, resting
    - 10 : start of ME trials
    - 11 : ME_SIT_STD  (executed sit-to-stand)
    - 12 : ME_STD_SIT  (executed stand-to-sit)
    - 13 : ME_R        (executed resting condition)
    - 20 : start of MI trials during sit
    - 21 : MI_SIT_STD  (imagined sit-to-stand)
    - 22 : MI_SIT_SIT  (imagined staying seated)
    - 23 : MI_R_SIT    (rest while sitting)
    - 30 : start of MI trials during stand
    - 31 : MI_STD_STD  (imagined staying standing)
    - 32 : MI_STD_SIT  (imagined stand-to-sit)
    - 33 : MI_R_STD    (rest while standing)

    Consistent with the ``imagery`` paradigm, this loader exposes the two
    imagined transition classes -- imagined sit-to-stand (code 21) versus
    imagined stand-to-sit (code 32) -- which are the transitions the dataset
    explicitly targets. A benchmark EEGNet classification reported in the
    accompanying readme reached ~70% accuracy for MI (and ~80% for ME).

    References
    ----------

    .. [1] Uengsawapak, B., Kongwudhikunakorn, S., Kiatthaveephong, S.,
       Polpakdee, W., Chaisaen, R., Manoonpong, P., Chuenchit, C.,
       Bhakdisongkhram, G., & Wilaiprasitporn, T. (2025). EEG-Based Dataset
       Explicitly Targeting the Transitions between Sitting and Standing for
       Exploring Neural Activation Patterns in Motor Imagery and Execution.
       Zenodo. DOI: https://doi.org/10.5281/zenodo.20348444

    .. versionadded:: 1.2.0

    """

    nemar_id = "EXEMPT"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1200.0,
            n_channels=60,
            channel_types={"eeg": 60, "eog": 2, "stim": 1},
            montage="10-05",
            reference=None,
            ground=None,
            sensors=EEG_CHANNELS,
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=2,
                eog_type=["horizontal", "vertical"],
                has_emg=True,
                emg_channels=6,
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=22,
            health_status="healthy",
            gender={"male": 14, "female": 8},
            age_min=18.0,
            age_max=30.0,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["sit_stand", "stand_sit"],
            trial_duration=4.0,
            events={"sit_stand": 21, "stand_sit": 32},
            study_design=(
                "Sit-to-stand and stand-to-sit transitions performed under both "
                "motor execution and motor imagery conditions; trigger codes at "
                "channel 63 mark each transition/rest event."
            ),
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.20348444",
            description=(
                "First public EEG dataset explicitly targeting sit-to-stand and "
                "stand-to-sit transitions during motor execution and motor imagery, "
                "from 22 healthy participants with 60-channel EEG, EOG and EMG."
            ),
            investigators=[
                "Benjakarn Uengsawapak",
                "Supavit Kongwudhikunakorn",
                "Suktipol Kiatthaveephong",
                "Wipamas Polpakdee",
                "Rattanaphon Chaisaen",
                "Poramate Manoonpong",
                "Chanitsada Chuenchit",
                "Gun Bhakdisongkhram",
                "Theerawit Wilaiprasitporn",
            ],
            institution="Vidyasirimedhi Institute of Science and Technology",
            country="TH",
            license="CC0-1.0",
            repository="Zenodo",
            publication_year=2025,
        ),
        sessions_per_subject=2,
        runs_per_session=1,
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
    )

    def __init__(self):
        # Subjects S01-S23 excluding S05 (poor signal quality) -> 22 subjects.
        subjects = [s for s in range(1, 24) if s != 5]
        super().__init__(
            subjects=subjects,
            sessions_per_subject=2,
            events={"sit_stand": 21, "stand_sit": 32},
            code="SitStand2026",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.5281/zenodo.20348444",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local paths of the two session ``.mat`` files for a subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level.

        Returns
        -------
        list of str
            Paths to ``S<ID>_S1.mat`` and ``S<ID>_S2.mat``.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        url = f"{SITSTAND_BASE_URL}v1_raw_S{subject:02d}.zip?download=1"
        path_zip = Path(dl.data_dl(url, self.code, path=path, force_update=force_update))
        extract_dir = path_zip.parent / f"S{subject:02d}"

        def resolve_mat_paths():
            resolved = []
            for sess in (1, 2):
                filename = f"S{subject:02d}_S{sess}.mat"
                direct = extract_dir / filename
                if direct.exists():
                    resolved.append(direct)
                    continue
                found = sorted(extract_dir.rglob(filename))
                if not found:
                    return []
                resolved.append(found[0])
            return resolved

        # Zenodo's archives contain an extra ``v1_raw_s<ID>`` directory. Check
        # that nested layout before opening the archive: otherwise every
        # subsequent ``get_data`` call needlessly extracts the same subject
        # again, which also breaks read-only/offline compute jobs.
        mat_paths = resolve_mat_paths()
        if force_update or len(mat_paths) != 2:
            with z.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            mat_paths = resolve_mat_paths()

        if len(mat_paths) != 2:
            raise FileNotFoundError(
                f"Could not find both session .mat files under {extract_dir}"
            )
        return [str(mat_path) for mat_path in mat_paths]

    def _build_raw(self, mat_path):
        """Build an :class:`mne.io.RawArray` from one session ``.mat`` file."""
        from scipy.io import loadmat

        mat = loadmat(mat_path, squeeze_me=True)
        eeg = np.asarray(mat["eeg"], dtype=float)
        sfreq = float(np.atleast_1d(mat["eeg_fs"]).ravel()[0])
        n_rows = eeg.shape[0]

        # Recover channel names from the file when available, else use the
        # documented acquisition order. The raw ``.mat`` stores these lower-cased
        # (e.g. ``fp1``, ``fcz``) with ``hEOG``/``vEOG``/``trigger`` for the three
        # auxiliary rows.
        documented = EEG_CHANNELS + EOG_CHANNELS + [TRIGGER_CHANNEL]
        if "eeg_channels" in mat:
            raw_names = [str(c).strip() for c in np.atleast_1d(mat["eeg_channels"])]
        else:
            raw_names = list(documented)

        # Validate that the stored names line up with the data matrix; fall back
        # to the documented acquisition order when they do not.
        if len(raw_names) != n_rows:
            if len(documented) == n_rows:
                raw_names = list(documented)
            else:
                raise ValueError(
                    f"eeg matrix has {n_rows} rows but {len(raw_names)} channel "
                    f"names were resolved for {mat_path}"
                )

        # Normalise casing against the 10-05 montage so channels are recognised,
        # and derive the MNE channel type from each (case-insensitive) name.
        # Unknown EEG labels are upper-cased as a best-effort fallback.
        montage = mne.channels.make_standard_montage("standard_1005")
        lower_to_std = {ch.lower(): ch for ch in montage.ch_names}

        ch_names, ch_types = [], []
        for name in raw_names:
            low = name.lower()
            if low in ("trigger", "stim", "sti"):
                ch_names.append("STI")
                ch_types.append("stim")
            elif low in ("heog", "veog", "eog"):
                ch_names.append("hEOG" if low == "heog" else "vEOG")
                ch_types.append("eog")
            else:
                ch_names.append(lower_to_std.get(low, name.upper()))
                ch_types.append("eeg")

        # Convert microvolts -> volts for the physiological (EEG/EOG) rows only;
        # the stim/trigger row carries raw integer event codes and must not be
        # rescaled.
        scale = np.array(
            [MICROVOLTS_TO_VOLTS if t in ("eeg", "eog") else 1.0 for t in ch_types]
        )
        eeg = eeg * scale[:, np.newaxis]

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(eeg, info, verbose=False)
        raw.set_montage(montage, on_missing="ignore", verbose=False)
        return raw

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        mat_paths = self.data_path(subject)
        sessions = {}
        for sess_idx, mat_path in enumerate(mat_paths):
            raw = self._build_raw(mat_path)
            sessions[str(sess_idx)] = {"0": raw}
        return sessions
