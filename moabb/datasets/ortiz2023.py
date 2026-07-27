"""Ortiz 2023 motor imagery during walking with a lower-limb exoskeleton dataset."""

import warnings
import zipfile as z
from collections import defaultdict
from pathlib import Path
from zipfile import BadZipFile

import mne
import numpy as np
from scipy.io import loadmat

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


# Single Figshare archive (article 21185362, v2) holding the whole database.
ORTIZ2023_URL = "https://ndownloader.figshare.com/files/37759473"

# MOABB subject id -> internal participant code (the M-code embedded in the .mat
# file names). Restricted to the EXPERIENCE (flat-ground) scenario, sorted by
# M-code. The subject_NN folder numbering in the archive is not a stable
# participant id (some folders hold two M-codes), so the M-code is used instead.
SUBJECT_MCODE = {
    1: "M05",
    2: "M06",
    3: "M07",
    4: "M08",
    5: "M09",
    6: "M10",
    7: "M11",
    8: "M17",
    9: "M20",
    10: "M21",
}

# 27 EEG electrodes in the row order of ``data_EEG`` (as listed in
# conf.acquisition.device.devices_EEG.electrodes_names_selected), normalised to
# MNE 10-05 capitalisation (FZ -> Fz, FCZ -> FCz, CZ -> Cz, CPZ -> CPz, PZ -> Pz).
EEG_CHANNELS = [
    "F3", "Fz", "FC1", "FCz", "C1", "Cz", "CP1", "CPz",
    "FC5", "FC3", "C5", "C3", "CP5", "CP3", "P3", "Pz",
    "F4", "FC2", "FC4", "FC6", "C2", "C4", "CP2", "CP4",
    "C6", "CP6", "P4",
]  # fmt: skip
# 4 EOG electrodes (rows 28-31 of ``data_EEG``), bipolar montage around one eye.
EOG_CHANNELS = ["HR", "HL", "VU", "VD"]

# Effective sample-wise task codes stored in ``task_EEG``. The transition /
# instruction codes (400, 401, 403, 405) and the baseline codes (709, 900) are
# not decoded as classes.
TASK_CODES = {402: "relax", 404: "motor_imagery", 406: "regressive_count"}
MI_CODE = 404

SFREQ = 200.0

# openloop file indices within a recording: 1 = sitting baseline, 2 = standing
# baseline, 3..18 = the 16 mental-task trials. Only the task trials are exposed.
FIRST_TASK_INDEX = 3


class Ortiz2023(BaseDataset):
    """Motor imagery during walking with a lower-limb exoskeleton [1]_.

    .. admonition:: Dataset summary

        ==========  =======  =======  ==========  =================  ============  ===============  ===========
        Name          #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate    #Sessions
        ==========  =======  =======  ==========  =================  ============  ===============  ===========
        Ortiz2023        10       27           3                 16             9s            200Hz          1-2
        ==========  =======  =======  ==========  =================  ============  ===============  ===========

    Dataset [1]_ recorded within the DECODED sub-project of the EU EUROBENCH
    project for the cognitive assessment of motor imagery during walking with a
    lower-limb exoskeleton. Able-bodied participants wore an H3 exoskeleton that
    provided fully assisted walking while they performed mental tasks. Because
    the exoskeleton produced the gait, the "motor imagery" class is the
    kinesthetic imagination of the limb movement rather than a voluntary
    execution of walking (the participant does not command the gait). Each task
    trial (open-loop) follows the sequence:

    - 15 s standing and relaxed (code 402, effective 10 s),
    - 24 s walking while imagining the limb movement (code 404, effective 20 s),
    - 22 s walking while performing a regressive-count mental task (code 406,
      effective 20 s),
    - 14 s standing and relaxed (code 402, effective 10 s).

    Three classes are exposed from the data-borne ``task_EEG`` code channel:
    ``relax``, ``motor_imagery`` and ``regressive_count``. The canonical motor
    imagery contrast reported in the paper is ``relax`` vs ``motor_imagery``
    (the "Motor Imagery Index"); ``regressive_count`` supports the secondary
    "Attention to Gait Index" (``motor_imagery`` vs ``regressive_count``).

    EEG was recorded with a Brain Products actiCHamp amplifier, 27 wet
    electrodes over fronto-central and parietal areas plus 4 EOG channels, at
    200 Hz, referenced/grounded to the earlobes. Only online hardware filtering
    was applied (0.1 Hz high-pass and 50 Hz notch); no artefact removal.

    This loader exposes only the **EXPERIENCE** (flat-ground) scenario, which is
    structurally consistent across participants: 10 participants, one recording
    per participant (M05 and M11 have a second recording exposed as a second
    session), each with 16 task trials (runs). The archive also contains a
    **SLOPES** (inclined-surface) scenario for a partly different set of
    participants; it uses variable-length trials, a different code table and
    duplicated ``_sync`` copies, and is not loaded here.

    References
    ----------

    .. [1] Ortiz, M., de la Ossa, L., Ianez, E., Torricelli, D., Tornero, J., &
       Azorin, J. M. (2023). An EEG database for the cognitive assessment of
       motor imagery during walking with a lower-limb exoskeleton. Scientific
       Data, 10, 343. https://doi.org/10.1038/s41597-023-02243-7

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=SFREQ,
            n_channels=27,
            channel_types={"eeg": 27, "eog": 4},
            sensors=EEG_CHANNELS,
            sensor_type="Ag/AgCl wet",
            reference="linked earlobes (A1, A2)",
            ground="earlobe",
            hardware="Brain Products actiCHamp",
            montage="standard_1005",
            line_freq=50.0,
            filters="0.1 Hz high-pass and 50 Hz notch (online, hardware)",
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=4,
                eog_type=["horizontal", "horizontal", "vertical", "vertical"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="healthy",
            species="homo sapiens",
            age_mean=28.7,
            age_std=4.8,
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["relax", "motor_imagery", "regressive_count"],
            trial_duration=9.0,
            study_design=(
                "Kinesthetic motor imagery of gait alternated with relaxation "
                "and a regressive-count distractor task while walking with a "
                "fully assisted lower-limb exoskeleton on flat ground."
            ),
            stimulus_type="auditory",
            stimulus_modalities=["audio"],
            synchronicity="cue-based",
            mode="offline",
            events={"relax": 1, "motor_imagery": 2, "regressive_count": 3},
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-023-02243-7",
            description=(
                "EEG database for the cognitive assessment of motor imagery "
                "during walking with a lower-limb exoskeleton (DECODED, a "
                "EUROBENCH sub-project); flat-ground (EXPERIENCE) scenario."
            ),
            investigators=[
                "Mario Ortiz",
                "Luis de la Ossa",
                "Eduardo Ianez",
                "Diego Torricelli",
                "Jesus Tornero",
                "Jose M. Azorin",
            ],
            institution="Universitas Miguel Hernandez de Elche",
            country="ES",
            repository="Figshare",
            data_url="https://doi.org/10.6084/m9.figshare.21185362.v2",
            license="CC-BY-4.0",
            publication_year=2023,
        ),
        sessions_per_subject=1,
        runs_per_session=16,
        sessions=["0", "1"],
        tags=Tags(pathology=["healthy"], modality=["motor"], type=["imagery"]),
        file_format="MAT",
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10 + 1)),
            sessions_per_subject=1,
            events={"relax": 1, "motor_imagery": 2, "regressive_count": 3},
            code="Ortiz2023",
            interval=[0, 9],
            paradigm="imagery",
            doi="10.1038/s41597-023-02243-7",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the archive (once) and return the extraction directory."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        path_zip = Path(
            dl.data_dl(ORTIZ2023_URL, self.code, path=path, force_update=force_update)
        )
        extract_dir = path_zip.parent / "MI_walking_figshare21185362"
        marker = extract_dir / "EXPERIENCE"
        if not marker.is_dir():
            try:
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
            except BadZipFile:
                warnings.warn(
                    "Corrupted zip file detected, re-downloading...", stacklevel=2
                )
                path_zip.unlink(missing_ok=True)
                path_zip = Path(
                    dl.data_dl(ORTIZ2023_URL, self.code, path=path, force_update=True)
                )
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
        return str(extract_dir)

    def _make_raw(self, mat_file):
        """Build a continuous mne.Raw from one open-loop task .mat file."""
        mat = loadmat(mat_file, struct_as_record=False, squeeze_me=True)["session"]

        # data_EEG is (31, n_samples): 27 EEG + 4 EOG, stored in microvolts.
        data = np.asarray(mat.data_EEG, dtype=float)
        ch_names = EEG_CHANNELS + EOG_CHANNELS
        ch_types = ["eeg"] * len(EEG_CHANNELS) + ["eog"] * len(EOG_CHANNELS)
        if data.shape[0] != len(ch_names):
            raise ValueError(
                f"{Path(mat_file).name}: expected {len(ch_names)} rows in "
                f"data_EEG, found {data.shape[0]}"
            )

        # microvolts -> volts (mne expects SI units).
        data = data * 1e-6

        info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_montage("standard_1005", on_missing="ignore", verbose=False)

        # Build annotations from the sample-wise task code channel.
        task = np.asarray(mat.task_EEG).ravel()
        onsets, durations, descriptions = [], [], []
        for code, label in TASK_CODES.items():
            mask = task == code
            if not mask.any():
                continue
            starts = np.where(np.diff(np.concatenate(([0], mask.astype(int)))) == 1)[0]
            ends = np.where(np.diff(np.concatenate((mask.astype(int), [0]))) == -1)[0]
            for s, e in zip(starts, ends):
                onsets.append(s / SFREQ)
                durations.append((e - s + 1) / SFREQ)
                descriptions.append(label)

        annotations = mne.Annotations(
            onset=onsets, duration=durations, description=descriptions
        )
        raw.set_annotations(annotations)
        return raw

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        root = Path(self.data_path(subject))
        mcode = SUBJECT_MCODE[subject]

        # All EXPERIENCE task files for this participant, regardless of whether
        # the recording-date sub-folder sits before or after ORIGINAL.
        mat_files = sorted(root.glob(f"EXPERIENCE/**/{mcode}_*_openloop_*.mat"))

        by_date = defaultdict(list)
        for f in mat_files:
            parts = f.stem.split("_")  # e.g. ["M05", "20210928", "openloop", "05"]
            if len(parts) < 4 or parts[0] != mcode:
                continue
            date = parts[1]
            try:
                idx = int(parts[3])
            except ValueError:
                continue
            if idx < FIRST_TASK_INDEX:
                continue  # skip the two resting baselines
            by_date[date].append((idx, f))

        sessions = {}
        for sess_idx, date in enumerate(sorted(by_date)):
            runs = {}
            for run_idx, (_, mat_file) in enumerate(sorted(by_date[date])):
                runs[str(run_idx)] = self._make_raw(mat_file)
            sessions[str(sess_idx)] = runs
        return sessions
