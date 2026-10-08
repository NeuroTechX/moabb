"""Closed-loop motor imagery EEG (live recordings) from Shin, Suma and He (2022)."""

import logging
import zipfile
from pathlib import Path

import mne
import numpy as np

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)
from .utils import safe_extract_zip


log = logging.getLogger(__name__)

# Figshare article 20383716 (v1): every subject in one ~353 MB ZIP
# ("study-2022-data-anonymized.zip").
_FIGSHARE_ZIP_URL = "https://ndownloader.figshare.com/files/36438960"

# The g.Nautilus RESEARCH records 16 dry (g.SAHARA) EEG channels.
_N_EEG = 16

# BCI2000 ``TargetCode`` state (constant within a trial) -> event name, following
# the He-lab convention of the identical LR task in Stieger2021.
_TARGETCODE_TO_EVENT = {1: "right_hand", 2: "left_hand"}

# Online source (0.5-30 Hz) filtering was disabled during acquisition
# (FilterEnabled=0 in the BCI2000 headers), so the stored signal is broadband.
# g.Nautilus stores calibrated microvolts (SourceChGain=1); convert to volts.
_UV_TO_V = 1e-6


class Shin2022(BaseDataset):
    """Closed-loop 1D left/right motor imagery EEG dataset (live recordings).

    Dataset from [1]_ (data record [2]_). Ten healthy adults performed a 1D
    left/right center-out sensorimotor-rhythm cursor task under continuous
    visual feedback, recorded with a 16-channel dry-electrode g.tec g.Nautilus
    at 250 Hz through BCI2000. Each subject has one session of 11 runs of 24
    trials (12 per class), one run per value of an online control parameter:
    normalization bin width (``BW30``..``BW120``), maximum cursor velocity
    (``CV200``..``CV350``) and trials carried for normalization (``NT0``,
    ``NT24``, ``NT48``). These only change the online dynamics, not the task.

    Labels are data-borne: the cued class is the BCI2000 ``TargetCode`` state
    and events are placed at the onset of the ``Feedback`` (active control)
    period, whose variable length (target hit or 6 s timeout) is stored as the
    annotation duration. Only the recorded ``LIVE`` EEG is loaded; the archive's
    ``SIMULATED`` tree (synthetic cursor telemetry in ``.csv``) is not EEG.

    Reading the ``.dat`` files requires ``pip install "moabb[bci2000]"``.

    Paper-vs-release note: the paper describes each session as "10 runs of 24
    1D LR center-out discrete trials", because "the 'NT = 0 trials' run
    ... was jointly represented by the 'BW = 60 s (default)' run, thus forming
    10 runs (4 + 2 + 4)". The archive's ``NT`` folder is documented here with
    three files (``NT0``, ``NT24``, ``NT48``), i.e. 11 runs; the loader
    enumerates whatever ``.dat`` files the archive holds, so the declared 11
    runs / 2640 trials remain to be confirmed against the release. The
    online pipeline used a "Notch filtered 58-62 Hz" stage (60 Hz mains).

    Notes
    -----
    The BCI2000 headers store no electrode labels, so the channels are named
    ``EEG1``..``EEG16`` and no montage is applied; the online classifier uses
    ``EEG7`` / ``EEG9`` as the C3 / C4 control pair. Signals are converted from
    microvolts to volts.

    References
    ----------
    .. [1] H. Shin, D. Suma and B. He, "Closed-loop motor imagery EEG
       simulation for brain-computer interfaces," Frontiers in Human
       Neuroscience, vol. 16, 951591, 2022.
       DOI: 10.3389/fnhum.2022.951591

    .. [2] H. Shin, D. Suma and B. He, "Data from: Closed-loop motor imagery
       EEG simulation for brain-computer interfaces," figshare, 2022.
       DOI: 10.6084/m9.figshare.20383716
    """

    nemar_id = "nm000315"

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            channel_types={"eeg": 16},
            sensor_type="dry",
            hardware="g.tec g.Nautilus RESEARCH (16 channel)",
            software="BCI2000",
            reference="unknown",
            line_freq=60.0,
            cap_manufacturer="g.tec medical engineering GmbH",
            cap_model="g.SAHARA dry electrode system",
            electrode_type="wire",
            filters="broadband (online 0.5-30 Hz filter disabled at source)",
        ),
        participants=ParticipantMetadata(
            n_subjects=10, health_status="healthy", species="human"
        ),
        experiment=ExperimentMetadata(
            events={"right_hand": 1, "left_hand": 2},
            paradigm="imagery",
            n_classes=2,
            class_labels=["right_hand", "left_hand"],
            trial_duration=6.0,
            study_design=(
                "Closed-loop 1D left/right (LR) center-out SMR cursor control. "
                "Left- vs right-hand motor imagery moves a cursor toward a left "
                "or right target under continuous visual feedback. 24 trials per "
                "run (12 left / 12 right), 11 runs per subject sweeping online "
                "control parameters (BW, CV, NT); one recording day per subject."
            ),
            feedback_type="visual",
            stimulus_type="visual cursor",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
            instructions=(
                "Imagine left- or right-hand movement to drive the cursor toward "
                "the cued left or right target."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnhum.2022.951591",
            investigators=["Hyonyoung Shin", "Daniel Suma", "Bin He"],
            senior_author="Bin He",
            institution="Carnegie Mellon University",
            institution_department="Department of Biomedical Engineering",
            country="US",
            data_url="https://doi.org/10.6084/m9.figshare.20383716",
            repository="Figshare",
            license="CC BY 4.0",
            publication_year=2022,
            associated_paper_doi="10.3389/fnhum.2022.951591",
            keywords=[
                "motor imagery",
                "EEG",
                "sensorimotor rhythm",
                "brain-computer interface",
                "closed-loop",
                "cursor control",
                "dry electrodes",
            ],
        ),
        preprocessing=PreprocessingMetadata(data_state="continuous"),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery", imagery_tasks=["right_hand", "left_hand"]
        ),
        data_structure=DataStructureMetadata(
            n_trials=2640,
            trials_context="10 subjects x 11 runs x 24 trials (12 left / 12 right)",
            n_trials_per_class={"right_hand": 12, "left_hand": 12},
        ),
        signal_processing=SignalProcessingMetadata(
            frequency_bands={"alpha": [8.0, 12.0]},
            spatial_filters="small surface Laplacian around C3 and C4",
            feature_extraction=["autoregressive_spectrum", "alpha_band_power"],
        ),
        cross_validation=CrossValidationMetadata(evaluation_type=["within_subject"]),
        bci_application=BCIApplicationMetadata(
            applications=["cursor_control"],
            environment="laboratory",
            online_feedback=True,
        ),
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Research"]),
        sessions_per_subject=1,
        runs_per_session=11,
        file_format="BCI2000",
    )

    _events = {"right_hand": 1, "left_hand": 2}

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 11)),
            sessions_per_subject=1,
            events=self._events,
            code="Shin2022",
            interval=[0, 3.0],
            paradigm="imagery",
            doi="10.3389/fnhum.2022.951591",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _read_bci2000_dat(self, dat_path):
        """Read a BCI2000 ``.dat`` file and return an MNE Raw with annotations."""
        try:
            from BCI2kReader.BCI2kReader import BCI2kReader
        except ImportError as err:
            raise ImportError(
                "BCI2kReader is required for Shin2022.  "
                'Install it with: pip install "moabb[bci2000]"'
            ) from err

        reader = BCI2kReader(dat_path)
        sfreq = float(reader.samplingrate)

        n_eeg = min(_N_EEG, reader.signals.shape[0])
        data = reader.signals[:n_eeg].astype(np.float64) * _UV_TO_V  # uV -> V

        ch_names = [f"EEG{i}" for i in range(1, n_eeg + 1)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)

        # One event per ``Feedback`` block (zero-padded, so a block still open
        # at the end of the file closes there), labelled by its ``TargetCode``.
        target = reader.states["TargetCode"].flatten().astype(int)
        feedback = reader.states["Feedback"].flatten().astype(int)
        edges = np.diff(np.r_[0, feedback == 1, 0].astype(int))
        onset_s, duration_s, description = [], [], []
        for onset, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if target[onset] in _TARGETCODE_TO_EVENT:
                onset_s.append(onset / sfreq)
                duration_s.append((end - onset) / sfreq)
                description.append(_TARGETCODE_TO_EVENT[target[onset]])

        raw.set_annotations(
            mne.Annotations(onset_s, duration_s, description), verbose=False
        )
        return raw

    def _get_single_subject_data(self, subject):
        """Return ``{session: {run: Raw}}`` for one subject (LIVE runs only)."""
        session_dir = Path(self.data_path(subject))
        runs = {}
        for condition in ("BW", "CV", "NT"):
            for dat_file in sorted(session_dir.glob(f"{condition}/*.dat")):
                # Run keys must start with an integer; keep the source stem.
                runs[f"{len(runs)}{dat_file.stem}"] = self._read_bci2000_dat(
                    str(dat_file)
                )
        if not runs:
            raise FileNotFoundError(
                f"No .dat runs found for subject {subject} under {session_dir}"
            )
        return {"0": runs}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download the archive (once), extract this subject's LIVE ``.dat`` runs.

        Returns the LIVE session directory holding the ``BW`` / ``CV`` / ``NT``
        run folders.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        zip_path = Path(
            dl.data_dl(_FIGSHARE_ZIP_URL, self.code, path, force_update, verbose)
        )
        extract_root = zip_path.parent
        with zipfile.ZipFile(zip_path) as zf:
            # LIVE folders are ``S{NN}_<date>``; ``SIMULATED_S{NN}_...`` never match.
            live = sorted(
                {
                    n.split("/")[0]
                    for n in zf.namelist()
                    if n.startswith(f"S{subject:02d}_")
                }
            )
            if not live:
                raise FileNotFoundError(
                    f"No LIVE recordings found for subject {subject} in the archive"
                )
            dat_members = [
                info
                for info in zf.infolist()
                if info.filename.startswith(live[0] + "/")
                and info.filename.lower().endswith(".dat")
            ]
            if not dat_members:
                raise FileNotFoundError(
                    f"No .dat files for subject {subject} in {live[0]}"
                )
            target_dir = extract_root / live[0]
            existing = list(target_dir.rglob("*.dat")) if target_dir.is_dir() else []
            if force_update or len(existing) < len(dat_members):
                safe_extract_zip(zf, extract_root, members=dat_members)
        # .../S{NN}_<date>/S{NN}_<date>_LR_S1001/{BW,CV,NT}/<run>.dat
        dat_files = sorted(target_dir.rglob("*.dat"))
        if not dat_files:
            raise FileNotFoundError(f"Extraction failed: no .dat under {target_dir}")
        return str(dat_files[0].parents[1])  # .../<LR session>/{BW,CV,NT}/<run>.dat
