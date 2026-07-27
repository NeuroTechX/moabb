"""KMI HandGrip 2025 graded-force kinesthetic motor imagery dataset."""

import json

import mne
import pandas as pd

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    Tags,
)


# Mendeley Data public-api record; the ``files`` array of the returned JSON
# carries a (long-lived) signed ``download_url`` for every file.
KMI_HANDGRIP2025_RECORD = "https://data.mendeley.com/public-api/datasets/msgzn862ns"

# The four graded-force conditions, encoded in the file name (Sxx_<level>_KMI.csv).
KMI_HANDGRIP2025_LEVELS = ["10", "40", "70", "100"]

# 20 Cognionics Quick-20m electrodes. The CSV header uses the legacy T3/T4
# labels; MNE's standard montages use the modern T7/T8, so we rename on load.
KMI_HANDGRIP2025_RENAME = {"T3": "T7", "T4": "T8"}

# Sampling rate: every recording is 84 s long with 42000 samples -> 500 Hz.
KMI_HANDGRIP2025_SFREQ = 500.0

# 4 s initial rest, then 10 repetitions of 4 s task + 4 s rest (= 84 s). The
# CSVs carry no trigger/marker channel, so this initial-rest-first ordering is
# inferred from the source description ("with initial and final resting
# period") rather than confirmed from an in-file marker; a task-first layout
# (onsets 0, 8, ..., 72) is arithmetically equivalent at 84 s.
KMI_HANDGRIP2025_N_TRIALS = 10
KMI_HANDGRIP2025_ONSET0 = 4.0
KMI_HANDGRIP2025_PERIOD = 8.0
KMI_HANDGRIP2025_TASK = 4.0


class KMIHandGrip2025(BaseDataset):
    """Graded-force kinesthetic motor imagery of hand-grip [1]_.

    **Dataset description**

    Raw EEG recorded during kinesthetic motor imagery (KMI) of a hand-grip at
    graded force levels of 10%, 40%, 70% and 100% of the maximal voluntary
    contraction (MVC). Fifty healthy right-handed university students
    (23 female, 27 male; 17-30 years, mean 21.78 +/- 2.66) took part; each
    subject performed the task with a single hand: subjects S01-S25 with the
    left hand and subjects S26-S50 with the right hand.

    For each of the four force levels a single 84 s recording was acquired,
    consisting of a 4 s initial resting period followed by 10 repetitions of
    4 s of imagery execution and 4 s of rest. The four force levels are treated
    here as the four decoding classes; each recording is loaded as one run and
    contributes 10 imagery trials of its class (40 trials per subject in total).

    EEG was recorded with a Cognionics Quick-20m dry-electrode headset carrying
    20 surface electrodes at Fp1, Fp2, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8,
    A2, P7, P3, Pz, P4, P8, O1 and O2 (the CSV files use the legacy T3/T4 labels
    for T7/T8), sampled at 500 Hz. Signals are provided fully raw (no filtering
    or preprocessing); the first row of every CSV holds the electrode names.
    Amplitudes are stored in microvolts and converted to volts on load.

    References
    ----------

    .. [1] Martinez Peon, D. C., & Perez Espinoza, M. (2025). EEG signals of
       KMI levels of the right and left hands. Mendeley Data, V1.
       DOI: https://doi.org/10.17632/msgzn862ns.1

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=20,
            channel_types={"eeg": 20},
            montage="10-20",
            hardware="Cognionics Quick-20m dry-electrode EEG headset",
            sensor_type="dry",
            electrode_type="dry",
            reference="A2",
            ground=None,
            sensors=[
                "A2",
                "C3",
                "C4",
                "Cz",
                "F3",
                "F4",
                "F7",
                "F8",
                "Fp1",
                "Fp2",
                "Fz",
                "O1",
                "O2",
                "P3",
                "P4",
                "P7",
                "P8",
                "Pz",
                "T7",
                "T8",
            ],
        ),
        participants=ParticipantMetadata(
            n_subjects=50,
            health_status="healthy",
            gender={"male": 27, "female": 23},
            age_mean=21.78,
            age_std=2.66,
            age_min=17.0,
            age_max=30.0,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=4,
            class_labels=["grip_10", "grip_40", "grip_70", "grip_100"],
            trials_per_class={
                "grip_10": 10,
                "grip_40": 10,
                "grip_70": 10,
                "grip_100": 10,
            },
            trial_duration=4.0,
            study_design=(
                "Kinesthetic motor imagery of a hand-grip at 10/40/70/100% of "
                "maximal voluntary contraction. One 84 s recording per force "
                "level: 4 s initial rest, then 10 x (4 s imagery + 4 s rest). "
                "S01-S25 left hand, S26-S50 right hand."
            ),
            feedback_type="none",
            synchronicity="cue-based",
            mode="offline",
            events={"grip_10": 1, "grip_40": 2, "grip_70": 3, "grip_100": 4},
        ),
        documentation=DocumentationMetadata(
            doi="10.17632/msgzn862ns.1",
            description=(
                "Raw EEG during graded-force kinesthetic motor imagery of "
                "hand-grip (10/40/70/100% MVC) from 50 healthy subjects, "
                "Cognionics Quick-20m, 20 channels, 500 Hz."
            ),
            investigators=["Dulce Citlalli Martinez Peon", "Marcos Perez Espinoza"],
            country="MX",
            data_url="https://data.mendeley.com/datasets/msgzn862ns/1",
            publication_year=2025,
            keywords=[
                "motor imagery",
                "kinesthetic motor imagery",
                "hand grip",
                "force level",
                "EEG",
                "BCI",
            ],
            license="CC-BY-NC-SA-4.0",
            repository="Mendeley Data",
        ),
        sessions_per_subject=1,
        runs_per_session=4,
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
        file_format="CSV",
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 50 + 1)),
            sessions_per_subject=1,
            events={"grip_10": 1, "grip_40": 2, "grip_70": 3, "grip_100": 4},
            code="KMIHandGrip2025",
            interval=(0, 4),
            paradigm="imagery",
            doi="10.17632/msgzn862ns.1",
        )

    def _file_map(self):
        """Return a mapping ``{filename: download_url}`` from the record JSON."""
        record_path = dl.data_dl(KMI_HANDGRIP2025_RECORD, self.code)
        if isinstance(record_path, (list, tuple)):
            record_path = record_path[0]
        with open(record_path, encoding="utf-8") as fid:
            record = json.load(fid)
        return {
            f["filename"]: f["content_details"]["download_url"] for f in record["files"]
        }

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local paths of the four force-level CSV files of a subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level.

        Returns
        -------
        list of str
            The four CSV paths, ordered 10/40/70/100% MVC.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        file_map = self._file_map()
        sub = f"S{subject:02d}"
        paths = []
        for level in KMI_HANDGRIP2025_LEVELS:
            filename = f"{sub}_{level}_KMI.csv"
            url = file_map[filename]
            local = dl.data_dl(url, self.code, force_update=force_update)
            if isinstance(local, (list, tuple)):
                local = local[0]
            paths.append(str(local))
        return paths

    def _read_run(self, file_path, event_code):
        """Build a single ``Raw`` (one force level) with 10 imagery events."""
        df = pd.read_csv(file_path)
        ch_names = [KMI_HANDGRIP2025_RENAME.get(c, c) for c in df.columns]
        # microvolts -> volts
        data = df.to_numpy(dtype=float).T * 1e-6
        info = mne.create_info(
            ch_names=ch_names, sfreq=KMI_HANDGRIP2025_SFREQ, ch_types="eeg"
        )
        raw = mne.io.RawArray(data, info, verbose=False)

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore", verbose=False)

        onsets = [
            KMI_HANDGRIP2025_ONSET0 + KMI_HANDGRIP2025_PERIOD * i
            for i in range(KMI_HANDGRIP2025_N_TRIALS)
        ]
        label = [k for k, v in self.event_id.items() if v == event_code][0]
        annotations = mne.Annotations(
            onset=onsets,
            duration=[KMI_HANDGRIP2025_TASK] * KMI_HANDGRIP2025_N_TRIALS,
            description=[label] * KMI_HANDGRIP2025_N_TRIALS,
        )
        raw.set_annotations(annotations, verbose=False)
        return raw

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}."""
        paths = self.data_path(subject)
        runs = {}
        for run_idx, (file_path, level) in enumerate(zip(paths, KMI_HANDGRIP2025_LEVELS)):
            event_code = self.event_id[f"grip_{level}"]
            runs[str(run_idx)] = self._read_run(file_path, event_code)
        return {"0": runs}
