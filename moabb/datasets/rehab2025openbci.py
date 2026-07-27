"""MI Neurorehabilitation OpenBCI dataset (Kment & Moucek, 2025)."""

import warnings
import zipfile as z
from datetime import datetime
from pathlib import Path

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


# Single Zenodo archive holding every recording (record 15399490, v1).
REHAB2025_URL = "https://zenodo.org/records/15399490/files/data.zip"

# OpenBCI Cyton, 8-channel configuration. Electrode positions are not
# documented in the record, so generic sensor names are used.
CHANNELS = [f"EEG{i}" for i in range(1, 9)]

# Sampling rate. The default OpenBCI Cyton rate is 250 Hz, but this dataset
# was recorded with the Cyton WiFi shield, which streams at 1000 Hz. Verified
# from the per-recording data.txt header lines
# ("%Sample Rate = 1000 Hz" / "%Board = OpenBCI_GUI$BoardCytonWifi").
SFREQ = 1000.0

# Class markers used for the 2-class motor-imagery vs rest problem.
# See the module note about the (inferred) S-code -> class mapping.
EVENTS = {"rest": 5, "motor_imagery": 6}

# Minimum rest-block length (s). The recordings contain both long rest
# periods (~10-20 s) and instantaneous S5 boundary markers (~0.03 s) that
# merely separate trials; only the former are kept as "rest" epochs.
_MIN_REST_DURATION = 4.0


class Rehab2025OpenBCI(BaseDataset):
    """Motor imagery EEG dataset for BCI-based neurorehabilitation [1]_.

    **Dataset description**

    EEG recordings acquired from 8 participants performing upper-limb motor
    imagery tasks under a structured rehabilitation scenario, using an OpenBCI
    Cyton board in an 8-channel configuration sampled at 1000 Hz. Acquisition
    was synchronised with a Unity-based 3D environment and the FourMotors
    robotic software to simulate realistic rehabilitation conditions. Within
    each recording the participant alternated between rest and motor imagery of
    an upper-limb movement, cued by ``S1``-``S6`` event markers stored in a
    per-recording ``labels.txt`` file with millisecond wall-clock timestamps.

    The archive holds 24 recording folders named
    ``<index><initial><ddmmyyyy><task>`` (e.g. ``01m09042025e``), each with an
    OpenBCI ``data.txt`` (raw EXG export) and a ``labels.txt``. Consecutive
    triples correspond to the same participant, giving 8 subjects x 3 runs
    (subject ``s`` -> folders ``3s-2, 3s-1, 3s``). The three ``z`` recordings
    fall exactly on one triple, corroborating this grouping.

    Each ``labels.txt`` contains a fixed protocol: one ``S1`` (start), nine
    ``S3`` (5 s cue), nine ``S4`` (3 s movement-onset), fourteen ``S5`` (rest)
    and four ``S6`` (16 s sustained imagery); each ``S<n>`` onset is paired with
    a matching ``R<n>`` offset.

    Notes
    -----
    The Zenodo record documents the ``S1``-``S6`` markers only as indicating
    "cue presentation, movement onset, and resting phases" and does not give an
    explicit code-to-class table. The mapping used here is inferred from the
    marker durations and frequencies: ``S6`` (the 16 s sustained block) is taken
    as motor imagery and ``S5`` (the variable-length relax block) as rest, with
    ``S1``/``S3``/``S4`` treated as start/cue/onset cues and not epoched.
    Electrode positions are not documented, so channels are named generically
    and no montage is attached.

    References
    ----------

    .. [1] Kment, T., & Moucek, R. (2025). Motor Imagery EEG Dataset for
        BCI-based Neurorehabilitation. Zenodo.
        DOI: https://doi.org/10.5281/zenodo.15399490

    .. versionadded:: 1.1.1
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=SFREQ,
            n_channels=8,
            channel_types={"eeg": 8},
            montage=None,
            hardware="OpenBCI Cyton (8-channel configuration)",
            reference=None,
            ground=None,
            sensors=list(CHANNELS),
        ),
        participants=ParticipantMetadata(n_subjects=8, species="homo sapiens"),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["rest", "motor_imagery"],
            study_design="Cued upper-limb motor imagery alternating with rest, "
            "synchronised with a Unity 3D environment and FourMotors robotic "
            "software for neurorehabilitation.",
            stimulus_modalities=["visual"],
            mode="offline",
            events=dict(EVENTS),
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.15399490",
            description="Motor imagery EEG dataset for BCI-based "
            "neurorehabilitation: 8 subjects, OpenBCI Cyton 8-channel, 1000 Hz, "
            "upper-limb motor imagery vs rest.",
            investigators=["Tomas Kment", "Roman Moucek"],
            institution="University of West Bohemia",
            country="CZ",
            data_url="https://doi.org/10.5281/zenodo.15399490",
            publication_year=2025,
            license="CC-BY-4.0",
            repository="Zenodo",
        ),
        sessions_per_subject=1,
        runs_per_session=3,
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
        file_format="OpenBCI text",
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 8 + 1)),
            sessions_per_subject=1,
            events=dict(EVENTS),
            code="Rehab2025OpenBCI",
            interval=(0, 4),
            paradigm="imagery",
            doi="10.5281/zenodo.15399490",
        )

    def _extracted_root(self, path=None, force_update=False):
        """Download and extract the archive, returning the ``data`` folder."""
        path_zip = Path(dl.data_dl(REHAB2025_URL, self.code, force_update=force_update))
        root = path_zip.parent
        data_dir = root / "data"
        if not data_dir.is_dir():
            with z.ZipFile(path_zip, "r") as zip_ref:
                zip_ref.extractall(root)
        return data_dir

    @staticmethod
    def _folder_index(name):
        """Leading integer index of a recording folder (e.g. ``01m...`` -> 1)."""
        digits = ""
        for ch in name:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            raise ValueError(f"Recording folder has no leading index: {name!r}")
        return int(digits)

    def _subject_folders(self, data_dir):
        """Return the 24 recording folder names ordered by their numeric index.

        Folder names are ``<index><initial><ddmmyyyy><task>`` (e.g.
        ``01m09042025e``). The recordings are grouped into subjects by their
        leading acquisition index in consecutive triples, so ordering is done on
        the parsed integer index rather than a lexicographic sort of the names
        (which would only coincide with the numeric order while the indices stay
        zero-padded).
        """
        names = [p.name for p in data_dir.iterdir() if p.is_dir()]
        folders = sorted(names, key=self._folder_index)
        return folders

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the data paths (the 3 recording folders) of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for (1-8).
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose()).

        Returns
        -------
        list of str
            The three recording-folder paths for the subject.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        data_dir = self._extracted_root(path=path, force_update=force_update)
        folders = self._subject_folders(data_dir)
        # Subject s -> the s-th consecutive triple of recording folders.
        start = (subject - 1) * 3
        subject_folders = folders[start : start + 3]
        if len(subject_folders) != 3:
            raise ValueError(
                f"Expected 3 recording folders for subject {subject}, "
                f"found {len(subject_folders)} in {data_dir}"
            )
        return [str(data_dir / f) for f in subject_folders]

    def _read_run(self, folder):
        """Build one annotated Raw from a recording folder."""
        folder = Path(folder)
        data_file = folder / "data.txt"
        labels_file = folder / "labels.txt"

        # Locate the tabular header line ("Sample Index, ...").
        header_idx = None
        with open(data_file, "r") as fh:
            for i, line in enumerate(fh):
                if line.startswith("Sample Index"):
                    header_idx = i
                    break
        if header_idx is None:
            raise ValueError(f"Malformed OpenBCI file: {data_file}")

        df = pd.read_csv(
            data_file, skiprows=header_idx + 1, header=None, skipinitialspace=True
        )

        # Columns 1-8 are the eight EXG channels (uV); last column is the
        # formatted wall-clock timestamp of each sample.
        eeg = df.iloc[:, 1:9].to_numpy(dtype=float).T * 1e-6  # uV -> V
        info = mne.create_info(ch_names=list(CHANNELS), sfreq=SFREQ, ch_types="eeg")
        raw = mne.io.RawArray(eeg, info, verbose=False)

        t0 = self._parse_data_time(str(df.iloc[0, -1]).strip())
        annotations = self._build_annotations(labels_file, t0, raw.n_times)
        raw.set_annotations(annotations)
        return raw

    @staticmethod
    def _parse_data_time(stamp):
        """Parse an OpenBCI formatted timestamp: '2025-04-09 11:07:24.365'."""
        return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S.%f")

    @staticmethod
    def _parse_label_time(stamp):
        """Parse a labels.txt timestamp: '25.04.09-11:08:17.242'."""
        return datetime.strptime(stamp, "%y.%m.%d-%H:%M:%S.%f")

    def _build_annotations(self, labels_file, t0, n_times):
        """Turn labels.txt into class annotations aligned to the recording."""
        rows = []
        with open(labels_file, "r") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2:
                    rows.append((self._parse_label_time(parts[0]), parts[1]))

        code_to_label = {v: k for k, v in EVENTS.items()}
        onsets, durations, descriptions = [], [], []
        for k, (t, marker) in enumerate(rows):
            if not (marker.startswith("S") and marker[1:].isdigit()):
                continue
            code = int(marker[1:])
            label = code_to_label.get(code)
            if label is None:
                continue

            # Duration to the matching R<code> offset marker.
            block_len = None
            for t2, m2 in rows[k + 1 :]:
                if m2 == f"R{code}":
                    block_len = (t2 - t).total_seconds()
                    break
            # Drop the instantaneous S5 boundary markers.
            if label == "rest" and (block_len is None or block_len < _MIN_REST_DURATION):
                continue

            sample = round((t - t0).total_seconds() * SFREQ)
            if 0 <= sample < n_times:
                onsets.append(sample / SFREQ)
                durations.append(0.0)
                descriptions.append(label)

        return mne.Annotations(onset=onsets, duration=durations, description=descriptions)

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for (1-8).

        Returns
        -------
        dict
            ``{"0": {"0": raw, "1": raw, "2": raw}}`` - one session, three runs.
        """
        folders = self.data_path(subject)
        runs = {}
        for run_idx, folder in enumerate(folders):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                runs[str(run_idx)] = self._read_run(folder)
        return {"0": runs}
