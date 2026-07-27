"""EEG Kinesthetic Motor Imagery force-level dataset (Martinez-Peon, 2024)."""

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


# Figshare article 25773342 hosts 60 plain-text files, one per
# (subject, force-level, attempt). File names encode the class:
# ``userNNN_<level>_<attempt>.txt`` (e.g. ``user004_70_2.txt`` = subject 4,
# 70% MVC, second attempt). The force level in the name is the data-borne
# class label. The Figshare per-file download IDs are pinned below (v1).
MARTINEZPEON2024_BASE_URL = "https://ndownloader.figshare.com/files/"

# FILE_IDS[subject][(level, attempt)] = Figshare file id (article 25773342, v1).
MARTINEZPEON2024_FILE_IDS = {
    1: {
        ("10", "1"): 46186644,
        ("10", "2"): 46186650,
        ("40", "1"): 46186647,
        ("40", "2"): 46186653,
        ("70", "1"): 46186656,
        ("70", "2"): 46186659,
    },
    2: {
        ("10", "1"): 46186662,
        ("10", "2"): 46186665,
        ("40", "1"): 46186668,
        ("40", "2"): 46186671,
        ("70", "1"): 46186674,
        ("70", "2"): 46186677,
    },
    3: {
        ("10", "1"): 46186680,
        ("10", "2"): 46186683,
        ("40", "1"): 46186686,
        ("40", "2"): 46186689,
        ("70", "1"): 46186692,
        ("70", "2"): 46186695,
    },
    4: {
        ("10", "1"): 46186698,
        ("10", "2"): 46186701,
        ("40", "1"): 46186704,
        ("40", "2"): 46186707,
        ("70", "1"): 46186710,
        ("70", "2"): 46186713,
    },
    5: {
        ("10", "1"): 46186716,
        ("10", "2"): 46186719,
        ("40", "1"): 46186722,
        ("40", "2"): 46186725,
        ("70", "1"): 46186728,
        ("70", "2"): 46186731,
    },
    6: {
        ("10", "1"): 46186734,
        ("10", "2"): 46186737,
        ("40", "1"): 46186740,
        ("40", "2"): 46186743,
        ("70", "1"): 46186746,
        ("70", "2"): 46186749,
    },
    7: {
        ("10", "1"): 46186752,
        ("10", "2"): 46186755,
        ("40", "1"): 46186758,
        ("40", "2"): 46186761,
        ("70", "1"): 46186764,
        ("70", "2"): 46186767,
    },
    8: {
        ("10", "1"): 46186770,
        ("10", "2"): 46186773,
        ("40", "1"): 46186776,
        ("40", "2"): 46186779,
        ("70", "1"): 46186782,
        ("70", "2"): 46186785,
    },
    9: {
        ("10", "1"): 46186788,
        ("10", "2"): 46186791,
        ("40", "1"): 46186794,
        ("40", "2"): 46186797,
        ("70", "1"): 46186800,
        ("70", "2"): 46186803,
    },
    10: {
        ("10", "1"): 46186806,
        ("10", "2"): 46186809,
        ("40", "1"): 46186812,
        ("40", "2"): 46186815,
        ("70", "1"): 46186818,
        ("70", "2"): 46186821,
    },
}

# Three graded kinesthetic-MI force levels (% of maximal voluntary contraction).
MARTINEZPEON2024_LEVELS = ["10", "40", "70"]
MARTINEZPEON2024_ATTEMPTS = ["1", "2"]

# Emotiv EPOC, 14 EEG channels, in file-column order (columns 3-16 of each row).
MARTINEZPEON2024_CHANNELS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]

# The 14 EEG channels occupy 0-based columns 2..15; col 0 = Time, col 1 = Sample,
# cols 16-17 = gyroscope (GX/GY), col 18 = time (s), cols 19+ = zeros.
MARTINEZPEON2024_EEG_COLS = list(range(2, 16))

MARTINEZPEON2024_SFREQ = 128.0

# Five kinesthetic-MI cues per file at fixed protocol times (s); each 5 s long.
MARTINEZPEON2024_ONSETS = [2.9, 10.9, 18.9, 26.9, 34.9]
MARTINEZPEON2024_TRIAL_DUR = 5.0


class MartinezPeon2024(BaseDataset):
    """Kinesthetic motor imagery at graded force levels [1]_.

    .. admonition:: Dataset summary

        ================  =======  =======  ==========  =================  ============  ===============  ===========
        Name                #Subj    #Chan    #Classes    #Trials / class    Trials len    Sampling rate      #Sessions
        ================  =======  =======  ==========  =================  ============  ===============  ===========
        MartinezPeon2024       10       14           3                 10             5s            128 Hz            1
        ================  =======  =======  ==========  =================  ============  ===============  ===========

    **Dataset description**

    EEG recorded while 10 healthy subjects performed kinesthetic motor imagery
    (KMI) of squeezing a ball with the right hand at three graded force levels:
    10%, 40% and 70% of their maximal voluntary contraction (MVC). KMI consists
    of imagining the somatosensory sensations of the movement rather than its
    visual appearance. The three force levels are treated here as the three
    decoding classes.

    Signals were acquired with an Emotiv EPOC headset (14 wet-saline channels
    AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4, arranged after
    the 10-10 system) at 128 Hz. Each recording lasts about 40 s and contains
    five KMI cues elicited at 2.9, 10.9, 18.9, 26.9 and 34.9 s, each of 5 s
    duration, giving five imagery trials per file. Every subject repeated each
    force level twice, so a subject contributes six recordings (three levels x
    two attempts) and 10 imagery trials per force level.

    The force level of every recording is carried by its file name
    (``userNNN_<level>_<attempt>.txt``); this is the data-borne class label. The
    five within-file cue onsets follow the fixed acquisition protocol described
    by the authors (there is no trigger/marker channel in the files) and are
    written as annotations at those times. Each recording is exposed as one run:
    session ``"0"`` holds six runs keyed ``"<level>_<attempt>"``.

    The published experiment additionally defines a fourth, "basal" (rest) class
    taken from the inter-cue rest periods; because those rest windows are not
    separately marked in the files, this loader exposes only the three graded
    force-level classes.

    Signals are provided raw (unfiltered); the stored amplitudes are in
    microvolts (Emotiv raw output, centred on a large DC offset) and are
    converted to volts on load. The gyroscope and auxiliary time columns are
    discarded.

    References
    ----------

    .. [1] Martinez-Peon, D. (2024). EEG Kinesthetic motor imagery levels.
       figshare. Dataset. DOI: https://doi.org/10.6084/m9.figshare.25773342
       Associated article: https://doi.org/10.1088/1741-2552/ad5f27

    Notes
    -----

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=128.0,
            n_channels=14,
            channel_types={"eeg": 14},
            montage="standard_1020",
            hardware="Emotiv EPOC",
            sensor_type="wet",
            electrode_type="saline",
            line_freq=60.0,
            sensors=list(MARTINEZPEON2024_CHANNELS),
        ),
        participants=ParticipantMetadata(
            n_subjects=10, health_status="healthy", species="homo sapiens"
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["level_10", "level_40", "level_70"],
            trials_per_class={"level_10": 10, "level_40": 10, "level_70": 10},
            trial_duration=5.0,
            study_design=(
                "Kinesthetic motor imagery of a right-hand ball squeeze at "
                "10/40/70% of maximal voluntary contraction. Each 40 s "
                "recording carries five KMI cues at 2.9/10.9/18.9/26.9/34.9 s "
                "(5 s each); each force level is recorded twice per subject. "
                "The force level is encoded in the file name."
            ),
            feedback_type="none",
            synchronicity="cue-based",
            mode="offline",
            events={"level_10": 1, "level_40": 2, "level_70": 3},
        ),
        documentation=DocumentationMetadata(
            doi="10.6084/m9.figshare.25773342.v1",
            description=(
                "EEG kinesthetic motor imagery at three graded hand-grip force "
                "levels (10/40/70% MVC) from 10 healthy subjects, Emotiv EPOC, "
                "14 channels, 128 Hz."
            ),
            investigators=["Dulce Martinez-Peon"],
            country="MX",
            data_url="https://doi.org/10.6084/m9.figshare.25773342",
            associated_paper_doi="10.1088/1741-2552/ad5f27",
            publication_year=2024,
            keywords=[
                "motor imagery",
                "kinesthetic motor imagery",
                "force level",
                "hand grip",
                "EEG",
                "BCI",
            ],
            license="CC-BY-4.0",
            repository="Figshare",
        ),
        sessions_per_subject=1,
        runs_per_session=6,
        tags=Tags(modality=["Motor"], type=["Motor Imagery"]),
        file_format="TXT",
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 10 + 1)),
            sessions_per_subject=1,
            events={"level_10": 1, "level_40": 2, "level_70": 3},
            code="MartinezPeon2024",
            interval=(0, 5),
            paradigm="imagery",
            doi="10.6084/m9.figshare.25773342.v1",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Download (if needed) and return the six .txt paths of one subject.

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
            The six file paths, ordered 10_1, 10_2, 40_1, 40_2, 70_1, 70_2.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        file_ids = MARTINEZPEON2024_FILE_IDS[subject]
        paths = []
        for level in MARTINEZPEON2024_LEVELS:
            for attempt in MARTINEZPEON2024_ATTEMPTS:
                url = MARTINEZPEON2024_BASE_URL + str(file_ids[(level, attempt)])
                local = dl.data_dl(url, self.code, path=path, force_update=force_update)
                if isinstance(local, (list, tuple)):
                    local = local[0]
                paths.append(str(local))
        return paths

    def _read_run(self, file_path, label):
        """Build one Raw (one force-level recording) with five KMI events."""
        # Whitespace-delimited, no header; keep only the 14 EEG columns.
        data = pd.read_csv(
            file_path, sep=r"\s+", header=None, usecols=MARTINEZPEON2024_EEG_COLS
        ).to_numpy(dtype=float)
        # (n_samples, 14) -> (14, n_samples); microvolts -> volts.
        data = data.T * 1e-6

        info = mne.create_info(
            ch_names=list(MARTINEZPEON2024_CHANNELS),
            sfreq=MARTINEZPEON2024_SFREQ,
            ch_types="eeg",
        )
        raw = mne.io.RawArray(data, info, verbose=False)
        # All 14 Emotiv EPOC channels are standard 10-10/10-20 sites and must
        # resolve to positions; assert this so a future name mismatch fails
        # loudly instead of silently dropping electrode locations.
        montage = mne.channels.make_standard_montage("standard_1020")
        montage_positions = montage.get_positions()["ch_pos"]
        unresolved = [
            ch for ch in MARTINEZPEON2024_CHANNELS if ch not in montage_positions
        ]
        assert not unresolved, (
            "standard_1020 montage lacks positions for channels: " + ", ".join(unresolved)
        )
        raw.set_montage(montage, on_missing="raise", verbose=False)

        duration = raw.n_times / MARTINEZPEON2024_SFREQ
        onsets = [o for o in MARTINEZPEON2024_ONSETS if o < duration]
        annotations = mne.Annotations(
            onset=onsets,
            duration=[MARTINEZPEON2024_TRIAL_DUR] * len(onsets),
            description=[label] * len(onsets),
        )
        raw.set_annotations(annotations, verbose=False)
        return raw

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as {session: {run: Raw}}.

        Runs are keyed ``"<index>lvl<level>rep<attempt>"`` (e.g.
        ``"0lvl10rep1"``): a unique 0-based recording index followed by a
        letters-and-digits description of the force level and attempt, as
        required by MOABB's run-name convention.
        """
        paths = self.data_path(subject)
        runs = {}
        idx = 0
        for level in MARTINEZPEON2024_LEVELS:
            label = f"level_{level}"
            for attempt in MARTINEZPEON2024_ATTEMPTS:
                run_key = f"{idx}lvl{level}rep{attempt}"
                runs[run_key] = self._read_run(Path(paths[idx]), label)
                idx += 1
        return {"0": runs}
