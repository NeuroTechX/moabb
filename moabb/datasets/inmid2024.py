"""InMID Indonesian Motor Imagery / Motor Execution dataset (Wirawan et al., 2024)."""

import warnings
import zipfile as z
from pathlib import Path
from zipfile import BadZipFile

import mne
import numpy as np
import scipy.io as sio

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


# Whole-record archive of the Mendeley Data record 10.17632/rjx76wd5v6
# (version 2). It is served through the Mendeley public-api zip endpoint, which
# 302-redirects to a freshly signed S3 URL; the per-class folders are not
# individually addressable through the public API, so the whole record (about
# 487 MB, which also bundles a 150 MB stimulus video and a few PDF/xlsx files)
# is fetched once and cached.
INMID2024_URL = "https://data.mendeley.com/public-api/zip/rjx76wd5v6/download/2"

# Top-level folder name inside the archive.
INMID2024_ROOT = "Indonesian Motor Imagery Dataset (InMID)"

# The 14 Emotiv EPOC X electrodes, in the channel order stored in each .mat
# (identical across every imagery and movement file).
INMID2024_CHANNELS = [
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

INMID2024_SFREQ = 128.0

# Activity folders. Each "Raising and Lowering ..." / "Standing and Sitting"
# folder groups both directions of one limb or postural movement into a single,
# reliably data-borne folder-level class. Inspection of the distributed .mat
# files shows the movement recordings additionally carry a data-borne
# ``labels_selfassessment`` array that splits raise (1) from lower (0), but the
# imagery recordings carry no such per-segment sublabel, so the loader exposes
# the folder-level class that is common to both modalities (see the class
# docstring Notes).
INMID2024_CLASSES = {
    "Raising and Lowering the Left-Hand": "left_hand",
    "Raising and Lowering the Right-Hand": "right_hand",
    "Standing and Sitting": "trunk",
}

# The two recording modalities, each stored in its own
# "<activity> <Modality> Dataset" folder and exposed as a MOABB session.
INMID2024_SESSIONS = {"imagery": "Imagery", "execution": "Movement"}

INMID2024_EVENTS = {"left_hand": 1, "right_hand": 2, "trunk": 3}


class InMID2024(BaseDataset):
    """Indonesian Motor Imagery Dataset (InMID) from Wirawan et al. 2024 [1]_.

    **Dataset description**

    The Indonesian Motor Imagery Dataset (InMID) was recorded from 23 healthy
    participants (12 men, 11 women) from diverse regions of Indonesia using an
    Emotiv EPOC X 14-channel wireless headset sampled at 128 Hz. The 14
    electrodes follow the international 10-20 system: AF3, F7, F3, FC5, T7, P7,
    O1, O2, P8, T8, FC6, F4, F8, AF4.

    Participants performed six activities, both as motor imagery and as motor
    execution: raising the right hand, lowering the right hand, raising the left
    hand, lowering the left hand, standing and sitting. The activities are
    distributed as one ``.mat`` file per participant inside six folders
    ("Raising and Lowering the Left-Hand", "Raising and Lowering the
    Right-Hand" and "Standing and Sitting", each in an "Imagery Dataset" and a
    "Movement Dataset" variant).

    The two modalities are exposed as two MOABB sessions: ``imagery`` (motor
    imagery) and ``execution`` (motor execution). Each imagery ``.mat`` stores
    four continuous segments (about 19 s each) in ``joined_data``; each movement
    ``.mat`` stores eight continuous segments (about 90-105 s each). Every
    segment is annotated with one event, placed at the segment onset, and
    labelled with the reliable, folder-level 3-class task: ``left_hand`` (1)
    from the left-hand folder, ``right_hand`` (2) from the right-hand folder and
    ``trunk`` (3) from the standing/sitting folder. The same class set is used
    for both sessions.

    The signals are stored in Emotiv raw units (micro-volts with a DC offset of
    roughly 4200 uV) and are rescaled to volts on load.

    Notes
    -----
    The movement (motor execution) ``.mat`` files additionally carry a
    ``labels_selfassessment`` array whose value is a perfectly fixed alternating
    pattern (``1, 0, 1, 0, ...``) across all 23 subjects and all three movement
    folders, i.e. a structural, data-borne raise (1) / lower (0, or stand / sit)
    sublabel for the eight segments. The imagery ``.mat`` files carry no such
    per-segment sublabel, so a raise/lower split is only recoverable for the
    execution session. To keep a single class set shared by both sessions the
    loader collapses each folder to one class; the folder membership is the
    data-borne label used here.

    This is a distinct recording from the same authors' MIMED dataset
    (:class:`moabb.datasets.Wirawan2024`, Mendeley 10.17632/zs25xxjkm9.3):
    InMID has 23 participants from diverse Indonesian regions, whereas MIMED has
    30 participants from the Bali region.

    References
    ----------

    .. [1] Wirawan, I. M. A., Maneetham, D., Darmawiguna, I. G. M.,
       Crisnapati, P. N., Thwe, Y., & Agustini, N. N. M. (2024). InMID: A
       dataset of EEG signal-based motor imagery in Indonesian student
       participants. Mendeley Data, V2. DOI:
       https://doi.org/10.17632/rjx76wd5v6.2

    .. versionadded:: 1.2.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=128.0,
            n_channels=14,
            channel_types={"eeg": 14},
            montage="standard_1020",
            hardware="Emotiv EPOC X",
            reference="CMS/DRL (P3/P4)",
            ground="DRL",
            sensor_type="saline felt (Ag/AgCl)",
            sensors=list(INMID2024_CHANNELS),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=23, health_status="healthy", species="homo sapiens"
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["left_hand", "right_hand", "trunk"],
            events=dict(INMID2024_EVENTS),
            stimulus_type="video",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.17632/rjx76wd5v6.2",
            related_paper_dois=["10.17632/zs25xxjkm9.3"],
            description=(
                "InMID: Indonesian Motor Imagery Dataset of six activities "
                "(raising and lowering each hand, standing and sitting) "
                "recorded as both motor imagery and motor execution from 23 "
                "subjects with an Emotiv EPOC X 14-channel headset at 128 Hz."
            ),
            investigators=[
                "I Made Agus Wirawan",
                "Dechrit Maneetham",
                "I Gede Mahendra Darmawiguna",
                "Padma Nyoman Crisnapati",
                "Yamin Thwe",
                "Ni Nyoman Mestri Agustini",
            ],
            country="ID",
            data_url="https://data.mendeley.com/datasets/rjx76wd5v6/2",
            publication_year=2024,
            license="CC-BY-4.0",
            repository="Mendeley Data",
        ),
        sessions_per_subject=2,
        runs_per_session=1,
        tags=Tags(modality=["Motor"], type=["Motor Imagery", "Motor Execution"]),
        file_format="MAT (converted from EDF)",
        abstract=(
            "The InMID dataset provides EEG recordings of motor imagery and "
            "motor execution for six activities (raising and lowering each "
            "hand, standing and sitting) from 23 Indonesian subjects, acquired "
            "with a 14-channel Emotiv EPOC X headset at 128 Hz. The two "
            "modalities are exposed as an imagery session and an execution "
            "session that share a data-borne, folder-level 3-class task "
            "(left_hand / right_hand / trunk)."
        ),
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 23 + 1)),
            sessions_per_subject=2,
            events=dict(INMID2024_EVENTS),
            code="InMID2024",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.17632/rjx76wd5v6.2",
        )

    def _extract_root(self):
        """Download the whole-record archive and return its extraction root."""
        path_zip = Path(dl.data_dl(INMID2024_URL, self.code))
        path_folder = path_zip.parent
        marker = path_folder / INMID2024_ROOT

        if not marker.is_dir():
            try:
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)
            except BadZipFile:
                warnings.warn(
                    "Corrupted zip file detected, re-downloading...", stacklevel=2
                )
                path_zip.unlink(missing_ok=True)
                path_zip = Path(dl.data_dl(INMID2024_URL, self.code, force_update=True))
                with z.ZipFile(path_zip, "r") as zip_ref:
                    zip_ref.extractall(path_folder)

        return path_folder

    def _class_file(self, root, subject, modality, activity):
        """Return the .mat path for one subject / modality / activity folder."""
        sub = f"P{subject:02d}"
        folder = f"{activity} {modality} Dataset"
        return root / INMID2024_ROOT / folder / f"{sub}.mat"

    @staticmethod
    def _trial_to_channels_by_samples(trial):
        """Return a valid InMID trial in MNE's channel-by-sample layout.

        InMID matrices are stored as ``(n_samples, 14)``.  The released
        P05 standing/sitting movement file has one corrupt square segment
        (13440 by 13440), which cannot represent the 14-channel recording.
        Returning ``None`` lets the caller retain every valid segment while
        excluding only an unusable one.
        """
        trial = np.asarray(trial)
        if trial.ndim != 2 or trial.shape[1] != len(INMID2024_CHANNELS):
            return None
        return trial.astype(np.float64, copy=False).T

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the list of class file paths for a single subject.

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
            If not None, override default verbose level (see mne.verbose()).

        Returns
        -------
        list of str
            One path per (modality, activity) file for the subject, ordered by
            session then class.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        root = self._extract_root()
        subject_paths = []
        for modality in INMID2024_SESSIONS.values():
            for activity in INMID2024_CLASSES:
                subject_paths.append(
                    str(self._class_file(root, subject, modality, activity))
                )
        return subject_paths

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            ``{"imagery": {"0": raw}, "execution": {"0": raw}}`` with all
            segments of the subject annotated with their folder-level class.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        root = self._extract_root()
        sessions = {}

        for session_idx, (session_key, modality) in enumerate(INMID2024_SESSIONS.items()):
            segments = []
            onset_samples = []
            onset_codes = []
            cursor = 0

            for activity, label in INMID2024_CLASSES.items():
                mat = sio.loadmat(
                    str(self._class_file(root, subject, modality, activity))
                )
                joined = mat["joined_data"]
                n_trials = joined.shape[1]

                for trial_idx in range(n_trials):
                    trial = self._trial_to_channels_by_samples(joined[0, trial_idx])
                    if trial is None:
                        shape = np.asarray(joined[0, trial_idx]).shape
                        warnings.warn(
                            "Skipping malformed InMID2024 segment "
                            f"(subject={subject}, modality={modality}, "
                            f"activity={activity!r}, trial={trial_idx}, "
                            f"shape={shape}); expected (n_samples, "
                            f"{len(INMID2024_CHANNELS)}).",
                            stacklevel=2,
                        )
                        continue
                    # trial is (n_channels, n_samples); one event per segment,
                    # placed at the segment onset and sharing the folder class.
                    onset_samples.append(cursor)
                    onset_codes.append(self.event_id[label])
                    segments.append(trial)
                    cursor += trial.shape[1]

            expected_codes = set(self.event_id.values())
            observed_codes = set(onset_codes)
            if observed_codes != expected_codes:
                missing = sorted(expected_codes - observed_codes)
                raise ValueError(
                    f"InMID2024 subject {subject} {modality} has no usable "
                    f"segments for event code(s) {missing}."
                )

            # Concatenate all segments into one continuous recording (volts).
            data = np.concatenate(segments, axis=1) * 1e-6

            info = mne.create_info(
                ch_names=list(INMID2024_CHANNELS), sfreq=INMID2024_SFREQ, ch_types="eeg"
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne.io.RawArray(data=data, info=info, verbose=False)
                raw.set_montage("standard_1020", on_missing="ignore", verbose=False)

            events = np.column_stack(
                (
                    np.asarray(onset_samples, dtype=int),
                    np.zeros(len(onset_samples), dtype=int),
                    np.asarray(onset_codes, dtype=int),
                )
            )
            event_desc = {code: name for name, code in self.event_id.items()}
            annotations = mne.annotations_from_events(
                events, sfreq=raw.info["sfreq"], event_desc=event_desc, verbose=False
            )
            raw.set_annotations(annotations)

            # MOABB requires the session key to start with an integer index
            # (optionally followed by a letters+digits description).
            sessions[f"{session_idx}{session_key}"] = {"0": raw}

        return sessions
