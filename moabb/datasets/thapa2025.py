"""Freewill reach-and-grasp motor-execution EEG dataset (Thapa et al. 2025).

Thapa, Boggess and Bae (2025), Scientific Data.
Article DOI: 10.1038/s41597-025-06039-9
Data: Figshare article 28632599 (single BIDS archive, CC BY 4.0).
"""

import csv
import re
import tempfile
import warnings
import zipfile
from pathlib import Path

import mne
from mne_bids import BIDSPath, get_entity_vals

from moabb.datasets import download as dl
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)

from .base import BaseDataset


# Single 13.6 GB BIDS archive on Figshare (article 28632599, file id 57518986).
THAPA2025_URL = "https://ndownloader.figshare.com/files/57518986"

# BIDS task label.
_TASK = "reachingandgrasping"

# Four target cups, freely chosen by the subject (own timing + target).
_EVENTS = {"Tgt1": 1, "Tgt2": 2, "Tgt3": 3, "Tgt4": 4}

# 31 scalp EEG channels, extended 10-20, Cz reference.
# fmt: off
_EEG_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FT9", "FC5", "FC1",
    "FC2", "FC6", "FT10", "T7", "C3", "C4", "T8", "TP9", "CP5", "CP1",
    "CP2", "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8", "O1", "Oz", "O2",
]
# fmt: on

# Non-EEG auxiliary channels to drop when building the EEG view:
# 4 EOG, 1 audio-cue trigger, 3-axis accelerometer.
_EOG_CHANNELS = ["EOGL", "EOGR", "EOGU", "EOGD"]
_AUX_DROP = [
    "TRIG",
    "ACCX",
    "ACCY",
    "ACCZ",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "X",
    "Y",
    "Z",
]


def _target_from_description(desc):
    """Map a raw annotation description to a Tgt{n} label, or None.

    The BIDS ``events.tsv`` stores the selected target per trial; the exact
    ``trial_type`` string is not documented, so accept the common variants
    ("Tgt1", "target1", "T1", or a bare "1") and normalise to ``Tgt{n}``.
    """
    text = str(desc).strip()
    m = re.search(r"(?:tgt|target|t)\s*_?\s*([1-4])", text, flags=re.IGNORECASE)
    if m is None:
        m = re.fullmatch(r"([1-4])", text)
    if m is None:
        return None
    return f"Tgt{m.group(1)}"


class Thapa2025(BaseDataset):
    """Freewill reach-and-grasp motor-execution dataset from Thapa et al. 2025.

    Dataset [1]_ from the study *A large electroencephalogram database of
    freewill reaching and grasping tasks for brain machine interfaces*.

    EEG from 23 healthy young adults (ages 18-24; 8 female, 15 male,
    right-handed) performing a self-paced, self-selected reach-and-grasp
    task. On each trial the subject freely chose one of four cups and, at
    a moment of their own choosing (~1-2 s after an audio start cue),
    reached out and grasped it. The four target cups define the four
    classes:

    - **Tgt1** / **Tgt3**: water-filled cups
    - **Tgt2** / **Tgt4**: empty cups

    Data were recorded with a 31-channel extended 10-20 montage
    (Cz reference, separate ground) plus 4 EOG channels, a single audio-cue
    trigger channel, and a 3-axis accelerometer at the distal ulna to mark
    movement onset. The sampling rate is 250 Hz for 21 subjects but 1000 Hz
    for sub-13 and sub-15, so it is read per subject from the BrainVision
    header rather than assumed.

    The data are distributed in BIDS (BrainVision ``.vhdr``/``.vmrk``/``.eeg``)
    as a single Figshare archive. Each subject has 2-3 sessions recorded on
    different days, each with 3-7 runs of ~30 trials; 49 sessions and 6808
    trials in total. MOABB session and run keys are the sorted BIDS
    ``ses-*`` / ``run-*`` entities remapped to integer-prefixed strings.

    Notes on events: the ``.vmrk`` marker files contain only a "New Segment"
    marker; the per-trial target selection lives in the BIDS ``events.tsv``
    sidecar, from which ``read_raw_bids`` builds the annotations. The exact
    ``trial_type`` label strings for the four cups are not documented, so the
    loader normalises common variants to ``Tgt1``..``Tgt4``.

    References
    ----------
    .. [1] Thapa, B. R., Boggess, J., & Bae, J. (2025). A large
           electroencephalogram database of freewill reaching and grasping
           tasks for brain machine interfaces. Scientific Data, 12(1).
           https://doi.org/10.1038/s41597-025-06039-9
    """

    # NEMAR exempt (Figshare-hosted, not deposited on NEMAR/OpenNeuro).
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=31,
            channel_types={"eeg": 31, "eog": 4, "misc": 4},
            montage="standard_1020",
            reference="Cz",
            ground="GND",
            sensors=list(_EEG_CHANNELS),
            sensor_type="scalp EEG",
            filters="none",
            line_freq=60.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=4,
                eog_type=["horizontal", "vertical"],
                other_physiological=["accelerometer", "audio_cue"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=23,
            health_status="healthy",
            gender={"female": 8, "male": 15},
            age_min=18.0,
            age_max=24.0,
            handedness="right",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=4,
            class_labels=list(_EVENTS.keys()),
            trial_duration=12.0,
            study_design=(
                "Self-paced freewill reach-and-grasp: the subject freely "
                "chooses one of four cups and the timing of movement. Four "
                "cups (two water-filled, two empty) define the four classes."
            ),
            feedback_type="none",
            stimulus_type="audio start/end cue",
            stimulus_modalities=["auditory"],
            primary_modality="auditory",
            synchronicity="self-paced",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-025-06039-9",
            description=(
                "A large EEG database of freewill reaching and grasping tasks "
                "for brain-machine interfaces: 23 subjects, 49 sessions, 6808 "
                "self-paced reach-and-grasp trials toward one of four cups."
            ),
            investigators=["Bhoj Raj Thapa", "John Boggess", "Jihye Bae"],
            senior_author="Jihye Bae",
            institution="University of Kentucky",
            institution_department="Department of Electrical and Computer Engineering",
            country="US",
            repository="Figshare",
            data_url="https://doi.org/10.6084/m9.figshare.28632599",
            publication_year=2025,
            license="CC-BY-4.0",
            keywords=[
                "EEG",
                "reach and grasp",
                "motor execution",
                "brain-machine interface",
                "freewill",
            ],
        ),
        sessions_per_subject=3,
        runs_per_session=5,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Research"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery", n_targets=4
        ),
        data_structure=DataStructureMetadata(
            n_trials=6808, trials_context="6808 trials across 23 subjects and 49 sessions"
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["within_subject", "cross_session"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control"], environment="laboratory"
        ),
        file_format="BrainVision (BIDS)",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 24)),
            sessions_per_subject=3,
            events=dict(_EVENTS),
            code="Thapa2025",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1038/s41597-025-06039-9",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _bids_root(self, path=None):
        """Download (once) and extract the Figshare archive; return BIDS root."""
        base = Path(dl.get_dataset_path(self.code, path)) / f"MNE-{self.code}-data"
        base.mkdir(parents=True, exist_ok=True)

        # Reuse an existing extraction if a BIDS root is already present.
        root = self._find_bids_root(base)
        if root is not None:
            return root

        zip_path = Path(dl.data_dl(THAPA2025_URL, self.code, path=path))
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(base)

        root = self._find_bids_root(base)
        if root is None:
            raise FileNotFoundError(
                "Could not locate dataset_description.json in the extracted "
                f"Thapa2025 archive under {base}."
            )
        return root

    @staticmethod
    def _find_bids_root(base):
        """Return the directory holding dataset_description.json, or None."""
        base = Path(base)
        if (base / "dataset_description.json").exists():
            return base
        for dd in base.rglob("dataset_description.json"):
            return dd.parent
        return None

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the BIDSPath objects for one subject's runs."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        root = self._bids_root(path)
        subj = f"{subject:02d}"

        # get_entity_vals() has no positive session= filter kwarg (only
        # ignore_sessions); take the dataset-wide union of session/run labels
        # and keep only the combinations that actually exist for this subject.
        #
        # Subjects have different numbers of sessions (e.g. sub-01 has ses-01
        # and ses-02 but no ses-03, sub-02 has only ses-01), so a session from
        # the dataset-wide union may be absent for this subject. Resolving a
        # BIDSPath into a missing session directory makes mne_bids os.scandir a
        # non-existent path and raise FileNotFoundError, so skip any session
        # whose directory is absent before scanning, and guard the per-run
        # existence check defensively.
        sessions = get_entity_vals(root, "session") or [None]
        all_runs = get_entity_vals(root, "run") or [None]
        bids_paths = []
        for ses in sessions:
            if (
                ses is not None
                and not (Path(root) / f"sub-{subj}" / f"ses-{ses}").is_dir()
            ):
                continue
            for run in all_runs:
                bids_path = BIDSPath(
                    subject=subj,
                    session=ses,
                    task=_TASK,
                    run=run,
                    suffix="eeg",
                    root=root,
                    check=True,
                )
                try:
                    exists = bids_path.fpath.exists()
                except (FileNotFoundError, OSError):
                    exists = False
                if exists:
                    bids_paths.append(bids_path)
        return bids_paths

    @staticmethod
    def _read_brainvision(vhdr_path):
        """Read a BIDS BrainVision run, repairing stale sibling references.

        Three files in the published archive retain an acquisition-time
        ``DataFile`` or ``MarkerFile`` name although the BIDS conversion
        renamed the adjacent ``.eeg``/``.vmrk`` file.  Repair only a missing
        reference for which the same-stem BIDS sibling exists, in a temporary
        header, leaving the downloaded archive unchanged.
        """
        vhdr_path = Path(vhdr_path)
        header = vhdr_path.read_text(encoding="utf-8")
        repaired = header
        for field, suffix in (("DataFile", ".eeg"), ("MarkerFile", ".vmrk")):
            match = re.search(rf"(?m)^{field}=(.+)$", repaired)
            if match is None:
                raise ValueError(f"Missing {field} entry in BrainVision header {vhdr_path}")
            referenced = vhdr_path.parent / match.group(1).strip()
            if referenced.exists():
                continue
            sibling = vhdr_path.with_suffix(suffix)
            if not sibling.exists():
                raise FileNotFoundError(
                    f"{vhdr_path} references missing {referenced.name}; expected BIDS "
                    f"sibling {sibling.name} is also absent."
                )
            repaired = re.sub(
                rf"(?m)^{field}=.+$", f"{field}={sibling.name}", repaired
            )

        temporary = None
        try:
            path = vhdr_path
            if repaired != header:
                with tempfile.NamedTemporaryFile(
                    "w",
                    suffix=".vhdr",
                    dir=vhdr_path.parent,
                    encoding="utf-8",
                    delete=False,
                ) as fout:
                    fout.write(repaired)
                    temporary = Path(fout.name)
                path = temporary
            return mne.io.read_raw_brainvision(path, preload=True, verbose=False)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _annotations_from_events(events_path):
        """Read BIDS events without assuming rectangular optional columns.

        The source's sub-13/run-0001 events.tsv has one target row without an
        empty ``stim_file`` field and the following row has an extra trailing
        tab. ``mne_bids`` delegates this to ``numpy.loadtxt`` and rejects the
        whole run. ``csv.DictReader`` preserves the required onset and target
        fields in both rows, so no protocol event needs to be discarded.
        """
        events_path = Path(events_path)
        onsets, durations, descriptions = [], [], []
        with events_path.open(encoding="utf-8", newline="") as fin:
            reader = csv.DictReader(fin, delimiter="\t")
            required = {"onset", "duration", "trial_type"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"Thapa2025 events file {events_path} lacks required columns "
                    f"{sorted(required)}; found {reader.fieldnames}."
                )
            for line_number, row in enumerate(reader, start=2):
                onset = (row.get("onset") or "").strip()
                trial_type = (row.get("trial_type") or "").strip()
                if not onset or not trial_type:
                    raise ValueError(
                        f"Thapa2025 events file {events_path} has no onset or "
                        f"trial_type at line {line_number}."
                    )
                try:
                    onset_value = float(onset)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid onset {onset!r} in {events_path} line {line_number}."
                    ) from exc
                duration = (row.get("duration") or "").strip()
                if duration.lower() in {"", "n/a", "na"}:
                    duration_value = 0.0
                else:
                    try:
                        duration_value = float(duration)
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid duration {duration!r} in {events_path} "
                            f"line {line_number}."
                        ) from exc
                onsets.append(onset_value)
                durations.append(duration_value)
                descriptions.append(_target_from_description(trial_type) or trial_type)
        return mne.Annotations(onsets, durations, descriptions)

    def _get_single_subject_data(self, subject):
        """Return {session: {run: Raw}} for one subject."""
        bids_paths = self.data_path(subject)

        # Group by BIDS session; remap ses/run entities to integer-prefixed keys.
        ses_ids = sorted({bp.session for bp in bids_paths}, key=lambda s: (s is None, s))
        ses_key = {s: str(i) for i, s in enumerate(ses_ids)}

        sessions = {}
        run_counter = {}
        for bids_path in bids_paths:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = self._read_brainvision(bids_path.fpath)

            # Type the auxiliary channels and (by default) keep EEG only.
            type_map = {ch: "eog" for ch in _EOG_CHANNELS if ch in raw.ch_names}
            for ch in _AUX_DROP:
                if ch in raw.ch_names:
                    type_map[ch] = "misc"
            if type_map:
                raw.set_channel_types(type_map)

            events_path = bids_path.copy().update(suffix="events", extension=".tsv")
            raw.set_annotations(self._annotations_from_events(events_path.fpath))

            if not self.return_all_modalities:
                raw.pick("eeg")

            skey = ses_key[bids_path.session]
            rkey = str(run_counter.get(skey, 0))
            run_counter[skey] = run_counter.get(skey, 0) + 1
            sessions.setdefault(skey, {})[rkey] = raw

        return sessions
