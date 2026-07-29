"""MIND multimodal fNIRS-EEG four-class directional motor imagery dataset.

Feng, Xu, Zhang, Lin, Deng, Tao, Liu, Jia, Duan, and Jia (2026).
*MIND: A Multimodal fNIRS-EEG Dataset for Unilateral Limb Motor Imagery.*
Preprint: arXiv 2602.04299.
Data DOI: 10.57760/sciencedb.34326 (ScienceDB).
"""

import csv
import logging
import re
from pathlib import Path

import mne

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


log = logging.getLogger(__name__)

# ScienceDB exposes a direct archive download via getZipFile (no login and no
# per-session traceId required). The whole dataset is a single ~74 GB zip
# (EEG + fNIRS + raw), so the fetch is heavy but fully automatic.
_SCIDB_ID = "d0f67f38c5bc4416bdcce619f33e57da"
MIND2026_DOI = "10.57760/sciencedb.34326"
MIND2026_URL = f"https://www.scidb.cn/detail?dataSetId={_SCIDB_ID}"
MIND2026_DOWNLOAD_URL = (
    f"https://china.scidb.cn/getZipFile?dataSetId={_SCIDB_ID}&version=V3"
)

# BIDS task label of the EEG BrainVision recordings (sub-NN/eeg/*_task-mi2d_*).
_TASK = "mi2d"

# The four directional right-upper-limb MI tasks. Trigger ``value`` codes 4-7 in
# the BIDS ``*_events.tsv`` sidecars (mi_l2r/mi_t2b/mi_ul2lr/mi_ur2ll); codes 1
# and 2 are eyes-open / eyes-closed rest and code 3 is the pre-imagery cue.
_EVENTS = {
    "left_to_right": 4,
    "up_to_down": 5,
    "upperleft_to_lowerright": 6,
    "upperright_to_lowerleft": 7,
}


class MIND2026(BaseDataset):
    """Multimodal fNIRS-EEG directional motor imagery dataset from Feng et al 2026.

    Dataset from *MIND: A Multimodal fNIRS-EEG Dataset for Unilateral Limb
    Motor Imagery* [1]_. This loader exposes the **EEG** modality only.

    MIND is built on a four-class directional motor imagery paradigm of the
    right upper limb. 64-channel EEG was recorded at 1000 Hz with a Neuroscan
    SynAmps system and an international 10-20 Ag/AgCl cap, simultaneously with
    51-channel fNIRS (47.62 Hz, not loaded here). 30 healthy participants
    (12 female, 18 male; aged 19-25 years) took part.

    Each participant completed three consecutive blocks (treated here as three
    runs of a single session). Every block lasted ~11 min and contained a
    60 s eyes-open rest, a 60 s eyes-closed rest, and 20 MI trials. Each MI
    trial lasted 27 s: a 2 s synchronous visual+auditory cue at 0 s, a 10 s
    motor-imagery execution period, and a 10 s post-imagery rest. Participants
    performed 120 MI trials in total (30 per class), for 3,600 EEG/fNIRS trials
    across the cohort.

    The four directional MI conditions (raw trigger codes) are:

    - **left_to_right** (code 4)
    - **up_to_down** (code 5)
    - **upperleft_to_lowerright** (code 6)
    - **upperright_to_lowerleft** (code 7)

    Codes 1 (eyes-open rest) and 2 (eyes-closed rest) are ignored.

    In the BIDS events sidecar, code 3 marks the 2 s pre-imagery cue and codes
    4-7 mark the subsequent motor-imagery execution onset. The imagery
    ``interval`` is therefore ``[0, 10]`` s relative to the code 4-7 event,
    covering exactly the 10 s execution period.

    .. note::

       ScienceDB serves the whole dataset as a single ~74 GB archive through a
       public ``getZipFile`` endpoint (no login and no per-session token), so
       :meth:`data_path` downloads and extracts it automatically on first use.
       The download is large and one-time. The archive ships the EEG in BIDS
       BrainVision form under ``MIND_BIDS/sub-NN/eeg/`` (a
       ``.vhdr``/``.vmrk``/``.eeg`` triple plus an ``*_events.tsv`` sidecar per
       run); this loader reads those with :func:`mne.io.read_raw_brainvision`
       and takes the per-trial MI class from the ``value`` column of the
       sidecar. Only the per-subject EEG BrainVision files are extracted (not
       the co-shipped raw Curry ``sourcedata`` or the fNIRS recordings). The raw
       Curry export is not used because its scidb V3 copy omits the ``.rs3``
       sensor-position files that ``read_raw_curry`` requires.

    References
    ----------
    .. [1] Feng, L., Xu, B., Zhang, H., Lin, B., Deng, Z., Tao, S., Liu, C.,
           Jia, S., Duan, L., & Jia, Z. (2026). MIND: A Multimodal fNIRS-EEG
           Dataset for Unilateral Limb Motor Imagery. arXiv:2602.04299.
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="10-20",
            hardware="Neuroscan SynAmps (64-channel Ag/AgCl, 10-20)",
            sensor_type="Ag/AgCl",
            electrode_type="wet",
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=30,
            health_status="healthy",
            gender={"female": 12, "male": 18},
            age_min=19.0,
            age_max=25.0,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=4,
            class_labels=list(_EVENTS.keys()),
            trials_per_class=dict.fromkeys(_EVENTS, 30),
            trial_duration=27.0,
            study_design=(
                "Four-class directional motor imagery of the right upper limb "
                "with simultaneous EEG and fNIRS recording. Three blocks per "
                "subject; each block: 60 s eyes-open rest, 60 s eyes-closed "
                "rest, 20 MI trials. Each 27 s trial: 2 s synchronous "
                "visual+auditory cue, 10 s MI execution, 10 s post-imagery rest."
            ),
            feedback_type="none",
            stimulus_type="visual instruction animation and auditory cue",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            instructions=(
                "Participants continuously imagined the cued directional upper-"
                "limb movement, mentally repeating it 2-4 times over the 10 s "
                "imagery interval."
            ),
        ),
        documentation=DocumentationMetadata(
            doi=MIND2026_DOI,
            description=(
                "Multimodal fNIRS-EEG dataset for four-class directional "
                "unilateral (right upper-limb) motor imagery from 30 healthy "
                "adults."
            ),
            investigators=[
                "Lufeng Feng",
                "Baomin Xu",
                "Haoran Zhang",
                "Bihai Lin",
                "Zuxuan Deng",
                "Sidi Tao",
                "Chenyu Liu",
                "Shifan Jia",
                "Li Duan",
                "Ziyu Jia",
            ],
            country="CN",
            repository="ScienceDB",
            data_url=MIND2026_URL,
            publication_year=2026,
            license="CC-BY-4.0",
            keywords=[
                "motor imagery",
                "directional motor imagery",
                "fNIRS",
                "EEG",
                "multimodal",
                "hybrid BCI",
                "brain-computer interface",
                "upper limb",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=3,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            notes=(
                "This loader reads the raw 1000 Hz EEG recordings from the "
                "BrainVision (BIDS) copies of the scidb export (the co-shipped "
                "Curry export omits the .rs3 sensor positions). The "
                "repository also ships a preprocessed derivative (250 Hz, "
                "MATLAB .mat epochs) and preprocessed fNIRS (HbO/HbR CSV) which "
                "are not exposed here."
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=None,
            feature_extraction=None,
            frequency_bands=None,
            spatial_filters=None,
        ),
        cross_validation=CrossValidationMetadata(evaluation_type=["within_subject"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_EVENTS.keys()),
            cue_duration_s=2.0,
            imagery_duration_s=10.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=120,
            n_trials_per_class=dict.fromkeys(_EVENTS, 30),
            trials_context=(
                "120 MI trials per subject (30 per class) split over three "
                "blocks/runs of 20 MI trials each; 3,600 trials across 30 "
                "subjects."
            ),
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control", "rehabilitation"],
            environment="laboratory",
            online_feedback=False,
        ),
        data_processed=False,
        file_format="BrainVision",
    )

    def __init__(self, subjects=None, sessions=None):
        # This loader is EEG-only: it reads just the Curry EEG blocks and never
        # touches the separate fNIRS modality. A ``return_all_modalities`` flag
        # would only pick channels within an already-loaded recording, so it
        # could not surface fNIRS here; it is therefore not exposed.
        super().__init__(
            subjects=list(range(1, 31)),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="MIND2026",
            interval=[0, 10],
            paradigm="imagery",
            doi=MIND2026_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return raw EEG data for a single subject as {session: {run: Raw}}."""
        vhdr_files = self._subject_vhdrs(subject)
        if not vhdr_files:
            raise FileNotFoundError(
                f"No BrainVision .vhdr files found for subject {subject}. "
                f"MIND is hosted on ScienceDB (DOI {MIND2026_DOI}); download it "
                "so the per-subject MIND_BIDS/sub-NN/eeg/*_eeg.vhdr recordings "
                "are present under MNE-mind2026-data/."
            )

        # Single session; the three consecutive blocks are the runs.
        runs = {}
        for run_idx, vhdr in enumerate(vhdr_files):
            raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)
            raw.set_annotations(self._mi_annotations(vhdr, raw))
            runs[str(run_idx)] = raw

        return {"0": runs}

    @staticmethod
    def _mi_annotations(vhdr, raw):
        """Directional-MI annotations for one run, from the events.tsv sidecar.

        The four MI conditions carry trigger ``value`` codes 4-7 in the BIDS
        ``*_events.tsv``. These events already mark motor-imagery execution
        onset, approximately 2 s after the separate pre-imagery cue (code 3);
        the rest/cue codes (1, 2, 3) are dropped. Onsets are in seconds from
        recording start, matching the BrainVision time axis. Falls back to the
        ``read_raw_brainvision`` marker annotations if the sidecar is missing,
        recovering the code from the marker description (e.g. ``"Comment/4"``).
        Descriptions are the :data:`_EVENTS` class labels, so the four
        directional codes map to the same classes the loader intends.
        """
        event_desc = {code: label for label, code in _EVENTS.items()}
        events_path = Path(str(vhdr).replace("_eeg.vhdr", "_events.tsv"))

        onsets, descs = [], []
        if events_path.exists():
            with open(events_path, newline="") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    try:
                        code = int(float(row["value"]))
                        onset = float(row["onset"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if code in event_desc:
                        onsets.append(onset)
                        descs.append(event_desc[code])

        if not onsets:
            # Recover the codes from the BrainVision marker annotations, whose
            # descriptions end in the numeric code (e.g. "Comment/4").
            for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
                m = re.search(r"(\d+)\s*$", str(desc))
                if m is not None and int(m.group(1)) in event_desc:
                    onsets.append(float(onset))
                    descs.append(event_desc[int(m.group(1))])

        if not onsets:
            raise RuntimeError(
                f"MIND2026: no MI trigger codes {sorted(_EVENTS.values())} found "
                f"for {Path(vhdr).name}; checked the sidecar {events_path.name} and "
                "the BrainVision markers."
            )

        return mne.Annotations(
            onset=onsets, duration=[0.0] * len(onsets), description=descs
        )

    def _subject_vhdrs(self, subject):
        """Sorted BrainVision ``.vhdr`` recordings for one subject (by run)."""
        return self._find_subject_vhdrs(Path(self.data_path(subject)[0]), subject)

    @staticmethod
    def _find_subject_vhdrs(data_dir, subject):
        subj = f"sub-{subject:02d}"
        pattern = f"{subj}_task-{_TASK}_run-*_eeg.vhdr"
        eeg_dir = Path(data_dir) / "MIND_BIDS" / subj / "eeg"
        if eeg_dir.is_dir():
            found = sorted(eeg_dir.glob(pattern))
            if found:
                return found
        # Fallback: search the whole data dir, but never the Curry sourcedata.
        return sorted(
            p for p in Path(data_dir).rglob(pattern) if "sourcedata" not in p.parts
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        sign = self.code
        data_dir = Path(dl.get_dataset_path(sign, path)) / f"MNE-{sign.lower()}-data"

        # Reuse the per-subject BrainVision recordings if already extracted.
        if self._find_subject_vhdrs(data_dir, subject):
            return [str(data_dir)]

        # Otherwise download the single ~74 GB ScienceDB archive (all subjects)
        # once and extract only the per-subject EEG BrainVision BIDS files
        # (.vhdr/.vmrk/.eeg + *_events.tsv); the raw Curry ``sourcedata`` and the
        # fNIRS recordings are intentionally skipped.
        import zipfile

        data_dir.mkdir(parents=True, exist_ok=True)
        archive = dl.data_dl(
            MIND2026_DOWNLOAD_URL, sign, path=path, force_update=force_update
        )
        with zipfile.ZipFile(archive) as zf:
            members = [
                n
                for n in zf.namelist()
                if "/eeg/" in n
                and "/sourcedata/" not in n
                and n.rsplit("/", 1)[-1].startswith("sub-")
                and (
                    n.endswith("_eeg.vhdr")
                    or n.endswith("_eeg.vmrk")
                    or n.endswith("_eeg.eeg")
                    or n.endswith("_events.tsv")
                )
            ]
            zf.extractall(data_dir, members=members)
        return [str(data_dir)]
