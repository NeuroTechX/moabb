"""PardoGarcia2026 mu/beta motor-imagery EEG dataset in chronic MCA stroke."""

import warnings
from pathlib import Path

import mne
from mne.channels import make_standard_montage

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


# Zenodo record 19599466 (version of concept record 19599465). Each recording is
# stored as three BrainVision files (.eeg/.vhdr/.vmrk) served individually. The
# plain "/records/<id>/files/<name>" endpoint returns the raw bytes and yields a
# distinct local filename per file (unlike "/content", which collides on the name
# "content"); data_dl places every file of a record in the same folder, so the
# .vhdr can resolve its sibling .eeg/.vmrk by their bare names.
PARDOGARCIA2026_BASE_URL = "https://zenodo.org/records/19599466/files"

# Explicit, data-borne subject -> {session_label: recording_stem} mapping, read
# directly from the record file listing. Chronic MCA stroke patients are stored
# as "PAC{n:02d}" (baseline, session "pre") and "PAC{n:02d}-POST" (after
# rehabilitation, session "post"). Patient 2 (PAC02) has no post recording (the
# participant was lost to follow-up, per the record note "nota post 02 y post
# 03.txt"). Healthy controls were recorded once and are stored under inconsistent
# stems ("C01", "02", "C03", "CONTROL04".."CONTROL08").
PARDOGARCIA2026_SUBJECTS = {
    1: {"pre": "PAC01", "post": "PAC01-POST"},
    2: {"pre": "PAC02"},
    3: {"pre": "PAC03", "post": "PAC03-POST"},
    4: {"pre": "PAC04", "post": "PAC04-POST"},
    5: {"pre": "PAC05", "post": "PAC05-POST"},
    6: {"pre": "PAC06", "post": "PAC06-POST"},
    7: {"pre": "PAC07", "post": "PAC07-POST"},
    8: {"pre": "PAC08", "post": "PAC08-POST"},
    9: {"pre": "PAC09", "post": "PAC09-POST"},
    10: {"pre": "PAC10", "post": "PAC10-POST"},
    11: {"pre": "C01"},
    12: {"pre": "02"},
    13: {"pre": "C03"},
    14: {"pre": "CONTROL04"},
    15: {"pre": "CONTROL05"},
    16: {"pre": "CONTROL06"},
    17: {"pre": "CONTROL07"},
    18: {"pre": "CONTROL08"},
}

# BrainVision annotation descriptions (as produced by mne.io.read_raw_brainvision)
# for the two image-onset cue markers, mapped to their class labels. The stimulus
# codes 1 and 2 index the cue image shown to the participant: 1 = a precision
# "pinza" (pinch) grip, 2 = a "puno cerrado" (closed fist). This code -> grip
# mapping is documented in the record file "bdf_IMAGEN.txt" ("Imagen, aparece una
# imagen de una pinza o un puno cerrado. {1; 2}"). Each trial also carries a
# second marker (S 11 / S 22) at roughly 2 s for the auditory go cue that starts
# overt execution of the same class. Those markers are left unmapped so the
# adapter exposes only one image-cue event per trial.
PARDOGARCIA2026_EVENT_RENAME = {"Stimulus/S  1": "pinch", "Stimulus/S  2": "fist"}

# Bipolar EOG channels present alongside the scalp EEG montage.
PARDOGARCIA2026_EOG = ("HEOGn", "HEOGp", "VEOGn", "VEOGp")

# 63 recording channels in acquisition order (59 EEG incl. the A1 mastoid, plus
# 4 EOG), taken verbatim from the BrainVision headers.
# fmt: off
PARDOGARCIA2026_CHANNELS = [
    "O2", "OZ", "O1", "PO8", "PO6", "PO4", "POZ", "PO3", "PO5", "PO7", "P8", "P6",
    "P4", "P2", "PZ", "P1", "P3", "P5", "P7", "TP8", "CP6", "CP4", "CP2", "CPZ",
    "CP1", "CP3", "CP5", "TP7", "HEOGn", "HEOGp", "VEOGn", "VEOGp", "FP1", "FPZ",
    "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FC5",
    "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "T7", "C5", "C3", "C1", "CZ", "C2",
    "C4", "C6", "T8", "A1",
]
# fmt: on


class PardoGarcia2026(BaseDataset):
    """Mu/beta motor-imagery EEG in chronic MCA stroke and healthy controls [1]_.

    **Dataset description**

    EEG recorded during a cued two-class hand motor-imagery task used to study
    mu (8-12 Hz) and beta (12-30 Hz) changes in 10 chronic middle cerebral
    artery (MCA) stroke patients (``PAC01``-``PAC10``) and 8 healthy controls
    (``C01``, ``02``, ``C03``, ``CONTROL04``-``CONTROL08``), with a 63-channel
    BrainVision system at 1000 Hz. Each trial shows an image of the grip to
    imagine, a precision pinch (``pinch``, code 1) or a closed fist (``fist``,
    code 2), as documented in the record file ``bdf_IMAGEN.txt``.

    Patients were recorded at baseline (session ``pre``) and after
    rehabilitation (session ``post``), except patient 2 (lost to follow-up);
    controls have ``pre`` only. Each session is one continuous run. The four
    bipolar EOG channels are typed ``eog`` and standard 10-05 template
    positions are attached. An auditory go-cue marker (``S 11``/``S 22``,
    earliest at 1.510 s) starts overt execution, so this loader maps only the
    image cues (``S 1``/``S 2``) and exposes 0-1.5 s, keeping one event per
    trial and excluding overt movement. Control recordings hold 50 trials per
    class, patient recordings about 70.

    Paper-vs-release note: the record description reports "7 right-handed
    healthy controls" and "a 64-channel system (10-20 international system)",
    while the released files hold eight control recordings (``C01``, ``02``,
    ``C03``, ``CONTROL04``-``CONTROL08``) and 63 channels in their BrainVision
    headers (59 EEG incl. ``A1`` + 4 EOG). The companion preprint on the same
    cohort describes "a 64-channel Ag/AgCl electrode cap (Electro-Cap
    International), following the international 10-20 system, with A2 as
    reference" and four ocular electrodes, sampled at 1000 Hz; the record's
    "140 trials per subject" matches the patient recordings. Recordings were
    made at the Instituto Pluridisciplinar, Universidad Complutense de Madrid.

    References
    ----------

    .. [1] Pardo-Garcia, R., Ruiz-Izquierdo, M., Garcia de la Vega, M., Calvillo,
       R., Kontaxakis, G., Moreno, E. M., and Pozo, M. A. (2026). Mu and Beta
       Oscillatory Changes during a motor task following Rehabilitation in Chronic
       MCA Stroke: Insights from EEG. Zenodo.
       DOI: https://doi.org/10.5281/zenodo.19599465

    Notes
    -----

    .. versionadded:: 1.8.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            channel_types={"eeg": 59, "eog": 4},
            montage="10-10",
            hardware="BrainVision (Brain Products GmbH)",
            cap_manufacturer="Electro-Cap International",
            sensor_type="Ag/AgCl",
            reference="A2 (right mastoid)",
            ground=None,
            sensors=list(PARDOGARCIA2026_CHANNELS),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=18,
            health_status="mixed",
            clinical_population="chronic middle cerebral artery (MCA) stroke",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=2,
            class_labels=["pinch", "fist"],
            trial_duration=1.5,
            study_design=(
                "Image-cued two-class hand motor imagery and preparation (precision "
                "pinch vs closed fist) before a later auditory go cue and overt "
                "execution. The adapter exposes only the pre-execution phase."
            ),
            feedback_type="none",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="offline",
            events={"pinch": 1, "fist": 2},
            instructions=(
                "Imagine and prepare the hand grip shown in the cue image (a "
                "precision pinch or a closed fist); overt execution begins only "
                "after the later auditory go cue, outside the exposed interval."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.19599465",
            description=(
                "Pre-execution, two-class hand motor-imagery EEG (pinch vs fist) "
                "from recordings that continue into cued overt execution: 10 "
                "chronic MCA stroke patients (baseline and post-rehabilitation) "
                "and 8 healthy controls, 63 BrainVision channels at 1000 Hz."
            ),
            investigators=[
                "Rebeca Pardo-Garcia",
                "Maria Ruiz-Izquierdo",
                "Mercedes Garcia de la Vega",
                "Rocio Calvillo",
                "George Kontaxakis",
                "Eva M. Moreno",
                "M. A. Pozo",
            ],
            institution=(
                "Instituto Pluridisciplinar, Universidad Complutense de Madrid; "
                "Universidad Politecnica de Madrid; Hospital Clinico San Carlos"
            ),
            related_paper_dois=["10.21203/rs.3.rs-6958817/v1", "10.31428/10317/13645"],
            country="ES",
            data_url="https://doi.org/10.5281/zenodo.19599465",
            publication_year=2026,
            license="CC-BY-4.0",
            repository="Zenodo",
            keywords=[
                "EEG",
                "motor imagery",
                "motor preparation",
                "motor execution",
                "chronic stroke",
                "MCA stroke",
                "rehabilitation",
                "mu rhythm",
                "beta rhythm",
                "ERD",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        tags=Tags(
            pathology=["stroke", "healthy"], modality=["Motor"], type=["Motor Imagery"]
        ),
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 18 + 1)),
            # Sessions vary: 9 patients have pre+post (2), while patient 2 and the
            # 8 controls have pre only (1). Per MOABB convention, the declared
            # count is the minimum (1); the true per-subject session set is read
            # data-borne from PARDOGARCIA2026_SUBJECTS in _get_single_subject_data.
            sessions_per_subject=1,
            events={"pinch": 1, "fist": 2},
            code="PardoGarcia2026",
            interval=[0, 1.5],
            paradigm="imagery",
            doi="10.5281/zenodo.19599465",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return one ``.vhdr`` path per session (``pre`` then ``post``)."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        paths = []
        for stem in PARDOGARCIA2026_SUBJECTS[subject].values():
            # Fetch the payload (.eeg) and markers (.vmrk) before the header (the
            # returned path) so its siblings are on disk once it lands.
            for ext in (".eeg", ".vmrk", ".vhdr"):
                url = f"{PARDOGARCIA2026_BASE_URL}/{stem}{ext}"
                local = dl.data_dl(url, self.code, path, force_update, verbose)
            paths.append(local)
        return paths

    def _load_raw(self, vhdr_path):
        """Read one BrainVision recording and standardize channels/events."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Some headers still name the pre-BIDS DataFile/MarkerFile.
            raw = mne.io.read_raw_brainvision(
                vhdr_path,
                preload=True,
                verbose=False,
                overrides={
                    "data_fname": Path(vhdr_path).with_suffix(".eeg").name,
                    "marker_fname": Path(vhdr_path).with_suffix(".vmrk").name,
                },
            )
            raw.set_channel_types(
                {ch: "eog" for ch in PARDOGARCIA2026_EOG if ch in raw.ch_names}
            )
            # Map upper-case header names (e.g. "CZ", "FCZ") to the 10-05
            # template spelling so the montage resolves the midline electrodes.
            montage = make_standard_montage("colin27_1005")
            canonical = {name.lower(): name for name in montage.ch_names}
            raw.rename_channels(
                {ch: canonical.get(ch.lower(), ch) for ch in raw.ch_names}
            )
            # Map only the image-onset cues; go-cue (S 11 / S 22) and block
            # (S255) markers are left as-is, exposing one event per trial.
            descriptions = set(raw.annotations.description)
            raw.annotations.rename(
                {
                    desc: label
                    for desc, label in PARDOGARCIA2026_EVENT_RENAME.items()
                    if desc in descriptions
                }
            )
            raw.set_montage(montage, on_missing="ignore", verbose=False)
        return raw

    def _get_single_subject_data(self, subject):
        # Session keys start with an integer index: "0pre", "1post".
        labels = PARDOGARCIA2026_SUBJECTS[subject]
        return {
            f"{idx}{label}": {"0": self._load_raw(Path(vhdr_path))}
            for idx, (label, vhdr_path) in enumerate(zip(labels, self.data_path(subject)))
        }
