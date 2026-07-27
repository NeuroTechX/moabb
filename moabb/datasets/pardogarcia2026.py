"""PardoGarcia2026 mu/beta motor-imagery EEG dataset in chronic MCA stroke."""

import re
import tempfile
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
# second marker (S 11 / S 22) at roughly 2 s that flags the motor-skill phase of
# the same class; those are left unmapped so a single event is kept per trial.
PARDOGARCIA2026_EVENT_RENAME = {"Stimulus/S  1": "pinch", "Stimulus/S  2": "fist"}

# Bipolar EOG channels present alongside the scalp EEG montage.
PARDOGARCIA2026_EOG = ("HEOGn", "HEOGp", "VEOGn", "VEOGp")

# 63 recording channels in acquisition order (59 EEG incl. the A1 mastoid, plus
# 4 EOG), taken verbatim from the BrainVision headers.
PARDOGARCIA2026_CHANNELS = [
    "O2",
    "OZ",
    "O1",
    "PO8",
    "PO6",
    "PO4",
    "POZ",
    "PO3",
    "PO5",
    "PO7",
    "P8",
    "P6",
    "P4",
    "P2",
    "PZ",
    "P1",
    "P3",
    "P5",
    "P7",
    "TP8",
    "CP6",
    "CP4",
    "CP2",
    "CPZ",
    "CP1",
    "CP3",
    "CP5",
    "TP7",
    "HEOGn",
    "HEOGp",
    "VEOGn",
    "VEOGp",
    "FP1",
    "FPZ",
    "FP2",
    "AF3",
    "AF4",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "A1",
]


class PardoGarcia2026(BaseDataset):
    """Mu/beta motor-imagery EEG in chronic MCA stroke and healthy controls [1]_.

    .. admonition:: Dataset summary

        ==============  =======  =======  ==========  =====================  ============  ===============  ===========
        Name              #Subj    #Chan    #Classes    #Trials / class        Trials len    Sampling rate      #Sessions
        ==============  =======  =======  ==========  =====================  ============  ===============  ===========
        PardoGarcia2026      18       63           2       50 (ctrl) / 70 (pat)           7s          1000 Hz          1-2
        ==============  =======  =======  ==========  =====================  ============  ===============  ===========

    **Dataset description**

    EEG recorded during a cued, two-class hand motor-imagery task used to study mu
    (8-12 Hz) and beta (12-30 Hz) oscillatory changes in chronic middle cerebral
    artery (MCA) stroke patients and healthy controls. On each trial an image of
    the grip to imagine is shown: a precision pinch (``pinch``, stimulus code 1)
    or a closed fist (``fist``, stimulus code 2); the participant then performs
    kinaesthetic motor imagery of that grip. The grip-to-code mapping is documented
    in the record file ``bdf_IMAGEN.txt``.

    The cohort comprises 10 chronic MCA stroke patients (stems ``PAC01``-``PAC10``)
    and 8 healthy controls (stems ``C01``, ``02``, ``C03`` and
    ``CONTROL04``-``CONTROL08``), for 18 subjects in total. Patients were recorded
    at baseline (session ``pre``) and, when available, again after a rehabilitation
    programme (session ``post``); every patient has a ``post`` session except
    patient 2 (lost to follow-up), so 9 of the 10 patients have two sessions.
    Controls were recorded once (session ``pre`` only). Each session is exposed as
    a single continuous run.

    Signals were acquired with a 63-channel BrainVision system at 1000 Hz. The
    montage holds 59 EEG electrodes (a 10-10 scalp layout plus the ``A1`` mastoid)
    and 4 bipolar EOG channels (``HEOGn``, ``HEOGp``, ``VEOGn``, ``VEOGp``), which
    this loader marks as ``eog``. Channel locations are not stored in the headers;
    standard 10-05 template positions are attached at load time.

    Each trial carries two markers of the same class: an image-onset cue
    (``S 1``/``S 2``, taken as t = 0) and a motor-skill marker (``S 11``/``S 22``)
    at roughly 2 s. This loader keeps one event per trial by mapping only the
    image-onset cues; the exposed interval spans 0-7 s after the cue, matching the
    epoching window described in the dataset analysis manual (baseline [-1, 0] s,
    analysis up to 7 s, with the imagery/ERD phase from about 2 s). Control
    recordings contain 50 trials per class and patient recordings about 70.

    References
    ----------

    .. [1] Pardo-Garcia, R., Ruiz-Izquierdo, M., Garcia de la Vega, M., Calvillo,
       R., Kontaxakis, G., Moreno, E. M., and Pozo, M. A. (2026). Mu and Beta
       Oscillatory Changes during a motor task following Rehabilitation in Chronic
       MCA Stroke: Insights from EEG. Zenodo.
       DOI: https://doi.org/10.5281/zenodo.19599465

    Notes
    -----

    .. versionadded:: 1.2.1

    """

    nemar_id = "EXEMPT"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=63,
            channel_types={"eeg": 59, "eog": 4},
            montage="10-10",
            hardware="BrainVision (Brain Products GmbH)",
            reference=None,
            ground=None,
            sensors=list(PARDOGARCIA2026_CHANNELS),
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
            trial_duration=7.0,
            study_design=(
                "Cued two-class hand motor imagery (precision pinch vs closed fist) "
                "used to study mu/beta oscillatory changes in chronic MCA stroke "
                "patients (baseline and post-rehabilitation) and healthy controls."
            ),
            feedback_type="none",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="offline",
            events={"pinch": 1, "fist": 2},
            instructions=(
                "Perform kinaesthetic motor imagery of the hand grip shown in the "
                "cue image (a precision pinch or a closed fist)."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.19599465",
            description=(
                "Two-class hand motor-imagery EEG (pinch vs fist) from 10 chronic "
                "MCA stroke patients (baseline and post-rehabilitation) and 8 "
                "healthy controls, 63 BrainVision channels at 1000 Hz, used to "
                "study mu and beta oscillatory changes."
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
                "Universidad Complutense de Madrid; Universidad Politecnica de "
                "Madrid; Universidad Nacional de Educacion a Distancia"
            ),
            country="ES",
            data_url="https://doi.org/10.5281/zenodo.19599465",
            publication_year=2026,
            license="CC-BY-4.0",
            repository="Zenodo",
            keywords=[
                "EEG",
                "motor imagery",
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
            interval=[0, 7],
            paradigm="imagery",
            doi="10.5281/zenodo.19599465",
        )

    def _download_recording(self, stem, path, force_update, verbose):
        """Download the three BrainVision files of one recording.

        Parameters
        ----------
        stem : str
            Recording stem (e.g. ``"PAC01"`` or ``"PAC01-POST"``).
        path : None | str
            Storage location override forwarded to :func:`data_dl`.
        force_update : bool
            Re-download even if a local copy exists.
        verbose : bool, str, int, or None
            Verbosity level.

        Returns
        -------
        str
            Local path to the recording's ``.vhdr`` header file.
        """
        vhdr_path = None
        # Fetch the payload (.eeg) and markers (.vmrk) before the header so that
        # every sibling referenced by the .vhdr is present on disk once it lands.
        for ext in (".eeg", ".vmrk", ".vhdr"):
            url = f"{PARDOGARCIA2026_BASE_URL}/{stem}{ext}"
            local = dl.data_dl(url, self.code, path, force_update, verbose)
            if ext == ".vhdr":
                vhdr_path = local
        return vhdr_path

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the ``.vhdr`` paths of a single subject's session(s).

        Parameters
        ----------
        subject : int
            Subject number (1-18).
        path : None | str
            Location of where to look for the data storing location. If None, the
            environment variable or config parameter MNE_(dataset) is used. If it
            doesn't exist, the "~/mne_data" directory is used. If the dataset is
            not found under the given path, the data will be automatically
            downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose()).

        Returns
        -------
        list of str
            One ``.vhdr`` path per session, ordered as in
            ``PARDOGARCIA2026_SUBJECTS`` (``pre`` then, when present, ``post``).
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        paths = []
        for stem in PARDOGARCIA2026_SUBJECTS[subject].values():
            paths.append(self._download_recording(stem, path, force_update, verbose))
        return paths

    def _load_raw(self, vhdr_path):
        """Read one BrainVision recording and standardize channels/events."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = self._read_brainvision(vhdr_path)

        # Mark the bipolar EOG derivations.
        eog = {ch: "eog" for ch in PARDOGARCIA2026_EOG if ch in raw.ch_names}
        if eog:
            raw.set_channel_types(eog)

        # The headers use upper-case names (e.g. "CZ", "FCZ", "FPZ"); map them to
        # the canonical 10-05 template spelling so the montage resolves positions
        # for the (sensorimotor-relevant) midline electrodes.
        montage = make_standard_montage("standard_1005")
        canonical = {name.lower(): name for name in montage.ch_names}
        rename = {
            ch: canonical[ch.lower()]
            for ch in raw.ch_names
            if ch.lower() in canonical and ch != canonical[ch.lower()]
        }
        if rename:
            raw.rename_channels(rename)

        # Map only the image-onset cue markers to their class labels; the paired
        # motor-skill markers (S 11 / S 22) and block markers (S255) are left as-is
        # so a single event per trial is exposed to the paradigm.
        present = {
            desc: label
            for desc, label in PARDOGARCIA2026_EVENT_RENAME.items()
            if desc in set(raw.annotations.description)
        }
        if present:
            raw.annotations.rename(present)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_montage(montage, on_missing="ignore", verbose=False)

        return raw

    @staticmethod
    def _read_brainvision(vhdr_path):
        """Read a header, repairing an unambiguous stale sibling reference."""
        vhdr_path = Path(vhdr_path)
        header = vhdr_path.read_text(encoding="utf-8")
        repaired_any = False

        def repair_reference(match):
            nonlocal repaired_any
            key, filename = match.groups()
            suffix = {"DataFile": ".eeg", "MarkerFile": ".vmrk"}[key]
            sibling = vhdr_path.with_suffix(suffix)
            if (vhdr_path.parent / filename).is_file() or not sibling.is_file():
                return match.group(0)
            repaired_any = True
            return f"{key}={sibling.name}"

        repaired = re.sub(
            r"(?m)^(DataFile|MarkerFile)=([^\r\n]+)",
            repair_reference,
            header,
        )
        if not repaired_any:
            return mne.io.read_raw_brainvision(
                str(vhdr_path), preload=True, verbose=False
            )

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".vhdr",
            prefix=f".{vhdr_path.stem}-",
            dir=vhdr_path.parent,
            delete=False,
        ) as stream:
            stream.write(repaired)
            repaired_path = Path(stream.name)
        try:
            return mne.io.read_raw_brainvision(
                str(repaired_path), preload=True, verbose=False
            )
        finally:
            repaired_path.unlink(missing_ok=True)

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject as ``{session: {run: Raw}}``.

        Parameters
        ----------
        subject : int
            Subject number (1-18).

        Returns
        -------
        dict
            ``{session_label: {"0": Raw}}`` with ``session_label`` in
            ``{"pre", "post"}``; patients have ``pre`` and (except patient 2)
            ``post``, controls have ``pre`` only.
        """
        session_labels = list(PARDOGARCIA2026_SUBJECTS[subject].keys())
        vhdr_paths = self.data_path(subject)

        sessions = {}
        for idx, (label, vhdr_path) in enumerate(zip(session_labels, vhdr_paths)):
            # MOABB requires the session key to start with an integer index
            # (optionally followed by a letters+digits description).
            sessions[f"{idx}{label}"] = {"0": self._load_raw(Path(vhdr_path))}
        return sessions
