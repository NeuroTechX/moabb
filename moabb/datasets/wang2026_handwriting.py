"""Chinese handwriting imagery dataset (CCS-HI and SV-HI) from Wang et al. 2026.

Wang et al. (2026), Scientific Data.
DOI: 10.1038/s41597-026-06708-3
Data DOI: 10.6084/m9.figshare.29987758.v4
"""

import json
import zipfile
from pathlib import Path

import mne_bids
import pandas as pd
from mne.channels import get_builtin_montages, make_standard_montage
from mne_bids import events_file_to_annotation_kwargs

from .base import BaseBIDSDataset
from .download import data_dl, get_dataset_path
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


# Sampling rate (Hz), from the BIDS ``task-handwritingimagery_eeg.json``
# sidecar.
_SFREQ = 1000.0

# 32 channel names in acquisition order, from the BIDS ``electrodes.tsv``
# sidecar. The dataset follows the 10-10 layout.
# fmt: off
WANG2026_CH_NAMES = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8",
    "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
    "Cz", "C3", "C4", "T7", "T8",
    "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "P7", "P8",
    "PO3", "PO4", "Oz", "O1", "O2",
]
# fmt: on

# The two tasks are published as separate BIDS tasks. Each is a 5- or 6-class
# imagery task whose trial_type labels carry the stroke/vowel name, so the
# events dictionary maps the BIDS ``value`` column to a stable class label.
WANG2026_EVENTS = {
    "horizontal": 1,
    "vertical": 2,
    "left_falling": 3,
    "right_falling": 4,
    "turning": 5,
    "a": 11,
    "o": 12,
    "e": 13,
    "i": 14,
    "u": 15,
    "v": 16,
}

# BIDS task names, in the order they appear in each subject's directory.
_TASKS = ("CCSHI", "SVHI")

# Trials per class and trial length, from the README and the events sidecars
# of the released data (200 events for CCS-HI, 240 for SV-HI, 4 s each).
_TRIALS_PER_CLASS = 40
_TRIAL_DURATION = 4.0

# figshare article and version hosting the BIDS release. The file ids are
# version-specific, so the loader resolves them through the public figshare
# API instead of hardcoding per-subject download urls.
_FIGSHARE_ARTICLE_ID = 29987758
_FIGSHARE_VERSION = 4

# Per-subject demographics from the BIDS ``participants.tsv`` sidecar,
# indexed by 1-based subject number. sub-02, sub-04, sub-09 and sub-16 are
# absent from the release (see the README); the remaining 21 subjects are
# numbered non-contiguously.
_SUBJECT_IDS = (
    1,
    3,
    5,
    6,
    7,
    8,
    10,
    11,
    12,
    13,
    14,
    15,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
)

_DEMOGRAPHICS = {
    1: ("male", 28, "novice"),
    3: ("male", 24, "expert"),
    5: ("male", 23, "expert"),
    6: ("male", 27, "novice"),
    7: ("female", 23, "novice"),
    8: ("male", 25, "novice"),
    10: ("female", 24, "novice"),
    11: ("male", 23, "novice"),
    12: ("male", 32, "expert"),
    13: ("male", 28, "expert"),
    14: ("male", 30, "expert"),
    15: ("male", 24, "novice"),
    17: ("female", 27, "expert"),
    18: ("male", 22, "novice"),
    19: ("male", 28, "novice"),
    20: ("male", 27, "expert"),
    21: ("female", 22, "novice"),
    22: ("female", 28, "novice"),
    23: ("female", 25, "novice"),
    24: ("female", 27, "novice"),
    25: ("female", 24, "novice"),
}

# The BIDS trial_type labels embed the class name in parentheses and, for one
# stroke, a trailing space. This maps them onto the stable labels of
# WANG2026_EVENTS.
_TRIAL_TYPE_TO_LABEL = {
    "horizontal('一')": "horizontal",
    "vertical('丨')": "vertical",
    "left-falling ('丿')": "left_falling",
    "left-falling ('丿') ": "left_falling",
    "left-falling('丿')": "left_falling",
    "right-falling('㇏')": "right_falling",
    "turning('𠃌')": "turning",
    "a(/a/)": "a",
    "o(/o/)": "o",
    "e(/e/)": "e",
    "i(/i/)": "i",
    "u(/u/)": "u",
    "v(/ü/)": "v",
}


class Wang2026Handwriting(BaseBIDSDataset):
    """Chinese handwriting imagery dataset (CCS-HI / SV-HI) from Wang et al. 2026 (cognitive paradigm).

    Dataset from [1]_.

    The dataset contains EEG recordings from 21 healthy, right-handed adults
    (10 female, 11 male, aged 22-32 years) performing two handwriting-imagery
    tasks:

    - **CCS-HI** (Chinese character strokes imagery): five stroke classes
      (horizontal, vertical, left-falling, right-falling, turning), 40 trials
      per class;
    - **SV-HI** (single vowels imagery): six Pinyin vowel classes
      (a, o, e, i, u, ü), 40 trials per class.

    Each subject completed both tasks in each of two sessions recorded on
    separate days. Every trial is a 4 s window of kinesthetic handwriting
    imagery cued by a visual stimulus, followed by a rest period. EEG was
    recorded from 32 Ag/AgCl electrodes (10-10 layout) at 1000 Hz, referenced
    to CPz with Fpz as ground, with a 0.1-100 Hz hardware band-pass and no
    software filtering applied before disclosure.

    The subjects split into expert and novice groups by their handwriting
    imagery performance (see the ``group`` column of the release's
    ``participants.tsv``), which makes the dataset usable for both
    within-session and expert/novice transfer analyses.

    .. important::

       The release flags two sessions with fewer valid trials than the
       protocol's standard count (sub-06 session 2 of SV-HI: 225 instead of
       240; sub-13 session 2 of CCS-HI: 162 instead of 200). The authors
       attribute this to trigger misfires during acquisition. MOABB keeps
       those sessions as released; the counts above describe the standard
       session, and the per-subject trial totals reported by
       :meth:`get_data` reflect what each session actually contains.

    This loader reads the authors' BIDS release hosted on figshare. Each
    subject is distributed as a single zip containing both tasks for both
    sessions; the loader downloads it, unpacks it beside the other BIDS
    sidecars it fetches from the same deposit, and reads the resulting BIDS
    tree with :class:`mne_bids`. Sessions are numbered ``"0"`` and ``"1"``
    in MOABB and runs are named after the BIDS task (``CCSHI`` / ``SVHI``).

    References
    ----------
    .. [1] F. Wang, Y. Chen, P. Wang, A. Gong, H. Xu, and Y. Fu, "An EEG
       dataset for handwriting imagery decoding of Chinese character strokes
       and Pinyin single vowels," Scientific Data, 2026.
       DOI: 10.1038/s41597-026-06708-3

    Notes
    -----
    .. versionadded:: 1.8.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=_SFREQ,
            channel_types={"eeg": 32},
            montage="standard_1005",
            hardware="Neuracle NeuSenW32 32-channel Ag/AgCl cap",
            sensor_type="Ag/AgCl",
            reference="CPz",
            ground="Fpz",
            sensors=list(WANG2026_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=len(_SUBJECT_IDS),
            health_status="healthy",
            gender={
                "male": sum(1 for s in _SUBJECT_IDS if _DEMOGRAPHICS[s][0] == "male"),
                "female": sum(1 for s in _SUBJECT_IDS if _DEMOGRAPHICS[s][0] == "female"),
            },
            age_mean=sum(_DEMOGRAPHICS[s][1] for s in _SUBJECT_IDS) / len(_SUBJECT_IDS),
            age_min=min(_DEMOGRAPHICS[s][1] for s in _SUBJECT_IDS),
            age_max=max(_DEMOGRAPHICS[s][1] for s in _SUBJECT_IDS),
            handedness={"right": len(_SUBJECT_IDS)},
            bci_experience="mixed (expert and novice groups)",
            ages=[_DEMOGRAPHICS[s][1] for s in _SUBJECT_IDS],
            sexes=[_DEMOGRAPHICS[s][0] for s in _SUBJECT_IDS],
        ),
        experiment=ExperimentMetadata(
            paradigm="cognitive",
            events=dict(WANG2026_EVENTS),
            n_classes=5,
            class_labels=[
                "horizontal",
                "vertical",
                "left_falling",
                "right_falling",
                "turning",
            ],
            trial_duration=_TRIAL_DURATION,
            study_design=(
                "Two handwriting-imagery tasks: Chinese character strokes "
                "(CCS-HI, 5 classes) and Pinyin single vowels (SV-HI, 6 "
                "classes); each task recorded in two sessions on separate days"
            ),
            study_domain="communication",
            stimulus_type="visual cue (stroke or vowel to imagine writing)",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            task_type="handwriting_imagery",
            feedback_type="none",
            has_training_test_split=False,
            instructions=(
                "Subjects imagined the kinesthetic act of writing the cued "
                "Chinese character stroke or Pinyin vowel for the duration of "
                "the 4 s trial."
            ),
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-026-06708-3",
            investigators=[
                "Fan Wang",
                "Yanxiao Chen",
                "Peng Wang",
                "Anmin Gong",
                "Hui Xu",
                "Yunfa Fu",
            ],
            senior_author="Yunfa Fu",
            institution="Minjiang University",
            country="CN",
            repository="figshare",
            data_url="https://doi.org/10.6084/m9.figshare.29987758.v4",
            license="CC-BY-4.0",
            publication_year=2026,
            funding=["National Natural Science Foundation of China (Grant No. 62376112)"],
            keywords=[
                "handwriting imagery",
                "EEG",
                "BCI",
                "Chinese character strokes",
                "Pinyin vowels",
            ],
        ),
        sessions_per_subject=2,
        runs_per_session=2,
        data_processed=False,
        file_format="BDF",
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            highpass_hz=0.1,
            lowpass_hz=100.0,
            filter_type="hardware",
            epoch_window=[0.0, _TRIAL_DURATION],
            notes=(
                "Recorded with a 0.1-100 Hz hardware band-pass; the release "
                "applies no software filter."
            ),
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="cognitive",
            imagery_tasks=[
                "horizontal",
                "vertical",
                "left_falling",
                "right_falling",
                "turning",
            ],
            imagery_duration_s=_TRIAL_DURATION,
        ),
        data_structure=DataStructureMetadata(
            n_trials=len(_SUBJECT_IDS) * 2 * (200 + 240),
            trials_context=(
                "21 subjects x 2 sessions x (200 CCS-HI + 240 SV-HI trials) "
                "= 18,480 trials at the protocol's standard counts; two "
                "sessions carry fewer valid trials as released (sub-06 "
                "session 2 SV-HI: 225, sub-13 session 2 CCS-HI: 162)"
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["EEGNet"], frequency_bands={"mu_beta": [8.0, 30.0]}
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold", cv_folds=10, evaluation_type=["within_session"]
        ),
        bci_application=BCIApplicationMetadata(
            environment="laboratory",
            online_feedback=False,
            applications=["communication"],
        ),
        tags=Tags(pathology=["healthy"], modality=["cognitive"], type=["cognitive"]),
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        # Each stored trial is exactly the 4 s imagery window (4000 samples
        # at 1000 Hz). Because mne.Epochs includes the tmax sample, an
        # interval of [0, 4] would request 4001 samples and thus borrow one
        # sample from the next trial; tmax = 4 - 1/sfreq selects exactly the
        # 4000 stored samples.
        super().__init__(
            subjects=list(range(1, len(_SUBJECT_IDS) + 1)),
            sessions_per_subject=2,
            events=dict(WANG2026_EVENTS),
            code="Wang2026Handwriting",
            interval=[0.0, _TRIAL_DURATION - 1.0 / _SFREQ],
            paradigm="cognitive",
            doi="10.1038/s41597-026-06708-3",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    # ------------------------------------------------------------------
    # Download: figshare BIDS release
    # ------------------------------------------------------------------
    def _figshare_files(self):
        """Return {filename: download_url} for the pinned figshare version."""
        import pooch

        url = (
            f"https://api.figshare.com/v2/articles/{_FIGSHARE_ARTICLE_ID}"
            f"/versions/{_FIGSHARE_VERSION}"
        )
        response = pooch.retrieve(
            url,
            known_hash=None,
            path=pooch.os_cache("moabb"),
            fname=f"figshare_{_FIGSHARE_ARTICLE_ID}_v{_FIGSHARE_VERSION}.json",
            progressbar=False,
        )
        with open(response) as f:
            payload = json.load(f)
        return {f["name"]: f["download_url"] for f in payload["files"]}

    def _download_subject(self, subject, path, force_update, update_path, verbose):
        """Download one subject's BIDS tree from figshare and return its root."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        bids_id = _SUBJECT_IDS[subject - 1]
        # data_dl places every file under MNE-wang2026-data; that directory is
        # the BIDS root the released zips unpack into.
        bids_root = Path(get_dataset_path("WANG2026", path)) / "MNE-wang2026-data"
        bids_root.mkdir(parents=True, exist_ok=True)

        files = self._figshare_files()
        # The subject zip carries the per-subject BIDS tree; the deposit's
        # top-level sidecars are fetched once alongside it.
        subject_zip = f"sub-{bids_id:02d}.zip"
        top_level = (
            "dataset_description.json",
            "participants.tsv",
            "participants.json",
            "task-CCSHI_events.json",
            "task-SVHI_events.json",
        )
        for name in [subject_zip, *top_level]:
            if name not in files:
                raise FileNotFoundError(
                    f"{name} is missing from the figshare release "
                    f"(article {_FIGSHARE_ARTICLE_ID} version {_FIGSHARE_VERSION})"
                )
            data_dl(
                files[name],
                "WANG2026",
                path=path,
                force_update=force_update,
                verbose=verbose,
                fname=name,
            )
        subject_dir = bids_root / subject_zip
        if not subject_dir.is_file():
            raise FileNotFoundError(f"Download of {subject_zip} failed")
        with zipfile.ZipFile(subject_dir) as zf:
            zf.extractall(bids_root)
        return str(bids_root)

    def download(
        self,
        subject_list=None,
        path=None,
        force_update=False,
        update_path=None,
        accept=False,
        verbose=None,
    ):
        """Download the BIDS trees of the requested subjects."""
        for subject in self.subject_list if subject_list is None else subject_list:
            self.data_path(subject, path, force_update, update_path, verbose)

    def _get_path_search_params(self, subject):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        return {
            "subjects": f"{_SUBJECT_IDS[subject - 1]:02d}",
            "tasks": None,
            "suffixes": "eeg",
            "datatypes": "eeg",
            "extensions": ".bdf",
        }

    def _get_single_subject_data(self, subject):
        # The two tasks share a session but have no run entity, so
        # BaseBIDSDataset's (session, run) keying would let one task overwrite
        # the other. Key the runs by the BIDS task name instead.
        bids_paths = self.bids_paths(subject)
        montage_name = (
            "colin27_1005"
            if "colin27_1005" in get_builtin_montages()
            else "standard_1005"
        )
        montage = make_standard_montage(montage_name)
        data = {}
        for bids_path in bids_paths:
            raw = mne_bids.read_raw_bids(
                bids_path, extra_params=self._get_read_extra_params(subject)
            )
            # The BDFs carry a 33rd acquisition channel (BioSemi-style status
            # marker) that the release's channels.tsv does not list; drop it so
            # that only the 32 EEG channels reach the paradigms.
            raw.drop_channels([ch for ch in raw.ch_names if ch not in WANG2026_CH_NAMES])
            raw.load_data()
            # The BIDS electrodes.tsv already carries the cap positions;
            # set_montage keeps them and only fills in the montage for
            # channels whose coordinates are missing.
            raw.set_montage(montage, match_case=False, on_missing="ignore")
            # The BIDS trial_type labels carry the stroke/vowel name in
            # parentheses (and, for one stroke, an inconsistent trailing
            # space); rewrite them to the stable class labels of
            # WANG2026_EVENTS so that paradigms can select them. Only this
            # task's own labels are renamed, since MNE's rename rejects keys
            # that are absent from the data.
            task_labels = {
                old: new
                for old, new in _TRIAL_TYPE_TO_LABEL.items()
                if old in raw.annotations.description
            }
            if task_labels:
                raw.annotations.rename(task_labels)
            # The trials are discontinuous: annotate the boundaries between
            # consecutive trials so that MNE's default filtering does not mix
            # neighbouring trials, as in the Ma2022 loader.
            onsets = sorted(
                onset
                for onset, description in zip(
                    raw.annotations.onset, raw.annotations.description
                )
                if description in self.event_id
            )
            if len(onsets) > 1:
                raw.annotations.append(
                    onsets[1:],
                    [0.0] * (len(onsets) - 1),
                    ["EDGE boundary"] * (len(onsets) - 1),
                )
            session = (
                "0" if bids_path.session is None else str(int(bids_path.session) - 1)
            )
            # MOABB run names must start with a unique integer index; the
            # BIDS task entity (CCSHI / SVHI) is kept as the descriptive
            # suffix, indexed by its position in _TASKS.
            run = f"{_TASKS.index(bids_path.task) + 1}{bids_path.task}"
            data.setdefault(session, {})[run] = raw
        return data

    def get_additional_metadata(self, subject, session, run):
        # BaseBIDSDataset matches the BIDS run entity, which this release does
        # not use; match the task instead.
        bids_path = next(
            p
            for p in self.bids_paths(subject)
            if int(p.session) == int(session) + 1 and p.task == run
        )
        events_file = bids_path.find_matching_sidecar(suffix="events", extension=".tsv")
        annotations = events_file_to_annotation_kwargs(events_file)
        return pd.DataFrame(
            {
                "onset": annotations["onset"],
                "duration": annotations["duration"],
                "trial_type": annotations["description"],
            }
        ).assign(subject=subject, session=session, run=run)
