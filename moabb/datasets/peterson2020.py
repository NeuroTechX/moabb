"""Peterson 2020 motor imagery vs rest low-cost EEG dataset (OpenNeuro ds003810)."""

from pathlib import Path

import mne_bids

from ._openneuro_mirror import OpenNeuroMirrorMixin
from .base import BaseBIDSDataset
from .bids_interface import StepType
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
from .preprocessing import FixedPipeline, SetRawAnnotations
from .utils import stim_channels_with_selected_ids


_S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds003810"

# Subjects present in the archive (sub-01 and sub-11 are absent).
_SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]

# MI-vs-Rest runs. RUN0 is a real-movement demonstration and is excluded.
_RUNS = ["1", "2", "3", "4"]

# 15 EEG channels (consumer-grade device, old 10-20 nomenclature).
_CH_NAMES = "Pz Cz T6 T4 F8 P4 C4 F4 Fz T5 T3 F7 P3 C3 F3".split()

_EVENTS = {"motor_imagery": 1, "rest": 2}

# OpenViBE GDF stimulation labels: OVTK_GDF_Right = MI cue, OVTK_GDF_Tongue = rest.
_ANNOT_TO_NAME = {"OVTK_GDF_Right": "motor_imagery", "OVTK_GDF_Tongue": "rest"}

# The EDF physical-dimension fields are blank even though channels.tsv records
# all 15 EEG signals as microvolts. Tell MNE the missing source unit so its EDF
# reader performs the standard microvolts-to-SI-volts conversion.
PETERSON2020_CACHE_VERSION = "edf-blank-physdim-uv-to-v-v1"


class _PetersonSetRawAnnotations(SetRawAnnotations):
    """Carry the unit repair version into MOABB's raw-cache fingerprint."""

    def __init__(self, event_id, interval, cache_version):
        self.cache_version = cache_version
        super().__init__(event_id, interval)


class Peterson2020(OpenNeuroMirrorMixin, BaseBIDSDataset):
    """Motor imagery vs rest low-cost EEG dataset from Peterson et al 2020 [1]_.

    10 novice participants (12 recruited, two excluded by the authors),
    15-channel consumer-grade EEG (OpenBCI Cyton + Daisy board with an
    Electro-Cap, reference left / ground right ear lobe) at 125 Hz, recorded
    in a non-shielded office. Subjects either imagined grasping with their
    dominant hand (**motor_imagery**) or stayed idle (**rest**) for 4 s after
    the cue; no feedback was presented. RUN0 (real-movement demonstration) is
    excluded; RUN1-RUN4 hold 20 MI + 20 rest trials each. Events are native
    EDF annotations (OpenViBE labels ``OVTK_GDF_Right`` = MI cue,
    ``OVTK_GDF_Tongue`` = rest).

    Notes
    -----
    The EDF physical-dimension fields are blank; the channels sidecars
    specify microvolts. The reader is given ``units="uV"`` so MNE returns
    SI volts without a second post-read scaling. Bad-channel flags are kept.

    The paper states that "during acquisition, the EEG signals were filtered
    between 0.5 and 45 Hz with a 3rd order Butterworth bandpass-filter"
    (OpenViBE), whereas the BIDS sidecars declare no hardware/software
    filters; whether the released EDF files carry that online filter has not
    been verified on the data. The 1-40 Hz 5th-order Butterworth filter of
    the paper was applied offline for the published analysis only.

    References
    ----------
    .. [1] Peterson, V., Galvan, C., Hernandez, H., & Spies, R. (2020).
           A feasibility study of a complete low-cost consumer-grade
           brain-computer interface system. Heliyon, 6(3), e03425.
           https://doi.org/10.1016/j.heliyon.2020.e03425
    """

    nemar_id = "on003810"
    nemar_subject_template = "{subject:02d}"
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=125.0,
            channel_types={"eeg": 15},
            montage="10-20",
            hardware="OpenBCI Cyton + Daisy (16-channel) with Electro-Cap System II",
            cap_manufacturer="Electro-Cap International",
            cap_model="Electro-Cap System II",
            reference="left ear lobe",
            ground="right ear lobe",
            filters=(
                "paper: 0.5-45 Hz 3rd-order Butterworth band-pass applied in "
                "OpenViBE during acquisition; BIDS sidecars: n/a"
            ),
            software="OpenViBE",
            sensors=list(_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="healthy",
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=list(_EVENTS.keys()),
            study_design=(
                "Binary kinesthetic motor imagery of dominant-hand grasping "
                "versus rest. RUN0 real-movement demo (excluded); RUN1-RUN4 "
                "MI-vs-rest (20 MI + 20 rest trials per run)."
            ),
            feedback_type="none",
            stimulus_type="visual cue (red arrow) after an auditory beep",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="cue-based",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.heliyon.2020.e03425",
            investigators=[
                "Victoria Peterson",
                "Catalina Maria Galvan",
                "Hugo Sacha Hernadez",
                "Ruben Spies",
            ],
            institution="IMAL, CONICET-UNL",
            institution_address="Santa Fe, Argentina",
            country="AR",
            data_url="https://openneuro.org/datasets/ds003810",
            publication_year=2020,
            license="CC0",
        ),
        sessions_per_subject=1,
        runs_per_session=4,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
            notes=(
                "BIDS sidecars declare no hardware/software filters; the paper "
                "reports a 0.5-45 Hz 3rd-order Butterworth band-pass applied in "
                "OpenViBE during acquisition and a 1-40 Hz 5th-order Butterworth "
                "post-processing filter for its analysis."
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA"],
            feature_extraction=["CSP", "bandpower"],
            spatial_filters=["CSP"],
        ),
        cross_validation=CrossValidationMetadata(evaluation_type=["within_subject"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_EVENTS.keys()),
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=160,
            n_trials_per_class={"motor_imagery": 80, "rest": 80},
            trials_context=(
                "Per subject: 4 MI-vs-rest runs x 40 trials (20 MI + 20 rest per run)."
            ),
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control", "rehabilitation"],
            environment="non-shielded office",
            online_feedback=False,
        ),
        data_processed=False,
        file_format="EDF (BIDS)",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(_SUBJECTS),
            sessions_per_subject=1,
            events=dict(_EVENTS),
            code="Peterson2020",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1016/j.heliyon.2020.e03425",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _create_process_pipeline(self):
        return FixedPipeline(
            [
                (
                    StepType.RAW,
                    _PetersonSetRawAnnotations(
                        self.event_id,
                        interval=self.interval,
                        cache_version=PETERSON2020_CACHE_VERSION,
                    ),
                )
            ]
        )

    def _get_path_search_params(self, subject):
        """Zero-padded subjects, MI runs only (exclude the RUN0 demo)."""
        out = {"extensions": [".edf"], "runs": list(_RUNS)}
        if subject is not None:
            out["subjects"] = f"{subject:02d}"
        return out

    def _get_read_extra_params(self, subject):
        """Declare the unit omitted from every published EEG EDF header."""
        return {"units": "uV"}

    def _get_single_subject_data(self, subject):
        """Load BIDS data and remap EDF annotation labels to class names."""
        result = {}
        for sess_key, session_runs in super()._get_single_subject_data(subject).items():
            for run_key, raw in session_runs.items():
                raw.drop_channels(
                    [
                        ch
                        for ch, kind in zip(raw.ch_names, raw.get_channel_types())
                        if kind == "stim"
                    ]
                )
                raw.annotations.rename(
                    {
                        k: v
                        for k, v in _ANNOT_TO_NAME.items()
                        if k in raw.annotations.description
                    }
                )
                result.setdefault(sess_key, {})[run_key] = (
                    stim_channels_with_selected_ids(raw, self.event_id)
                )
        return result

    def _download_subject(self, subject, path, force_update, update_path, verbose) -> str:
        """Download BIDS data from OpenNeuro S3 and return the BIDS root path."""
        mirror_root = self._mirror_root(subject, path, force_update, update_path, verbose)
        if mirror_root is not None:
            return mirror_root

        bids_root = Path(get_dataset_path("Peterson2020", path))
        bids_root = bids_root / "MNE-peterson2020-data"
        bids_root.mkdir(parents=True, exist_ok=True)
        subj_str = f"sub-{subject:02d}"
        for run in _RUNS:
            stem = f"{subj_str}/eeg/{subj_str}_task-MIvsRest_run-{run}"
            for suffix in ("eeg.edf", "eeg.json", "channels.tsv"):
                rel_path = f"{stem}_{suffix}"
                data_dl(
                    f"{_S3_BASE}/{rel_path}",
                    "Peterson2020",
                    path=path,
                    force_update=force_update,
                    verbose=verbose,
                    fname=rel_path,
                )
        mne_bids.make_dataset_description(
            path=bids_root,
            name="Motor Imagery vs Rest - Low-Cost EEG System",
            authors=list(self.METADATA.documentation.investigators),
            doi="doi:10.18112/openneuro.ds003810.v2.0.2",
            data_license="CC0",
            overwrite=False,
            verbose=False,
        )
        return str(bids_root)
