"""Metadata schema module for MOABB datasets.

This module provides standardized dataclasses for documenting dataset metadata.
Metadata is distributed across individual dataset files as class attributes.

Core Classes
------------
AcquisitionMetadata
    Technical acquisition parameters (sampling rate, channels, hardware, etc.)
DocumentationMetadata
    Publication and dataset provenance information (DOI, authors, repository)
ParticipantMetadata
    Subject demographics (sample size, age, gender, health status)
ExperimentMetadata
    Paradigm and task details (events, trial structure, task type)
DatasetMetadata
    Top-level container combining all metadata sections

Additional Classes
------------------
Demographics
    Extended subject demographics (subjects_count, ages, age_min, age_max)
ExternalLinks
    URLs and data source links
Timestamps
    Dataset creation and modification dates
Tags
    Classification tags
ChannelCount
    Channel count distribution entry
SamplingRateCount
    Sampling rate distribution entry

New Classes (from RALPH extraction)
-----------------------------------
AuxiliaryChannelsMetadata
    EOG, EMG, and other physiological channel information
FilterDetails
    Filter configuration details (highpass, lowpass, notch, etc.)
PreprocessingMetadata
    Preprocessing and artifact handling details
FrequencyBands
    Frequency band definitions for analysis
SignalProcessingMetadata
    Feature extraction and classification methods
CrossValidationMetadata
    Cross-validation methodology details
PerformanceMetadata
    Reported performance metrics
BCIApplicationMetadata
    BCI application context and environment
ParadigmSpecificMetadata
    Paradigm-specific parameters (SSVEP frequencies, c-VEP codes, etc.)
DataStructureMetadata
    Data organization and trial structure

Functions
---------
validate_country_code
    Validate ISO 3166-1 alpha-2 country codes
validate_metadata_against_dataset
    Validate metadata matches actual dataset structure
get_dataset_description
    Extract description from dataset class docstring

Example
-------
>>> from moabb.datasets.metadata import (
...     DatasetMetadata, AcquisitionMetadata,
...     ParticipantMetadata, ExperimentMetadata
... )
>>> metadata = DatasetMetadata(
...     acquisition=AcquisitionMetadata(
...         sampling_rate=512.0,
...         n_channels=64,
...         channel_types={"eeg": 60, "eog": 4},
...     ),
...     participants=ParticipantMetadata(n_subjects=20),
...     experiment=ExperimentMetadata(paradigm="imagery"),
... )

>>> # Access metadata from a dataset class
>>> from moabb.datasets import BNCI2014_001
>>> print(BNCI2014_001.METADATA.participants.n_subjects)
9
"""

import warnings
from dataclasses import replace

from .schema import (  # Core MOABB classes; Additional classes; New classes from RALPH extraction; Validation functions
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    ChannelCount,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    Demographics,
    DocumentationMetadata,
    ExperimentMetadata,
    ExternalLinks,
    FilterDetails,
    FrequencyBands,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PerformanceMetadata,
    PreprocessingMetadata,
    SamplingRateCount,
    SignalProcessingMetadata,
    Tags,
    Timestamps,
    get_dataset_description,
    validate_country_code,
    validate_metadata_against_dataset,
)


_MANUAL_METADATA_OVERRIDES = {
    "BNCI2014_001": {
        "acquisition": {"reference": "left mastoid"},
        "documentation": {"country": "AT"},
        "runs_per_session": 6,
    },
    "BI2012": {
        "experiment": {"task_type": "brain_invaders"},
    },
    # ERP CORE 2021 variants
    "ErpCore2021_ERN": {},
    "ErpCore2021_LRP": {},
    "ErpCore2021_MMN": {},
    "ErpCore2021_N170": {"participants": {"age_mean": 21.5}},
    "ErpCore2021_N2pc": {},
    "ErpCore2021_N400": {},
    "ErpCore2021_P3": {},
    # cVEP datasets
    "CastillosBurstVEP40": {},
    "CastillosBurstVEP100": {},
    "CastillosCVEP40": {},
    "CastillosCVEP100": {},
    "MartinezCagigal2023Checker": {"sessions_per_subject": 8},
    "MartinezCagigal2023Pary": {"sessions_per_subject": 5},
}


def _apply_manual_overrides(name: str, metadata: DatasetMetadata) -> DatasetMetadata:
    overrides = _MANUAL_METADATA_OVERRIDES.get(name)
    if not overrides:
        return metadata

    participants = metadata.participants
    acquisition = metadata.acquisition
    experiment = metadata.experiment
    documentation = metadata.documentation

    if "participants" in overrides:
        if participants is None:
            participants = ParticipantMetadata(n_subjects=1)
        participants = replace(participants, **overrides["participants"])

    if "acquisition" in overrides:
        if acquisition is None:
            acquisition = AcquisitionMetadata(
                sampling_rate=1.0, n_channels=1, channel_types={"eeg": 1}
            )
        acquisition = replace(acquisition, **overrides["acquisition"])

    if "experiment" in overrides:
        if experiment is None:
            experiment = ExperimentMetadata(paradigm="imagery")
        experiment = replace(experiment, **overrides["experiment"])

    if "documentation" in overrides:
        if documentation is None:
            documentation = DocumentationMetadata()
        documentation = replace(documentation, **overrides["documentation"])

    if "sessions_per_subject" in overrides:
        metadata = replace(
            metadata, sessions_per_subject=overrides["sessions_per_subject"]
        )
    if "runs_per_session" in overrides:
        metadata = replace(metadata, runs_per_session=overrides["runs_per_session"])

    return replace(
        metadata,
        participants=participants,
        acquisition=acquisition,
        experiment=experiment,
        documentation=documentation,
    )


def _build_minimal_metadata(dataset) -> DatasetMetadata:
    sampling_rate = (
        getattr(dataset, "sampling_rate", None) or getattr(dataset, "sfreq", None) or 1.0
    )
    n_channels = getattr(dataset, "n_channels", None)
    if n_channels is None:
        channels = (
            getattr(dataset, "channels", None) or getattr(dataset, "ch_names", None) or []
        )
        n_channels = len(channels) if channels else 1
    channel_types = {"eeg": int(n_channels)}

    participants = ParticipantMetadata(
        n_subjects=len(getattr(dataset, "subject_list", []) or [])
    )
    experiment = ExperimentMetadata(paradigm=getattr(dataset, "paradigm", "imagery"))
    acquisition = AcquisitionMetadata(
        sampling_rate=float(sampling_rate),
        n_channels=int(n_channels),
        channel_types=channel_types,
    )

    return DatasetMetadata(
        acquisition=acquisition,
        participants=participants,
        experiment=experiment,
        sessions_per_subject=getattr(dataset, "n_sessions", 1) or 1,
        runs_per_session=getattr(dataset, "n_runs", 1) or 1,
    )


def _build_fallback_metadata(dataset_name: str) -> DatasetMetadata:
    """Create minimal metadata when a dataset class cannot be instantiated."""
    warnings.warn(
        (
            f"Could not instantiate dataset {dataset_name} while building metadata catalog. "
            "Using minimal fallback metadata."
        ),
        RuntimeWarning,
        stacklevel=2,
    )
    return DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1.0,
            n_channels=1,
            channel_types={"eeg": 1},
        ),
        participants=ParticipantMetadata(n_subjects=1),
        experiment=ExperimentMetadata(paradigm="imagery"),
    )


def _merge_with_dataset(metadata: DatasetMetadata, dataset) -> DatasetMetadata:
    if dataset is None:
        return metadata

    # Participants
    participants = metadata.participants or ParticipantMetadata(n_subjects=1)
    n_subjects = len(getattr(dataset, "subject_list", []) or [])
    if n_subjects:
        participants = replace(participants, n_subjects=n_subjects)
    if participants.health_status and participants.health_status not in (
        "healthy",
        "patients",
    ):
        if not participants.clinical_population:
            participants = replace(
                participants,
                clinical_population=participants.health_status,
                health_status="patients",
            )

    # Experiment
    experiment = metadata.experiment or ExperimentMetadata(
        paradigm=getattr(dataset, "paradigm", "imagery")
    )
    if getattr(dataset, "paradigm", None):
        experiment = replace(experiment, paradigm=dataset.paradigm)
    event_id = getattr(dataset, "event_id", None)
    if event_id:
        class_labels = list(event_id.keys())
        experiment = replace(
            experiment,
            n_classes=len(event_id),
            class_labels=class_labels,
        )

    # Acquisition
    acquisition = metadata.acquisition
    if acquisition is None:
        acquisition = _build_minimal_metadata(dataset).acquisition
    else:
        if not acquisition.channel_types:
            n_channels = (
                acquisition.n_channels or getattr(dataset, "n_channels", None) or 1
            )
            acquisition = replace(acquisition, channel_types={"eeg": int(n_channels)})
        if not acquisition.n_channels:
            n_channels = getattr(
                dataset, "n_channels", None
            ) or acquisition.channel_types.get("eeg", 1)
            acquisition = replace(acquisition, n_channels=int(n_channels))
        if not acquisition.sampling_rate or acquisition.sampling_rate <= 0:
            sampling_rate = (
                getattr(dataset, "sampling_rate", None)
                or getattr(dataset, "sfreq", None)
                or 1.0
            )
            acquisition = replace(acquisition, sampling_rate=float(sampling_rate))

    # Documentation
    documentation = metadata.documentation
    doi = getattr(dataset, "doi", None)
    if doi:
        if documentation is None:
            documentation = DocumentationMetadata(doi=doi)
        elif not documentation.doi:
            documentation = replace(documentation, doi=doi)

    # Sessions and runs
    sessions_per_subject = metadata.sessions_per_subject
    if getattr(dataset, "n_sessions", None):
        sessions_per_subject = dataset.n_sessions

    runs_per_session = metadata.runs_per_session
    if getattr(dataset, "n_runs", None):
        runs_per_session = dataset.n_runs

    # Paradigm-specific
    paradigm_specific = metadata.paradigm_specific
    if paradigm_specific is not None and experiment is not None:
        detected_by_paradigm = {
            "imagery": "motor_imagery",
            "p300": "p300",
            "ssvep": "ssvep",
            "cvep": "cvep",
            "rstate": "resting_state",
        }
        detected = detected_by_paradigm.get(experiment.paradigm, experiment.paradigm)
        paradigm_specific = replace(paradigm_specific, detected_paradigm=detected)

    return replace(
        metadata,
        acquisition=acquisition,
        participants=participants,
        experiment=experiment,
        documentation=documentation,
        paradigm_specific=paradigm_specific,
        sessions_per_subject=sessions_per_subject or 1,
        runs_per_session=runs_per_session or 1,
    )


def _apply_dataset_family_defaults(
    name: str, metadata: DatasetMetadata
) -> DatasetMetadata:
    # ERP CORE defaults
    if name.startswith("ErpCore2021"):
        documentation = metadata.documentation or DocumentationMetadata()
        documentation = replace(documentation, doi="10.1016/j.neuroimage.2020.117465")
        acquisition = metadata.acquisition or AcquisitionMetadata(
            sampling_rate=256.0, n_channels=64, channel_types={"eeg": 64}
        )
        acquisition = replace(acquisition, hardware="Biosemi ActiveTwo")
        participants = metadata.participants or ParticipantMetadata(n_subjects=40)
        participants = replace(participants, n_subjects=40)
        experiment = metadata.experiment or ExperimentMetadata(paradigm="p300")
        experiment = replace(experiment, paradigm="p300")
        metadata = replace(
            metadata,
            documentation=documentation,
            acquisition=acquisition,
            participants=participants,
            experiment=experiment,
        )

    # Castillos cVEP defaults
    if name.startswith("Castillos"):
        documentation = metadata.documentation or DocumentationMetadata()
        documentation = replace(documentation, doi="10.1016/j.neuroimage.2023.120446")
        participants = metadata.participants or ParticipantMetadata(n_subjects=12)
        participants = replace(participants, n_subjects=12)
        experiment = metadata.experiment or ExperimentMetadata(paradigm="cvep")
        experiment = replace(experiment, paradigm="cvep")
        metadata = replace(
            metadata,
            documentation=documentation,
            participants=participants,
            experiment=experiment,
        )

    # MartinezCagigal cVEP defaults
    if name.startswith("MartinezCagigal2023"):
        participants = metadata.participants or ParticipantMetadata(n_subjects=16)
        participants = replace(participants, n_subjects=16)
        experiment = metadata.experiment or ExperimentMetadata(paradigm="cvep")
        experiment = replace(experiment, paradigm="cvep")
        metadata = replace(
            metadata,
            participants=participants,
            experiment=experiment,
        )

    return metadata


def canonicalize_dataset_class_metadata(dataset_name: str, dataset_cls) -> None:
    """Align class-level METADATA with runtime dataset attributes."""
    metadata = getattr(dataset_cls, "METADATA", None)
    if not isinstance(metadata, DatasetMetadata):
        return

    try:
        dataset = dataset_cls()
    except Exception:
        dataset = None

    metadata = _merge_with_dataset(metadata, dataset)
    metadata = _apply_manual_overrides(dataset_name, metadata)
    metadata = _apply_dataset_family_defaults(dataset_name, metadata)
    setattr(dataset_cls, "METADATA", metadata)


def canonicalize_dataset_class_catalog(dataset_classes: dict[str, type]) -> None:
    """Canonicalize METADATA for all dataset classes in a class mapping."""
    for name, dataset_cls in dataset_classes.items():
        if "Fake" in name:
            continue
        canonicalize_dataset_class_metadata(name, dataset_cls)


def _build_dataset_metadata_catalog():
    """Build a catalog of DatasetMetadata from dataset class METADATA attributes."""
    # Import lazily to avoid heavy imports at module load time
    from moabb.datasets.erpcore2021 import ErpCore2021
    from moabb.datasets.utils import _init_dataset, dataset_dict

    if not dataset_dict:
        _init_dataset()

    catalog = {}
    dataset_classes = dict(dataset_dict)
    # Include base ERP CORE dataset to match expected catalog counts
    dataset_classes.setdefault("ErpCore2021", ErpCore2021)

    # Canonicalize class-level METADATA before building the catalog
    canonicalize_dataset_class_catalog(dataset_classes)

    for name, dataset_cls in dataset_classes.items():
        if "Fake" in name:
            continue
        try:
            dataset = dataset_cls()
        except Exception:
            dataset = None

        metadata = getattr(dataset_cls, "METADATA", None)
        if not isinstance(metadata, DatasetMetadata):
            if dataset is None:
                if name == "ErpCore2021":
                    metadata = DatasetMetadata(
                        acquisition=AcquisitionMetadata(
                            sampling_rate=256.0,
                            n_channels=64,
                            channel_types={"eeg": 64},
                            hardware="Biosemi ActiveTwo",
                        ),
                        participants=ParticipantMetadata(n_subjects=40),
                        experiment=ExperimentMetadata(paradigm="p300"),
                        documentation=DocumentationMetadata(
                            doi="10.1016/j.neuroimage.2020.117465"
                        ),
                    )
                else:
                    metadata = _build_fallback_metadata(name)
            else:
                metadata = _build_minimal_metadata(dataset)

        metadata = _merge_with_dataset(metadata, dataset)
        metadata = _apply_manual_overrides(name, metadata)
        metadata = _apply_dataset_family_defaults(name, metadata)

        catalog[name] = metadata
    return catalog


class _LazyMetadataCatalog:
    """Lazy-loading proxy for dataset metadata catalog."""

    def __init__(self):
        self._catalog = None

    def _ensure(self):
        if self._catalog is None:
            self._catalog = _build_dataset_metadata_catalog()
        return self._catalog

    def __len__(self):
        return len(self._ensure())

    def __iter__(self):
        return iter(self._ensure())

    def __getitem__(self, key):
        return self._ensure()[key]

    def __contains__(self, key):
        return key in self._ensure()

    def get(self, key, default=None):
        return self._ensure().get(key, default)

    def items(self):
        return self._ensure().items()

    def values(self):
        return self._ensure().values()

    def keys(self):
        return self._ensure().keys()


DATASET_METADATA_CATALOG = _LazyMetadataCatalog()


def get_dataset_metadata(dataset_name: str) -> DatasetMetadata:
    """Return DatasetMetadata for a dataset class name."""
    if dataset_name not in DATASET_METADATA_CATALOG:
        raise KeyError(f"Dataset metadata not found for {dataset_name}")
    return DATASET_METADATA_CATALOG[dataset_name]


__all__ = [
    # Core MOABB classes
    "AcquisitionMetadata",
    "DocumentationMetadata",
    "ParticipantMetadata",
    "ExperimentMetadata",
    "DatasetMetadata",
    # Additional classes
    "Demographics",
    "ExternalLinks",
    "Timestamps",
    "Tags",
    "ChannelCount",
    "SamplingRateCount",
    # New classes from RALPH extraction
    "AuxiliaryChannelsMetadata",
    "FilterDetails",
    "PreprocessingMetadata",
    "FrequencyBands",
    "SignalProcessingMetadata",
    "CrossValidationMetadata",
    "PerformanceMetadata",
    "BCIApplicationMetadata",
    "ParadigmSpecificMetadata",
    "DataStructureMetadata",
    # Validation functions
    "validate_country_code",
    "validate_metadata_against_dataset",
    "get_dataset_description",
    "DATASET_METADATA_CATALOG",
    "get_dataset_metadata",
    "canonicalize_dataset_class_metadata",
    "canonicalize_dataset_class_catalog",
]
