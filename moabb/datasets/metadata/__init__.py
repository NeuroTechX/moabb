"""Metadata schema module for MOABB datasets.

This module provides standardized dataclasses for documenting dataset metadata.

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

Functions
---------
get_dataset_metadata
    Retrieve pre-defined metadata for a specific MOABB dataset
validate_country_code
    Validate ISO 3166-1 alpha-2 country codes
validate_metadata_against_dataset
    Validate metadata matches actual dataset structure
get_dataset_description
    Extract description from dataset class docstring

Constants
---------
DATASET_METADATA_CATALOG
    Dictionary mapping dataset names to their DatasetMetadata instances

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

>>> # Get pre-defined metadata for a dataset
>>> from moabb.datasets.metadata import get_dataset_metadata
>>> bnci_metadata = get_dataset_metadata("BNCI2014_001")
>>> print(bnci_metadata.participants.n_subjects)
9
"""

from .catalog import DATASET_METADATA_CATALOG, get_dataset_metadata
from .schema import (
    AcquisitionMetadata,
    ChannelCount,
    DatasetMetadata,
    Demographics,
    DocumentationMetadata,
    ExperimentMetadata,
    ExternalLinks,
    ParticipantMetadata,
    SamplingRateCount,
    Tags,
    Timestamps,
    get_dataset_description,
    validate_country_code,
    validate_metadata_against_dataset,
)


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
    # Catalog
    "DATASET_METADATA_CATALOG",
    "get_dataset_metadata",
    # Validation functions
    "validate_country_code",
    "validate_metadata_against_dataset",
    "get_dataset_description",
]
