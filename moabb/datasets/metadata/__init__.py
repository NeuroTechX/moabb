"""Metadata schema module for MOABB datasets.

This module provides standardized dataclasses for documenting dataset metadata,
combining MOABB's paradigm-focused structure with EEGDash's comprehensive
schema for compatibility with the broader EEG data ecosystem.

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

EEGDash-Compatible Classes
--------------------------
Demographics
    Extended subject demographics (subjects_count, ages, age_min, age_max)
ExternalLinks
    URLs and data source links
Timestamps
    Dataset creation and modification dates
Tags
    Classification tags with confidence scores
TagConfidence
    Confidence scores for each tag category
TagReasoning
    Reasoning explanations for tag assignments
ChannelCount
    Channel count distribution entry
SamplingRateCount
    Sampling rate distribution entry

Functions
---------
get_dataset_metadata
    Retrieve pre-defined metadata for a specific MOABB dataset

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
...     dataset_id="my_dataset",
...     source="OpenNeuro",
... )

>>> # Get pre-defined metadata for a dataset
>>> from moabb.datasets.metadata import get_dataset_metadata
>>> bnci_metadata = get_dataset_metadata("BNCI2014_001")
>>> print(bnci_metadata.participants.n_subjects)
9

References
----------
- EEGDash API: https://eegdash.org/
- EEGDash Data API: https://data.eegdash.org/api/eegdash/datasets/summary/
"""

from .catalog import DATASET_METADATA_CATALOG, get_dataset_metadata
from .schema import (  # Core MOABB classes; EEGDash-compatible classes
    AcquisitionMetadata,
    ChannelCount,
    DatasetMetadata,
    Demographics,
    DocumentationMetadata,
    ExperimentMetadata,
    ExternalLinks,
    ParticipantMetadata,
    SamplingRateCount,
    TagConfidence,
    TagReasoning,
    Tags,
    Timestamps,
)


__all__ = [
    # Core MOABB classes
    "AcquisitionMetadata",
    "DocumentationMetadata",
    "ParticipantMetadata",
    "ExperimentMetadata",
    "DatasetMetadata",
    # EEGDash-compatible classes
    "Demographics",
    "ExternalLinks",
    "Timestamps",
    "Tags",
    "TagConfidence",
    "TagReasoning",
    "ChannelCount",
    "SamplingRateCount",
    # Catalog
    "DATASET_METADATA_CATALOG",
    "get_dataset_metadata",
]
