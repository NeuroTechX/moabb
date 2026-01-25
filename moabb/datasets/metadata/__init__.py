"""Metadata schema module for MOABB datasets.

This module provides standardized dataclasses for documenting dataset metadata,
inspired by the Eegle.jl library's comprehensive YAML-based metadata format.

Classes
-------
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
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
)


__all__ = [
    "AcquisitionMetadata",
    "DocumentationMetadata",
    "ParticipantMetadata",
    "ExperimentMetadata",
    "DatasetMetadata",
    "DATASET_METADATA_CATALOG",
    "get_dataset_metadata",
]
