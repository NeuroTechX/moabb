"""Metadata schema dataclasses for MOABB datasets.

This module provides a standardized way to document dataset metadata,
inspired by the Eegle.jl library's YAML-based metadata format.

The schema is organized into logical sections:
- AcquisitionMetadata: Technical recording parameters
- DocumentationMetadata: Publication and provenance information
- ParticipantMetadata: Subject demographics
- ExperimentMetadata: Paradigm and task details
- DatasetMetadata: Top-level container combining all sections
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AcquisitionMetadata:
    """Technical acquisition parameters.

    Inspired by Eegle.jl acquisition section. Captures hardware, software,
    and recording settings used during data collection.

    Parameters
    ----------
    sampling_rate : float
        Sampling frequency in Hz.
    n_channels : int
        Total number of recorded channels.
    channel_types : Dict[str, int]
        Channel type counts, e.g., {"eeg": 60, "eog": 4}.
    sensors : List[str], optional
        List of sensor/channel names. Default is empty list.
    sensor_type : str, optional
        Electrode type, e.g., "Ag/AgCl wet", "dry", "active".
    reference : str, optional
        Reference electrode(s) used, e.g., "earlobes", "Cz", "average".
    ground : str, optional
        Ground electrode location.
    hardware : str, optional
        Recording system/amplifier, e.g., "BrainAmp DC", "g.USBamp".
    software : str, optional
        Recording software used.
    filters : str, optional
        Online filters applied during recording, e.g., "0.1-100 Hz bandpass".
    line_freq : float
        Power line frequency in Hz. Default is 50.0.
    montage : str
        Standard montage name for channel positions. Default is "standard_1005".
    """

    sampling_rate: float
    n_channels: int
    channel_types: Dict[str, int]
    sensors: List[str] = field(default_factory=list)
    sensor_type: Optional[str] = None
    reference: Optional[str] = None
    ground: Optional[str] = None
    hardware: Optional[str] = None
    software: Optional[str] = None
    filters: Optional[str] = None
    line_freq: float = 50.0
    montage: str = "standard_1005"


@dataclass
class DocumentationMetadata:
    """Publication and dataset provenance information.

    Inspired by Eegle.jl documentation section. Captures citation info,
    data repository links, and institutional details.

    Parameters
    ----------
    doi : str, optional
        Digital Object Identifier for the dataset or associated publication.
    description : str, optional
        Brief description of the dataset.
    investigators : List[str], optional
        Names of principal investigators or dataset creators.
    institution : str, optional
        Institution where data was collected.
    country : str, optional
        Country where data was collected.
    repository : str, optional
        Data repository name, e.g., "BNCI Horizon 2020", "PhysioNet".
    data_url : str, optional
        URL to download the dataset.
    license : str, optional
        Data license, e.g., "CC BY 4.0", "ODC-BY".
    publication_year : int, optional
        Year of dataset publication or associated paper.
    """

    doi: Optional[str] = None
    description: Optional[str] = None
    investigators: Optional[List[str]] = None
    institution: Optional[str] = None
    country: Optional[str] = None
    repository: Optional[str] = None
    data_url: Optional[str] = None
    license: Optional[str] = None
    publication_year: Optional[int] = None


@dataclass
class ParticipantMetadata:
    """Participant demographics information.

    Captures subject pool characteristics including sample size,
    demographics, and health status.

    Parameters
    ----------
    n_subjects : int
        Number of subjects in the dataset.
    health_status : str
        General health status, e.g., "healthy", "patients", "mixed".
        Default is "healthy".
    gender : Dict[str, int], optional
        Gender distribution, e.g., {"male": 12, "female": 8}.
    age_mean : float, optional
        Mean age of participants in years.
    age_std : float, optional
        Standard deviation of participant ages.
    handedness : Dict[str, int], optional
        Handedness distribution, e.g., {"right": 18, "left": 2}.
    clinical_population : str, optional
        Clinical diagnosis if patient population,
        e.g., "stroke", "ALS", "spinal cord injury".
    """

    n_subjects: int
    health_status: str = "healthy"
    gender: Optional[Dict[str, int]] = None
    age_mean: Optional[float] = None
    age_std: Optional[float] = None
    handedness: Optional[Dict[str, int]] = None
    clinical_population: Optional[str] = None


@dataclass
class ExperimentMetadata:
    """Experimental paradigm and task details.

    Inspired by Eegle.jl id and stim sections. Captures the experimental
    design, event codes, and trial structure.

    Parameters
    ----------
    paradigm : str
        BCI paradigm type: "imagery", "p300", "ssvep", "cvep", or "rstate".
    task_type : str, optional
        Specific task variant, e.g., "left_right_hand", "4_class",
        "row_col_speller".
    events : Dict[str, int]
        Event name to code mapping, e.g., {"left_hand": 1, "right_hand": 2}.
        Default is empty dict.
    n_classes : int, optional
        Number of classes/conditions.
    trials_per_class : Dict[str, int], optional
        Number of trials per class/condition.
    trial_duration : float, optional
        Duration of each trial in seconds.
    """

    paradigm: str
    task_type: Optional[str] = None
    events: Dict[str, int] = field(default_factory=dict)
    n_classes: Optional[int] = None
    trials_per_class: Optional[Dict[str, int]] = None
    trial_duration: Optional[float] = None


@dataclass
class DatasetMetadata:
    """Complete dataset metadata combining all sections.

    This is the top-level container that aggregates all metadata sections
    into a single, comprehensive dataset description.

    Parameters
    ----------
    acquisition : AcquisitionMetadata
        Technical acquisition parameters.
    participants : ParticipantMetadata
        Participant demographics information.
    experiment : ExperimentMetadata
        Experimental paradigm details.
    documentation : DocumentationMetadata, optional
        Publication and provenance information.
    sessions_per_subject : int
        Number of sessions per subject. Default is 1.
    runs_per_session : int
        Number of runs per session. Default is 1.
    format_version : str
        Metadata schema version. Default is "1.0.0".

    Examples
    --------
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
    ...     experiment=ExperimentMetadata(
    ...         paradigm="imagery",
    ...         events={"left_hand": 1, "right_hand": 2},
    ...     ),
    ... )
    """

    acquisition: AcquisitionMetadata
    participants: ParticipantMetadata
    experiment: ExperimentMetadata
    documentation: Optional[DocumentationMetadata] = None
    sessions_per_subject: int = 1
    runs_per_session: int = 1
    format_version: str = "1.0.0"
