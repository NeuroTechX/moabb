"""Metadata schema dataclasses for MOABB datasets.

This module provides a standardized way to document dataset metadata.

The schema is organized into logical sections:
- AcquisitionMetadata: Technical recording parameters
- DocumentationMetadata: Publication and provenance information
- ParticipantMetadata: Subject demographics
- ExperimentMetadata: Paradigm and task details
- DatasetMetadata: Top-level container combining all sections

Additional classes:
- Demographics: Extended subject demographics
- ExternalLinks: URLs and data sources
- Timestamps: Dataset creation/modification dates
- Tags: Classification tags
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pycountry


def validate_country_code(code: str) -> bool:
    """Validate that a string is a valid ISO 3166-1 alpha-2 country code.

    Parameters
    ----------
    code : str
        Country code to validate (e.g., "US", "FR", "DE").

    Returns
    -------
    bool
        True if the code is a valid ISO 3166-1 alpha-2 country code.

    Examples
    --------
    >>> validate_country_code("US")
    True
    >>> validate_country_code("FR")
    True
    >>> validate_country_code("XX")
    False
    """
    if not isinstance(code, str) or len(code) != 2:
        return False
    return pycountry.countries.get(alpha_2=code.upper()) is not None


# =============================================================================
# Additional nested classes
# =============================================================================


@dataclass
class Demographics:
    """Subject demographics information following EEGDash schema.

    This extends ParticipantMetadata with additional EEGDash fields.

    Parameters
    ----------
    subjects_count : int
        Number of subjects in the dataset.
    age_min : float, optional
        Minimum age of participants in years.
    age_max : float, optional
        Maximum age of participants in years.
    ages : List[int], optional
        List of ages for each subject.
    gender : Dict[str, int], optional
        Gender distribution, e.g., {"male": 12, "female": 8}.
    handedness : Dict[str, int], optional
        Handedness distribution, e.g., {"right": 18, "left": 2}.
    clinical_population : str, optional
        Clinical diagnosis if patient population.
    """

    subjects_count: int
    age_min: Optional[float] = None
    age_max: Optional[float] = None
    ages: Optional[List[int]] = None
    gender: Optional[Dict[str, int]] = None
    handedness: Optional[Dict[str, int]] = None
    clinical_population: Optional[str] = None


@dataclass
class ExternalLinks:
    """External URLs and data source links following EEGDash schema.

    Parameters
    ----------
    source_url : str, optional
        Primary URL to the dataset source (e.g., OpenNeuro page).
    ftp_url : str, optional
        FTP server URL if available.
    alternative_urls : Dict[str, str], optional
        Additional URLs mapped by name.
    """

    source_url: Optional[str] = None
    ftp_url: Optional[str] = None
    alternative_urls: Optional[Dict[str, str]] = None


@dataclass
class Timestamps:
    """Dataset timestamp information following EEGDash schema.

    Parameters
    ----------
    dataset_created_at : datetime, optional
        When the dataset was first created/published.
    dataset_modified_at : datetime, optional
        When the dataset was last modified.
    ingested_at : datetime, optional
        When the dataset was ingested into the system.
    """

    dataset_created_at: Optional[datetime] = None
    dataset_modified_at: Optional[datetime] = None
    ingested_at: Optional[datetime] = None


@dataclass
class Tags:
    """Classification tags for the dataset following EEGDash schema.

    Parameters
    ----------
    pathology : List[str], optional
        Health status tags, e.g., ["healthy"] or ["epilepsy", "stroke"].
    modality : List[str], optional
        Stimulus/task modality, e.g., ["visual", "auditory", "motor"].
    type : List[str], optional
        Experiment type, e.g., ["perception", "imagery", "resting_state"].
    """

    pathology: Optional[List[str]] = None
    modality: Optional[List[str]] = None
    type: Optional[List[str]] = None


@dataclass
class ChannelCount:
    """Channel count entry for frequency distribution.

    Parameters
    ----------
    val : int
        Number of channels.
    count : int
        Number of recordings with this channel count.
    """

    val: int
    count: int


@dataclass
class SamplingRateCount:
    """Sampling rate count entry for frequency distribution.

    Parameters
    ----------
    val : float
        Sampling rate in Hz.
    count : int
        Number of recordings with this sampling rate.
    """

    val: float
    count: int


# =============================================================================
# Core MOABB metadata classes
# =============================================================================


@dataclass
class AcquisitionMetadata:
    """Technical acquisition parameters.

    Captures hardware, software, and recording settings used during
    data collection.

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

    Captures citation info, data repository links, and institutional details.

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
    senior_author : str, optional
        Senior/corresponding author (EEGDash field).
    contact_info : List[str], optional
        Contact information (EEGDash field).
    associated_paper_doi : str, optional
        DOI for associated publication (EEGDash field).
    funding : List[str], optional
        Funding sources (EEGDash field).
    readme : str, optional
        Dataset README content (EEGDash field).
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
    # EEGDash additional fields
    senior_author: Optional[str] = None
    contact_info: Optional[List[str]] = None
    associated_paper_doi: Optional[str] = None
    funding: Optional[List[str]] = None
    readme: Optional[str] = None


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
    age_min : float, optional
        Minimum age (EEGDash field).
    age_max : float, optional
        Maximum age (EEGDash field).
    ages : List[int], optional
        Per-subject ages (EEGDash field).
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
    # EEGDash additional fields
    age_min: Optional[float] = None
    age_max: Optional[float] = None
    ages: Optional[List[int]] = None
    handedness: Optional[Dict[str, int]] = None
    clinical_population: Optional[str] = None


@dataclass
class ExperimentMetadata:
    """Experimental paradigm and task details.

    Captures the experimental design, event codes, and trial structure.

    Parameters
    ----------
    paradigm : str
        BCI paradigm type: "imagery", "p300", "ssvep", "cvep", "erp", or "rstate".
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
    tasks : List[str], optional
        List of task names (EEGDash field).
    study_design : str, optional
        Study design description (EEGDash field).
    study_domain : str, optional
        Research domain (EEGDash field).
    """

    paradigm: str
    task_type: Optional[str] = None
    events: Dict[str, int] = field(default_factory=dict)
    n_classes: Optional[int] = None
    trials_per_class: Optional[Dict[str, int]] = None
    trial_duration: Optional[float] = None
    # EEGDash additional fields
    tasks: Optional[List[str]] = None
    study_design: Optional[str] = None
    study_domain: Optional[str] = None


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
    sessions : List[str], optional
        List of session identifiers.
    contributing_labs : List[str], optional
        Contributing laboratories.
    n_contributing_labs : int, optional
        Number of contributing labs.
    data_processed : bool
        Whether data has been preprocessed. Default is False.
    external_links : ExternalLinks, optional
        External URLs and data sources.
    timestamps : Timestamps, optional
        Dataset creation and modification dates.
    tags : Tags, optional
        Classification tags.
    nchans_counts : List[ChannelCount], optional
        Distribution of channel counts across recordings.
    sfreq_counts : List[SamplingRateCount], optional
        Distribution of sampling rates across recordings.

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

    # Core MOABB fields (required)
    acquisition: AcquisitionMetadata
    participants: ParticipantMetadata
    experiment: ExperimentMetadata
    documentation: Optional[DocumentationMetadata] = None
    sessions_per_subject: int = 1
    runs_per_session: int = 1

    # Sessions
    sessions: Optional[List[str]] = None

    # Institutions
    contributing_labs: Optional[List[str]] = None
    n_contributing_labs: Optional[int] = None

    # Processing status
    data_processed: bool = False

    # Nested objects
    external_links: Optional[ExternalLinks] = None
    timestamps: Optional[Timestamps] = None
    tags: Optional[Tags] = None

    # Distributions
    nchans_counts: Optional[List[ChannelCount]] = None
    sfreq_counts: Optional[List[SamplingRateCount]] = None


def validate_metadata_against_dataset(dataset, metadata: DatasetMetadata) -> List[str]:
    """Validate metadata matches actual dataset structure.

    Compares metadata values against properties of a dataset object to ensure
    consistency between documented metadata and actual data.

    Parameters
    ----------
    dataset : BaseDataset
        The dataset object to validate against.
    metadata : DatasetMetadata
        The metadata to validate.

    Returns
    -------
    List[str]
        List of validation error messages. Empty if validation passes.

    Examples
    --------
    >>> from moabb.datasets import BNCI2014_001
    >>> from moabb.datasets.metadata import get_dataset_metadata
    >>> dataset = BNCI2014_001()
    >>> metadata = get_dataset_metadata("BNCI2014_001")
    >>> errors = validate_metadata_against_dataset(dataset, metadata)
    >>> if errors:
    ...     print("Validation errors:", errors)
    """
    errors = []

    # Validate sessions_per_subject
    if hasattr(dataset, "n_sessions"):
        if metadata.sessions_per_subject != dataset.n_sessions:
            errors.append(
                f"sessions_per_subject mismatch: metadata={metadata.sessions_per_subject}, "
                f"dataset={dataset.n_sessions}"
            )

    # Validate n_subjects
    if hasattr(dataset, "subject_list"):
        n_subjects = len(dataset.subject_list)
        if metadata.participants.n_subjects != n_subjects:
            errors.append(
                f"n_subjects mismatch: metadata={metadata.participants.n_subjects}, "
                f"dataset={n_subjects}"
            )

    # Validate country code if present
    if (
        metadata.documentation
        and metadata.documentation.country
        and not validate_country_code(metadata.documentation.country)
    ):
        errors.append(
            f"Invalid country code: {metadata.documentation.country}. "
            "Must be a valid ISO 3166-1 alpha-2 code."
        )

    return errors


def get_dataset_description(dataset_class) -> Optional[str]:
    """Extract description from dataset class docstring.

    Parameters
    ----------
    dataset_class : type
        The dataset class to extract description from.

    Returns
    -------
    str or None
        The first paragraph of the docstring as description, or None if
        no docstring is available.

    Examples
    --------
    >>> from moabb.datasets import BNCI2014_001
    >>> desc = get_dataset_description(BNCI2014_001)
    >>> print(desc[:50])
    BNCI 2014-001 Motor Imagery dataset.
    """
    if dataset_class.__doc__:
        # Parse first paragraph as abstract
        doc = dataset_class.__doc__.strip()
        # Split on double newlines to get paragraphs
        paragraphs = doc.split("\n\n")
        if paragraphs:
            # Clean up the first paragraph
            first_para = paragraphs[0].strip()
            # Remove any leading indentation from subsequent lines
            lines = first_para.split("\n")
            cleaned_lines = [lines[0]] + [line.strip() for line in lines[1:]]
            return " ".join(cleaned_lines)
    return None
