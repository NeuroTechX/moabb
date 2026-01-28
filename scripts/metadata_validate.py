#!/usr/bin/env python
"""Async metadata validation against actual data files.

This script validates catalog metadata against actual data files in ~/mne_data,
comparing documented values with values extracted from MNE Raw objects.

Requirements:
    pip install mne

Usage:
    python scripts/metadata_validate.py --output results/metadata_validation_report.md
    python scripts/metadata_validate.py --dataset BNCI2014_001
    python scripts/metadata_validate.py --skip-urls --format json
    python scripts/metadata_validate.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# Add parent directory to path if running as script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Known valid values for validation
VALID_PARADIGMS = frozenset({"imagery", "p300", "ssvep", "cvep", "erp", "rstate"})
VALID_HEALTH_STATUS = frozenset({"healthy", "patients", "mixed"})

# Known SPDX license identifiers (subset)
VALID_LICENSES = frozenset(
    {
        "CC0",
        "CC0-1.0",
        "CC BY 4.0",
        "CC-BY-4.0",
        "CC BY-NC 4.0",
        "CC-BY-NC-4.0",
        "CC BY-SA 4.0",
        "CC-BY-SA-4.0",
        "ODC-BY",
        "ODbL",
        "PDDL",
        "MIT",
        "Apache-2.0",
        "GPL-3.0",
        "Open Data Commons Attribution License v1.0",
    }
)

# ISO 3166-1 country codes (subset of common ones)
VALID_COUNTRIES = frozenset(
    {
        "Austria",
        "Belgium",
        "Brazil",
        "Canada",
        "China",
        "France",
        "Germany",
        "India",
        "Italy",
        "Japan",
        "Netherlands",
        "Poland",
        "Portugal",
        "Russia",
        "South Korea",
        "Spain",
        "Sweden",
        "Switzerland",
        "Taiwan",
        "UK",
        "USA",
        "United Kingdom",
        "United States",
    }
)


class Severity(str, Enum):
    """Issue severity levels."""

    ERROR = "error"  # Critical mismatch
    WARNING = "warning"  # Potential issue
    INFO = "info"  # Informational / suggestion


@dataclass
class ValidationIssue:
    """A single validation issue."""

    field: str
    severity: Severity
    message: str
    catalog_value: Any = None
    actual_value: Any = None
    suggestion: Optional[str] = None


@dataclass
class DatasetValidationResult:
    """Validation result for a single dataset."""

    dataset_name: str
    issues: List[ValidationIssue] = field(default_factory=list)
    data_available: bool = False
    data_path: Optional[Path] = None
    extracted_values: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)


@dataclass
class FullValidationReport:
    """Complete validation report for all datasets."""

    results: List[DatasetValidationResult]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_datasets(self) -> int:
        return len(self.results)

    @property
    def datasets_with_errors(self) -> int:
        return sum(1 for r in self.results if r.has_errors)

    @property
    def datasets_with_warnings(self) -> int:
        return sum(1 for r in self.results if r.has_warnings)

    @property
    def datasets_with_data(self) -> int:
        return sum(1 for r in self.results if r.data_available)

    @property
    def clean_datasets(self) -> int:
        return sum(1 for r in self.results if not r.issues)


# =============================================================================
# Data Path Resolution
# =============================================================================


def get_mne_data_path() -> Path:
    """Get the MNE data directory path."""
    # Check environment variable first
    mne_data = os.environ.get("MNE_DATA")
    if mne_data:
        return Path(mne_data)

    # Default to ~/mne_data
    return Path.home() / "mne_data"


# Mapping from catalog dataset names to data directory patterns
DATASET_DATA_PATHS = {
    "AlexMI": ["MNE-alexeeg-data"],
    "BNCI2014_001": ["MNE-bnci-data/001-2014", "MNE-BIDS-bnci2014-001"],
    "BNCI2014_002": ["MNE-bnci-data/002-2014"],
    "BNCI2014_004": ["MNE-bnci-data/004-2014"],
    "BNCI2014_008": ["MNE-bnci-data/008-2014"],
    "BNCI2014_009": ["MNE-bnci-data/009-2014"],
    "BNCI2015_001": ["MNE-bnci-data/001-2015"],
    "BNCI2015_003": ["MNE-bnci-data/003-2015"],
    "BNCI2015_004": ["MNE-bnci-data/004-2015"],
    "BNCI2015_006": ["MNE-bnci-data/006-2015"],
    "BNCI2015_007": ["MNE-bnci-data/007-2015"],
    "BNCI2015_008": ["MNE-bnci-data/008-2015"],
    "BNCI2015_009": ["MNE-bnci-data/009-2015"],
    "BNCI2015_010": ["MNE-bnci-data/010-2015"],
    "BNCI2015_012": ["MNE-bnci-data/012-2015"],
    "BNCI2015_013": ["MNE-bnci-data/013-2015"],
    "BNCI2016_002": ["MNE-bnci-data/002-2016"],
    "BNCI2019_001": ["MNE-bnci-data/001-2019"],
    "BNCI2020_001": ["MNE-bnci-data/001-2020"],
    "BNCI2020_002": ["MNE-bnci-data/002-2020"],
    "BNCI2003_004": ["MNE-bnci-data/004-2003"],
    "BNCI2022_001": ["MNE-bnci-data/001-2022"],
    "BNCI2024_001": ["MNE-bnci-data/001-2024"],
    "BNCI2025_001": ["MNE-bnci-data/001-2025"],
    "BNCI2025_002": ["MNE-bnci-data/002-2025"],
    "Beetl2021_A": ["MNE-Beetl2021-A-data"],
    "Beetl2021_B": ["MNE-Beetl2021-B-data"],
    "Cho2017": ["MNE-gigadb-data"],
    "Dreyer2023": ["MNE-Dreyer2023-data"],
    "Dreyer2023A": ["MNE-Dreyer2023A-data"],
    "Dreyer2023B": ["MNE-Dreyer2023B-data"],
    "Dreyer2023C": ["MNE-Dreyer2023C-data"],
    "Lee2019_MI": ["MNE-lee2019-mi-data"],
    "Liu2024": ["MNE-liu2024-data"],
    "PhysionetMI": ["MNE-eegbci-data"],
    "Schirrmeister2017": ["MNE-schirrmeister2017-data"],
    "Stieger2021": ["MNE-Stieger2021-data"],
    "Weibo2014": ["MNE-weibo-2014"],
    "Zhou2016": ["MNE-BIDS-zhou2016"],
    "BI2012": ["MNE-braininvaders2012-data"],
    "BI2013a": ["MNE-braininvaders2013a-data"],
    "BI2014a": ["MNE-braininvaders2014a-data"],
    "BI2014b": ["MNE-braininvaders2014b-data"],
    "BI2015a": ["MNE-braininvaders2015a-data"],
    "BI2015b": ["MNE-braininvaders2015b-data"],
    "Cattan2019_VR": ["MNE-virtualreality-data"],
    "Cattan2019_PHMD": ["MNE-headmounted-data"],
    "Hinss2021": ["MNE-neuroergonomics2021-data"],
    "MAMEM1": ["MNE-ssvepexo-data"],
    "MAMEM2": ["MNE-ssvepexo-data"],
    "MAMEM3": ["MNE-ssvepexo-data"],
    # ERPCore datasets
    "ErpCore2021_ERN": ["MNE-erpcorep32021-data", "ds005505-bdf"],
    "ErpCore2021_LRP": ["MNE-erpcorep32021-data", "ds005506-bdf"],
    "ErpCore2021_MMN": ["MNE-erpcorep32021-data", "ds005507-bdf"],
    "ErpCore2021_N170": ["MNE-erpcorep32021-data", "ds005508-bdf"],
    "ErpCore2021_N2pc": ["MNE-erpcorep32021-data", "ds005509-bdf"],
    "ErpCore2021_N400": ["MNE-erpcorep32021-data", "ds005510-bdf"],
    "ErpCore2021_P3": ["MNE-erpcorep32021-data", "ds005511-bdf"],
}


def find_data_path(dataset_name: str) -> Optional[Path]:
    """Find the data path for a dataset if it exists."""
    mne_data = get_mne_data_path()

    # Check specific paths first
    if dataset_name in DATASET_DATA_PATHS:
        for path_suffix in DATASET_DATA_PATHS[dataset_name]:
            path = mne_data / path_suffix
            if path.exists():
                return path

    # Try common patterns
    patterns = [
        f"MNE-{dataset_name.lower()}-data",
        f"MNE-{dataset_name}-data",
        dataset_name.lower(),
    ]

    for pattern in patterns:
        path = mne_data / pattern
        if path.exists():
            return path

    return None


# =============================================================================
# Data Loading via MOABB
# =============================================================================


def load_sample_raw_via_moabb(dataset_name: str) -> Optional[Any]:
    """Load a sample Raw object using MOABB's dataset classes.

    This ensures we validate against the actual data that MOABB uses.
    Returns None if loading fails.
    """
    try:
        import mne

        mne.set_log_level("ERROR")

        # Import dataset classes
        from moabb.datasets.utils import dataset_list

        # Find the dataset class
        dataset_class = None
        for ds_class in dataset_list:
            if (
                ds_class.__name__ == dataset_name
                or getattr(ds_class, "code", None) == dataset_name
            ):
                dataset_class = ds_class
                break

        if dataset_class is None:
            return None

        # Instantiate and get data for first subject
        ds_instance = dataset_class()
        if not ds_instance.subject_list:
            return None

        first_subject = ds_instance.subject_list[0]
        data = ds_instance.get_data(subjects=[first_subject])

        # Extract first Raw object
        for subj, sessions in data.items():
            for sess, runs in sessions.items():
                for run, raw in runs.items():
                    return raw

        return None

    except Exception:
        # Silently fail - data may not be downloaded
        return None


def extract_raw_metadata(raw: Any) -> Dict[str, Any]:
    """Extract metadata from an MNE Raw object."""
    extracted = {}

    # Sampling rate
    extracted["sampling_rate"] = raw.info["sfreq"]

    # Channel count
    extracted["n_channels"] = len(raw.ch_names)

    # Channel names
    extracted["sensors"] = list(raw.ch_names)

    # Channel types
    try:
        ch_types = raw.get_channel_types()
        type_counts = {}
        for ch_type in ch_types:
            type_counts[ch_type] = type_counts.get(ch_type, 0) + 1
        extracted["channel_types"] = type_counts
    except Exception:
        pass

    # Line frequency
    if raw.info.get("line_freq"):
        extracted["line_freq"] = raw.info["line_freq"]

    # Montage/digitization
    if raw.info.get("dig"):
        extracted["has_montage"] = True
    else:
        extracted["has_montage"] = False

    return extracted


# =============================================================================
# Validation Functions
# =============================================================================


def validate_acquisition_metadata(
    catalog_meta, extracted: Dict[str, Any], issues: List[ValidationIssue]
) -> None:
    """Validate acquisition metadata against extracted values."""
    acq = catalog_meta.acquisition

    # Sampling rate
    if "sampling_rate" in extracted:
        catalog_sr = acq.sampling_rate
        actual_sr = extracted["sampling_rate"]
        if catalog_sr != actual_sr:
            # Allow small tolerance for floating point
            if abs(catalog_sr - actual_sr) > 0.1:
                issues.append(
                    ValidationIssue(
                        field="acquisition.sampling_rate",
                        severity=Severity.ERROR,
                        message=f"Sampling rate mismatch: catalog={catalog_sr}, actual={actual_sr}",
                        catalog_value=catalog_sr,
                        actual_value=actual_sr,
                        suggestion=f"Update catalog: sampling_rate={actual_sr}",
                    )
                )

    # Number of channels
    if "n_channels" in extracted:
        catalog_nc = acq.n_channels
        actual_nc = extracted["n_channels"]
        if catalog_nc != actual_nc:
            issues.append(
                ValidationIssue(
                    field="acquisition.n_channels",
                    severity=Severity.WARNING,
                    message=f"Channel count mismatch: catalog={catalog_nc}, actual={actual_nc}",
                    catalog_value=catalog_nc,
                    actual_value=actual_nc,
                    suggestion=f"Update catalog: n_channels={actual_nc}",
                )
            )

    # Channel types consistency
    if acq.channel_types:
        total_from_types = sum(acq.channel_types.values())
        if total_from_types != acq.n_channels:
            issues.append(
                ValidationIssue(
                    field="acquisition.channel_types",
                    severity=Severity.WARNING,
                    message=f"Channel type sum ({total_from_types}) != n_channels ({acq.n_channels})",
                    catalog_value=acq.channel_types,
                    actual_value=acq.n_channels,
                )
            )

    # Line frequency validation
    if acq.line_freq not in [50.0, 60.0]:
        issues.append(
            ValidationIssue(
                field="acquisition.line_freq",
                severity=Severity.WARNING,
                message=f"Unusual line frequency: {acq.line_freq} (expected 50 or 60 Hz)",
                catalog_value=acq.line_freq,
            )
        )


def validate_participants_metadata(
    catalog_meta, dataset_class: Optional[Any], issues: List[ValidationIssue]
) -> None:
    """Validate participant metadata."""
    part = catalog_meta.participants

    # n_subjects consistency with dataset class
    if dataset_class is not None:
        try:
            actual_n = len(dataset_class.subject_list)
            if part.n_subjects != actual_n:
                issues.append(
                    ValidationIssue(
                        field="participants.n_subjects",
                        severity=Severity.ERROR,
                        message=f"Subject count mismatch: catalog={part.n_subjects}, dataset_class={actual_n}",
                        catalog_value=part.n_subjects,
                        actual_value=actual_n,
                        suggestion=f"Update catalog: n_subjects={actual_n}",
                    )
                )
        except Exception:
            pass

    # Gender distribution consistency
    if part.gender:
        gender_total = sum(part.gender.values())
        if gender_total != part.n_subjects:
            issues.append(
                ValidationIssue(
                    field="participants.gender",
                    severity=Severity.WARNING,
                    message=f"Gender sum ({gender_total}) != n_subjects ({part.n_subjects})",
                    catalog_value=part.gender,
                    actual_value=part.n_subjects,
                )
            )

    # Health status validation
    if part.health_status not in VALID_HEALTH_STATUS:
        issues.append(
            ValidationIssue(
                field="participants.health_status",
                severity=Severity.WARNING,
                message=f"Unknown health status: '{part.health_status}'",
                catalog_value=part.health_status,
                suggestion=f"Use one of: {', '.join(sorted(VALID_HEALTH_STATUS))}",
            )
        )

    # Age consistency
    if part.age_mean is not None and part.age_std is not None:
        if part.age_min is not None and part.age_max is not None:
            # Basic sanity check
            if part.age_min > part.age_mean or part.age_max < part.age_mean:
                issues.append(
                    ValidationIssue(
                        field="participants.age",
                        severity=Severity.WARNING,
                        message=f"Age mean ({part.age_mean}) outside min/max range ({part.age_min}-{part.age_max})",
                        catalog_value={
                            "mean": part.age_mean,
                            "min": part.age_min,
                            "max": part.age_max,
                        },
                    )
                )


def validate_experiment_metadata(
    catalog_meta, dataset_class: Optional[Any], issues: List[ValidationIssue]
) -> None:
    """Validate experiment metadata."""
    exp = catalog_meta.experiment

    # Paradigm validation
    if exp.paradigm not in VALID_PARADIGMS:
        issues.append(
            ValidationIssue(
                field="experiment.paradigm",
                severity=Severity.ERROR,
                message=f"Unknown paradigm: '{exp.paradigm}'",
                catalog_value=exp.paradigm,
                suggestion=f"Use one of: {', '.join(sorted(VALID_PARADIGMS))}",
            )
        )

    # Event consistency with dataset class
    if dataset_class is not None:
        try:
            class_events = dataset_class.event_id
            if exp.events and class_events:
                catalog_keys = set(exp.events.keys())
                class_keys = set(class_events.keys())

                if catalog_keys != class_keys:
                    missing_in_catalog = class_keys - catalog_keys
                    extra_in_catalog = catalog_keys - class_keys

                    if missing_in_catalog:
                        issues.append(
                            ValidationIssue(
                                field="experiment.events",
                                severity=Severity.WARNING,
                                message=f"Events missing from catalog: {missing_in_catalog}",
                                catalog_value=exp.events,
                                actual_value=class_events,
                            )
                        )
                    if extra_in_catalog:
                        issues.append(
                            ValidationIssue(
                                field="experiment.events",
                                severity=Severity.WARNING,
                                message=f"Extra events in catalog not in class: {extra_in_catalog}",
                                catalog_value=exp.events,
                                actual_value=class_events,
                            )
                        )
        except Exception:
            pass

    # n_classes consistency (with exceptions for specific paradigms)
    if exp.events and exp.n_classes is not None:
        n_events = len(exp.events)
        if n_events != exp.n_classes:
            # cVEP paradigms: events are Target/NonTarget, n_classes is symbol count
            # This is expected and not an error
            if exp.paradigm == "cvep" and set(exp.events.keys()) == {
                "Target",
                "NonTarget",
            }:
                pass  # Expected for cVEP
            # "rest" event is often not counted as a class
            elif "rest" in exp.events and n_events - 1 == exp.n_classes:
                pass  # Expected when rest is not a class
            else:
                issues.append(
                    ValidationIssue(
                        field="experiment.n_classes",
                        severity=Severity.WARNING,
                        message=f"n_classes ({exp.n_classes}) != len(events) ({n_events})",
                        catalog_value=exp.n_classes,
                        actual_value=n_events,
                    )
                )


def validate_documentation_metadata(catalog_meta, issues: List[ValidationIssue]) -> None:
    """Validate documentation metadata."""
    doc = catalog_meta.documentation

    if doc is None:
        issues.append(
            ValidationIssue(
                field="documentation",
                severity=Severity.WARNING,
                message="No documentation metadata provided",
            )
        )
        return

    # DOI format validation
    if doc.doi:
        doi_pattern = re.compile(r"^10\.\d{4,9}/[^\s]+$")
        # Clean DOI
        cleaned_doi = doc.doi
        for prefix in [
            "https://doi.org/",
            "http://doi.org/",
            "doi.org/",
            "doi:",
            "DOI:",
        ]:
            if cleaned_doi.startswith(prefix):
                cleaned_doi = cleaned_doi[len(prefix) :]

        if not doi_pattern.match(cleaned_doi):
            issues.append(
                ValidationIssue(
                    field="documentation.doi",
                    severity=Severity.ERROR,
                    message=f"Invalid DOI format: '{doc.doi}'",
                    catalog_value=doc.doi,
                    suggestion="DOI should match pattern: 10.XXXX/...",
                )
            )
    else:
        issues.append(
            ValidationIssue(
                field="documentation.doi",
                severity=Severity.INFO,
                message="No DOI provided",
            )
        )

    # License validation
    if doc.license:
        if doc.license not in VALID_LICENSES:
            issues.append(
                ValidationIssue(
                    field="documentation.license",
                    severity=Severity.INFO,
                    message=f"Non-standard license identifier: '{doc.license}'",
                    catalog_value=doc.license,
                    suggestion="Consider using SPDX identifier",
                )
            )
    else:
        issues.append(
            ValidationIssue(
                field="documentation.license",
                severity=Severity.INFO,
                message="No license specified",
            )
        )

    # Country validation
    if doc.country and doc.country not in VALID_COUNTRIES:
        issues.append(
            ValidationIssue(
                field="documentation.country",
                severity=Severity.INFO,
                message=f"Unrecognized country: '{doc.country}'",
                catalog_value=doc.country,
            )
        )


def validate_cross_field_consistency(catalog_meta, issues: List[ValidationIssue]) -> None:
    """Validate consistency across different metadata sections."""
    # Sessions consistency
    if hasattr(catalog_meta, "sessions_per_subject"):
        if catalog_meta.sessions_per_subject < 1:
            issues.append(
                ValidationIssue(
                    field="sessions_per_subject",
                    severity=Severity.ERROR,
                    message=f"Invalid sessions_per_subject: {catalog_meta.sessions_per_subject}",
                    catalog_value=catalog_meta.sessions_per_subject,
                )
            )

    # Runs consistency
    if hasattr(catalog_meta, "runs_per_session"):
        if catalog_meta.runs_per_session < 1:
            issues.append(
                ValidationIssue(
                    field="runs_per_session",
                    severity=Severity.ERROR,
                    message=f"Invalid runs_per_session: {catalog_meta.runs_per_session}",
                    catalog_value=catalog_meta.runs_per_session,
                )
            )


# =============================================================================
# Main Validation Logic
# =============================================================================


def validate_dataset_sync(
    dataset_name: str, catalog_meta, verbose: bool = False
) -> DatasetValidationResult:
    """Validate a single dataset (synchronous version for process pool)."""
    result = DatasetValidationResult(dataset_name=dataset_name)
    issues = []

    # Try to load data via MOABB (this properly loads the actual data)
    raw = load_sample_raw_via_moabb(dataset_name)
    if raw is not None:
        result.data_available = True
        extracted = extract_raw_metadata(raw)
        result.extracted_values = extracted

        # Validate against extracted values
        validate_acquisition_metadata(catalog_meta, extracted, issues)
    else:
        # Fallback: check if data path exists (may not be downloaded)
        data_path = find_data_path(dataset_name)
        if data_path:
            result.data_path = data_path

    # Load dataset class if possible (for subject_list, event_id)
    dataset_class = None
    try:
        from moabb.datasets.utils import dataset_list

        for ds_class in dataset_list:
            if (
                ds_class.__name__ == dataset_name
                or getattr(ds_class, "code", None) == dataset_name
            ):
                # Don't instantiate, just use class
                dataset_class = ds_class
                break

        if dataset_class is not None:
            # Instantiate to get subject_list etc.
            try:
                ds_instance = dataset_class()
                validate_participants_metadata(catalog_meta, ds_instance, issues)
                validate_experiment_metadata(catalog_meta, ds_instance, issues)
            except Exception:
                # Validate without instance
                validate_participants_metadata(catalog_meta, None, issues)
                validate_experiment_metadata(catalog_meta, None, issues)
        else:
            validate_participants_metadata(catalog_meta, None, issues)
            validate_experiment_metadata(catalog_meta, None, issues)
    except Exception:
        validate_participants_metadata(catalog_meta, None, issues)
        validate_experiment_metadata(catalog_meta, None, issues)

    # Validate documentation (doesn't need data)
    validate_documentation_metadata(catalog_meta, issues)

    # Cross-field consistency
    validate_cross_field_consistency(catalog_meta, issues)

    result.issues = issues
    return result


async def validate_dataset(
    dataset_name: str,
    catalog_meta,
    executor: ProcessPoolExecutor,
    verbose: bool = False,
) -> DatasetValidationResult:
    """Validate a single dataset asynchronously."""
    loop = asyncio.get_event_loop()

    # Run validation in process pool
    result = await loop.run_in_executor(
        executor, validate_dataset_sync, dataset_name, catalog_meta, verbose
    )

    if verbose:
        status = "OK" if not result.issues else f"{len(result.issues)} issues"
        data_status = "data found" if result.data_available else "no data"
        print(f"  {dataset_name}: {status} ({data_status})", file=sys.stderr)

    return result


async def validate_all_datasets(
    catalog: Dict,
    max_workers: int = 4,
    verbose: bool = False,
) -> FullValidationReport:
    """Validate all datasets in the catalog."""
    results = []

    if verbose:
        print(f"Validating {len(catalog)} datasets...", file=sys.stderr)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = []
        for name, metadata in catalog.items():
            task = validate_dataset(name, metadata, executor, verbose)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    return FullValidationReport(results=list(results))


# =============================================================================
# Report Generation
# =============================================================================


def generate_markdown_report(report: FullValidationReport) -> str:
    """Generate a Markdown report."""
    lines = []

    lines.append("# Metadata Validation Report")
    lines.append("")
    lines.append(f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Datasets | {report.total_datasets} |")
    lines.append(f"| Datasets with Errors | {report.datasets_with_errors} |")
    lines.append(f"| Datasets with Warnings | {report.datasets_with_warnings} |")
    lines.append(f"| Datasets with Data Available | {report.datasets_with_data} |")
    lines.append(f"| Clean (No Issues) | {report.clean_datasets} |")
    lines.append("")

    # Count by severity
    total_errors = sum(r.error_count for r in report.results)
    total_warnings = sum(r.warning_count for r in report.results)
    total_info = sum(
        sum(1 for i in r.issues if i.severity == Severity.INFO) for r in report.results
    )

    lines.append("### Issue Counts by Severity")
    lines.append("")
    lines.append("| Severity | Count |")
    lines.append("|----------|-------|")
    lines.append(f"| Errors | {total_errors} |")
    lines.append(f"| Warnings | {total_warnings} |")
    lines.append(f"| Info | {total_info} |")
    lines.append("")

    # Datasets with errors
    error_datasets = [r for r in report.results if r.has_errors]
    if error_datasets:
        lines.append("## Datasets with Errors")
        lines.append("")
        for result in sorted(error_datasets, key=lambda r: r.dataset_name):
            lines.append(f"### {result.dataset_name}")
            lines.append("")
            if result.data_available:
                lines.append(f"Data path: `{result.data_path}`")
                lines.append("")
            for issue in result.issues:
                if issue.severity == Severity.ERROR:
                    lines.append(f"- **ERROR** `{issue.field}`: {issue.message}")
                    if issue.suggestion:
                        lines.append(f"  - Suggestion: {issue.suggestion}")
            lines.append("")

    # Datasets with warnings only (no errors)
    warning_only = [r for r in report.results if r.has_warnings and not r.has_errors]
    if warning_only:
        lines.append("## Datasets with Warnings")
        lines.append("")
        for result in sorted(warning_only, key=lambda r: r.dataset_name):
            lines.append(f"### {result.dataset_name}")
            lines.append("")
            for issue in result.issues:
                if issue.severity == Severity.WARNING:
                    lines.append(f"- **WARNING** `{issue.field}`: {issue.message}")
                    if issue.suggestion:
                        lines.append(f"  - Suggestion: {issue.suggestion}")
            lines.append("")

    # Missing fields summary
    lines.append("## Missing/Optional Field Summary")
    lines.append("")
    lines.append("Fields that could be populated:")
    lines.append("")

    missing_fields: Dict[str, List[str]] = {}
    for result in report.results:
        for issue in result.issues:
            if issue.severity == Severity.INFO and "No " in issue.message:
                if issue.field not in missing_fields:
                    missing_fields[issue.field] = []
                missing_fields[issue.field].append(result.dataset_name)

    for field_name, datasets in sorted(missing_fields.items()):
        lines.append(f"- `{field_name}`: {len(datasets)} datasets")

    lines.append("")

    # Extracted values for datasets with data
    lines.append("## Extracted Values (from available data)")
    lines.append("")
    for result in sorted(report.results, key=lambda r: r.dataset_name):
        if result.extracted_values:
            lines.append(f"### {result.dataset_name}")
            lines.append("")
            lines.append("```")
            for key, value in result.extracted_values.items():
                if key == "sensors":
                    lines.append(f"  {key}: [{len(value)} channels]")
                else:
                    lines.append(f"  {key}: {value}")
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def generate_json_report(report: FullValidationReport) -> str:
    """Generate a JSON report."""

    def serialize_issue(issue: ValidationIssue) -> dict:
        return {
            "field": issue.field,
            "severity": issue.severity.value,
            "message": issue.message,
            "catalog_value": (
                str(issue.catalog_value) if issue.catalog_value is not None else None
            ),
            "actual_value": (
                str(issue.actual_value) if issue.actual_value is not None else None
            ),
            "suggestion": issue.suggestion,
        }

    def serialize_result(result: DatasetValidationResult) -> dict:
        return {
            "dataset_name": result.dataset_name,
            "data_available": result.data_available,
            "data_path": str(result.data_path) if result.data_path else None,
            "error_count": result.error_count,
            "warning_count": result.warning_count,
            "issues": [serialize_issue(i) for i in result.issues],
            "extracted_values": {
                k: v if k != "sensors" else f"[{len(v)} channels]"
                for k, v in result.extracted_values.items()
            },
        }

    data = {
        "metadata": {
            "generated": report.timestamp.isoformat(),
            "format_version": "1.0.0",
        },
        "summary": {
            "total_datasets": report.total_datasets,
            "datasets_with_errors": report.datasets_with_errors,
            "datasets_with_warnings": report.datasets_with_warnings,
            "datasets_with_data": report.datasets_with_data,
            "clean_datasets": report.clean_datasets,
        },
        "results": [serialize_result(r) for r in report.results],
    }

    return json.dumps(data, indent=2)


# =============================================================================
# CLI
# =============================================================================


def main():
    """Run metadata validation."""
    parser = argparse.ArgumentParser(
        description="Validate MOABB dataset metadata against actual data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --output results/metadata_validation_report.md
  %(prog)s --dataset BNCI2014_001
  %(prog)s --format json --output results/metadata_validation_report.json
  %(prog)s --verbose --workers 8
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (default: stdout)",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        help="Validate only a specific dataset",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show progress during validation",
    )

    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    parser.add_argument(
        "--skip-urls",
        action="store_true",
        help="Skip URL validation (faster)",
    )

    args = parser.parse_args()

    # Import MOABB catalog
    try:
        from moabb.datasets.metadata import DATASET_METADATA_CATALOG
    except ImportError as e:
        print(f"Error importing MOABB: {e}", file=sys.stderr)
        print(
            "Make sure you are in the moabb directory and have installed dependencies.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Filter catalog if specific dataset requested
    catalog = dict(DATASET_METADATA_CATALOG)
    if args.dataset:
        if args.dataset not in catalog:
            print(
                f"Error: Dataset '{args.dataset}' not found in catalog", file=sys.stderr
            )
            print(
                f"Available datasets: {', '.join(sorted(catalog.keys()))}",
                file=sys.stderr,
            )
            sys.exit(1)
        catalog = {args.dataset: catalog[args.dataset]}

    # Run validation
    if args.verbose:
        print(f"MNE data path: {get_mne_data_path()}", file=sys.stderr)

    report = asyncio.run(
        validate_all_datasets(
            catalog,
            max_workers=args.workers,
            verbose=args.verbose,
        )
    )

    # Generate report
    if args.format == "json":
        output = generate_json_report(report)
    else:
        output = generate_markdown_report(report)

    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        if args.verbose:
            print(f"Report written to: {output_path}", file=sys.stderr)
    else:
        print(output)

    # Exit status
    if report.datasets_with_errors > 0:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
