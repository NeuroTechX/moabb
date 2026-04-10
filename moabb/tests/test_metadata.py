"""Tests for the metadata schema module."""

import csv
import dataclasses
import json
import typing
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pytest

# Module-level imports used as monkeypatch targets (setattr requires the module object)
import moabb.datasets as datasets_module  # noqa: F401
import moabb.datasets.metadata as metadata_module  # noqa: F401
import moabb.datasets.utils as dataset_utils  # noqa: F401

# Named imports for direct use in test assertions and setup
from moabb.datasets.bids_interface import _update_participants_tsv
from moabb.datasets.lee2021_mobile import Lee2021Mobile
from moabb.datasets.metadata import (
    DATASET_METADATA_CATALOG,
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    get_dataset_metadata,
)
from moabb.datasets.utils import _init_dataset, build_raw_from_epochs, dataset_dict


class TestAcquisitionMetadata:
    """Tests for AcquisitionMetadata dataclass."""

    def test_required_fields_only(self):
        """Test instantiation with only required fields."""
        acq = AcquisitionMetadata(
            sampling_rate=512.0, n_channels=64, channel_types={"eeg": 60, "eog": 4}
        )
        assert acq.sampling_rate == 512.0
        assert acq.n_channels == 64
        assert acq.channel_types == {"eeg": 60, "eog": 4}
        # Check defaults
        assert acq.sensors == []
        assert acq.sensor_type is None
        assert acq.reference is None
        assert acq.ground is None
        assert acq.hardware is None
        assert acq.software is None
        assert acq.filters is None
        assert acq.line_freq == 50.0
        assert acq.montage == "standard_1005"

    def test_all_fields(self):
        """Test instantiation with all fields."""
        acq = AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=128,
            channel_types={"eeg": 120, "eog": 4, "emg": 4},
            sensors=["Fp1", "Fp2", "F3", "F4"],
            sensor_type="Ag/AgCl wet",
            reference="average",
            ground="AFz",
            hardware="BrainAmp DC",
            software="BrainVision Recorder",
            filters="0.1-100 Hz bandpass",
            line_freq=60.0,
            montage="standard_1020",
        )
        assert acq.sampling_rate == 1000.0
        assert acq.n_channels == 128
        assert acq.sensor_type == "Ag/AgCl wet"
        assert acq.reference == "average"
        assert acq.ground == "AFz"
        assert acq.hardware == "BrainAmp DC"
        assert acq.line_freq == 60.0
        assert acq.montage == "standard_1020"

    def test_sensors_mutable_default(self):
        """Test that sensors default list is not shared between instances."""
        acq1 = AcquisitionMetadata(
            sampling_rate=512.0, n_channels=64, channel_types={"eeg": 64}
        )
        acq2 = AcquisitionMetadata(
            sampling_rate=256.0, n_channels=32, channel_types={"eeg": 32}
        )
        acq1.sensors.append("Cz")
        assert acq1.sensors == ["Cz"]
        assert acq2.sensors == []


class TestDocumentationMetadata:
    """Tests for DocumentationMetadata dataclass."""

    def test_all_defaults(self):
        """Test instantiation with all defaults."""
        doc = DocumentationMetadata()
        assert doc.doi is None
        assert doc.description is None
        assert doc.investigators is None
        assert doc.institution is None
        assert doc.country is None
        assert doc.repository is None
        assert doc.data_url is None
        assert doc.license is None
        assert doc.publication_year is None

    def test_all_fields(self):
        """Test instantiation with all fields."""
        doc = DocumentationMetadata(
            doi="10.1234/example",
            description="A motor imagery dataset",
            investigators=["John Doe", "Jane Smith"],
            institution="University of Example",
            country="Germany",
            repository="BNCI Horizon 2020",
            data_url="http://example.com/data",
            license="CC BY 4.0",
            publication_year=2020,
        )
        assert doc.doi == "10.1234/example"
        assert doc.description == "A motor imagery dataset"
        assert doc.investigators == ["John Doe", "Jane Smith"]
        assert doc.institution == "University of Example"
        assert doc.country == "Germany"
        assert doc.repository == "BNCI Horizon 2020"
        assert doc.data_url == "http://example.com/data"
        assert doc.license == "CC BY 4.0"
        assert doc.publication_year == 2020


class TestParticipantMetadata:
    """Tests for ParticipantMetadata dataclass."""

    def test_required_fields_only(self):
        """Test instantiation with only required fields."""
        part = ParticipantMetadata(n_subjects=20)
        assert part.n_subjects == 20
        assert part.health_status == "healthy"
        assert part.gender is None
        assert part.age_mean is None
        assert part.age_std is None
        assert part.handedness is None
        assert part.clinical_population is None

    def test_all_fields(self):
        """Test instantiation with all fields."""
        part = ParticipantMetadata(
            n_subjects=30,
            health_status="patients",
            gender={"male": 18, "female": 12},
            age_mean=45.5,
            age_std=12.3,
            handedness={"right": 27, "left": 3},
            clinical_population="stroke",
        )
        assert part.n_subjects == 30
        assert part.health_status == "patients"
        assert part.gender == {"male": 18, "female": 12}
        assert part.age_mean == 45.5
        assert part.age_std == 12.3
        assert part.handedness == {"right": 27, "left": 3}
        assert part.clinical_population == "stroke"


class TestExperimentMetadata:
    """Tests for ExperimentMetadata dataclass."""

    def test_required_fields_only(self):
        """Test instantiation with only required fields."""
        exp = ExperimentMetadata(paradigm="imagery")
        assert exp.paradigm == "imagery"
        assert exp.task_type is None
        assert exp.events == {}
        assert exp.n_classes is None
        assert exp.trials_per_class is None
        assert exp.trial_duration is None

    def test_all_fields(self):
        """Test instantiation with all fields."""
        exp = ExperimentMetadata(
            paradigm="p300",
            task_type="row_col_speller",
            events={"target": 1, "non_target": 2},
            n_classes=2,
            trials_per_class={"target": 100, "non_target": 500},
            trial_duration=0.8,
        )
        assert exp.paradigm == "p300"
        assert exp.task_type == "row_col_speller"
        assert exp.events == {"target": 1, "non_target": 2}
        assert exp.n_classes == 2
        assert exp.trials_per_class == {"target": 100, "non_target": 500}
        assert exp.trial_duration == 0.8

    def test_events_mutable_default(self):
        """Test that events default dict is not shared between instances."""
        exp1 = ExperimentMetadata(paradigm="imagery")
        exp2 = ExperimentMetadata(paradigm="ssvep")
        exp1.events["left_hand"] = 1
        assert exp1.events == {"left_hand": 1}
        assert exp2.events == {}


class TestDatasetMetadata:
    """Tests for DatasetMetadata dataclass."""

    @pytest.fixture
    def minimal_acquisition(self):
        """Create minimal AcquisitionMetadata for testing."""
        return AcquisitionMetadata(
            sampling_rate=512.0, n_channels=64, channel_types={"eeg": 60, "eog": 4}
        )

    @pytest.fixture
    def minimal_participants(self):
        """Create minimal ParticipantMetadata for testing."""
        return ParticipantMetadata(n_subjects=20)

    @pytest.fixture
    def minimal_experiment(self):
        """Create minimal ExperimentMetadata for testing."""
        return ExperimentMetadata(paradigm="imagery")

    def test_required_fields_only(
        self, minimal_acquisition, minimal_participants, minimal_experiment
    ):
        """Test instantiation with only required fields."""
        meta = DatasetMetadata(
            acquisition=minimal_acquisition,
            participants=minimal_participants,
            experiment=minimal_experiment,
        )
        assert meta.acquisition == minimal_acquisition
        assert meta.participants == minimal_participants
        assert meta.experiment == minimal_experiment
        assert meta.documentation is None
        assert meta.sessions_per_subject == 1
        assert meta.runs_per_session == 1

    def test_all_fields(
        self, minimal_acquisition, minimal_participants, minimal_experiment
    ):
        """Test instantiation with all fields."""
        doc = DocumentationMetadata(doi="10.1234/example", description="Test dataset")
        meta = DatasetMetadata(
            acquisition=minimal_acquisition,
            participants=minimal_participants,
            experiment=minimal_experiment,
            documentation=doc,
            sessions_per_subject=3,
            runs_per_session=2,
        )
        assert meta.documentation == doc
        assert meta.sessions_per_subject == 3
        assert meta.runs_per_session == 2

    def test_nested_access(
        self, minimal_acquisition, minimal_participants, minimal_experiment
    ):
        """Test accessing nested metadata fields."""
        meta = DatasetMetadata(
            acquisition=minimal_acquisition,
            participants=minimal_participants,
            experiment=minimal_experiment,
        )
        assert meta.acquisition.sampling_rate == 512.0
        assert meta.participants.n_subjects == 20
        assert meta.experiment.paradigm == "imagery"


class TestMetadataIntegration:
    """Integration tests for the complete metadata schema."""

    def test_realistic_motor_imagery_dataset(self):
        """Test creating metadata for a realistic motor imagery dataset."""
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=512.0,
                n_channels=22,
                channel_types={"eeg": 22},
                sensor_type="Ag/AgCl wet",
                reference="left earlobe",
                ground="right mastoid",
                hardware="g.USBamp",
                line_freq=50.0,
                montage="standard_1020",
            ),
            participants=ParticipantMetadata(
                n_subjects=9,
                health_status="healthy",
                gender={"male": 6, "female": 3},
                age_mean=27.2,
                age_std=3.1,
                handedness={"right": 9},
            ),
            experiment=ExperimentMetadata(
                paradigm="imagery",
                task_type="left_right_hand",
                events={"left_hand": 1, "right_hand": 2, "feet": 3, "tongue": 4},
                n_classes=4,
                trial_duration=4.0,
            ),
            documentation=DocumentationMetadata(
                doi="10.3389/fnins.2012.00055",
                description="BCI Competition IV Dataset 2a",
                investigators=["C. Brunner", "R. Leeb", "G. Mueller-Putz"],
                institution="Graz University of Technology",
                country="Austria",
                repository="BNCI Horizon 2020",
                license="CC BY 4.0",
                publication_year=2012,
            ),
            sessions_per_subject=2,
            runs_per_session=6,
        )

        # Verify key fields
        assert metadata.acquisition.sampling_rate == 512.0
        assert metadata.participants.n_subjects == 9
        assert metadata.experiment.paradigm == "imagery"
        assert metadata.experiment.n_classes == 4
        assert metadata.documentation.doi == "10.3389/fnins.2012.00055"
        assert metadata.sessions_per_subject == 2
        assert metadata.runs_per_session == 6

    def test_realistic_p300_dataset(self):
        """Test creating metadata for a realistic P300 dataset."""
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256.0,
                n_channels=64,
                channel_types={"eeg": 64},
                hardware="BioSemi ActiveTwo",
                reference="CMS/DRL",
            ),
            participants=ParticipantMetadata(n_subjects=8, health_status="healthy"),
            experiment=ExperimentMetadata(
                paradigm="p300",
                task_type="row_col_speller",
                events={"target": 1, "non_target": 2},
                n_classes=2,
                trials_per_class={"target": 180, "non_target": 900},
            ),
        )

        assert metadata.experiment.paradigm == "p300"
        assert metadata.experiment.task_type == "row_col_speller"
        assert metadata.acquisition.hardware == "BioSemi ActiveTwo"

    def test_realistic_ssvep_dataset(self):
        """Test creating metadata for a realistic SSVEP dataset."""
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=250.0,
                n_channels=8,
                channel_types={"eeg": 8},
                sensors=["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"],
            ),
            participants=ParticipantMetadata(n_subjects=35, health_status="healthy"),
            experiment=ExperimentMetadata(
                paradigm="ssvep",
                events={"8Hz": 1, "10Hz": 2, "12Hz": 3, "14Hz": 4},
                n_classes=4,
                trial_duration=5.0,
            ),
        )

        assert metadata.experiment.paradigm == "ssvep"
        assert len(metadata.experiment.events) == 4
        assert metadata.acquisition.sensors[0] == "PO7"


class TestMetadataCatalog:
    """Tests for the pre-defined metadata catalog."""

    def test_catalog_not_empty(self):
        """Test that the catalog contains datasets."""
        assert len(DATASET_METADATA_CATALOG) > 0

    def test_all_catalog_entries_are_dataset_metadata(self):
        """Test that all catalog entries are DatasetMetadata instances."""
        for name, metadata in DATASET_METADATA_CATALOG.items():
            assert isinstance(metadata, DatasetMetadata), f"{name} is not DatasetMetadata"

    def test_get_dataset_metadata_valid(self):
        """Test retrieving valid dataset metadata."""
        metadata = get_dataset_metadata("BNCI2014_001")
        assert isinstance(metadata, DatasetMetadata)
        assert metadata.participants.n_subjects == 9
        assert metadata.acquisition.sampling_rate == 250.0
        assert metadata.experiment.paradigm == "imagery"

    def test_get_dataset_metadata_invalid(self):
        """Test retrieving invalid dataset raises KeyError."""
        with pytest.raises(KeyError) as exc_info:
            get_dataset_metadata("NonExistentDataset")
        assert "NonExistentDataset" in str(exc_info.value)

    def test_bnci2014_001_metadata_complete(self):
        """Test BNCI2014_001 metadata has expected fields."""
        metadata = get_dataset_metadata("BNCI2014_001")
        # Acquisition
        assert metadata.acquisition.n_channels == 25
        assert metadata.acquisition.reference == "left mastoid"
        assert "eeg" in metadata.acquisition.channel_types
        # Participants
        assert metadata.participants.n_subjects == 9
        assert metadata.participants.health_status == "healthy"
        # Experiment
        assert metadata.experiment.n_classes == 4
        # Note: events are now extracted dynamically from dataset.event_id
        # Documentation
        assert metadata.documentation is not None
        assert "10.3389" in metadata.documentation.doi
        assert metadata.documentation.country == "AT"  # ISO alpha-2 code
        # Structure
        assert metadata.sessions_per_subject == 2
        assert metadata.runs_per_session == 6

    def test_physionetmi_metadata(self):
        """Test PhysionetMI metadata."""
        metadata = get_dataset_metadata("PhysionetMI")
        assert metadata.participants.n_subjects == 109
        assert metadata.acquisition.sampling_rate == 160.0
        assert metadata.experiment.paradigm == "imagery"
        # Note: events are now extracted dynamically from dataset.event_id

    def test_lee2019_mi_metadata(self):
        """Test Lee2019_MI metadata."""
        metadata = get_dataset_metadata("Lee2019_MI")
        assert metadata.participants.n_subjects == 54
        assert metadata.acquisition.sampling_rate == 1000.0
        assert metadata.sessions_per_subject == 2

    def test_bi2012_metadata(self):
        """Test BI2012 (Brain Invaders) metadata."""
        metadata = get_dataset_metadata("BI2012")
        assert metadata.participants.n_subjects == 25
        assert metadata.experiment.paradigm == "p300"
        assert metadata.experiment.task_type == "brain_invaders"

    def test_wang2016_ssvep_metadata(self):
        """Test Wang2016 SSVEP metadata."""
        metadata = get_dataset_metadata("Wang2016")
        assert metadata.participants.n_subjects == 34
        assert metadata.experiment.paradigm == "ssvep"
        assert metadata.experiment.n_classes == 40
        assert metadata.acquisition.n_channels == 64

    def test_nakanishi2015_metadata(self):
        """Test Nakanishi2015 SSVEP metadata."""
        metadata = get_dataset_metadata("Nakanishi2015")
        assert metadata.participants.n_subjects == 9
        assert metadata.experiment.paradigm == "ssvep"
        assert metadata.experiment.n_classes == 12
        assert len(metadata.acquisition.sensors) == 8

    def test_erpcore2021_n170_metadata(self):
        """Test ErpCore2021_N170 metadata."""
        metadata = get_dataset_metadata("ErpCore2021_N170")
        assert metadata.participants.n_subjects == 40
        assert metadata.participants.age_mean == 21.5
        assert metadata.acquisition.hardware == "Biosemi ActiveTwo"
        assert metadata.experiment.paradigm == "p300"

    def test_dreyer2023_metadata(self):
        """Test Dreyer2023 metadata."""
        metadata = get_dataset_metadata("Dreyer2023")
        assert metadata.participants.n_subjects == 87
        assert metadata.acquisition.n_channels == 27
        assert metadata.documentation.country == "FR"  # ISO alpha-2 code
        assert "10.1038/s41597-023-02445-z" in metadata.documentation.doi

    @pytest.mark.parametrize(
        "paradigm,expected_datasets",
        [
            ("imagery", ["BNCI2014_001", "PhysionetMI", "Lee2019_MI"]),
            ("p300", ["BI2012", "BNCI2014_008", "Lee2019_ERP"]),
            ("ssvep", ["Wang2016", "Nakanishi2015", "Kalunga2016"]),
        ],
    )
    def test_paradigm_consistency(self, paradigm, expected_datasets):
        """Test that datasets have correct paradigm assignment."""
        for name in expected_datasets:
            metadata = get_dataset_metadata(name)
            assert metadata.experiment.paradigm == paradigm, (
                f"{name} should have paradigm '{paradigm}'"
            )

    def test_all_datasets_have_required_fields(self):
        """Test that all catalog datasets have required metadata fields."""
        for name, metadata in DATASET_METADATA_CATALOG.items():
            # Acquisition required fields
            assert metadata.acquisition.sampling_rate > 0, f"{name} missing sampling_rate"
            assert metadata.acquisition.n_channels > 0, f"{name} missing n_channels"
            assert len(metadata.acquisition.channel_types) > 0, (
                f"{name} missing channel_types"
            )
            # Participants required field
            assert metadata.participants.n_subjects > 0, f"{name} missing n_subjects"
            # Experiment required field
            assert metadata.experiment.paradigm in [
                "imagery",
                "p300",
                "ssvep",
                "cvep",
                "rstate",
            ], f"{name} has invalid paradigm"

    def test_catalog_dataset_count(self):
        """Test that catalog contains expected number of datasets."""
        assert len(DATASET_METADATA_CATALOG) == 148

    def test_bnci2015_006_metadata(self):
        """Test BNCI2015_006 music BCI metadata."""
        metadata = get_dataset_metadata("BNCI2015_006")
        assert metadata.participants.n_subjects == 11
        assert metadata.experiment.paradigm == "p300"
        assert "10.1088/1741-2560/11/2/026009" in metadata.documentation.doi

    def test_bnci2019_001_metadata(self):
        """Test BNCI2019_001 spinal cord injury metadata."""
        metadata = get_dataset_metadata("BNCI2019_001")
        assert metadata.participants.n_subjects == 10
        assert metadata.participants.health_status == "patients"
        assert metadata.participants.clinical_population == "spinal cord injury"
        assert metadata.experiment.paradigm == "imagery"

    def test_castillos_cvep_metadata(self):
        """Test Castillos cVEP datasets metadata."""
        for name in [
            "CastillosBurstVEP40",
            "CastillosBurstVEP100",
            "CastillosCVEP40",
            "CastillosCVEP100",
        ]:
            metadata = get_dataset_metadata(name)
            assert metadata.participants.n_subjects == 12
            assert metadata.experiment.paradigm == "cvep"
            assert "10.1016/j.neuroimage.2023.120446" in metadata.documentation.doi

    def test_martinezcagigal_cvep_metadata(self):
        """Test MartinezCagigal cVEP datasets metadata."""
        checker = get_dataset_metadata("MartinezCagigal2023Checker")
        assert checker.participants.n_subjects == 16
        assert checker.experiment.paradigm == "cvep"
        assert checker.sessions_per_subject == 8

        # codespell:ignore pary
        pary = get_dataset_metadata("MartinezCagigal2023Pary")
        assert pary.participants.n_subjects == 16
        assert pary.experiment.paradigm == "cvep"
        assert pary.sessions_per_subject == 5

    def test_erpcore2021_all_variants(self):
        """Test all ERP CORE 2021 dataset variants."""
        variants = [
            "ErpCore2021_ERN",
            "ErpCore2021_LRP",
            "ErpCore2021_MMN",
            "ErpCore2021_N170",
            "ErpCore2021_N2pc",
            "ErpCore2021_N400",
            "ErpCore2021_P3",
        ]
        for name in variants:
            metadata = get_dataset_metadata(name)
            assert metadata.participants.n_subjects == 40
            assert metadata.acquisition.hardware == "Biosemi ActiveTwo"
            assert metadata.experiment.paradigm == "p300"
            assert "10.1016/j.neuroimage.2020.117465" in metadata.documentation.doi

    @pytest.mark.parametrize("paradigm", ["imagery", "p300", "ssvep", "cvep", "rstate"])
    def test_paradigm_counts(self, paradigm):
        """Cross-check catalog paradigm counts against summary CSV rows."""
        catalog_names = {
            name
            for name, meta in DATASET_METADATA_CATALOG.items()
            if meta.experiment.paradigm == paradigm
        }

        summary_path = (
            Path(__file__).resolve().parents[1] / "datasets" / f"summary_{paradigm}.csv"
        )
        with open(summary_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            summary_names = {
                row["Dataset"].strip() for row in reader if row.get("Dataset")
            }

        # Every summary entry should correspond to a catalog entry.
        assert summary_names <= catalog_names

        # Catalog entries omitted from summary must be umbrella datasets
        # represented by one or more explicit variants (e.g., Name_*).
        missing_from_summary = catalog_names - summary_names
        allowed_omissions = {
            name
            for name in missing_from_summary
            if any(candidate.startswith(f"{name}_") for candidate in catalog_names)
        }
        assert missing_from_summary == allowed_omissions
        assert len(catalog_names) == len(summary_names) + len(allowed_omissions)

    def test_detected_paradigm_matches_experiment(self):
        """Test that detected_paradigm agrees with experiment.paradigm."""
        expected_by_paradigm = {
            "imagery": "motor_imagery",
            "p300": "p300",
            "ssvep": "ssvep",
            "cvep": "cvep",
            "rstate": "resting_state",
        }
        for name, metadata in DATASET_METADATA_CATALOG.items():
            if (
                metadata.paradigm_specific
                and metadata.paradigm_specific.detected_paradigm
            ):
                assert (
                    metadata.paradigm_specific.detected_paradigm
                    == expected_by_paradigm.get(
                        metadata.experiment.paradigm, metadata.experiment.paradigm
                    )
                ), (
                    f"{name}: detected_paradigm="
                    f"'{metadata.paradigm_specific.detected_paradigm}' "
                    f"!= experiment.paradigm='{metadata.experiment.paradigm}'"
                )

    def test_catalog_supports_get(self):
        """Test dict-style get support on lazy catalog wrapper."""
        assert DATASET_METADATA_CATALOG.get("BNCI2014_001") is not None
        assert DATASET_METADATA_CATALOG.get("DoesNotExist") is None

    def test_class_level_metadata_core_fields_match_dataset(self):
        """Ensure class-level METADATA is authoritative for core runtime fields."""
        _init_dataset()
        for name, dataset_cls in dataset_dict.items():
            if "Fake" in name:
                continue

            metadata = getattr(dataset_cls, "METADATA", None)
            if not isinstance(metadata, DatasetMetadata):
                continue

            dataset = dataset_cls()
            assert metadata.experiment.paradigm == dataset.paradigm
            assert metadata.participants.n_subjects == len(dataset.subject_list)
            assert metadata.experiment.n_classes == len(dataset.event_id)
            assert metadata.experiment.class_labels == list(dataset.event_id.keys())
            assert metadata.sessions_per_subject == dataset.n_sessions

    def test_catalog_fallback_for_uninstantiable_dataset(self, monkeypatch):
        """Catalog should include a fallback entry when class init fails."""

        class BrokenDataset:
            def __init__(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(
            dataset_utils, "dataset_dict", {"BrokenDataset": BrokenDataset}
        )
        monkeypatch.setattr(dataset_utils, "_init_dataset", lambda: None)

        with pytest.warns(RuntimeWarning, match="BrokenDataset"):
            catalog = metadata_module._build_dataset_metadata_catalog()

        assert "BrokenDataset" in catalog
        fallback = catalog["BrokenDataset"]
        assert fallback.participants.n_subjects == 1
        assert fallback.experiment.paradigm == "imagery"

    def test_removed_dataset_error_is_explicit(self):
        with pytest.raises(AttributeError, match="DemonsP300 has been removed"):
            _ = datasets_module.DemonsP300

    def test_metadata_type_consistency(self):
        """Test that all METADATA values match their declared types."""

        def _check_type(value, expected_type, path):
            errors = []
            if value is None:
                return errors
            origin = getattr(expected_type, "__origin__", None)
            if origin is typing.Union:
                args = [
                    a
                    for a in getattr(expected_type, "__args__", ())
                    if a is not type(None)
                ]
                if len(args) == 1:
                    return _check_type(value, args[0], path)
                return errors
            if origin is list:
                if not isinstance(value, list):
                    return [f"{path}: expected list, got {type(value).__name__}"]
                inner = getattr(expected_type, "__args__", ())
                if inner:
                    for i, item in enumerate(value):
                        errors.extend(_check_type(item, inner[0], f"{path}[{i}]"))
                return errors
            if origin is dict:
                if not isinstance(value, dict):
                    return [f"{path}: expected dict, got {type(value).__name__}"]
                args = getattr(expected_type, "__args__", ())
                if args and len(args) == 2:
                    for k, v in value.items():
                        errors.extend(_check_type(k, args[0], f"{path}.key"))
                        errors.extend(_check_type(v, args[1], f"{path}[{k!r}]"))
                return errors
            if expected_type is float:
                if not isinstance(value, (int, float)):
                    errors.append(f"{path}: expected float, got {type(value).__name__}")
            elif expected_type is int:
                if not isinstance(value, int) or isinstance(value, bool):
                    errors.append(f"{path}: expected int, got {type(value).__name__}")
            elif expected_type is str:
                if not isinstance(value, str):
                    errors.append(f"{path}: expected str, got {type(value).__name__}")
            elif expected_type is bool:
                if not isinstance(value, bool):
                    errors.append(f"{path}: expected bool, got {type(value).__name__}")
            elif dataclasses.is_dataclass(expected_type):
                if not isinstance(value, expected_type):
                    errors.append(
                        f"{path}: expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
                else:
                    for f in dataclasses.fields(value):
                        v = getattr(value, f.name)
                        if v is not None:
                            errors.extend(_check_type(v, f.type, f"{path}.{f.name}"))
            return errors

        all_errors = []
        for name, metadata in DATASET_METADATA_CATALOG.items():
            for f in dataclasses.fields(metadata):
                v = getattr(metadata, f.name)
                if v is not None:
                    all_errors.extend(_check_type(v, f.type, f"{name}.{f.name}"))
        assert all_errors == [], (
            f"Found {len(all_errors)} type violations:\n" + "\n".join(all_errors[:20])
        )


class TestBuildRawFromEpochsValidation:
    def test_valid_build_raw_from_epochs(self):
        data = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
        raw = build_raw_from_epochs(
            data=data,
            ch_names=["C3", "Cz", "C4"],
            sfreq=128.0,
            event_ids=[1, 2],
            montage_name="standard_1005",
            onset_sample=1,
            buffer_samples=0,
        )

        stim = raw.get_data(picks=[raw.ch_names.index("STI")])[0]
        assert raw.info["nchan"] == 4
        assert np.where(stim > 0)[0].tolist() == [1, 5]
        assert stim[1] == 1
        assert stim[5] == 2

    def test_rejects_invalid_data_shape(self):
        with pytest.raises(ValueError, match="data must have shape"):
            build_raw_from_epochs(
                data=np.zeros((3, 4)),
                ch_names=["C3", "Cz", "C4"],
                sfreq=128.0,
                event_ids=[1, 2, 3],
                montage_name="standard_1005",
            )

    def test_rejects_scalar_event_ids(self):
        with pytest.raises(ValueError, match="event_ids must be a 1D array-like"):
            build_raw_from_epochs(
                data=np.zeros((2, 3, 4)),
                ch_names=["C3", "Cz", "C4"],
                sfreq=128.0,
                event_ids=1,
                montage_name="standard_1005",
            )

    def test_rejects_negative_onset_sample(self):
        with pytest.raises(ValueError, match="onset_sample .* must be between 0"):
            build_raw_from_epochs(
                data=np.zeros((2, 3, 4)),
                ch_names=["C3", "Cz", "C4"],
                sfreq=128.0,
                event_ids=[1, 2],
                montage_name="standard_1005",
                onset_sample=-1,
            )


class TestLee2021MobileSessionNormalization:
    @staticmethod
    def _make_mock_raw():
        info = mne.create_info(["Oz"], sfreq=500.0, ch_types=["eeg"])
        raw = mne.io.RawArray(np.zeros((1, 50)), info, verbose=False)
        raw.set_annotations(
            mne.Annotations(
                onset=[0.0, 0.01, 0.02],
                duration=[0.0, 0.0, 0.0],
                description=["Stimulus/S 11", "Stimulus/S 12", "Stimulus/S 13"],
            )
        )
        return raw

    def test_selected_sessions_accept_unpadded_integer(self, monkeypatch):
        dataset = Lee2021Mobile(paradigm="ssvep", subjects=[1], sessions=[2])
        fake_files = [
            "/tmp/sub-01_ses-02_task-SSVEP_eeg.vhdr",
            "/tmp/sub-01_ses-03_task-SSVEP_eeg.vhdr",
        ]
        monkeypatch.setattr(dataset, "data_path", lambda subject: fake_files)
        monkeypatch.setattr(
            "moabb.datasets.lee2021_mobile.mne.io.read_raw_brainvision",
            lambda *args, **kwargs: self._make_mock_raw(),
        )

        subject_sessions = dataset._get_single_subject_data(1)
        assert set(subject_sessions) == {"2", "3"}

        monkeypatch.setattr(
            dataset,
            "_get_single_subject_data_using_cache",
            lambda subject, cache_config, process_pipeline: subject_sessions,
        )
        data = dataset.get_data(subjects=[1])
        assert set(data[1]) == {"2"}


class TestParticipantsResolutionOrdering:
    @staticmethod
    def _write_participants_tsv(tmp_path, participant_ids):
        tsv_path = tmp_path / "participants.tsv"
        with open(tsv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["participant_id"], delimiter="\t")
            writer.writeheader()
            for pid in participant_ids:
                writer.writerow({"participant_id": pid})
        return tsv_path

    @staticmethod
    def _read_participants_tsv(tsv_path):
        with open(tsv_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    @staticmethod
    def _make_raw(subject_info=None, age=None):
        raw = SimpleNamespace(info={})
        if subject_info is not None:
            raw.info["subject_info"] = subject_info
        if age is not None:
            raw._moabb_subject_age = age
        return raw

    def test_age_resolution_priority(self, tmp_path):
        tsv_path = self._write_participants_tsv(tmp_path, ["sub-1", "sub-2"])
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=128.0, n_channels=1, channel_types={"eeg": 1}
            ),
            participants=ParticipantMetadata(
                n_subjects=2, ages=[25, None], age_mean=44.0
            ),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )

        _update_participants_tsv(tmp_path, 1, metadata, raw=self._make_raw(age=31))
        _update_participants_tsv(tmp_path, 2, metadata, raw=self._make_raw(age=31))
        rows = self._read_participants_tsv(tsv_path)

        assert rows[0]["age"] == "25"
        assert rows[1]["age"] == "31"

    def test_sex_and_hand_resolution_priority_and_parsing(self, tmp_path):
        tsv_path = self._write_participants_tsv(tmp_path, ["sub-1", "sub-2"])
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=128.0, n_channels=1, channel_types={"eeg": 1}
            ),
            participants=ParticipantMetadata(
                n_subjects=2,
                sexes=["male", None],
                handedness_list=[None, None],
                handedness={"left": 2},
            ),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )

        # Subject 1: metadata list has priority over raw sex.
        _update_participants_tsv(
            tmp_path, 1, metadata, raw=self._make_raw(subject_info={"sex": 2, "hand": 2})
        )
        # Subject 2: fallback to raw subject_info with numeric strings.
        _update_participants_tsv(
            tmp_path,
            2,
            metadata,
            raw=self._make_raw(subject_info={"sex": "2", "hand": "1"}),
        )
        rows = self._read_participants_tsv(tsv_path)

        assert rows[0]["sex"] == "male"
        assert rows[0]["hand"] == "left"
        assert rows[1]["sex"] == "female"
        assert rows[1]["hand"] == "right"

    def test_doi_cache_metadata_total_matches_entries(self):
        cache_path = Path(__file__).resolve().parent / "doi_cache.json"
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        assert "_metadata" in cache
        assert "total" in cache["_metadata"]
        total = cache["_metadata"]["total"]
        actual = sum(1 for key in cache if key != "_metadata")
        assert total == actual
