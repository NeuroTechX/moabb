"""Tests for the metadata schema module."""

import pytest

from moabb.datasets.metadata import (
    DATASET_METADATA_CATALOG,
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    get_dataset_metadata,
)


class TestAcquisitionMetadata:
    """Tests for AcquisitionMetadata dataclass."""

    def test_required_fields_only(self):
        """Test instantiation with only required fields."""
        acq = AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=64,
            channel_types={"eeg": 60, "eog": 4},
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
            sampling_rate=512.0,
            n_channels=64,
            channel_types={"eeg": 60, "eog": 4},
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
        doc = DocumentationMetadata(
            doi="10.1234/example",
            description="Test dataset",
        )
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
            participants=ParticipantMetadata(
                n_subjects=8,
                health_status="healthy",
            ),
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
            participants=ParticipantMetadata(
                n_subjects=35,
                health_status="healthy",
            ),
            experiment=ExperimentMetadata(
                paradigm="ssvep",
                events={
                    "8Hz": 1,
                    "10Hz": 2,
                    "12Hz": 3,
                    "14Hz": 4,
                },
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
        assert metadata.acquisition.n_channels == 22
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
            assert (
                metadata.experiment.paradigm == paradigm
            ), f"{name} should have paradigm '{paradigm}'"

    def test_all_datasets_have_required_fields(self):
        """Test that all catalog datasets have required metadata fields."""
        for name, metadata in DATASET_METADATA_CATALOG.items():
            # Acquisition required fields
            assert metadata.acquisition.sampling_rate > 0, f"{name} missing sampling_rate"
            assert metadata.acquisition.n_channels > 0, f"{name} missing n_channels"
            assert (
                len(metadata.acquisition.channel_types) > 0
            ), f"{name} missing channel_types"
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
        assert len(DATASET_METADATA_CATALOG) == 84

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

    @pytest.mark.parametrize(
        "paradigm,expected_count",
        [
            ("imagery", 31),
            ("p300", 35),
            ("ssvep", 7),
            ("cvep", 8),
            ("rstate", 3),
        ],
    )
    def test_paradigm_counts(self, paradigm, expected_count):
        """Test that each paradigm has expected number of datasets."""
        count = sum(
            1
            for m in DATASET_METADATA_CATALOG.values()
            if m.experiment.paradigm == paradigm
        )
        assert (
            count == expected_count
        ), f"Expected {expected_count} {paradigm} datasets, found {count}"
