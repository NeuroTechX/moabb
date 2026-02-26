"""Tests for BIDS enrichment helpers in bids_interface.py."""

import csv
import json
from unittest.mock import MagicMock

import mne
import numpy as np
import pytest

from moabb.datasets.bids_interface import (
    _build_dataset_description_kwargs,
    _build_hed_sidecar_annotations,
    _build_readme,
    _build_sidecar_enrichment,
    _enrich_raw_info_from_metadata,
    _extract_references_from_docstring,
    _hed_element_to_tree,
    _render_hed_tree,
    _split_hed_top_level,
    _split_manufacturer,
    _update_dataset_description_extra,
    _update_electrodes_tsv,
    _update_events_json_sidecar,
    _update_participants_tsv,
)
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
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


# ============================================================
# _split_manufacturer
# ============================================================


class TestSplitManufacturer:
    def test_known_amplifier(self):
        assert _split_manufacturer("BrainAmp") == ("Brain Products", "BrainAmp")

    def test_known_amplifier_case_insensitive(self):
        assert _split_manufacturer("brainamp DC") == ("Brain Products", "brainamp DC")

    def test_biosemi(self):
        assert _split_manufacturer("Biosemi ActiveTwo") == (
            "BioSemi",
            "Biosemi ActiveTwo",
        )

    def test_unknown_amplifier(self):
        assert _split_manufacturer("CustomAmp v3") == ("CustomAmp v3", "CustomAmp v3")

    def test_none(self):
        assert _split_manufacturer(None) == (None, None)

    def test_empty_string(self):
        assert _split_manufacturer("") == (None, None)

    def test_gtec(self):
        assert _split_manufacturer("g.USBamp") == ("g.tec", "g.USBamp")


# ============================================================
# _enrich_raw_info_from_metadata
# ============================================================


class TestEnrichRawInfoFromMetadata:
    def _make_raw(self):
        """Create a minimal raw object for testing."""
        info = mne.create_info(ch_names=["C3", "C4"], sfreq=256, ch_types="eeg")
        raw = mne.io.RawArray(np.zeros((2, 256)), info)
        return raw

    def test_none_metadata_is_noop(self):
        raw = self._make_raw()
        _enrich_raw_info_from_metadata(raw, None, 1)
        assert raw.info.get("subject_info") is None

    def test_line_freq_from_metadata(self):
        raw = self._make_raw()
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256,
                n_channels=2,
                channel_types={"eeg": 2},
                line_freq=60.0,
            ),
            participants=ParticipantMetadata(n_subjects=1),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        # raw.info line_freq is None by default — metadata should fill it
        assert raw.info["line_freq"] is None
        _enrich_raw_info_from_metadata(raw, metadata, 1)
        assert raw.info["line_freq"] == 60.0

    def test_sex_all_male(self):
        raw = self._make_raw()
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(n_subjects=5, gender={"male": 5}),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        _enrich_raw_info_from_metadata(raw, metadata, 1)
        assert raw.info["subject_info"]["sex"] == 1

    def test_sex_all_female(self):
        raw = self._make_raw()
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(n_subjects=5, gender={"female": 5}),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        _enrich_raw_info_from_metadata(raw, metadata, 1)
        assert raw.info["subject_info"]["sex"] == 2

    def test_sex_mixed_not_set(self):
        raw = self._make_raw()
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(
                n_subjects=10, gender={"male": 5, "female": 5}
            ),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        _enrich_raw_info_from_metadata(raw, metadata, 1)
        subj = raw.info.get("subject_info") or {}
        assert "sex" not in subj

    def test_hand_all_right_dict(self):
        raw = self._make_raw()
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(n_subjects=5, handedness={"right": 5}),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        _enrich_raw_info_from_metadata(raw, metadata, 1)
        assert raw.info["subject_info"]["hand"] == 1

    def test_hand_string_right(self):
        raw = self._make_raw()
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(n_subjects=5, handedness="right-handed"),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        _enrich_raw_info_from_metadata(raw, metadata, 1)
        assert raw.info["subject_info"]["hand"] == 1

    def test_no_participants(self):
        raw = self._make_raw()
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(n_subjects=1),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        # No gender or handedness set — should not crash
        _enrich_raw_info_from_metadata(raw, metadata, 1)

    def test_per_subject_sex(self):
        raw = self._make_raw()
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(
                n_subjects=3, sexes=["male", "female", "male"]
            ),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        # Subject 2 → index 1 → female → code 2
        _enrich_raw_info_from_metadata(raw, metadata, 2)
        assert raw.info["subject_info"]["sex"] == 2

    def test_per_subject_handedness(self):
        raw = self._make_raw()
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(
                n_subjects=3, handedness_list=["right", "left", "ambidextrous"]
            ),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        # Subject 2 → index 1 → left → code 2
        _enrich_raw_info_from_metadata(raw, metadata, 2)
        assert raw.info["subject_info"]["hand"] == 2

        # Subject 3 → index 2 → ambidextrous → code 3
        raw2 = self._make_raw()
        _enrich_raw_info_from_metadata(raw2, metadata, 3)
        assert raw2.info["subject_info"]["hand"] == 3

    def test_per_subject_sex_overrides_aggregate(self):
        raw = self._make_raw()
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(
                n_subjects=3,
                gender={"male": 3},  # aggregate says all male
                sexes=["male", "female", "male"],  # per-subject says #2 is female
            ),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        # sexes should take priority: subject 2 → female → code 2
        _enrich_raw_info_from_metadata(raw, metadata, 2)
        assert raw.info["subject_info"]["sex"] == 2


# ============================================================
# _build_sidecar_enrichment
# ============================================================


class TestBuildSidecarEnrichment:
    def test_none_metadata(self):
        assert _build_sidecar_enrichment(None) == {}

    def test_basic_acquisition(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256,
                n_channels=22,
                channel_types={"eeg": 22},
                reference="left mastoid",
                ground="AFz",
                hardware="BrainAmp",
                software="BCI2000",
                montage="standard_1020",
                sensor_type="Ag/AgCl",
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        entries = _build_sidecar_enrichment(metadata)
        assert entries["EEGReference"] == "left mastoid"
        assert entries["EEGGround"] == "AFz"
        assert entries["Manufacturer"] == "Brain Products"
        assert entries["ManufacturersModelName"] == "BrainAmp"
        assert entries["SoftwareVersions"] == "BCI2000"
        assert entries["EEGPlacementScheme"] == "10-20 system"
        assert entries["CapManufacturer"] == "Ag/AgCl"
        assert entries["RecordingType"] == "continuous"
        assert entries["SoftwareFilters"] == "n/a"

    def test_filter_details_bandpass_dict(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=22, channel_types={"eeg": 22}
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="imagery"),
            preprocessing=PreprocessingMetadata(
                bandpass={"low_cutoff_hz": 0.5, "high_cutoff_hz": 100.0},
                notch_hz=50,
            ),
        )
        entries = _build_sidecar_enrichment(metadata)
        assert "HardwareFilters" in entries
        hw = entries["HardwareFilters"]
        assert "Bandpass" in hw
        assert hw["Bandpass"]["low_cutoff_hz"] == 0.5
        assert "Notch" in hw
        assert hw["Notch"]["CutoffFrequency"] == [50]

    def test_filter_details_bandpass_list(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=22, channel_types={"eeg": 22}
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="imagery"),
            preprocessing=PreprocessingMetadata(
                bandpass=[0.5, 100.0],
            ),
        )
        entries = _build_sidecar_enrichment(metadata)
        bp = entries["HardwareFilters"]["Bandpass"]
        assert bp["LowCutoffFrequency"] == 0.5
        assert bp["HighCutoffFrequency"] == 100.0

    def test_filter_details_individual_hp_lp(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=22, channel_types={"eeg": 22}
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="imagery"),
            preprocessing=PreprocessingMetadata(
                highpass_hz=0.1,
                lowpass_hz=40.0,
            ),
        )
        entries = _build_sidecar_enrichment(metadata)
        bp = entries["HardwareFilters"]["Bandpass"]
        assert bp["LowCutoffFrequency"] == 0.1
        assert bp["HighCutoffFrequency"] == 40.0

    def test_task_description(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=22, channel_types={"eeg": 22}
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(
                paradigm="imagery",
                study_design="Four-class motor imagery",
            ),
        )
        entries = _build_sidecar_enrichment(metadata)
        assert entries["TaskDescription"] == "Four-class motor imagery"

    def test_institution(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=22, channel_types={"eeg": 22}
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="imagery"),
            documentation=DocumentationMetadata(
                institution="Graz University of Technology"
            ),
        )
        entries = _build_sidecar_enrichment(metadata)
        assert entries["InstitutionName"] == "Graz University of Technology"

    def test_montage_fallback(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256,
                n_channels=22,
                channel_types={"eeg": 22},
                montage="custom_montage",
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        entries = _build_sidecar_enrichment(metadata)
        assert entries["EEGPlacementScheme"] == "custom_montage"

    def test_cap_manufacturer_and_model(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256,
                n_channels=22,
                channel_types={"eeg": 22},
                cap_manufacturer="EasyCap",
                cap_model="actiCAP snap",
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        entries = _build_sidecar_enrichment(metadata)
        assert entries["CapManufacturer"] == "EasyCap"
        assert entries["CapManufacturersModelName"] == "actiCAP snap"

    def test_instructions(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=22, channel_types={"eeg": 22}
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(
                paradigm="imagery",
                instructions="Imagine moving your left or right hand.",
            ),
        )
        entries = _build_sidecar_enrichment(metadata)
        assert entries["Instructions"] == "Imagine moving your left or right hand."

    def test_cog_atlas_explicit(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=22, channel_types={"eeg": 22}
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(
                paradigm="imagery",
                cog_atlas_id="https://www.cognitiveatlas.org/task/id/custom123/",
            ),
        )
        entries = _build_sidecar_enrichment(metadata)
        assert (
            entries["CogAtlasID"] == "https://www.cognitiveatlas.org/task/id/custom123/"
        )

    def test_cog_atlas_fallback(self):
        # Motor imagery paradigm has a known fallback
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=22, channel_types={"eeg": 22}
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        entries = _build_sidecar_enrichment(metadata)
        assert (
            entries["CogAtlasID"]
            == "https://www.cognitiveatlas.org/task/id/trm_4c8a834779883/"
        )

        # P300 paradigm has a known fallback
        metadata_p300 = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=22, channel_types={"eeg": 22}
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="p300"),
        )
        entries_p300 = _build_sidecar_enrichment(metadata_p300)
        assert (
            entries_p300["CogAtlasID"]
            == "https://www.cognitiveatlas.org/task/id/tsk_GxjZBNiJorj1K/"
        )

        # SSVEP has no fallback
        metadata_ssvep = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=22, channel_types={"eeg": 22}
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="ssvep"),
        )
        entries_ssvep = _build_sidecar_enrichment(metadata_ssvep)
        assert "CogAtlasID" not in entries_ssvep

    def test_institution_address(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=22, channel_types={"eeg": 22}
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="imagery"),
            documentation=DocumentationMetadata(
                institution="TU Graz",
                institution_address="Inffeldgasse 13, 8010 Graz, Austria",
                institution_department="Institute of Neural Engineering",
            ),
        )
        entries = _build_sidecar_enrichment(metadata)
        assert entries["InstitutionName"] == "TU Graz"
        assert entries["InstitutionAddress"] == "Inffeldgasse 13, 8010 Graz, Austria"
        assert entries["InstitutionalDepartmentName"] == "Institute of Neural Engineering"

    def test_cap_manufacturer_fallback_to_sensor_type(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256,
                n_channels=22,
                channel_types={"eeg": 22},
                sensor_type="Ag/AgCl",
                # cap_manufacturer is None → should fall back to sensor_type
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        entries = _build_sidecar_enrichment(metadata)
        assert entries["CapManufacturer"] == "Ag/AgCl"

    def test_cap_manufacturer_overrides_sensor_type(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256,
                n_channels=22,
                channel_types={"eeg": 22},
                sensor_type="Ag/AgCl",
                cap_manufacturer="EasyCap",
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        entries = _build_sidecar_enrichment(metadata)
        # cap_manufacturer should win over sensor_type
        assert entries["CapManufacturer"] == "EasyCap"


# ============================================================
# _build_dataset_description_kwargs
# ============================================================


class TestBuildDatasetDescriptionKwargs:
    def _make_dataset(self, metadata=None):
        ds = MagicMock()
        ds.code = "TestDataset"
        ds.doi = "10.1234/test"
        ds.metadata = metadata
        return ds

    def test_without_metadata(self):
        ds = self._make_dataset(metadata=None)
        kwargs = _build_dataset_description_kwargs(ds)
        assert kwargs["name"] == "TestDataset"
        assert kwargs["dataset_type"] == "derivative"
        assert kwargs["source_datasets"][0]["DOI"] == "10.1234/test"
        assert "data_license" not in kwargs
        assert "authors" not in kwargs

    def test_with_documentation(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=22, channel_types={"eeg": 22}
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="imagery"),
            documentation=DocumentationMetadata(
                license="CC BY 4.0",
                investigators=["Alice", "Bob"],
                funding=["Grant 123"],
                data_url="https://example.com/data",
                associated_paper_doi="10.5678/paper",
                doi="10.9999/dataset",
            ),
        )
        ds = self._make_dataset(metadata=metadata)
        kwargs = _build_dataset_description_kwargs(ds)
        assert kwargs["data_license"] == "CC BY 4.0"
        assert kwargs["authors"] == ["Alice", "Bob"]
        assert kwargs["funding"] == ["Grant 123"]
        assert kwargs["source_datasets"][0]["URL"] == "https://example.com/data"
        assert "https://example.com/data" in kwargs["references_and_links"]
        assert "10.5678/paper" in kwargs["references_and_links"]
        assert kwargs["doi"] == "10.9999/dataset"

    def test_partial_documentation(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=22, channel_types={"eeg": 22}
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="imagery"),
            documentation=DocumentationMetadata(license="MIT"),
        )
        ds = self._make_dataset(metadata=metadata)
        kwargs = _build_dataset_description_kwargs(ds)
        assert kwargs["data_license"] == "MIT"
        assert "authors" not in kwargs
        assert "funding" not in kwargs
        assert "references_and_links" not in kwargs

    def test_hed_version_present(self):
        ds = self._make_dataset(metadata=None)
        kwargs = _build_dataset_description_kwargs(ds)
        assert kwargs["hed_version"] == "8.4.0"

    def test_ethics_and_acknowledgements(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=22, channel_types={"eeg": 22}
            ),
            participants=ParticipantMetadata(n_subjects=9),
            experiment=ExperimentMetadata(paradigm="imagery"),
            documentation=DocumentationMetadata(
                acknowledgements="Thanks to all participants.",
                how_to_acknowledge="Please cite doi:10.1234/foo.",
                ethics_approval=["IRB-2020-001", "EC-2020-042"],
            ),
        )
        ds = self._make_dataset(metadata=metadata)
        kwargs = _build_dataset_description_kwargs(ds)
        assert kwargs["acknowledgements"] == "Thanks to all participants."
        assert kwargs["how_to_acknowledge"] == "Please cite doi:10.1234/foo."
        assert kwargs["ethics_approvals"] == ["IRB-2020-001", "EC-2020-042"]


# ============================================================
# _update_participants_tsv
# ============================================================


class TestUpdateParticipantsTsv:
    def test_no_tsv_file(self, tmp_path):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(
                n_subjects=3, ages=[25, 30, 35], health_status="healthy"
            ),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        # Should not crash even if no file exists
        _update_participants_tsv(tmp_path, 1, metadata)

    def test_adds_age_and_group(self, tmp_path):
        # Create a minimal participants.tsv
        tsv_path = tmp_path / "participants.tsv"
        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["participant_id"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"participant_id": "sub-1"})

        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(
                n_subjects=1, ages=[25], health_status="healthy"
            ),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        _update_participants_tsv(tmp_path, 1, metadata)

        # Read back
        with open(tsv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["age"] == "25"
        assert rows[0]["group"] == "healthy"

    def test_uses_age_mean_fallback(self, tmp_path):
        tsv_path = tmp_path / "participants.tsv"
        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["participant_id"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"participant_id": "sub-1"})

        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(n_subjects=1, age_mean=27.5),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        _update_participants_tsv(tmp_path, 1, metadata)

        with open(tsv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        assert rows[0]["age"] == "27.5"

    def test_preserves_existing_data(self, tmp_path):
        tsv_path = tmp_path / "participants.tsv"
        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["participant_id", "age"], delimiter="\t"
            )
            writer.writeheader()
            writer.writerow({"participant_id": "sub-1", "age": "30"})

        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(
                n_subjects=1, ages=[25], health_status="healthy"
            ),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        _update_participants_tsv(tmp_path, 1, metadata)

        with open(tsv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        # Should preserve existing age=30, not overwrite with 25
        assert rows[0]["age"] == "30"
        assert rows[0]["group"] == "healthy"

    def test_updates_participants_json(self, tmp_path):
        tsv_path = tmp_path / "participants.tsv"
        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["participant_id"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"participant_id": "sub-1"})

        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(
                n_subjects=1, ages=[25], health_status="healthy"
            ),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        _update_participants_tsv(tmp_path, 1, metadata)

        json_path = tmp_path / "participants.json"
        assert json_path.exists()
        with open(json_path) as f:
            sidecar = json.load(f)
        assert "age" in sidecar
        assert sidecar["age"]["Units"] == "years"
        assert "group" in sidecar
        assert "sex" in sidecar
        assert "hand" in sidecar
        assert "species" in sidecar

    def test_none_metadata(self, tmp_path):
        _update_participants_tsv(tmp_path, 1, None)

    def test_none_participants(self, tmp_path):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(n_subjects=1),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        # health_status defaults to "healthy", but no ages — should still work
        tsv_path = tmp_path / "participants.tsv"
        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["participant_id"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"participant_id": "sub-1"})
        _update_participants_tsv(tmp_path, 1, metadata)

        with open(tsv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        # age should be n/a, group should be "healthy"
        assert rows[0]["age"] == "n/a"
        assert rows[0]["group"] == "healthy"

    def test_per_subject_sex_and_hand(self, tmp_path):
        tsv_path = tmp_path / "participants.tsv"
        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["participant_id"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"participant_id": "sub-1"})
            writer.writerow({"participant_id": "sub-2"})

        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(
                n_subjects=2,
                sexes=["male", "female"],
                handedness_list=["right", "left"],
            ),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        _update_participants_tsv(tmp_path, 1, metadata)
        _update_participants_tsv(tmp_path, 2, metadata)

        with open(tsv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        assert rows[0]["sex"] == "male"
        assert rows[0]["hand"] == "right"
        assert rows[1]["sex"] == "female"
        assert rows[1]["hand"] == "left"

    def test_species_column(self, tmp_path):
        tsv_path = tmp_path / "participants.tsv"
        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["participant_id"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"participant_id": "sub-1"})

        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(n_subjects=1),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        _update_participants_tsv(tmp_path, 1, metadata)

        with open(tsv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        assert rows[0]["species"] == "homo sapiens"


# ============================================================
# _update_electrodes_tsv
# ============================================================


class TestUpdateElectrodesTsv:
    def _make_bids_path(self, tmp_path, subject="1"):
        bp = MagicMock()
        bp.root = str(tmp_path)
        bp.subject = subject
        return bp

    def test_adds_material_and_type(self, tmp_path):
        # Create electrode TSV in the expected location
        elec_dir = tmp_path / "sub-1" / "ses-1" / "eeg"
        elec_dir.mkdir(parents=True)
        tsv_path = elec_dir / "sub-1_ses-1_electrodes.tsv"
        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "x", "y", "z"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"name": "C3", "x": "0", "y": "0", "z": "0"})
            writer.writerow({"name": "C4", "x": "1", "y": "1", "z": "1"})

        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256,
                n_channels=2,
                channel_types={"eeg": 2},
                electrode_type="cup",
                electrode_material="Ag/AgCl",
            ),
            participants=ParticipantMetadata(n_subjects=1),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        bp = self._make_bids_path(tmp_path)
        _update_electrodes_tsv(bp, metadata)

        with open(tsv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        assert rows[0]["type"] == "cup"
        assert rows[0]["material"] == "Ag/AgCl"
        assert rows[1]["type"] == "cup"
        assert rows[1]["material"] == "Ag/AgCl"

    def test_no_electrodes_file(self, tmp_path):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256,
                n_channels=2,
                channel_types={"eeg": 2},
                electrode_type="cup",
            ),
            participants=ParticipantMetadata(n_subjects=1),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        bp = self._make_bids_path(tmp_path)
        # Should not crash
        _update_electrodes_tsv(bp, metadata)

    def test_preserves_existing_columns(self, tmp_path):
        elec_dir = tmp_path / "sub-1" / "ses-1" / "eeg"
        elec_dir.mkdir(parents=True)
        tsv_path = elec_dir / "sub-1_ses-1_electrodes.tsv"
        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["name", "x", "y", "z", "type", "material"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerow(
                {
                    "name": "C3",
                    "x": "0",
                    "y": "0",
                    "z": "0",
                    "type": "ring",
                    "material": "Gold",
                }
            )

        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256,
                n_channels=2,
                channel_types={"eeg": 2},
                electrode_type="cup",
                electrode_material="Ag/AgCl",
            ),
            participants=ParticipantMetadata(n_subjects=1),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        bp = self._make_bids_path(tmp_path)
        _update_electrodes_tsv(bp, metadata)

        with open(tsv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        # Columns already existed — should NOT overwrite
        assert rows[0]["type"] == "ring"
        assert rows[0]["material"] == "Gold"

    def test_material_fallback_to_sensor_type(self, tmp_path):
        elec_dir = tmp_path / "sub-1" / "ses-1" / "eeg"
        elec_dir.mkdir(parents=True)
        tsv_path = elec_dir / "sub-1_ses-1_electrodes.tsv"
        with open(tsv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "x", "y", "z"], delimiter="\t")
            writer.writeheader()
            writer.writerow({"name": "C3", "x": "0", "y": "0", "z": "0"})

        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256,
                n_channels=2,
                channel_types={"eeg": 2},
                sensor_type="Tin",
                # electrode_material is None → should fall back to sensor_type
            ),
            participants=ParticipantMetadata(n_subjects=1),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        bp = self._make_bids_path(tmp_path)
        _update_electrodes_tsv(bp, metadata)

        with open(tsv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        assert rows[0]["material"] == "Tin"

    def test_none_metadata(self, tmp_path):
        bp = self._make_bids_path(tmp_path)
        _update_electrodes_tsv(bp, None)


# ============================================================
# _update_dataset_description_extra
# ============================================================


class TestUpdateDatasetDescriptionExtra:
    def test_adds_keywords(self, tmp_path):
        desc_path = tmp_path / "dataset_description.json"
        with open(desc_path, "w") as f:
            json.dump({"Name": "Test", "BIDSVersion": "1.11.0"}, f)

        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(n_subjects=1),
            experiment=ExperimentMetadata(paradigm="imagery"),
            documentation=DocumentationMetadata(keywords=["BCI", "EEG", "imagery"]),
        )
        _update_dataset_description_extra(tmp_path, metadata)

        with open(desc_path) as f:
            desc = json.load(f)
        assert desc["Keywords"] == ["BCI", "EEG", "imagery"]
        # Should preserve existing fields
        assert desc["Name"] == "Test"

    def test_keywords_from_tags(self, tmp_path):
        desc_path = tmp_path / "dataset_description.json"
        with open(desc_path, "w") as f:
            json.dump({"Name": "Test"}, f)

        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(n_subjects=1),
            experiment=ExperimentMetadata(paradigm="imagery"),
            tags=Tags(
                pathology=["healthy"],
                modality=["motor"],
                type=["imagery"],
            ),
        )
        _update_dataset_description_extra(tmp_path, metadata)

        with open(desc_path) as f:
            desc = json.load(f)
        assert "Keywords" in desc
        assert "healthy" in desc["Keywords"]
        assert "motor" in desc["Keywords"]
        assert "imagery" in desc["Keywords"]

    def test_no_overwrite_existing(self, tmp_path):
        desc_path = tmp_path / "dataset_description.json"
        with open(desc_path, "w") as f:
            json.dump(
                {"Name": "Test", "Keywords": ["existing_keyword"]},
                f,
            )

        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(n_subjects=1),
            experiment=ExperimentMetadata(paradigm="imagery"),
            documentation=DocumentationMetadata(keywords=["new_keyword"]),
        )
        _update_dataset_description_extra(tmp_path, metadata)

        with open(desc_path) as f:
            desc = json.load(f)
        # Should preserve existing keywords, not overwrite
        assert desc["Keywords"] == ["existing_keyword"]

    def test_none_metadata(self, tmp_path):
        _update_dataset_description_extra(tmp_path, None)

    def test_no_description_file(self, tmp_path):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(n_subjects=1),
            experiment=ExperimentMetadata(paradigm="imagery"),
            documentation=DocumentationMetadata(keywords=["BCI"]),
        )
        # Should not crash
        _update_dataset_description_extra(tmp_path, metadata)


# ============================================================
# _build_hed_sidecar_annotations
# ============================================================


class TestBuildHedSidecarAnnotations:
    def _make_dataset(self, paradigm, event_id, metadata=None):
        ds = MagicMock()
        ds.paradigm = paradigm
        ds.event_id = event_id
        ds.metadata = metadata
        return ds

    def test_imagery_default_tags(self):
        ds = self._make_dataset("imagery", {"left_hand": 1, "right_hand": 2, "feet": 3})
        hed = _build_hed_sidecar_annotations(ds)
        assert "left_hand" in hed
        assert "Imagine" in hed["left_hand"]
        assert "Experimental-stimulus" in hed["left_hand"]
        assert "right_hand" in hed
        assert "Experimental-stimulus" in hed["right_hand"]
        assert "feet" in hed
        assert "Experimental-stimulus" in hed["feet"]

    def test_p300_default_tags(self):
        ds = self._make_dataset("p300", {"Target": 2, "NonTarget": 1})
        hed = _build_hed_sidecar_annotations(ds)
        assert "Target" in hed
        assert "Target" in hed["Target"]
        assert "NonTarget" in hed
        assert "Non-target" in hed["NonTarget"]

    def test_ssvep_dynamic_tags(self):
        ds = self._make_dataset("ssvep", {"9.25": 1, "11.25": 2, "rest": 3})
        hed = _build_hed_sidecar_annotations(ds)
        assert "9.25" in hed
        assert "Visual-presentation" in hed["9.25"]
        assert "Label/9_25" in hed["9.25"]
        assert "11.25" in hed
        assert "Label/11_25" in hed["11.25"]
        assert "rest" in hed
        assert "Rest" in hed["rest"]

    def test_cvep_default_tags(self):
        ds = self._make_dataset("cvep", {"1.0": 101, "0.0": 100})
        hed = _build_hed_sidecar_annotations(ds)
        assert "1.0" in hed
        assert "Label/intensity_1_0" in hed["1.0"]
        assert "0.0" in hed
        assert "Label/intensity_0_0" in hed["0.0"]

    def test_rstate_default_tags(self):
        ds = self._make_dataset("rstate", {"closed": 1, "open": 2})
        hed = _build_hed_sidecar_annotations(ds)
        assert "closed" in hed
        assert "Close" in hed["closed"]
        assert "open" in hed
        assert "Open" in hed["open"]

    def test_metadata_override(self):
        custom_tags = {
            "left_hand": "Custom-tag, Left",
            "right_hand": "Custom-tag, Right",
        }
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(n_subjects=1),
            experiment=ExperimentMetadata(paradigm="imagery", hed_tags=custom_tags),
        )
        ds = self._make_dataset(
            "imagery", {"left_hand": 1, "right_hand": 2}, metadata=metadata
        )
        hed = _build_hed_sidecar_annotations(ds)
        # Custom tags take priority
        assert hed["left_hand"] == "Custom-tag, Left"
        assert hed["right_hand"] == "Custom-tag, Right"

    def test_metadata_override_partial_merges_defaults(self):
        """Partial hed_tags override merges with paradigm defaults."""
        custom_tags = {
            "left_hand": "Custom-tag, Left",
        }
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256, n_channels=2, channel_types={"eeg": 2}
            ),
            participants=ParticipantMetadata(n_subjects=1),
            experiment=ExperimentMetadata(paradigm="imagery", hed_tags=custom_tags),
        )
        ds = self._make_dataset(
            "imagery",
            {"left_hand": 1, "right_hand": 2, "unknown_event": 99},
            metadata=metadata,
        )
        hed = _build_hed_sidecar_annotations(ds)
        # Custom tag takes priority
        assert hed["left_hand"] == "Custom-tag, Left"
        # Non-overridden event still gets paradigm default
        assert "Imagine" in hed["right_hand"]
        # Unknown event gets Label fallback
        assert hed["unknown_event"] == "Sensory-event, (Label/unknown_event)"

    def test_unknown_events_get_label_fallback(self):
        ds = self._make_dataset("imagery", {"left_hand": 1, "unknown_event": 99})
        hed = _build_hed_sidecar_annotations(ds)
        assert "left_hand" in hed
        assert "Imagine" in hed["left_hand"]
        # Unknown events get Label fallback
        assert "unknown_event" in hed
        assert hed["unknown_event"] == "Sensory-event, (Label/unknown_event)"

    def test_no_metadata_uses_defaults(self):
        ds = self._make_dataset("imagery", {"left_hand": 1, "right_hand": 2})
        ds.metadata = None
        hed = _build_hed_sidecar_annotations(ds)
        assert "left_hand" in hed
        assert "right_hand" in hed

    def test_unknown_paradigm_uses_label_fallback(self):
        ds = self._make_dataset("unknown_paradigm", {"event1": 1, "event2": 2})
        hed = _build_hed_sidecar_annotations(ds)
        assert hed == {
            "event1": "Sensory-event, (Label/event1)",
            "event2": "Sensory-event, (Label/event2)",
        }

    def test_upper_limb_events(self):
        ds = self._make_dataset(
            "imagery",
            {
                "right_elbow_flexion": 1536,
                "right_hand_open": 1541,
                "rest": 1542,
            },
        )
        hed = _build_hed_sidecar_annotations(ds)
        assert "Flex" in hed["right_elbow_flexion"]
        assert "Elbow" in hed["right_elbow_flexion"]
        assert "Experimental-stimulus" in hed["right_elbow_flexion"]
        assert "Open" in hed["right_hand_open"]
        assert "Experimental-stimulus" in hed["right_hand_open"]
        assert "Rest" in hed["rest"]
        assert "Experimental-stimulus" in hed["rest"]

    def test_reaching_events(self):
        ds = self._make_dataset(
            "imagery",
            {"up_slow_near": 1, "down_fast_far": 8, "left_fast_near": 11},
        )
        hed = _build_hed_sidecar_annotations(ds)
        assert "Reach" in hed["up_slow_near"]
        assert "Upward" in hed["up_slow_near"]
        assert "Experimental-stimulus" in hed["up_slow_near"]
        assert "Downward" in hed["down_fast_far"]
        assert "Experimental-stimulus" in hed["down_fast_far"]
        assert "Left" in hed["left_fast_near"]
        assert "Experimental-stimulus" in hed["left_fast_near"]

    def test_letter_events(self):
        ds = self._make_dataset("imagery", {"letter_a": 1, "letter_v": 10})
        hed = _build_hed_sidecar_annotations(ds)
        assert "Write" in hed["letter_a"]
        assert "Label/a" in hed["letter_a"]
        assert "Experimental-stimulus" in hed["letter_a"]
        assert "Label/v" in hed["letter_v"]
        assert "Experimental-stimulus" in hed["letter_v"]

    def test_mental_task_events(self):
        ds = self._make_dataset(
            "imagery",
            {
                "math": 1,
                "letter": 2,
                "rotation": 3,
                "count": 4,
                "baseline": 5,
                "subtraction": 6,
            },
        )
        hed = _build_hed_sidecar_annotations(ds)
        # Each mental task must be distinguishable via Label qualifier
        assert (
            hed["math"]
            == "Sensory-event, Experimental-stimulus, Cue, (Imagine, Think, (Label/math))"
        )
        assert (
            hed["letter"]
            == "Sensory-event, Experimental-stimulus, Cue, (Imagine, Think, (Label/letter))"
        )
        assert (
            hed["rotation"]
            == "Sensory-event, Experimental-stimulus, Cue, (Imagine, Think, (Label/rotation))"
        )
        assert (
            hed["subtraction"]
            == "Sensory-event, Experimental-stimulus, Cue, (Imagine, Think, (Label/subtraction))"
        )
        assert "Count" in hed["count"]
        assert "Experimental-stimulus" in hed["count"]
        assert "Rest" in hed["baseline"]
        assert "Experimental-stimulus" in hed["baseline"]

    def test_label_fallback_sanitizes_dots(self):
        """Label fallback sanitizes dots for HED nameClass compliance."""
        ds = self._make_dataset(
            paradigm="unknown_paradigm",
            event_id={"13.5": 1, "event name": 2},
        )
        hed = _build_hed_sidecar_annotations(ds)
        assert hed["13.5"] == "Sensory-event, (Label/13_5)"
        assert hed["event name"] == "Sensory-event, (Label/event_name)"

    def test_all_events_covered_no_gaps(self):
        """Every event from dataset.event_id must appear in the output."""
        ds = self._make_dataset(
            "imagery",
            {"left_hand": 1, "right_hand": 2, "novel_future_event": 99},
        )
        hed = _build_hed_sidecar_annotations(ds)
        assert set(hed.keys()) == {"left_hand", "right_hand", "novel_future_event"}


# ============================================================
# _update_events_json_sidecar
# ============================================================


class TestUpdateEventsJsonWithHed:
    def _make_bids_path(self, tmp_path, events_json_content=None):
        """Create a mock BIDSPath that points to a real events.json file."""
        events_json_path = tmp_path / "sub-1_task-imagery_events.json"
        if events_json_content is not None:
            with open(events_json_path, "w") as f:
                json.dump(events_json_content, f)

        bp = MagicMock()
        # .copy().update(suffix="events", extension=".json").fpath
        mock_copy = MagicMock()
        mock_update = MagicMock()
        mock_update.fpath = events_json_path
        mock_copy.update.return_value = mock_update
        bp.copy.return_value = mock_copy
        return bp, events_json_path

    def test_adds_hed_to_trial_type(self, tmp_path):
        content = {"trial_type": {"Description": "Event type."}}
        bp, json_path = self._make_bids_path(tmp_path, content)
        hed_tags = {"left_hand": "Sensory-event, Cue, (Imagine, (Move, Hand))"}
        _update_events_json_sidecar(bp, hed_tags, None)

        with open(json_path) as f:
            sidecar = json.load(f)
        assert "HED" in sidecar["trial_type"]
        assert sidecar["trial_type"]["HED"]["left_hand"] == hed_tags["left_hand"]

    def test_existing_hed_preserved_on_conflict(self, tmp_path):
        existing_hed = {"left_hand": "Existing-tag"}
        content = {
            "trial_type": {
                "Description": "Event type.",
                "HED": existing_hed,
            }
        }
        bp, json_path = self._make_bids_path(tmp_path, content)
        hed_tags = {"left_hand": "New-tag", "right_hand": "Right-tag"}
        _update_events_json_sidecar(bp, hed_tags, None)

        with open(json_path) as f:
            sidecar = json.load(f)
        # Existing entries preserved on conflict
        assert sidecar["trial_type"]["HED"]["left_hand"] == "Existing-tag"
        # New entries merged in
        assert sidecar["trial_type"]["HED"]["right_hand"] == "Right-tag"

    def test_creates_trial_type_if_missing(self, tmp_path):
        content = {"onset": {"Description": "Event onset."}}
        bp, json_path = self._make_bids_path(tmp_path, content)
        hed_tags = {"Target": "Sensory-event, Target"}
        _update_events_json_sidecar(bp, hed_tags, None)

        with open(json_path) as f:
            sidecar = json.load(f)
        assert "trial_type" in sidecar
        assert "HED" in sidecar["trial_type"]
        assert sidecar["trial_type"]["HED"]["Target"] == hed_tags["Target"]

    def test_no_events_json_no_op(self, tmp_path):
        bp, json_path = self._make_bids_path(tmp_path, events_json_content=None)
        # File doesn't exist — should be a no-op
        hed_tags = {"left_hand": "Sensory-event, Cue"}
        _update_events_json_sidecar(bp, hed_tags, None)
        assert not json_path.exists()

    def test_empty_hed_tags_no_op(self, tmp_path):
        content = {"trial_type": {"Description": "Event type."}}
        bp, json_path = self._make_bids_path(tmp_path, content)
        _update_events_json_sidecar(bp, {}, None)

        with open(json_path) as f:
            sidecar = json.load(f)
        # Should not have added HED key
        assert "HED" not in sidecar.get("trial_type", {})


# ============================================================
# _extract_references_from_docstring
# ============================================================


class TestExtractReferencesFromDocstring:
    def test_single_reference(self):
        doc = """My dataset.

        references
        ----------

        .. [1] Smith, J., 2020. A great paper. Journal, 1(2), 3-4.
               https://doi.org/10.1234/example
        """
        refs = _extract_references_from_docstring(doc)
        assert "Smith, J., 2020." in refs
        assert "https://doi.org/10.1234/example" in refs
        # Should be joined into one reference
        assert refs.count("\n\n") == 0

    def test_multiple_references(self):
        doc = """Dataset.

        references
        ----------

        .. [1] Author A, 2020. Paper one. Journal A.

        .. [2] Author B, 2021. Paper two. Journal B.
        """
        refs = _extract_references_from_docstring(doc)
        assert "Author A" in refs
        assert "Author B" in refs
        # Two references separated by blank line
        assert "\n\n" in refs

    def test_no_references(self):
        doc = "A dataset with no references section."
        refs = _extract_references_from_docstring(doc)
        assert refs == ""

    def test_empty_docstring(self):
        assert _extract_references_from_docstring("") == ""
        assert _extract_references_from_docstring(None) == ""


# ============================================================
# HED tree visualization helpers
# ============================================================


class TestSplitHedTopLevel:
    def test_flat_tags(self):
        result = _split_hed_top_level("Sensory-event, Cue, Rest")
        assert result == ["Sensory-event", "Cue", "Rest"]

    def test_single_group(self):
        result = _split_hed_top_level("Sensory-event, (Imagine, (Move, Hand))")
        assert result == ["Sensory-event", "(Imagine, (Move, Hand))"]

    def test_nested_groups(self):
        s = "A, (B, (C, D)), E"
        result = _split_hed_top_level(s)
        assert result == ["A", "(B, (C, D))", "E"]

    def test_multiple_groups(self):
        s = "(A, B), (C, D)"
        result = _split_hed_top_level(s)
        assert result == ["(A, B)", "(C, D)"]

    def test_empty_string(self):
        assert _split_hed_top_level("") == []

    def test_single_tag(self):
        assert _split_hed_top_level("Sensory-event") == ["Sensory-event"]

    def test_whitespace_handling(self):
        result = _split_hed_top_level("  A , B ,  C ")
        assert result == ["A", "B", "C"]


class TestHedElementToTree:
    def test_simple_tag(self):
        label, children = _hed_element_to_tree("Sensory-event")
        assert label == "Sensory-event"
        assert children == []

    def test_leaf_only_group_inline(self):
        """Groups with no sub-groups are rendered as inline labels."""
        label, children = _hed_element_to_tree("(Right, Hand)")
        assert label == "Right, Hand"
        assert children == []

    def test_group_with_subgroup(self):
        label, children = _hed_element_to_tree("(Move, (Right, Hand))")
        assert label == "Move"
        assert len(children) == 1
        assert children[0] == ("Right, Hand", [])

    def test_deeply_nested(self):
        tag = "(Imagine, (Move, (Right, Hand)))"
        label, children = _hed_element_to_tree(tag)
        assert label == "Imagine"
        assert len(children) == 1
        assert children[0][0] == "Move"
        assert children[0][1] == [("Right, Hand", [])]

    def test_multiple_subgroups(self):
        tag = "(Imagine, (Move, (Right, Hand)), (Move, (Right, Foot)))"
        label, children = _hed_element_to_tree(tag)
        assert label == "Imagine"
        assert len(children) == 2
        assert children[0][0] == "Move"
        assert children[1][0] == "Move"

    def test_mixed_tags_and_subgroups(self):
        tag = "(Grasp, Hand, (Label/lateral))"
        label, children = _hed_element_to_tree(tag)
        assert label == "Grasp"
        assert len(children) == 2
        assert children[0] == ("Hand", [])
        assert children[1] == ("Label/lateral", [])

    def test_label_tag(self):
        tag = "(Label/9_25)"
        label, children = _hed_element_to_tree(tag)
        assert label == "Label/9_25"
        assert children == []


class TestRenderHedTree:
    def test_single_leaf(self):
        nodes = [("Sensory-event", [])]
        lines = _render_hed_tree(nodes)
        assert len(lines) == 1
        assert "└─ Sensory-event" in lines[0]

    def test_multiple_leaves(self):
        nodes = [("A", []), ("B", []), ("C", [])]
        lines = _render_hed_tree(nodes)
        assert len(lines) == 3
        assert "├─ A" in lines[0]
        assert "├─ B" in lines[1]
        assert "└─ C" in lines[2]

    def test_nested_tree(self):
        nodes = [
            ("Sensory-event", []),
            ("Cue", []),
            ("Imagine", [("Move", [("Right, Hand", [])])]),
        ]
        lines = _render_hed_tree(nodes)
        assert len(lines) == 5
        assert "├─ Sensory-event" in lines[0]
        assert "├─ Cue" in lines[1]
        assert "└─ Imagine" in lines[2]
        assert "└─ Move" in lines[3]
        assert "└─ Right, Hand" in lines[4]

    def test_box_characters(self):
        """Verify proper box-drawing characters are used."""
        nodes = [("A", []), ("B", [("C", [])])]
        lines = _render_hed_tree(nodes)
        assert "\u251c\u2500" in lines[0]  # ├─
        assert "\u2514\u2500" in lines[1]  # └─
        assert "\u2514\u2500" in lines[2]  # └─

    def test_branching_tree(self):
        """Multiple children at same level produce │ continuation lines."""
        nodes = [("Imagine", [("Move", [("Right, Hand", [])]), ("Move", [("Foot", [])])])]
        lines = _render_hed_tree(nodes)
        assert len(lines) == 5
        # Should have │ continuation character for first child branch
        assert "\u2502" in lines[2]


class TestReadmeHedSection:
    """Test that the HED Event Annotations section renders in the README."""

    def _make_dataset(self):
        ds = MagicMock()
        ds.code = "TestDataset"
        ds.paradigm = "imagery"
        ds.event_id = {"left_hand": 1, "right_hand": 2}
        ds.interval = [0, 3]
        ds.n_sessions = 1
        ds.doi = None
        type(ds).__doc__ = "Test dataset."
        type(ds).__name__ = "TestDataset"
        ds.metadata = None
        return ds

    def test_hed_section_present(self):
        ds = self._make_dataset()
        readme = _build_readme(ds)
        assert "HED Event Annotations" in readme
        assert "Schema: HED 8.4.0" in readme

    def test_hed_event_names_shown(self):
        ds = self._make_dataset()
        readme = _build_readme(ds)
        assert "left_hand" in readme
        assert "right_hand" in readme

    def test_hed_tree_has_box_chars(self):
        ds = self._make_dataset()
        readme = _build_readme(ds)
        assert "\u251c\u2500" in readme  # ├─
        assert "\u2514\u2500" in readme  # └─

    def test_hed_tree_shows_tag_content(self):
        ds = self._make_dataset()
        readme = _build_readme(ds)
        assert "Sensory-event" in readme
        assert "Imagine" in readme

    def test_hed_schema_link(self):
        ds = self._make_dataset()
        readme = _build_readme(ds)
        assert "hedtags.org/hed-schema-browser" in readme

    def test_no_hed_section_for_unknown_paradigm(self):
        ds = self._make_dataset()
        ds.paradigm = "unknown_paradigm_xyz"
        ds.event_id = {"event_a": 1}
        readme = _build_readme(ds)
        # Should still have HED section due to Label fallback
        assert "HED Event Annotations" in readme
        assert "Label/event_a" in readme

    def test_p300_events(self):
        ds = self._make_dataset()
        ds.paradigm = "p300"
        ds.event_id = {"Target": 1, "NonTarget": 2}
        readme = _build_readme(ds)
        assert "Target" in readme
        assert "Visual-presentation" in readme

    def test_resting_state_events(self):
        ds = self._make_dataset()
        ds.paradigm = "rstate"
        ds.event_id = {"open": 1, "closed": 2}
        readme = _build_readme(ds)
        assert "Experiment-structure" in readme
        assert "Rest" in readme


# ============================================================
# _build_readme
# ============================================================


class TestBuildReadme:
    def _make_dataset(self, metadata=None):
        """Create a mock dataset with typical attributes."""
        ds = MagicMock()
        ds.code = "TestDataset"
        ds.paradigm = "imagery"
        ds.doi = "10.1234/test"
        ds.subject_list = list(range(1, 11))
        ds.n_sessions = 2
        ds.event_id = {"left_hand": 1, "right_hand": 2}
        ds.interval = [0, 4]
        type(
            ds
        ).__doc__ = """Test Motor Imagery dataset.

        A dataset for testing BIDS README generation.

        references
        ----------

        .. [1] Test Author, 2024. Test paper. Test Journal, 1, 1-10.
        """
        if metadata is not None:
            ds.metadata = metadata
        else:
            ds.metadata = None
        return ds

    def test_basic_readme_no_metadata(self):
        ds = self._make_dataset()
        readme = _build_readme(ds)
        assert "TestDataset" in readme
        assert "==========" in readme
        assert "Test Motor Imagery dataset." in readme
        assert "Code: TestDataset" in readme
        assert "Paradigm: imagery" in readme
        assert "DOI: 10.1234/test" in readme
        assert "Subjects: 10" in readme
        assert "Sessions per subject: 2" in readme
        assert "left_hand=1" in readme
        assert "Trial interval: [0, 4] s" in readme
        assert "Generated by MOABB" in readme

    def test_readme_with_acquisition(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=512.0,
                n_channels=64,
                channel_types={"eeg": 60, "eog": 4},
                hardware="BrainAmp DC",
                reference="FCz",
                montage="standard_1005",
                sensors=["Fp1", "Fp2", "F3"],
            ),
            participants=ParticipantMetadata(n_subjects=10),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        ds = self._make_dataset(metadata)
        readme = _build_readme(ds)
        assert "Acquisition" in readme
        assert "Sampling rate: 512.0 Hz" in readme
        assert "Number of channels: 64" in readme
        assert "eeg=60" in readme
        assert "eog=4" in readme
        assert "Hardware: BrainAmp DC" in readme
        assert "Reference: FCz" in readme
        assert "Fp1, Fp2, F3" in readme

    def test_readme_with_participants(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256.0, n_channels=32, channel_types={"eeg": 32}
            ),
            participants=ParticipantMetadata(
                n_subjects=20,
                health_status="healthy",
                age_mean=25.5,
                age_std=4.2,
                gender={"male": 12, "female": 8},
                handedness="all right-handed",
                bci_experience="naive",
            ),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        ds = self._make_dataset(metadata)
        readme = _build_readme(ds)
        assert "Participants" in readme
        assert "Number of subjects: 20" in readme
        assert "Health status: healthy" in readme
        assert "mean=25.5" in readme
        assert "std=4.2" in readme
        assert "male=12" in readme
        assert "Handedness: all right-handed" in readme
        assert "BCI experience: naive" in readme

    def test_readme_with_experiment(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256.0, n_channels=32, channel_types={"eeg": 32}
            ),
            participants=ParticipantMetadata(n_subjects=10),
            experiment=ExperimentMetadata(
                paradigm="p300",
                task_type="row_col_speller",
                n_classes=2,
                class_labels=["Target", "NonTarget"],
                study_design="Visual P300 speller",
                feedback_type="visual",
                stimulus_type="character flash",
                synchronicity="synchronous",
            ),
        )
        ds = self._make_dataset(metadata)
        readme = _build_readme(ds)
        assert "Experimental Protocol" in readme
        assert "Task type: row_col_speller" in readme
        assert "Study design: Visual P300 speller" in readme
        assert "Synchronicity: synchronous" in readme

    def test_readme_with_documentation(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256.0, n_channels=32, channel_types={"eeg": 32}
            ),
            participants=ParticipantMetadata(n_subjects=10),
            experiment=ExperimentMetadata(paradigm="imagery"),
            documentation=DocumentationMetadata(
                doi="10.1234/test",
                license="CC BY 4.0",
                investigators=["Alice", "Bob"],
                institution="Test University",
                country="DE",
                repository="PhysioNet",
                data_url="https://example.com/data",
                publication_year=2024,
                funding=["Grant A", "Grant B"],
            ),
        )
        ds = self._make_dataset(metadata)
        readme = _build_readme(ds)
        assert "Documentation" in readme
        assert "License: CC BY 4.0" in readme
        assert "Alice, Bob" in readme
        assert "Test University" in readme
        assert "PhysioNet" in readme
        assert "Publication year: 2024" in readme
        assert "Grant A; Grant B" in readme

    def test_readme_with_preprocessing(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256.0, n_channels=32, channel_types={"eeg": 32}
            ),
            participants=ParticipantMetadata(n_subjects=10),
            experiment=ExperimentMetadata(paradigm="imagery"),
            preprocessing=PreprocessingMetadata(
                data_state="raw",
                artifact_methods=["ICA", "threshold"],
                re_reference="average",
            ),
        )
        ds = self._make_dataset(metadata)
        readme = _build_readme(ds)
        assert "Preprocessing" in readme
        assert "Data state: raw" in readme
        assert "ICA, threshold" in readme
        assert "Re-reference: average" in readme

    def test_readme_with_signal_processing(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256.0, n_channels=32, channel_types={"eeg": 32}
            ),
            participants=ParticipantMetadata(n_subjects=10),
            experiment=ExperimentMetadata(paradigm="imagery"),
            signal_processing=SignalProcessingMetadata(
                classifiers=["LDA", "SVM"],
                feature_extraction=["CSP", "PSD"],
            ),
        )
        ds = self._make_dataset(metadata)
        readme = _build_readme(ds)
        assert "Signal Processing" in readme
        assert "LDA, SVM" in readme
        assert "CSP, PSD" in readme

    def test_readme_with_cross_validation(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256.0, n_channels=32, channel_types={"eeg": 32}
            ),
            participants=ParticipantMetadata(n_subjects=10),
            experiment=ExperimentMetadata(paradigm="imagery"),
            cross_validation=CrossValidationMetadata(
                cv_method="5-fold",
                cv_folds=5,
                evaluation_type=["within-subject"],
            ),
        )
        ds = self._make_dataset(metadata)
        readme = _build_readme(ds)
        assert "Cross-Validation" in readme
        assert "Method: 5-fold" in readme
        assert "Folds: 5" in readme
        assert "within-subject" in readme

    def test_readme_with_bci_application(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256.0, n_channels=32, channel_types={"eeg": 32}
            ),
            participants=ParticipantMetadata(n_subjects=10),
            experiment=ExperimentMetadata(paradigm="imagery"),
            bci_application=BCIApplicationMetadata(
                applications=["speller", "wheelchair"],
                environment="lab",
                online_feedback=True,
            ),
        )
        ds = self._make_dataset(metadata)
        readme = _build_readme(ds)
        assert "BCI Application" in readme
        assert "speller, wheelchair" in readme
        assert "Environment: lab" in readme
        assert "Online feedback: True" in readme

    def test_readme_with_tags(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256.0, n_channels=32, channel_types={"eeg": 32}
            ),
            participants=ParticipantMetadata(n_subjects=10),
            experiment=ExperimentMetadata(paradigm="imagery"),
            tags=Tags(
                pathology=["Healthy"],
                modality=["Motor", "Visual"],
                type=["Motor Imagery"],
            ),
        )
        ds = self._make_dataset(metadata)
        readme = _build_readme(ds)
        assert "Tags" in readme
        assert "Pathology: Healthy" in readme
        assert "Motor, Visual" in readme
        assert "Motor Imagery" in readme

    def test_readme_with_paradigm_specific(self):
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256.0, n_channels=32, channel_types={"eeg": 32}
            ),
            participants=ParticipantMetadata(n_subjects=10),
            experiment=ExperimentMetadata(paradigm="ssvep"),
            paradigm_specific=ParadigmSpecificMetadata(
                detected_paradigm="ssvep",
                stimulus_frequencies_hz=[9.25, 11.25, 13.25],
            ),
        )
        ds = self._make_dataset(metadata)
        readme = _build_readme(ds)
        assert "Paradigm-Specific Parameters" in readme
        assert "Detected paradigm: ssvep" in readme
        assert "9.25, 11.25, 13.25" in readme

    def test_readme_includes_references(self):
        ds = self._make_dataset()
        readme = _build_readme(ds)
        assert "References" in readme
        # Original docstring reference
        assert "Test Author, 2024" in readme
        # Standard BIDS references
        assert "MNE-BIDS" in readme
        assert "EEG-BIDS" in readme

    def test_readme_includes_footer(self):
        ds = self._make_dataset()
        readme = _build_readme(ds)
        assert "---" in readme
        assert "Generated by MOABB" in readme
        assert "github.com/NeuroTechX/moabb" in readme

    def test_empty_sections_omitted(self):
        """Sections with no populated fields should not appear."""
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256.0, n_channels=32, channel_types={"eeg": 32}
            ),
            participants=ParticipantMetadata(n_subjects=10),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        ds = self._make_dataset(metadata)
        readme = _build_readme(ds)
        # These sections should NOT appear since no data is provided
        assert "Signal Processing\n" not in readme
        assert "Cross-Validation\n" not in readme
        assert "BCI Application\n" not in readme
        assert "Paradigm-Specific Parameters\n" not in readme

    def test_na_values_omitted(self):
        """Values set to 'n/a' should not appear in the README."""
        metadata = DatasetMetadata(
            acquisition=AcquisitionMetadata(
                sampling_rate=256.0,
                n_channels=32,
                channel_types={"eeg": 32},
                hardware="n/a",
                ground="n/a",
            ),
            participants=ParticipantMetadata(n_subjects=10),
            experiment=ExperimentMetadata(paradigm="imagery"),
        )
        ds = self._make_dataset(metadata)
        readme = _build_readme(ds)
        assert "Hardware: n/a" not in readme
        assert "Ground: n/a" not in readme


# ============================================================
# Parametrized field-coverage tests for _build_readme
# ============================================================


def _minimal_metadata(**overrides):
    """Build a DatasetMetadata with only required fields plus overrides."""
    kwargs = dict(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0, n_channels=32, channel_types={"eeg": 32}
        ),
        participants=ParticipantMetadata(n_subjects=10),
        experiment=ExperimentMetadata(paradigm="imagery"),
    )
    kwargs.update(overrides)
    return DatasetMetadata(**kwargs)


def _mock_ds(metadata=None):
    ds = MagicMock()
    ds.code = "TestDS"
    ds.paradigm = "imagery"
    ds.doi = None
    ds.subject_list = list(range(1, 11))
    ds.n_sessions = 1
    ds.event_id = {"left_hand": 1, "right_hand": 2}
    ds.interval = [0, 4]
    type(ds).__doc__ = "Test dataset."
    ds.metadata = metadata
    return ds


class TestReadmeAcquisitionFields:
    """Parametrize over every AcquisitionMetadata field."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("sampling_rate", 512.0, "512.0 Hz"),
            ("n_channels", 64, "64"),
            ("channel_types", {"eeg": 60, "eog": 4}, "eeg=60, eog=4"),
            ("sensors", ["Fp1", "Fp2"], "Fp1, Fp2"),
            ("sensor_type", "Ag/AgCl wet", "Ag/AgCl wet"),
            ("reference", "linked mastoids", "linked mastoids"),
            ("ground", "AFz", "AFz"),
            ("hardware", "BrainAmp DC", "BrainAmp DC"),
            ("software", "BrainVision Recorder", "BrainVision Recorder"),
            ("filters", "0.1-100 Hz bandpass", "0.1-100 Hz bandpass"),
            ("line_freq", 60.0, "60.0 Hz"),
            ("montage", "standard_1020", "standard_1020"),
            ("impedance_threshold_kohm", 20.0, "20.0 kOhm"),
            ("cap_manufacturer", "EasyCap", "EasyCap"),
            ("cap_model", "actiCAP", "actiCAP"),
            ("electrode_type", "ring", "ring"),
            ("electrode_material", "Tin", "Tin"),
        ],
    )
    def test_field_present(self, field, value, expected):
        kwargs = dict(sampling_rate=256.0, n_channels=32, channel_types={"eeg": 32})
        kwargs[field] = value
        meta = _minimal_metadata(acquisition=AcquisitionMetadata(**kwargs))
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmeAuxiliaryChannelsFields:
    """Parametrize over AuxiliaryChannelsMetadata fields."""

    @pytest.mark.parametrize(
        "aux_kwargs,expected",
        [
            (
                {"has_eog": True, "eog_channels": 2},
                "EOG (2 ch)",
            ),
            (
                {
                    "has_eog": True,
                    "eog_channels": 4,
                    "eog_type": ["horizontal", "vertical"],
                },
                "EOG (4 ch, horizontal, vertical)",
            ),
            (
                {"has_emg": True, "emg_channels": 3},
                "EMG (3 ch)",
            ),
            (
                {"other_physiological": ["ECG", "respiration"]},
                "ECG",
            ),
        ],
    )
    def test_aux_field(self, aux_kwargs, expected):
        acq = AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=32,
            channel_types={"eeg": 32},
            auxiliary_channels=AuxiliaryChannelsMetadata(**aux_kwargs),
        )
        meta = _minimal_metadata(acquisition=acq)
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmeParticipantFields:
    """Parametrize over every ParticipantMetadata field."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("n_subjects", 20, "20"),
            ("health_status", "patients", "patients"),
            ("clinical_population", "stroke", "stroke"),
            ("age_mean", 30.5, "mean=30.5"),
            ("age_std", 5.2, "std=5.2"),
            ("age_min", 18.0, "min=18.0"),
            ("age_max", 55.0, "max=55.0"),
            ("gender", {"male": 10, "female": 10}, "male=10"),
            ("handedness", "all right-handed", "all right-handed"),
            ("bci_experience", "experienced", "experienced"),
            ("species", "mus musculus", "mus musculus"),
        ],
    )
    def test_field_present(self, field, value, expected):
        kwargs = {"n_subjects": 10}
        kwargs[field] = value
        meta = _minimal_metadata(participants=ParticipantMetadata(**kwargs))
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmeExperimentFields:
    """Parametrize over every ExperimentMetadata field."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("paradigm", "ssvep", "ssvep"),
            ("task_type", "frequency_detection", "frequency_detection"),
            ("n_classes", 4, "4"),
            ("class_labels", ["A", "B"], "A, B"),
            ("trial_duration", 5.0, "5.0 s"),
            ("trials_per_class", {"A": 40, "B": 40}, "A=40"),
            ("tasks", ["task1", "task2"], "task1, task2"),
            ("study_design", "Oddball paradigm", "Oddball paradigm"),
            ("study_domain", "cognitive neuroscience", "cognitive neuroscience"),
            ("feedback_type", "auditory", "auditory"),
            ("stimulus_type", "visual flash", "visual flash"),
            ("stimulus_modalities", ["visual", "auditory"], "visual, auditory"),
            ("primary_modality", "visual", "visual"),
            ("synchronicity", "asynchronous", "asynchronous"),
            ("mode", "online", "online"),
            ("has_training_test_split", True, "True"),
            ("instructions", "Imagine moving your left hand", "Imagine moving"),
            ("cog_atlas_id", "https://cogat.org/123", "cogat.org/123"),
            ("cog_po_id", "https://cogpo.org/456", "cogpo.org/456"),
            (
                "stimulus_presentation",
                {"SoftwareName": "PsychoPy", "SoftwareVersion": "3.0"},
                "SoftwareName=PsychoPy",
            ),
        ],
    )
    def test_field_present(self, field, value, expected):
        kwargs = {"paradigm": "imagery"}
        kwargs[field] = value
        meta = _minimal_metadata(experiment=ExperimentMetadata(**kwargs))
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmeDocumentationFields:
    """Parametrize over every DocumentationMetadata field."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("doi", "10.1234/test", "10.1234/test"),
            ("description", "A motor imagery dataset", "A motor imagery dataset"),
            ("investigators", ["Alice", "Bob"], "Alice, Bob"),
            ("institution", "MIT", "MIT"),
            ("country", "US", "US"),
            ("repository", "OpenNeuro", "OpenNeuro"),
            ("data_url", "https://example.com/data", "https://example.com/data"),
            ("license", "CC BY 4.0", "CC BY 4.0"),
            ("publication_year", 2023, "2023"),
            ("senior_author", "Prof. Smith", "Prof. Smith"),
            ("contact_info", ["smith@mit.edu"], "smith@mit.edu"),
            ("associated_paper_doi", "10.5678/paper", "10.5678/paper"),
            ("funding", ["NIH R01", "ERC"], "NIH R01; ERC"),
            ("institution_address", "77 Mass Ave", "77 Mass Ave"),
            ("institution_department", "BCS", "BCS"),
            ("ethics_approval", ["IRB-2023-001"], "IRB-2023-001"),
            ("acknowledgements", "Thanks to lab members", "Thanks to lab members"),
            ("how_to_acknowledge", "Please cite Smith 2023", "Please cite Smith 2023"),
            ("keywords", ["EEG", "BCI", "motor imagery"], "EEG, BCI, motor imagery"),
        ],
    )
    def test_field_present(self, field, value, expected):
        kwargs = {field: value}
        meta = _minimal_metadata(documentation=DocumentationMetadata(**kwargs))
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmePreprocessingFields:
    """Parametrize over every PreprocessingMetadata field."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("data_state", "raw", "raw"),
            ("preprocessing_applied", True, "True"),
            ("preprocessing_steps", ["bandpass", "ICA"], "bandpass, ICA"),
            ("artifact_methods", ["ICA", "ASR"], "ICA, ASR"),
            ("re_reference", "average", "average"),
            ("downsampled_to_hz", 128.0, "128.0 Hz"),
            ("epoch_window", [-0.2, 0.8], "[-0.2, 0.8]"),
            ("notes", "Data was collected in two batches", "two batches"),
        ],
    )
    def test_field_present(self, field, value, expected):
        kwargs = {field: value}
        meta = _minimal_metadata(preprocessing=PreprocessingMetadata(**kwargs))
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmeFilterDetailsFields:
    """Parametrize over filter detail fields (now flat on PreprocessingMetadata)."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("highpass_hz", 0.1, "0.1 Hz"),
            ("lowpass_hz", 40.0, "40.0 Hz"),
            ("bandpass", [0.5, 45.0], "[0.5, 45.0]"),
            ("notch_hz", 50.0, "50.0 Hz"),
            ("filter_type", "butterworth", "butterworth"),
            ("filter_order", 4, "4"),
        ],
    )
    def test_field_present(self, field, value, expected):
        kwargs = {field: value}
        meta = _minimal_metadata(preprocessing=PreprocessingMetadata(**kwargs))
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmeSignalProcessingFields:
    """Parametrize over every SignalProcessingMetadata field."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("classifiers", ["LDA", "SVM"], "LDA, SVM"),
            ("feature_extraction", ["CSP", "PSD"], "CSP, PSD"),
            ("spatial_filters", ["Laplacian", "CCA"], "Laplacian, CCA"),
        ],
    )
    def test_field_present(self, field, value, expected):
        kwargs = {field: value}
        meta = _minimal_metadata(signal_processing=SignalProcessingMetadata(**kwargs))
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmeFrequencyBandsFields:
    """Parametrize over frequency_bands dict entries."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("delta", [0.5, 4.0], "delta=[0.5, 4.0] Hz"),
            ("theta", [4.0, 8.0], "theta=[4.0, 8.0] Hz"),
            ("alpha", [8.0, 13.0], "alpha=[8.0, 13.0] Hz"),
            ("mu", [8.0, 12.0], "mu=[8.0, 12.0] Hz"),
            ("beta", [13.0, 30.0], "beta=[13.0, 30.0] Hz"),
            ("gamma", [30.0, 100.0], "gamma=[30.0, 100.0] Hz"),
            ("analyzed_range", [1.0, 45.0], "analyzed=[1.0, 45.0] Hz"),
        ],
    )
    def test_field_present(self, field, value, expected):
        kwargs = {field: value}
        meta = _minimal_metadata(
            signal_processing=SignalProcessingMetadata(frequency_bands=kwargs)
        )
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmeCrossValidationFields:
    """Parametrize over every CrossValidationMetadata field."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("cv_method", "leave-one-out", "leave-one-out"),
            ("cv_folds", 10, "10"),
            ("evaluation_type", ["within-subject", "cross-subject"], "within-subject"),
        ],
    )
    def test_field_present(self, field, value, expected):
        kwargs = {field: value}
        meta = _minimal_metadata(cross_validation=CrossValidationMetadata(**kwargs))
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmePerformanceFields:
    """Parametrize over performance dict entries."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("accuracy_percent", 85.5, "85.5%"),
            ("itr_bits_per_min", 42.3, "42.3 bits/min"),
            ("auc", 0.92, "0.92"),
            ("kappa", 0.75, "0.75"),
            ("other_metrics", {"f1": 0.88}, "f1=0.88"),
        ],
    )
    def test_field_present(self, field, value, expected):
        kwargs = {field: value}
        meta = _minimal_metadata(performance=kwargs)
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmeBCIApplicationFields:
    """Parametrize over every BCIApplicationMetadata field."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("applications", ["speller", "wheelchair"], "speller, wheelchair"),
            ("environment", "clinical", "clinical"),
            ("online_feedback", True, "True"),
        ],
    )
    def test_field_present(self, field, value, expected):
        kwargs = {field: value}
        meta = _minimal_metadata(bci_application=BCIApplicationMetadata(**kwargs))
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmeParadigmSpecificFields:
    """Parametrize over every ParadigmSpecificMetadata field."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("detected_paradigm", "ssvep", "ssvep"),
            ("stimulus_frequencies_hz", [9.25, 11.25], "[9.25, 11.25] Hz"),
            ("frequency_resolution_hz", 0.25, "0.25 Hz"),
            ("code_type", "m-sequence", "m-sequence"),
            ("code_length", 127, "127"),
            ("n_targets", 36, "36"),
            ("n_repetitions", 15, "15"),
            ("isi_ms", 125.0, "125.0 ms"),
            ("soa_ms", 200.0, "200.0 ms"),
            ("imagery_tasks", ["left_hand", "feet"], "left_hand, feet"),
            ("cue_duration_s", 1.25, "1.25 s"),
            ("imagery_duration_s", 4.0, "4.0 s"),
        ],
    )
    def test_field_present(self, field, value, expected):
        kwargs = {field: value}
        meta = _minimal_metadata(paradigm_specific=ParadigmSpecificMetadata(**kwargs))
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmeDataStructureFields:
    """Parametrize over every DataStructureMetadata field."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("n_trials", 200, "200"),
            ("n_trials_per_class", {"A": 100, "B": 100}, "A=100"),
            ("n_blocks", 6, "6"),
            ("block_duration_s", 60.0, "60.0 s"),
            ("trials_context", "per_class", "per_class"),
        ],
    )
    def test_field_present(self, field, value, expected):
        kwargs = {field: value}
        meta = _minimal_metadata(data_structure=DataStructureMetadata(**kwargs))
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmeTagsFields:
    """Parametrize over every Tags field."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("pathology", ["Healthy", "Epilepsy"], "Healthy, Epilepsy"),
            ("modality", ["Motor", "Visual"], "Motor, Visual"),
            ("type", ["Motor Imagery"], "Motor Imagery"),
        ],
    )
    def test_field_present(self, field, value, expected):
        kwargs = {field: value}
        meta = _minimal_metadata(tags=Tags(**kwargs))
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmeExternalLinksFields:
    """Parametrize over external_links dict entries."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("source_url", "https://openneuro.org/ds001", "openneuro.org/ds001"),
            ("ftp_url", "ftp://example.com/data", "ftp://example.com/data"),
            (
                "alternative_urls",
                {"Mirror": "https://mirror.com/data"},
                "https://mirror.com/data",
            ),
        ],
    )
    def test_field_present(self, field, value, expected):
        kwargs = {field: value}
        meta = _minimal_metadata(external_links=kwargs)
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme


class TestReadmeTopLevelFields:
    """Parametrize over DatasetMetadata top-level fields."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("file_format", "GDF", "GDF"),
            ("abstract", "This study explores motor imagery BCI.", "This study explores"),
            ("methodology", "We recorded 64-channel EEG.", "64-channel EEG"),
            ("sessions", ["train", "test"], "train, test"),
            ("contributing_labs", ["Lab A", "Lab B"], "Lab A, Lab B"),
            ("n_contributing_labs", 5, "5"),
        ],
    )
    def test_field_present(self, field, value, expected):
        meta = _minimal_metadata(**{field: value})
        readme = _build_readme(_mock_ds(meta))
        assert expected in readme

    def test_runs_per_session_shown_when_gt_1(self):
        meta = _minimal_metadata(runs_per_session=3)
        readme = _build_readme(_mock_ds(meta))
        assert "Runs per session: 3" in readme

    def test_runs_per_session_hidden_when_1(self):
        meta = _minimal_metadata(runs_per_session=1)
        readme = _build_readme(_mock_ds(meta))
        assert "Runs per session" not in readme

    def test_data_processed_shown_when_true(self):
        meta = _minimal_metadata(data_processed=True)
        readme = _build_readme(_mock_ds(meta))
        assert "Data preprocessed: True" in readme

    def test_data_processed_hidden_when_false(self):
        meta = _minimal_metadata(data_processed=False)
        readme = _build_readme(_mock_ds(meta))
        assert "Data preprocessed" not in readme
