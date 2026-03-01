import inspect
import logging
import re
import warnings

import mne
import numpy as np
import pandas as pd
import pytest

import moabb.datasets as db
import moabb.datasets.compound_dataset as db_compound
from moabb.datasets import (
    BNCI2014_001,
    Cattan2019_VR,
    Kojima2024A,
    Kojima2024B,
    Shin2017A,
    Shin2017B,
)
from moabb.datasets.base import (
    BaseDataset,
    LocalBIDSDataset,
    _summary_table,
    is_abbrev,
    is_camel_kebab_case,
)
from moabb.datasets.braininvaders import BI2012, BI2013a
from moabb.datasets.compound_dataset import CompoundDataset
from moabb.datasets.compound_dataset.utils import compound_dataset_list
from moabb.datasets.fake import FakeDataset, FakeVirtualRealityDataset
from moabb.datasets.kojima2024b import EVENTS
from moabb.datasets.metadata import (
    DATASET_METADATA_CATALOG,
    AcquisitionMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    get_dataset_metadata,
)
from moabb.datasets.physionet_mi import PhysionetMI
from moabb.datasets.upper_limb import Ofner2017
from moabb.datasets.utils import bids_metainfo, block_rep, dataset_list
from moabb.paradigms import P300
from moabb.utils import aliases_list


_ = mne.set_log_level("CRITICAL")


class TestRegex:
    def test_is_abbrev(self):
        assert is_abbrev("a", "a-")
        assert is_abbrev("a", "a0")
        assert is_abbrev("a", "ab")
        assert not is_abbrev("a", "aA")
        assert not is_abbrev("a", "Aa")
        assert not is_abbrev("a", "-a")
        assert not is_abbrev("a", "0a")
        assert not is_abbrev("a", "ba")
        assert not is_abbrev("a", "a ")

    def test_is_camell_kebab_case(self):
        assert is_camel_kebab_case("Aa")
        assert is_camel_kebab_case("aAa")
        assert is_camel_kebab_case("Aa-a")
        assert is_camel_kebab_case("1Aa-1a1")
        assert is_camel_kebab_case("AB")
        assert not is_camel_kebab_case("A ")
        assert not is_camel_kebab_case(" A")
        assert not is_camel_kebab_case("A A")
        assert not is_camel_kebab_case("A_")
        assert not is_camel_kebab_case("_A")
        assert not is_camel_kebab_case("A_A")


class Test_Datasets:
    @pytest.mark.parametrize("paradigm", ["imagery", "p300", "ssvep"])
    def test_fake_dataset(self, paradigm):
        """This test will insure the basedataset works."""
        n_subjects = 3
        n_sessions = 2
        n_runs = 2

        ds = FakeDataset(
            n_sessions=n_sessions,
            n_runs=n_runs,
            n_subjects=n_subjects,
            paradigm=paradigm,
        )
        data = ds.get_data()

        # we should get a dict
        assert isinstance(data, dict)

        # we get the right number of subject
        assert len(data) == n_subjects

        # right number of session
        assert len(data[1]) == n_sessions

        # right number of run
        assert len(data[1]["0"]) == n_runs

        # We should get a raw array at the end
        assert isinstance(data[1]["0"]["0"], mne.io.BaseRaw)

        # bad subject id must raise error
        with pytest.raises(ValueError):
            ds.get_data([1000])

    @pytest.mark.parametrize("paradigm", ["imagery", "p300", "ssvep"])
    def test_fake_dataset_seed(self, paradigm):
        """this test will insure the fake dataset's random seed works"""
        n_subjects = 3
        n_sessions = 2
        n_runs = 2
        seed = 12

        ds1 = FakeDataset(
            n_sessions=n_sessions,
            n_runs=n_runs,
            n_subjects=n_subjects,
            paradigm=paradigm,
            seed=seed,
        )
        ds2 = FakeDataset(
            n_sessions=n_sessions,
            n_runs=n_runs,
            n_subjects=n_subjects,
            paradigm=paradigm,
            seed=seed,
        )
        X1, _, _ = ds1.get_data()
        X2, _, _ = ds2.get_data()
        X3, _, _ = ds2.get_data()

        # All the arrays should be equal:
        assert np.allclose(X1, X2)
        assert np.allclose(X3, X3)

    @pytest.mark.parametrize("paradigm", ["imagery", "p300", "ssvep"])
    @pytest.mark.filterwarnings("ignore:TSV file is empty.*:RuntimeWarning")
    @pytest.mark.filterwarnings("ignore:Converting data files to EDF.*:RuntimeWarning")
    def test_cache_dataset(self, paradigm, tmp_path, caplog):
        """This test will ensure that the cache is working."""
        dataset = FakeDataset(paradigm=paradigm)

        # Save cache:
        with caplog.at_level(logging.INFO):
            _ = dataset.get_data(
                subjects=[1],
                cache_config={
                    "save_raw": True,
                    "use": True,
                    "overwrite_raw": False,
                    "path": tmp_path,
                },
            )
        expected = [
            "Attempting to retrieve cache .* suffix-eeg",
            "No cache found at",
            "Starting caching .* suffix-eeg",
            "Finished caching .* suffix-eeg",
        ]
        assert len(expected) == len(caplog.messages)
        for i, regex in enumerate(expected):
            assert re.search(regex, caplog.messages[i])

        # Load cache:
        caplog.clear()
        with caplog.at_level(logging.INFO):
            _ = dataset.get_data(
                subjects=[1],
                cache_config={
                    "save_raw": True,
                    "use": True,
                    "overwrite_raw": False,
                    "path": tmp_path,
                },
            )
        expected = [
            "Attempting to retrieve cache .* suffix-eeg",
            "Finished reading cache .* suffix-eeg",
        ]
        assert len(expected) == len(caplog.messages)
        for i, regex in enumerate(expected):
            assert re.search(regex, caplog.messages[i])

        # Overwrite cache:
        caplog.clear()
        with caplog.at_level(logging.INFO):
            _ = dataset.get_data(
                subjects=[1],
                cache_config={
                    "save_raw": True,
                    "use": True,
                    "overwrite_raw": True,
                    "path": tmp_path,
                },
            )
        expected = [
            "Starting erasing cache .* suffix-eeg",
            "Finished erasing cache .* suffix-eeg",
            "Starting caching .* suffix-eeg",
            "Finished caching .* suffix-eeg",
        ]
        assert len(expected) == len(caplog.messages)
        for i, regex in enumerate(expected):
            assert re.search(regex, caplog.messages[i])

        metainfo = bids_metainfo(tmp_path)
        dataframe = pd.DataFrame(metainfo).T
        subjects = dataframe["subject"].unique()
        assert len(subjects) == 1
        assert subjects[0] == "1"
        assert len(dataframe["session"].unique()) == 2

    def test_dataset_accept(self):
        """Verify that accept licence is working."""
        # Only BaseShin2017 (bbci_eeg_fnirs) for now
        for ds in [Shin2017A(), Shin2017B()]:
            # if the data is already downloaded:
            if mne.get_config("MNE_DATASETS_BBCIFNIRS_PATH") is None:
                with pytest.raises(AttributeError):
                    ds.get_data([1])

    def test_datasets_init(self, caplog):
        codes = []
        deprecated_list, _, _ = zip(*aliases_list)

        logger_name = "moabb.datasets.base"
        log_level = "WARNING"
        caplog.set_level(log_level, logger=logger_name)

        for ds in dataset_list:
            kwargs = {}
            if inspect.signature(ds).parameters.get("accept"):
                kwargs["accept"] = True

            caplog.clear()
            obj = ds(**kwargs)

            if type(obj).__name__ not in deprecated_list:
                assert len(caplog.records) == 0

            assert obj is not None
            if type(obj).__name__ not in deprecated_list:
                codes.append(obj.code)

        # Check that all codes are unique:
        assert len(codes) == len(set(codes))

    def test_depreciated_datasets_init(self, caplog):
        depreciated_names, _, _ = zip(*aliases_list)
        for ds in db.__dict__.values():
            if ds in dataset_list:
                continue
            if not (inspect.isclass(ds) and issubclass(ds, BaseDataset)):
                continue
            kwargs = {}
            if inspect.signature(ds).parameters.get("accept"):
                kwargs["accept"] = True
            caplog.set_level("WARNING", logger="moabb.utils")
            caplog.clear()
            obj = ds(**kwargs)
            assert len(caplog.records) > 0
            assert obj is not None
            assert ds.__name__ in depreciated_names

    def test_dataset_docstring_table(self):
        # The dataset summary table will be automatically added to the docstring of
        # all the datasets listed in the moabb/datasets/summary_*.csv files.
        depreciated_names, _, _ = zip(*aliases_list)
        for ds in dataset_list:
            if "Fake" in ds.__name__:
                continue
            if ds.__name__ in depreciated_names:
                continue
            assert ".. admonition:: Dataset summary" in ds.__doc__

    def test_metadata_docstring_sections_auto_generated(self):
        class AutoDocMetadataDataset(BaseDataset):
            """Dataset doc for auto metadata generation."""

            METADATA = DatasetMetadata(
                acquisition=AcquisitionMetadata(
                    sampling_rate=256.0,
                    n_channels=8,
                    channel_types={"eeg": 8},
                    hardware="BrainAmp",
                    sensor_type="Ag/AgCl",
                    montage="10-20",
                ),
                participants=ParticipantMetadata(
                    n_subjects=4,
                    health_status="healthy",
                    age_mean=29.5,
                    age_min=24,
                    age_max=35,
                    handedness={"right": 4},
                    bci_experience="experienced",
                ),
                experiment=ExperimentMetadata(
                    paradigm="imagery",
                    task_type="left_right_hand",
                    feedback_type="visual",
                ),
                documentation=DocumentationMetadata(doi="10.1093/gigascience/giz002"),
                preprocessing=PreprocessingMetadata(
                    bandpass=[0.5, 40.0],
                    preprocessing_steps=["common average reference"],
                ),
            )

            def __init__(self):
                super().__init__(
                    subjects=[1],
                    sessions_per_subject=1,
                    events={"left_hand": 1, "right_hand": 2},
                    code="AutoDocMetadataDataset",
                    interval=[0, 1],
                    paradigm="imagery",
                )

            def _get_single_subject_data(self, subject):
                return {}

            def data_path(
                self,
                subject,
                path=None,
                force_update=False,
                update_path=None,
                verbose=None,
            ):
                return []

        doc = AutoDocMetadataDataset.__doc__
        assert ".. admonition:: Participants" in doc
        assert "- **Population**: healthy" in doc
        assert "- **Age**: 29.5 (range: 24-35) years" in doc
        assert ".. admonition:: Equipment" in doc
        assert "- **Amplifier**: BrainAmp" in doc
        assert ".. admonition:: Preprocessing" in doc
        assert "- **Bandpass filter**: 0.5-40 Hz" in doc
        assert ".. admonition:: Data Access" in doc
        assert "- **DOI**: 10.1093/gigascience/giz002" in doc
        assert ".. admonition:: Experimental Protocol" in doc
        assert "- **Paradigm**: imagery" in doc

    def test_metadata_docstring_sections_do_not_duplicate_manual_admonitions(self):
        class ManualParticipantsDataset(BaseDataset):
            """
            Dataset doc with a manual participants section.

            .. admonition:: Participants

                - **Population**: healthy
            """

            METADATA = DatasetMetadata(
                acquisition=AcquisitionMetadata(
                    sampling_rate=256.0,
                    n_channels=8,
                    channel_types={"eeg": 8},
                ),
                participants=ParticipantMetadata(n_subjects=4, health_status="healthy"),
                experiment=ExperimentMetadata(paradigm="imagery"),
            )

            def __init__(self):
                super().__init__(
                    subjects=[1],
                    sessions_per_subject=1,
                    events={"left_hand": 1, "right_hand": 2},
                    code="ManualParticipantsDataset",
                    interval=[0, 1],
                    paradigm="imagery",
                )

            def _get_single_subject_data(self, subject):
                return {}

            def data_path(
                self,
                subject,
                path=None,
                force_update=False,
                update_path=None,
                verbose=None,
            ):
                return []

        doc = ManualParticipantsDataset.__doc__
        assert doc.count(".. admonition:: Participants") == 1

    def test_feedback_section_auto_generated(self):
        class FeedbackTestDataset(BaseDataset):
            """A test dataset for feedback section."""

            def __init__(self):
                super().__init__(
                    subjects=[1],
                    sessions_per_subject=1,
                    events={"left_hand": 1, "right_hand": 2},
                    code="FeedbackTestDataset",
                    interval=[0, 1],
                    paradigm="imagery",
                )

            def _get_single_subject_data(self, subject):
                return {}

            def data_path(
                self,
                subject,
                path=None,
                force_update=False,
                update_path=None,
                verbose=None,
            ):
                return []

        doc = FeedbackTestDataset.__doc__
        assert "Found an issue with this dataset?" in doc
        assert "https://github.com/NeuroTechX/moabb/issues/new" in doc
        assert "FeedbackTestDataset" in doc
        assert "Report an Issue on GitHub" in doc

    def test_feedback_section_not_duplicated(self):
        class FeedbackNoDupDataset(BaseDataset):
            """A test dataset.

            .. admonition:: Found an issue with this dataset?
               :class: tip

               Custom feedback section already present.
            """

            def __init__(self):
                super().__init__(
                    subjects=[1],
                    sessions_per_subject=1,
                    events={"left_hand": 1, "right_hand": 2},
                    code="FeedbackNoDupDataset",
                    interval=[0, 1],
                    paradigm="imagery",
                )

            def _get_single_subject_data(self, subject):
                return {}

            def data_path(
                self,
                subject,
                path=None,
                force_update=False,
                update_path=None,
                verbose=None,
            ):
                return []

        doc = FeedbackNoDupDataset.__doc__
        assert doc.count("Found an issue with this dataset?") == 1

    def test_feedback_section_not_added_to_fake_datasets(self):
        assert "Found an issue with this dataset?" not in (FakeDataset.__doc__ or "")

    def test_completeness_summary_table(self):
        # The dataset summary table will be automatically added to the docstring of
        # all the datasets listed in the moabb/datasets/summary_*.csv files.
        depreciated_names, _, _ = zip(*aliases_list)
        for ds in dataset_list:
            if "Fake" in ds.__name__:
                continue
            if ds.__name__ in depreciated_names:
                continue
            assert ds.__name__ in _summary_table.index

    def test_dataset_list(self):
        if aliases_list:
            depreciated_list, _, _ = zip(*aliases_list)
        else:
            pass
        all_datasets = [
            c
            for c in db.__dict__.values()
            if (
                inspect.isclass(c)
                and issubclass(c, BaseDataset)
                # and c.__name__ not in depreciated_list
            )
        ]
        assert len(dataset_list) == len(all_datasets)
        assert set(dataset_list) == set(all_datasets)

    def test_bad_subject_name(self):
        ds = FakeDataset()
        ds.subject_list = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match=r"Subject names must be "):
            ds.get_data()

    def test_bad_session_name(self):
        class BadSessionDataset(FakeDataset):
            def _get_single_subject_data(self, subject):
                data = super()._get_single_subject_data(subject)
                data["session_0"] = data.pop("0")
                return data

        ds = BadSessionDataset()
        with pytest.raises(ValueError, match=r"Session names must be "):
            ds.get_data()

    def test_bad_run_name(self):
        class BadRunDataset(FakeDataset):
            def _get_single_subject_data(self, subject):
                data = super()._get_single_subject_data(subject)
                data["0"]["run_0"] = data["0"].pop("0")
                return data

        ds = BadRunDataset()
        with pytest.raises(ValueError, match=r"Run names must be "):
            ds.get_data()


class TestSubjectSessionFiltering:
    """Test subject and session filtering at construction time."""

    @pytest.mark.parametrize(
        "dataset_cls, kwargs, expected_len, all_len",
        [
            (PhysionetMI, dict(subjects=[1, 2, 3]), 3, 109),
            (BNCI2014_001, dict(subjects=[1, 5, 9]), 3, 9),
            (Ofner2017, dict(subjects=[1, 2]), 2, 15),
            (FakeDataset, dict(subjects=[1, 2, 3]), 3, 10),
        ],
        ids=["PhysionetMI", "BNCI2014_001", "Ofner2017", "FakeDataset"],
    )
    def test_subject_filtering(self, dataset_cls, kwargs, expected_len, all_len):
        ds = dataset_cls(**kwargs)
        assert ds.subject_list == kwargs["subjects"]
        assert len(ds.subject_list) == expected_len
        assert len(ds.all_subjects) == all_len

    @pytest.mark.parametrize(
        "dataset_cls, kwargs",
        [
            (PhysionetMI, dict(subjects=[999])),
            (BNCI2014_001, dict(subjects=[0, 100])),
            (FakeDataset, dict(subjects=[50])),
        ],
    )
    def test_invalid_subjects_raises(self, dataset_cls, kwargs):
        with pytest.raises(ValueError, match="Invalid subjects"):
            dataset_cls(**kwargs)

    def test_default_backward_compat(self):
        assert PhysionetMI().subject_list == list(range(1, 110))
        assert BNCI2014_001().subject_list == list(range(1, 10))

    def test_session_filtering_at_construction(self):
        ds = FakeDataset(n_subjects=2, n_sessions=3, sessions=[0])
        data = ds.get_data()
        for sess_data in data.values():
            assert list(sess_data.keys()) == ["0"]

    def test_combined_subject_and_session_filtering(self):
        ds = FakeDataset(n_subjects=5, n_sessions=3, subjects=[1, 2], sessions=[0, 1])
        assert ds.subject_list == [1, 2]
        data = ds.get_data()
        assert set(data.keys()) == {1, 2}
        for sess_data in data.values():
            assert set(sess_data.keys()) == {"0", "1"}

    def test_all_subjects_is_immutable_copy(self):
        ds = PhysionetMI(subjects=[1, 2])
        ds.all_subjects.append(999)
        assert 999 not in ds.all_subjects

    def test_all_datasets_accept_subjects_param(self):
        """Every dataset class in dataset_list accepts subjects or sessions."""
        for cls in dataset_list:
            sig = inspect.signature(cls.__init__)
            params = set(sig.parameters.keys())
            # Check own params or inherited from parent
            parent = cls.__mro__[1]
            if parent.__name__ not in ("object", "ABC"):
                psig = inspect.signature(parent.__init__)
                params |= set(psig.parameters.keys())
            assert (
                "subjects" in params or "sessions" in params
            ), f"{cls.__name__} missing subjects/sessions param"


class TestVirtualRealityDataset:
    def test_canary(self):
        assert Cattan2019_VR() is not None

    def test_warning_if_parameters_false(self):
        with pytest.warns(UserWarning):
            Cattan2019_VR(virtual_reality=False, screen_display=False)

    def test_get_block_repetition(self):
        ds = FakeVirtualRealityDataset()
        subject = 5
        block = 3
        repetition = 4
        _, _, ret = ds.get_block_repetition(P300(), [subject], [block], [repetition])
        assert ret.subject.unique()[0] == subject
        assert ret.run.unique()[0] == block_rep(block, repetition, ds.n_repetitions)


class TestDeprecatedParams:
    """Test deprecated PascalCase parameter names and new defaults."""

    def test_bi2012_deprecated_training(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = BI2012(Training=True)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "Training" in str(dep_warnings[0].message)
            assert "training" in str(dep_warnings[0].message)
            assert ds.training is True

    def test_bi2012_deprecated_online(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = BI2012(Online=False)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1
            assert "Online" in str(dep_warnings[0].message)
            assert ds.online is False

    def test_bi2012_new_defaults(self):
        ds = BI2012()
        assert ds.training is True
        assert ds.online is False

    def test_bi2012_snake_case_params(self):
        ds = BI2012(training=False, online=True)
        assert ds.training is False
        assert ds.online is True

    def test_bi2013a_deprecated_params(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = BI2013a(NonAdaptive=True, Adaptive=False, Training=True, Online=False)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 4
            assert ds.non_adaptive is True
            assert ds.adaptive is False
            assert ds.training is True
            assert ds.online is False

    def test_bi2013a_new_defaults(self):
        ds = BI2013a()
        assert ds.non_adaptive is True
        assert ds.adaptive is False
        assert ds.training is True
        assert ds.online is False

    def test_bi2013a_snake_case_params(self):
        ds = BI2013a(non_adaptive=False, adaptive=True, training=False, online=True)
        assert ds.non_adaptive is False
        assert ds.adaptive is True
        assert ds.training is False
        assert ds.online is True

    def test_bi2012_unexpected_kwarg_raises(self):
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            BI2012(Foo=True)

    def test_bi2013a_unexpected_kwarg_raises(self):
        with pytest.raises(TypeError, match="unexpected keyword arguments"):
            BI2013a(Foo=True)

    def test_physionet_new_defaults(self):
        ds = PhysionetMI()
        assert ds.imagined is True
        assert ds.executed is False
        assert len(ds.hand_runs) == 3
        assert len(ds.feet_runs) == 3

    def test_physionet_explicit_old_values(self):
        ds = PhysionetMI(imagined=True, executed=False)
        assert ds.imagined is True
        assert ds.executed is False
        assert len(ds.hand_runs) == 3
        assert len(ds.feet_runs) == 3

    def test_ofner2017_new_defaults(self):
        ds = Ofner2017()
        assert ds.imagined is True
        assert ds.executed is True
        assert ds.n_sessions == 2

    def test_ofner2017_explicit_old_values(self):
        ds = Ofner2017(imagined=True, executed=False)
        assert ds.imagined is True
        assert ds.executed is False
        assert ds.n_sessions == 1

    def test_cattan2019_vr_new_defaults(self):
        ds = Cattan2019_VR()
        assert ds.virtual_reality is True
        assert ds.personal_computer is True


class TestCompoundDataset:
    def setup_method(self):
        self.paradigm = "p300"
        self.n_sessions = 2
        self.n_subjects = 2
        self.n_runs = 2
        self.ds = FakeDataset(
            n_sessions=self.n_sessions,
            n_runs=self.n_runs,
            n_subjects=self.n_subjects,
            event_list=["Target", "NonTarget"],
            paradigm=self.paradigm,
        )

    @pytest.mark.parametrize("sessions, runs", [(None, None), ("0", "0"), (["0"], ["0"])])
    def test_fake_dataset(self, sessions, runs):
        """This test will insure the compoundataset works."""
        subjects_list = [(self.ds, 1, sessions, runs)]
        compound_data = CompoundDataset(
            subjects_list,
            code="CompoundDataset-test",
            interval=[0, 1],
        )

        data = compound_data.get_data()

        # Check event_id is correctly set
        assert compound_data.event_id == self.ds.event_id

        # Check data origin is correctly set
        assert data[1]["data_origin"] == subjects_list[0]

        # Check data type
        assert isinstance(data, dict)
        assert isinstance(data[1]["0"]["0"], mne.io.BaseRaw)

        # Check data size
        assert len(data) == 1
        expected_session_number = self.n_sessions if sessions is None else 1
        assert len(data[1]) == expected_session_number
        expected_runs_number = self.n_runs if runs is None else 1
        assert len(data[1]["0"]) == expected_runs_number

    def test_get_data_invalid_subject(self):
        """Test that requesting data for a non-existing subject raises ValueError"""
        subjects_list = [(self.ds, 1, None, None)]
        compound_data = CompoundDataset(
            subjects_list, code="CompoundDataset-test-invalid", interval=[0, 1]
        )

        # bad subject id must raise error
        with pytest.raises(ValueError):
            compound_data.get_data([1000])  # Request data for subject 1000

    def test_compound_dataset_composition(self):
        # Test we can compound two instance of CompoundDataset into a new one.

        # Create an instance of CompoundDataset with one subject
        subjects_list = [(self.ds, 1, None, None)]
        compound_dataset = CompoundDataset(
            subjects_list,
            code="CompoundDataset-test",
            interval=[0, 1],
        )

        # Add it two time to a subjects_list
        subjects_list = [compound_dataset, compound_dataset]
        compound_data = CompoundDataset(
            subjects_list,
            code="CompoundDataset-test",
            interval=[0, 1],
        )

        # Assert there is only one source dataset in the compound dataset
        assert len(compound_data.datasets) == 1

        # Assert that the coumpouned dataset has two times more subject than the original one.
        data = compound_data.get_data()
        assert len(data) == 2

    def test_get_sessions_per_subject(self):
        # define a new fake dataset with two times more sessions:
        self.ds2 = FakeDataset(
            n_sessions=self.n_sessions * 2,
            n_runs=self.n_runs,
            n_subjects=self.n_subjects,
            event_list=["Target", "NonTarget"],
            paradigm=self.ds.paradigm,
        )

        # Add the two datasets to a CompoundDataset
        subjects_list = [(self.ds, 1, None, None), (self.ds2, 1, None, None)]
        compound_dataset = CompoundDataset(
            subjects_list,
            code="CompoundDataset",
            interval=[0, 1],
        )

        # Assert there are two source datasets (ds and ds2) in the compound dataset
        assert len(compound_dataset.datasets) == 2

        # Test private method _get_sessions_per_subject returns the minimum number of sessions per subjects
        assert compound_dataset._get_sessions_per_subject() == self.n_sessions

    def test_event_id_correctly_updated(self):
        # define a new fake dataset with different event_id
        self.ds2 = FakeDataset(
            n_sessions=self.n_sessions,
            n_runs=self.n_runs,
            n_subjects=self.n_subjects,
            event_list=["Target2", "NonTarget2"],
            paradigm=self.ds.paradigm,
        )

        # Add the two datasets to a CompoundDataset
        subjects_list = [(self.ds, 1, None, None), (self.ds2, 1, None, None)]

        compound_dataset = CompoundDataset(
            subjects_list,
            code="CompoundDataset",
            interval=[0, 1],
        )

        # Check that the event_id of the compound_dataset is the same has the first dataset
        assert compound_dataset.event_id == self.ds.event_id

        # Check event_id get correctly updated when taking a subject from dataset 2
        data = compound_dataset.get_data(subjects=[2])
        assert compound_dataset.event_id == self.ds2.event_id
        assert len(data.keys()) == 1

        # Check event_id is correctly put back when taking a subject from the first dataset
        data = compound_dataset.get_data(subjects=[1])
        assert compound_dataset.event_id == self.ds.event_id
        assert len(data.keys()) == 1

    def test_datasets_init(self):
        codes = []
        for ds in compound_dataset_list:
            kwargs = {}
            if inspect.signature(ds).parameters.get("accept"):
                kwargs["accept"] = True
            obj = ds(**kwargs)
            assert obj is not None
            codes.append(obj.code)

        # Check that all codes are unique:
        assert len(codes) == len(set(codes))

    def test_dataset_list(self):
        if aliases_list:
            depreciated_list, _, _ = zip(*aliases_list)
        else:
            depreciated_list = []
        all_datasets = [
            c
            for c in db_compound.__dict__.values()
            if (
                inspect.isclass(c)
                and issubclass(c, CompoundDataset)
                and c.__name__ not in depreciated_list
                and c.__name__ != "CompoundDataset"
            )
        ]
        assert len(compound_dataset_list) == len(all_datasets)
        assert set(compound_dataset_list) == set(all_datasets)


class TestData:
    @pytest.fixture
    def dataset(self):
        return BNCI2014_001()

    @pytest.fixture
    def data(self, dataset):
        return dataset.get_data(subjects=[1])

    def test_epochs(self, data, dataset):
        # values computed form moabb 0.5:
        # using raw = data[1]['session_T']['run_0']
        raw = data[1]["0train"]["0"]
        assert len(raw) == 96735
        events = np.array(
            [
                [250, 0, 4],
                [2253, 0, 3],
                [4171, 0, 2],
            ]
        )
        np.testing.assert_array_equal(mne.find_events(raw)[:3], events)
        X = np.array(
            [
                0.34179688,
                0.24414062,
                -3.22265625,
                -7.86132812,
                -6.15234375,
                -4.83398437,
                0.9765625,
                -6.34765625,
                -10.59570312,
                -11.96289062,
                -8.93554688,
                -7.08007812,
                0.14648438,
                -11.23046875,
                -12.01171875,
                -10.40039062,
                -10.30273438,
                -7.12890625,
                -8.54492188,
                -7.51953125,
                -6.98242188,
                -3.56445312,
                10.25390625,
                20.5078125,
                5.859375,
                0.0,
            ]
        )
        np.testing.assert_array_almost_equal(
            raw.get_data()[:, 0] * dataset.unit_factor, X
        )
        onset = np.array(
            [3.0, 11.012, 18.684]
        )  # events times offset by dataset.interval[0]
        np.testing.assert_array_equal(raw.annotations.onset[:3], onset)
        np.testing.assert_array_equal(raw.annotations.duration, np.ones(48) * 4.0)
        description = ["tongue", "feet", "right_hand"]
        assert all([a == b for a, b in zip(raw.annotations.description[:3], description)])


class TestBIDSDataset:
    @pytest.fixture(scope="class")
    def cached_dataset_root(self, tmpdir_factory):
        root = tmpdir_factory.mktemp("fake_bids")
        dataset = FakeDataset(
            event_list=["fake1", "fake2"], n_sessions=2, n_subjects=2, n_runs=1
        )
        dataset.get_data(cache_config=dict(save_raw=True, overwrite_raw=False, path=root))
        return root / "MNE-BIDS-fake-dataset-imagery-2-2--60--120--fake1-fake2--c3-cz-c4"

    @pytest.mark.filterwarnings("ignore:Converting data files to EDF.*:RuntimeWarning")
    def test_local_bids_dataset(self, cached_dataset_root, caplog):
        with caplog.at_level(logging.WARNING):
            dataset = LocalBIDSDataset(
                cached_dataset_root,
                events={"fake1": 1, "fake2": 2},
                interval=[0, 3],
                paradigm="imagery",
            )
        # raw data
        raw_data = dataset.get_data()
        assert raw_data.keys() == {"1", "2"}
        for subject_data in raw_data.values():
            assert subject_data.keys() == {"0", "1"}
            for session_data in subject_data.values():
                assert session_data.keys() == {"0"}
                assert isinstance(session_data["0"], mne.io.BaseRaw)

    @pytest.mark.filterwarnings("ignore:Converting data files to EDF.*:RuntimeWarning")
    def test_convert_to_bids(self, tmp_path):
        """Test that convert_to_bids saves BIDS files without a desc hash."""
        dataset = FakeDataset(
            event_list=["fake1", "fake2"], n_sessions=2, n_subjects=2, n_runs=1
        )
        bids_root = dataset.convert_to_bids(
            path=tmp_path, subjects=[1, 2], overwrite=False
        )

        # The returned path should exist
        assert bids_root.exists()

        # There should be no files with 'desc-' in their names
        bids_files = list(bids_root.rglob("*"))
        for f in bids_files:
            assert "desc-" not in f.name, f"Unexpected desc entity in BIDS file: {f}"

        # No lock files should be written (lock files are part of the cache mechanism only)
        assert not list(bids_root.rglob("*lockfile*")), "Lock files should not be written"

        # EEG EDF files should be present for both subjects
        edf_files = list(bids_root.rglob("*.edf"))
        assert len(edf_files) > 0, "No EDF files were written to BIDS root"
        subjects_found = {f.parent.parent.parent.name for f in edf_files}
        assert subjects_found == {"sub-1", "sub-2"}

        # Calling again with overwrite=False should skip (EDF files already exist)
        bids_root2 = dataset.convert_to_bids(
            path=tmp_path, subjects=[1, 2], overwrite=False
        )
        assert bids_root2 == bids_root

        # Calling again with overwrite=True should succeed
        bids_root3 = dataset.convert_to_bids(path=tmp_path, subjects=[1], overwrite=True)
        assert bids_root3 == bids_root

    @pytest.mark.filterwarnings(
        "ignore:Converting data files to BrainVision.*:RuntimeWarning"
    )
    @pytest.mark.filterwarnings("ignore:Converting data files to EDF.*:RuntimeWarning")
    @pytest.mark.parametrize(
        "format, ext",
        [("EDF", ".edf"), ("BrainVision", ".vhdr")],
    )
    def test_convert_to_bids_format(self, tmp_path, format, ext):
        """Test that convert_to_bids respects the format parameter."""
        dataset = FakeDataset(
            event_list=["fake1", "fake2"], n_sessions=1, n_subjects=1, n_runs=1
        )
        bids_root = dataset.convert_to_bids(path=tmp_path, subjects=[1], format=format)

        data_files = list(bids_root.rglob(f"*{ext}"))
        assert len(data_files) > 0, f"No {ext} files were written for format={format}"

        # Calling again with overwrite=False should skip
        bids_root2 = dataset.convert_to_bids(
            path=tmp_path, subjects=[1], format=format, overwrite=False
        )
        assert bids_root2 == bids_root

    def test_convert_to_bids_invalid_format(self, tmp_path):
        """Test that convert_to_bids raises on invalid format."""
        dataset = FakeDataset(
            event_list=["fake1", "fake2"], n_sessions=1, n_subjects=1, n_runs=1
        )
        with pytest.raises(ValueError, match="Unsupported format"):
            dataset.convert_to_bids(path=tmp_path, subjects=[1], format="INVALID")


class TestKojima2024A:
    def test_convert_subject_to_subject_id(self):
        ds = Kojima2024A()
        assert ds.convert_subject_to_subject_id(1) == "A"
        assert ds.convert_subject_to_subject_id(3) == "C"
        assert ds.convert_subject_to_subject_id(list(range(1, 12))) == [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
        ]
        with pytest.raises(TypeError):
            ds.convert_subject_to_subject_id(1.5)

    @pytest.mark.skip(
        reason="Skipping due to network/download issues with dataverse.harvard.edu"
    )
    def test_data_shape(self):
        ds = Kojima2024A()
        paradigm = P300()
        X, labels, meta = paradigm.get_data(dataset=ds, subjects=[1])

        # number of channels
        assert X.shape[1] == 64

        # number of samples
        assert X.shape[0] == len(labels)


class TestKojima2024B:
    def test_convert_subject_to_subject_id(self):
        ds = Kojima2024B(
            events={"Target": EVENTS["Target"], "NonTarget": EVENTS["NonTarget"]}
        )
        assert ds.convert_subject_to_subject_id(1) == "A"
        assert ds.convert_subject_to_subject_id(3) == "C"
        assert ds.convert_subject_to_subject_id(list(range(1, 16))) == [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
        ]
        with pytest.raises(TypeError):
            ds.convert_subject_to_subject_id(1.5)

    @pytest.mark.skip(
        reason="Skipping due to network/download issues with dataverse.harvard.edu"
    )
    def test_get_task_run(self):
        ds = Kojima2024B(
            events={"Target": EVENTS["Target"], "NonTarget": EVENTS["NonTarget"]}
        )
        paradigm = P300(ignore_relabelling=True)
        X, labels, _meta = ds.get_block_repetition(
            paradigm, [1], ["2stream"], [1, 2, 3, 4, 5, 6]
        )

        # number of channels
        assert X.shape[1] == 64

        # number of samples
        assert X.shape[0] == len(labels)
        assert X.shape[0] == 1440

    @pytest.mark.skip(
        reason="Skipping due to network/download issues with dataverse.harvard.edu"
    )
    def test_other_events_than_target(self):
        ds = Kojima2024B(
            events={"D1": EVENTS["D1"], "D2": EVENTS["D2"], "S1": EVENTS["S1"]}
        )
        paradigm = P300(events=["D1", "D2", "S1"])
        _X, Y, _meta = paradigm.get_data(dataset=ds, subjects=[1])
        assert len(np.unique(Y)) == 3
        assert "D1" in Y
        assert "D2" in Y
        assert "S1" in Y


class TestDatasetMetadata:
    """Tests for the metadata property on BaseDataset."""

    def test_metadata_property_returns_datasetmetadata_or_none(self):
        """Ensure metadata property returns DatasetMetadata or None."""
        dataset = BNCI2014_001()
        metadata = dataset.metadata

        # Should return DatasetMetadata for datasets in the catalog
        assert metadata is not None
        assert isinstance(metadata, DatasetMetadata)

    def test_metadata_property_is_cached(self):
        """Ensure metadata property uses caching."""
        dataset = BNCI2014_001()

        # Access metadata twice
        metadata1 = dataset.metadata
        metadata2 = dataset.metadata

        # Should be the same object (cached)
        assert metadata1 is metadata2

    def test_metadata_has_required_fields(self):
        """Ensure metadata has required fields populated."""
        dataset = BNCI2014_001()
        metadata = dataset.metadata

        assert metadata is not None

        # Check required fields exist and are valid
        assert metadata.participants is not None
        assert metadata.participants.n_subjects > 0

        assert metadata.acquisition is not None
        assert metadata.acquisition.sampling_rate > 0

        assert metadata.experiment is not None
        assert metadata.experiment.paradigm in [
            "imagery",
            "p300",
            "ssvep",
            "cvep",
            "rstate",
        ]

    def test_metadata_matches_dataset_properties(self):
        """Ensure metadata is consistent with dataset properties."""
        dataset = BNCI2014_001()
        metadata = dataset.metadata

        assert metadata is not None

        # Number of subjects should match
        assert metadata.participants.n_subjects == len(dataset.subject_list)

        # Paradigm should match
        assert metadata.experiment.paradigm == dataset.paradigm

    @pytest.mark.parametrize("dataset_class", dataset_list)
    def test_all_datasets_have_metadata_property(self, dataset_class):
        """Ensure every dataset class has the metadata property."""
        kwargs = {}
        if inspect.signature(dataset_class).parameters.get("accept"):
            kwargs["accept"] = True

        dataset = dataset_class(**kwargs)

        # All datasets should have the metadata property
        assert hasattr(dataset, "metadata")

        # The property should return DatasetMetadata or None
        metadata = dataset.metadata
        assert metadata is None or isinstance(metadata, DatasetMetadata)

    def test_fake_dataset_metadata_is_none(self):
        """Ensure FakeDataset returns None for metadata (not in catalog)."""
        dataset = FakeDataset()
        metadata = dataset.metadata

        # FakeDataset is not in the catalog, should return None
        assert metadata is None

    def test_get_dataset_metadata_consistency(self):
        """Ensure get_dataset_metadata and .metadata property return same data."""
        dataset = BNCI2014_001()
        metadata_from_property = dataset.metadata
        metadata_from_function = get_dataset_metadata("BNCI2014_001")

        assert metadata_from_property is not None
        assert metadata_from_function is not None

        # Should have same values (but may be different objects after serialization)
        assert (
            metadata_from_property.participants.n_subjects
            == metadata_from_function.participants.n_subjects
        )
        assert (
            metadata_from_property.acquisition.sampling_rate
            == metadata_from_function.acquisition.sampling_rate
        )
        assert (
            metadata_from_property.experiment.paradigm
            == metadata_from_function.experiment.paradigm
        )

    @pytest.mark.parametrize("dataset_class", dataset_list)
    def test_all_datasets_have_license(self, dataset_class):
        """Ensure every dataset has a license in its documentation metadata."""
        kwargs = {}
        if inspect.signature(dataset_class).parameters.get("accept"):
            kwargs["accept"] = True

        dataset = dataset_class(**kwargs)
        metadata = dataset.metadata

        if metadata is None:
            pytest.skip(f"{dataset_class.__name__} has no metadata catalog entry")

        assert (
            metadata.documentation is not None
        ), f"{dataset_class.__name__} has no documentation metadata defined"
        assert (
            metadata.documentation.license is not None
        ), f"{dataset_class.__name__} is missing a license in its documentation metadata"

    @pytest.mark.download
    def test_n_channels_matches_raw_data(self):
        """Ensure metadata n_channels matches actual raw data channel count."""
        dataset = BNCI2014_001()
        metadata = dataset.metadata
        assert metadata is not None

        data = dataset.get_data(subjects=[dataset.subject_list[0]])
        subject_data = data[dataset.subject_list[0]]

        # Check first session, first run
        first_session = next(iter(subject_data.values()))
        first_run = next(iter(first_session.values()))

        # Exclude stim channels (added by MOABB, not actually recorded)
        n_channels = sum(
            1 for ch_type in first_run.get_channel_types() if ch_type != "stim"
        )
        assert n_channels == metadata.acquisition.n_channels, (
            f"Channel count mismatch for {dataset.code}: "
            f"raw has {n_channels} non-stim channels, "
            f"metadata says {metadata.acquisition.n_channels}"
        )

    @pytest.mark.download
    @pytest.mark.parametrize(
        "dataset_class",
        [ds for ds in dataset_list if ds.__name__ in DATASET_METADATA_CATALOG],
    )
    def test_metadata_matches_raw_data(self, dataset_class):
        """Ensure metadata matches actual raw data (data is ground truth)."""
        from collections import Counter

        kwargs = {}
        if inspect.signature(dataset_class).parameters.get("accept"):
            kwargs["accept"] = True

        dataset = dataset_class(**kwargs)
        metadata = dataset.metadata
        name = dataset_class.__name__

        data = dataset.get_data(subjects=[dataset.subject_list[0]])
        subject_data = data[dataset.subject_list[0]]

        first_session = next(iter(subject_data.values()))
        first_run = next(iter(first_session.values()))

        # --- Sampling rate ---
        assert first_run.info["sfreq"] == metadata.acquisition.sampling_rate, (
            f"Sampling rate mismatch for {name}: "
            f"data has {first_run.info['sfreq']} Hz, "
            f"metadata says {metadata.acquisition.sampling_rate} Hz"
        )

        # --- Channel counts by type (exclude stim, added by MOABB) ---
        raw_types = dict(zip(first_run.ch_names, first_run.get_channel_types()))
        raw_counts = Counter(raw_types.values())
        raw_counts.pop("stim", None)

        n_non_stim = sum(raw_counts.values())
        assert n_non_stim == metadata.acquisition.n_channels, (
            f"Channel count mismatch for {name}: "
            f"data has {n_non_stim} non-stim channels, "
            f"metadata says {metadata.acquisition.n_channels}"
        )

        for ch_type, meta_count in metadata.acquisition.channel_types.items():
            raw_count = raw_counts.get(ch_type, 0)
            assert raw_count == meta_count, (
                f"Channel type '{ch_type}' count mismatch for {name}: "
                f"data has {raw_count}, metadata says {meta_count}"
            )

        # --- Channel names (exclude stim, added by MOABB) ---
        raw_non_stim_names = sorted(
            n for n, ch_type in raw_types.items() if ch_type != "stim"
        )
        if metadata.acquisition.sensors:
            assert sorted(metadata.acquisition.sensors) == raw_non_stim_names, (
                f"Channel name mismatch for {name}: "
                f"only in metadata: "
                f"{set(metadata.acquisition.sensors) - set(raw_non_stim_names)}, "
                f"only in data: "
                f"{set(raw_non_stim_names) - set(metadata.acquisition.sensors)}"
            )

        # --- Number of sessions ---
        n_sessions_data = len(subject_data)
        assert n_sessions_data == metadata.sessions_per_subject, (
            f"Sessions per subject mismatch for {name}: "
            f"data has {n_sessions_data}, metadata says {metadata.sessions_per_subject}"
        )

        # --- Number of runs per session ---
        n_runs_data = len(first_session)
        assert n_runs_data == metadata.runs_per_session, (
            f"Runs per session mismatch for {name}: "
            f"data has {n_runs_data}, metadata says {metadata.runs_per_session}"
        )

        # --- Number of subjects ---
        n_subjects_data = len(dataset.subject_list)
        assert n_subjects_data == metadata.participants.n_subjects, (
            f"Number of subjects mismatch for {name}: "
            f"data has {n_subjects_data}, metadata says {metadata.participants.n_subjects}"
        )


def _make_dataset(dataset_cls, **extra_kwargs):
    """Instantiate a dataset, handling special constructor args like accept."""
    kwargs = dict(extra_kwargs)
    if inspect.signature(dataset_cls).parameters.get("accept"):
        kwargs["accept"] = True
    return dataset_cls(**kwargs)


def _is_valid_event_value(v):
    """Check that v is int (not bool) or a list/tuple of such."""
    if isinstance(v, (list, tuple)):
        return all(
            isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in v
        )
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


# Datasets that use subjects= to rebuild the subject pool rather than filter
_CUSTOM_SUBJECT_HANDLING = {"RomaniBF2025ERP"}
_ALIAS_NAMES = {old for old, _, _ in aliases_list}


@pytest.mark.parametrize("dataset_cls", dataset_list)
def test_constructor_defaults_and_properties(dataset_cls):
    """Default instantiation: all subjects exposed, n_sessions > 0."""
    ds = _make_dataset(dataset_cls)
    name = dataset_cls.__name__
    assert ds.subject_list == ds.all_subjects
    assert len(ds.all_subjects) > 0, f"{name} has no subjects"
    assert ds.n_sessions > 0, f"{name} has n_sessions <= 0"


@pytest.mark.parametrize("dataset_cls", dataset_list)
def test_constructor_events_integrity(dataset_cls):
    """event_id keys must be str, values must be int or list[int]."""
    ds = _make_dataset(dataset_cls)
    name = dataset_cls.__name__
    assert isinstance(ds.event_id, dict), f"{name}: event_id not a dict"
    for k, v in ds.event_id.items():
        assert isinstance(k, str), f"{name}: event key {k!r} is not str"
        assert _is_valid_event_value(v), f"{name}: event {k!r}={v!r} is not int/list[int]"


@pytest.mark.parametrize("dataset_cls", dataset_list)
def test_constructor_subject_filtering(dataset_cls):
    """subjects= should filter subject_list without mutating all_subjects."""
    name = dataset_cls.__name__
    if "subjects" not in inspect.signature(dataset_cls).parameters:
        pytest.skip(f"{name} has no 'subjects' parameter")
    if name in _CUSTOM_SUBJECT_HANDLING:
        pytest.skip(f"{name} uses custom subject handling")

    ds_full = _make_dataset(dataset_cls)
    if len(ds_full.all_subjects) < 2:
        pytest.skip(f"{name} has fewer than 2 subjects")

    first_two = ds_full.all_subjects[:2]
    ds_filtered = _make_dataset(dataset_cls, subjects=first_two)
    assert ds_filtered.subject_list == first_two
    assert ds_filtered.all_subjects == ds_full.all_subjects


@pytest.mark.parametrize("dataset_cls", dataset_list)
def test_constructor_summary_table_cross_ref(dataset_cls):
    """Cross-check subject/session counts against summary CSV."""
    name = dataset_cls.__name__
    if not hasattr(dataset_cls, "_summary_table"):
        pytest.skip(f"{name} not in summary CSV")
    if "Fake" in name or name in _ALIAS_NAMES:
        pytest.skip(f"{name} is fixture or alias")

    ds = _make_dataset(dataset_cls)
    table = dataset_cls._summary_table
    mismatches = []
    for col, actual in [("#Subj", len(ds.all_subjects)), ("#Sessions", ds.n_sessions)]:
        try:
            expected = int(table.get(col, ""))
            if actual != expected:
                mismatches.append(f"{col}: code={actual}, CSV={expected}")
        except (ValueError, TypeError):
            pass
    if mismatches:
        warnings.warn(
            f"{name} summary CSV mismatch: {'; '.join(mismatches)}", stacklevel=1
        )
