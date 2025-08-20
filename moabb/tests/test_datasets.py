import inspect
import logging
import re

import mne
import numpy as np
import pandas as pd
import pytest

import moabb.datasets as db
import moabb.datasets.compound_dataset as db_compound
from moabb.datasets import BNCI2014_001, Cattan2019_VR, Kojima2024A, Shin2017A, Shin2017B
from moabb.datasets.base import (
    BaseDataset,
    LocalBIDSDataset,
    _summary_table,
    is_abbrev,
    is_camel_kebab_case,
)
from moabb.datasets.compound_dataset import CompoundDataset
from moabb.datasets.compound_dataset.utils import compound_dataset_list
from moabb.datasets.fake import FakeDataset, FakeVirtualRealityDataset
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

    def test_data_shape(self):
        ds = Kojima2024A()
        paradigm = P300()
        X, labels, meta = paradigm.get_data(dataset=ds, subjects=[1])

        # number of channels
        assert X.shape[1] == 64

        # number of samples
        assert X.shape[0] == len(labels)
