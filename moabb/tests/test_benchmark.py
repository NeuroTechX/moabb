import shutil
from pathlib import Path

import pytest

from moabb import benchmark
from moabb.datasets.fake import FakeDataset
from moabb.evaluations.base import optuna_available
from moabb.paradigms import FakeImageryParadigm, FakeP300Paradigm


class TestBenchmark:
    @classmethod
    def setup_class(cls):
        cls.pp_dir = Path.cwd() / Path("moabb/tests/test_pipelines/")

    @classmethod
    def teardown_class(cls):
        rep_dir = Path.cwd() / Path("benchmark/")
        shutil.rmtree(rep_dir)

    def test_benchmark_strdataset(self):
        res = benchmark(
            pipelines=str(self.pp_dir),
            evaluations=["WithinSession"],
            paradigms=["FakeP300Paradigm", "FakeImageryParadigm"],
            include_datasets=[
                "FakeDataset-p300-10-2--60-60--120-120--target-nontarget--c3-cz-c4",
                "FakeDataset-imagery-10-2--60-60--120-120--lefthand-righthand--c3-cz-c4",
            ],
            overwrite=True,
        )
        assert len(res) == 60

    def test_benchmark_objdataset(self):
        res = benchmark(
            pipelines=str(self.pp_dir),
            evaluations=["WithinSession"],
            include_datasets=[
                FakeDataset(
                    ["left_hand", "right_hand"], paradigm="imagery", n_subjects=2
                ),
                FakeDataset(["Target", "NonTarget"], paradigm="p300", n_subjects=2),
                FakeDataset(["13", "15"], paradigm="ssvep", n_subjects=2),
                FakeDataset(["1.0", "0.0"], paradigm="cvep", n_subjects=2),
            ],
            overwrite=True,
        )
        assert len(res) == 16

    def test_nodataset(self):
        with pytest.raises(Exception):  # noqa: B017
            benchmark(
                pipelines=str(self.pp_dir),
                exclude_datasets=["NonExistingDatasetCode"],
                overwrite=True,
            )

    def test_selectparadigm(self):
        ds_imagery = FakeImageryParadigm().datasets[0]
        ds_p300 = FakeP300Paradigm().datasets[0]
        res = benchmark(
            pipelines=str(self.pp_dir),
            evaluations=["WithinSession"],
            paradigms=["FakeImageryParadigm"],
            include_datasets=[ds_imagery, ds_p300],
            overwrite=True,
        )
        # Flattened process path no longer produces the previous inflated count.
        assert len(res) == 80

    def test_include_exclude(self):
        with pytest.raises(ValueError):
            benchmark(
                pipelines=str(self.pp_dir),
                include_datasets=["Dataset1"],
                exclude_datasets=["Dataset2"],
                overwrite=True,
            )

    def test_include_unique(self):
        with pytest.raises(ValueError):
            benchmark(
                pipelines=str(self.pp_dir),
                include_datasets=["Dataset1", "Dataset1"],
                overwrite=True,
            )

    def test_include_two_types(self):
        with pytest.raises(TypeError):
            benchmark(
                pipelines=str(self.pp_dir),
                include_datasets=[
                    "Dataset1",
                    FakeDataset(["left_hand", "right_hand"], paradigm="imagery"),
                ],
                overwrite=True,
            )

    def test_optuna(self):
        if not optuna_available:
            pytest.skip("Optuna is not installed")
        ds = FakeImageryParadigm().datasets[0]
        res = benchmark(
            pipelines=str(self.pp_dir),
            evaluations=["WithinSession"],
            paradigms=["FakeImageryParadigm"],
            include_datasets=[ds],
            overwrite=True,
            optuna=True,
        )
        # Flattened process path no longer produces the previous inflated count.
        assert len(res) == 80


def test_include_datasets_keeps_user_instances():
    """Dataset objects in include_datasets are used as given, not re-created."""
    from moabb.benchmark import _inc_exc_datasets
    from moabb.datasets import BNCI2014_001, Zhou2016

    fresh = [BNCI2014_001(), Zhou2016()]
    user = BNCI2014_001()
    user.subject_list = [1, 2]  # a caller customisation

    selected = _inc_exc_datasets(fresh, "LeftRightImagery", [user], None)
    assert len(selected) == 1
    assert selected[0] is user
    assert selected[0].subject_list == [1, 2]

    # Codes still select the paradigm's own instances.
    by_code = _inc_exc_datasets(fresh, "LeftRightImagery", ["BNCI2014-001"], None)
    assert by_code[0] is fresh[0]
def test_epoch_pipelines_routed_by_type_not_name(tmp_path):
    """A template SSVEP classifier keeps its frequency labels under any name."""
    from moabb.datasets.fake import FakeDataset
    from moabb.pipelines.classification import SSVEP_CCA

    ds = FakeDataset(["13", "15"], n_sessions=1, n_runs=1, n_subjects=1, paradigm="ssvep")
    seen = {}

    class SpyCCA(SSVEP_CCA):
        def fit(self, X, y):
            seen["labels"] = sorted(set(map(str, y)))
            return super().fit(X, y)

    benchmark(
        pipelines=[{"name": "MyCCA", "paradigms": ["SSVEP"], "pipeline": SpyCCA()}],
        evaluations=["WithinSession"],
        paradigms=["SSVEP"],
        include_datasets=[ds],
        results=str(tmp_path / "res"),
        output=str(tmp_path / "out"),
        n_jobs=1,
        overwrite=True,
    )
    # Routed to the epochs branch (mne_labels=True): CCA sees the stimulus
    # frequencies, not label-encoded integers 0/1.
    assert seen["labels"] == ["13", "15"]


@pytest.mark.parametrize(
    "name",
    [
        "SSVEP_CCA",
        "SSVEP_TRCA",
        "SSVEP_MsetCCA",
        "SSVEP_itCCA",
        "SSVEP_eCCA",
        "SSVEP_TRCA_R",
        "SSVEP_SSCOR",
        "SSVEP_TDCA",
    ],
)
def test_every_epochs_only_ssvep_classifier_is_detected(name):
    from moabb.evaluations.utils import _pipeline_requires_epochs
    from moabb.pipelines import classification

    assert _pipeline_requires_epochs(getattr(classification, name)())
