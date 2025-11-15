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
        with pytest.raises(ValueError):
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
            include_datasets=[
                ds_imagery,
                ds_p300
            ],
            overwrite=True,
        )
        assert len(res) == 8

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
            include_datasets=[
                ds,
            ],
            overwrite=True,
            optuna=True,
        )
        assert len(res) == 8
