import shutil
from pathlib import Path

import pytest

from moabb import benchmark
from moabb.datasets.fake import FakeDataset
from moabb.evaluations.base import optuna_available


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
            include_datasets=[
                "FakeDataset-imagery-10-2--60-60--120-120--lefthand-righthand--c3-cz-c4",
                "FakeDataset-p300-10-2--60-60--120-120--target-nontarget--c3-cz-c4",
                "FakeDataset-ssvep-10-2--60-60--120-120--13-15--c3-cz-c4",
                "FakeDataset-cvep-10-2--60-60--120-120--10-00--c3-cz-c4",
            ],
            overwrite=True,
        )
        assert len(res) == 80

    def test_benchmark_objdataset(self):
        res = benchmark(
            pipelines=str(self.pp_dir),
            evaluations=["WithinSession"],
            include_datasets=[
                FakeDataset(["left_hand", "right_hand"], paradigm="imagery"),
                FakeDataset(["Target", "NonTarget"], paradigm="p300"),
                FakeDataset(["13", "15"], paradigm="ssvep"),
                FakeDataset(["1.0", "0.0"], paradigm="cvep"),
            ],
            overwrite=True,
        )
        assert len(res) == 80

    def test_nodataset(self):
        with pytest.raises(ValueError):
            benchmark(
                pipelines=str(self.pp_dir),
                exclude_datasets=["FakeDataset"],
                overwrite=True,
            )

    def test_selectparadigm(self):
        res = benchmark(
            pipelines=str(self.pp_dir),
            evaluations=["WithinSession"],
            paradigms=["FakeImageryParadigm"],
            overwrite=True,
        )
        assert len(res) == 40

    def test_include_exclude(self):
        with pytest.raises(AttributeError):
            benchmark(
                pipelines=str(self.pp_dir),
                include_datasets=["FakeDataset"],
                exclude_datasets=["AnotherDataset"],
                overwrite=True,
            )

    def test_optuna(self):
        if not optuna_available:
            pytest.skip("Optuna is not installed")
        res = benchmark(
            pipelines=str(self.pp_dir),
            evaluations=["WithinSession"],
            paradigms=["FakeImageryParadigm"],
            overwrite=True,
            optuna=True,
        )
        assert len(res) == 40
