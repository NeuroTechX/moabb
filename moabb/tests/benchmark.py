import shutil
import unittest
from pathlib import Path

from moabb import benchmark
from moabb.datasets.fake import FakeDataset


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pp_dir = Path.cwd() / Path("moabb/tests/test_pipelines/")

    @classmethod
    def tearDownClass(cls):
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
        self.assertEqual(len(res), 80)

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
        self.assertEqual(len(res), 80)

    def test_nodataset(self):
        self.assertRaises(
            Exception,
            benchmark,
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
        self.assertEqual(len(res), 40)

    def test_include_exclude(self):
        self.assertRaises(
            AttributeError,
            benchmark,
            pipelines=str(self.pp_dir),
            include_datasets=["FakeDataset"],
            exclude_datasets=["AnotherDataset"],
            overwrite=True,
        )

    def test_optuna(self):
        res = benchmark(
            pipelines=str(self.pp_dir),
            evaluations=["WithinSession"],
            paradigms=["FakeImageryParadigm"],
            overwrite=True,
            optuna=True,
        )
        self.assertEqual(len(res), 40)


if __name__ == "__main__":
    unittest.main()
