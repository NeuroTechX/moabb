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
                "FakeDataset-imagery-10-2-2-lefthandrighthand-c3czc4",
                "FakeDataset-p300-10-2-2-targetnontarget-c3czc4",
                "FakeDataset-ssvep-10-2-2-1315-c3czc4",
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


if __name__ == "__main__":
    unittest.main()
