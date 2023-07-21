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
                "FakeDataset_imagery_10_2_2__left_hand_right_hand__C3_Cz_C4",
                "FakeDataset_p300_10_2_2__Target_NonTarget__C3_Cz_C4",
                "FakeDataset_ssvep_10_2_2__13_15__C3_Cz_C4",
            ],
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
        )
        self.assertEqual(len(res), 80)

    def test_nodataset(self):
        self.assertRaises(
            Exception,
            benchmark,
            pipelines=str(self.pp_dir),
            exclude_datasets=["FakeDataset"],
        )

    def test_selectparadigm(self):
        res = benchmark(
            pipelines=str(self.pp_dir),
            evaluations=["WithinSession"],
            paradigms=["FakeImageryParadigm"],
        )
        self.assertEqual(len(res), 40)

    def test_include_exclude(self):
        self.assertRaises(
            AttributeError,
            benchmark,
            pipelines=str(self.pp_dir),
            include_datasets=["FakeDataset"],
            exclude_datasets=["AnotherDataset"],
        )


if __name__ == "__main__":
    unittest.main()
