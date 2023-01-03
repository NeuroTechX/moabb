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

    def test_benchmark(self):
        res = benchmark(
            pipelines=str(self.pp_dir),
            evaluations=["WithinSession"],
            include_datasets=["FakeDataset"],
        )
        self.assertEqual(len(res), 80)
        res = benchmark(
            pipelines=str(self.pp_dir),
            evaluations=["WithinSession"],
            include_datasets=[FakeDataset()],
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
