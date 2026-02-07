import os.path as osp
import tempfile
import unittest

import pytest
from joblib import Parallel, delayed
from mne import get_config, set_config

from moabb.datasets import utils
from moabb.utils import aliases_list, depreciated_alias, set_download_dir


class TestDownload(unittest.TestCase):
    def test_set_download_dir(self):
        original_path = get_config("MNE_DATA")
        new_path = osp.join(osp.expanduser("~"), "mne_data_test")
        set_download_dir(new_path)

        # Check if the mne config has been changed correctly
        self.assertTrue(get_config("MNE_DATA") == new_path)

        # Check if the folder has been created
        self.assertTrue(osp.isdir(new_path))

        # Set back to usual
        set_download_dir(original_path)


@pytest.mark.skip(
    reason="This test is only when you have already " "download the datasets."
)
class Test_Utils(unittest.TestCase):

    def test_channel_intersection_fun(self):
        print(utils.find_intersecting_channels([d() for d in utils.dataset_list])[0])

    def test_dataset_search_fun(self):
        found = utils.dataset_search("imagery", multi_session=True)
        print([type(dataset).__name__ for dataset in found])
        found = utils.dataset_search("imagery", multi_session=False)
        print([type(dataset).__name__ for dataset in found])
        res = utils.dataset_search(
            "imagery", events=["right_hand", "left_hand", "feet", "tongue", "rest"]
        )
        for out in res:
            print("multiclass: {}".format(out.event_id.keys()))

        res = utils.dataset_search(
            "imagery", events=["right_hand", "feet"], has_all_events=True
        )
        for out in res:
            self.assertTrue(set(["right_hand", "feet"]) <= set(out.event_id.keys()))

    def test_dataset_channel_search(self):
        chans = ["C3", "Cz"]
        All = utils.dataset_search(
            "imagery", events=["right_hand", "left_hand", "feet", "tongue", "rest"]
        )
        has_chans = utils.dataset_search(
            "imagery",
            events=["right_hand", "left_hand", "feet", "tongue", "rest"],
            channels=chans,
        )
        has_types = set([type(x) for x in has_chans])
        for d in has_chans:
            s1 = d.get_data([1])[1]
            sess1 = s1[list(s1.keys())[0]]
            raw = sess1[list(sess1.keys())[0]]
            self.assertTrue(set(chans) <= set(raw.info["ch_names"]))
        for d in All:
            if type(d) not in has_types:
                s1 = d.get_data([1])[1]
                sess1 = s1[list(s1.keys())[0]]
                raw = sess1[list(sess1.keys())[0]]
                self.assertFalse(set(chans) <= set(raw.info["ch_names"]))


class TestDepreciatedAlias(unittest.TestCase):
    def test_class_alias(self):
        @depreciated_alias("DummyB", expire_version="0.1")
        class DummyA:
            """DummyA class"""

            def __init__(self, a, b=1):
                self.a = a
                self.b = b

            def c(self):
                return self.a

        self.assertIn(("DummyB", "DummyA", "0.1"), aliases_list)

        # assertNoLogs was added in Python 3.10
        # https://bugs.python.org/issue39385
        if hasattr(self, "assertNoLogs"):
            with self.assertNoLogs(logger="moabb.utils", level="WARN") as cm:
                a = DummyA(2, b=2)
            self.assertEqual(
                a.__doc__,
                "DummyA class\n\n    Notes\n    -----\n\n"
                "    .. note:: ``DummyA`` was previously named ``DummyB``. "
                "``DummyB`` will be removed in  version 0.1.\n",
            )

        with self.assertLogs(logger="moabb.utils", level="WARN") as cm:
            b = DummyB(2, b=2)  # noqa: F821

        self.assertEqual(1, len(cm.output))
        expected = (
            "DummyB has been renamed to DummyA. DummyB will be removed in version 0.1."
        )
        self.assertRegex(cm.output[0], expected)
        # attributes:
        self.assertEqual(b.a, 2)
        self.assertEqual(b.b, 2)
        # methods:
        self.assertEqual(b.c(), 2)
        # class name and type:
        self.assertEqual(DummyB.__name__, "DummyB")  # noqa: F821
        self.assertEqual(b.__class__.__name__, "DummyB")
        self.assertIsInstance(b, DummyB)  # noqa: F821
        self.assertIsInstance(b, DummyA)

    def test_class_alias_notes(self):
        @depreciated_alias("DummyB", expire_version="0.1")
        class DummyA:
            """DummyA class

            Notes
            -----

            a note"""

            def __init__(self, a, b=1):
                self.a = a
                self.b = b

            def c(self):
                return self.a

        self.assertIn(("DummyB", "DummyA", "0.1"), aliases_list)

        if hasattr(self, "assertNoLogs"):
            with self.assertNoLogs(logger="moabb.utils", level="WARN"):
                a = DummyA(2, b=2)
            self.assertEqual(
                a.__doc__,
                "DummyA class\n\n            Notes\n            -----\n\n"
                "            .. note:: ``DummyA`` was previously named ``DummyB``. "
                "``DummyB`` will be removed in  version 0.1.\n\n"
                "            a note",
            )

    def test_function_alias(self):
        @depreciated_alias("dummy_b", expire_version="0.1")
        def dummy_a(a, b=1):
            """Dummy function"""
            return a + b

        self.assertIn(("dummy_b", "dummy_a", "0.1"), aliases_list)

        if hasattr(self, "assertNoLogs"):
            with self.assertNoLogs(logger="moabb.utils", level="WARN") as cm:
                self.assertEqual(dummy_a(2, b=2), 4)

        self.assertEqual(
            dummy_a.__doc__,
            # "Dummy function\n\nNotes\n-----\n"
            # "``dummy_a`` was previously named ``dummy_b``. "
            # "``dummy_b`` will be removed in  version 0.1.",
            "Dummy function\n\n    Notes\n    -----\n\n"
            "    .. note:: ``dummy_a`` was previously named ``dummy_b``. "
            "``dummy_b`` will be removed in  version 0.1.\n",
        )

        with self.assertLogs(logger="moabb.utils", level="WARN") as cm:
            self.assertEqual(dummy_b(2, b=2), 4)  # noqa: F821

        self.assertEqual(1, len(cm.output))
        expected = (
            "dummy_b has been renamed to dummy_a. dummy_b will be removed in version 0.1."
        )
        self.assertRegex(cm.output[0], expected)
        # class name and type:
        self.assertEqual(dummy_b.__name__, "dummy_b")  # noqa: F821


@pytest.fixture(autouse=True)
def reset_mne_config():
    """Fixture to reset MNE_DATA config before and after each test."""
    original_config = get_config("MNE_DATA")
    yield
    if original_config is not None:
        set_config("MNE_DATA", original_config, set_env=True)
    else:
        # Remove the config if it was not set originally
        set_config("MNE_DATA", None, set_env=True)


def test_set_download_dir_none_not_set(capsys):
    """Test setting download directory to None when MNE_DATA is not set."""
    # Ensure MNE_DATA is not set
    set_config("MNE_DATA", None)

    set_download_dir(None)

    captured = capsys.readouterr()
    expected_path = osp.join(osp.expanduser("~"), "mne_data")
    assert "MNE_DATA is not already configured" in captured.out
    assert "default location in the home directory" in captured.out
    assert "mne_data" in captured.out

    assert get_config("MNE_DATA") == expected_path


def test_set_download_dir_none_already_set(capsys):
    """Test setting download directory to None when MNE_DATA is already set."""
    predefined_path = "/existing/mne_data_path"
    set_config("MNE_DATA", predefined_path)

    set_download_dir(None)

    captured = capsys.readouterr()
    # No print should occur since MNE_DATA is already set
    assert captured.out == ""
    assert get_config("MNE_DATA") == predefined_path


def test_set_download_dir_existing_path(capsys):
    """Test setting download directory to an existing path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        set_download_dir(tmpdir)
        captured = capsys.readouterr()
        # No print should occur since the directory exists
        assert captured.out == ""
        assert get_config("MNE_DATA") == tmpdir


def test_set_download_dir_nonexistent_path(capsys):
    """Test setting download directory to a non-existent path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        non_existent_path = osp.join(tmpdir, "new_mne_data")

        # Ensure the path does not exist
        assert not osp.exists(non_existent_path)

        set_download_dir(non_existent_path)

        captured = capsys.readouterr()
        assert "The path given does not exist, creating it.." in captured.out
        assert osp.isdir(non_existent_path)
        assert get_config("MNE_DATA") == non_existent_path


@pytest.mark.parametrize("path_exists", [True, False])
def test_set_download_dir_parallel(path_exists, tmp_path, capsys):
    """Test setting download directory in parallel with joblib."""
    if path_exists:
        path = tmp_path / "existing_dir"
        path.mkdir()
    else:
        path = tmp_path / "non_existing_dir"

    def worker(p):
        set_download_dir(p)
        mne_data_value = get_config("MNE_DATA")
        return mne_data_value

    results = Parallel(n_jobs=10)(delayed(worker)(path) for _ in range(100))

    for mne_data_value in results:
        assert mne_data_value == str(path)


if __name__ == "__main__":
    unittest.main()
