import os.path as osp
import unittest
from unittest.mock import MagicMock, patch

import pytest
from mne import get_config

from moabb.datasets import utils
from moabb.utils import (
    _set_tensorflow_seed,
    _set_torch_seed,
    set_download_dir,
    setup_seed,
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


def test_setup_seed_without_tensorflow():
    with patch("builtins.print") as mock_print:
        with patch.dict("sys.modules", {"tensorflow": None}):
            assert not _set_tensorflow_seed(42)
            mock_print.assert_any_call(
                "We try to set the tensorflow seeds, but it seems that tensorflow is not installed. Please refer to `https://www.tensorflow.org/` to install if you need to use this deep learning module."
            )


def test_setup_seed_without_torch():
    with patch("builtins.print") as mock_print:
        with patch.dict("sys.modules", {"torch": None}):
            assert not _set_torch_seed(42)
            mock_print.assert_any_call(
                "We try to set the torch seeds, but it seems that torch is not installed. Please refer to `https://pytorch.org/` to install if you need to use this deep learning module."
            )


@pytest.mark.parametrize(
    "tf_installed, torch_installed, expected_result",
    [
        (True, True, None),
        (False, True, False),
        (True, False, False),
        (False, False, False),
    ],
)
def test_setup_seed(tf_installed, torch_installed, expected_result):
    with patch.dict(
        "sys.modules",
        {
            "tensorflow": MagicMock() if tf_installed else None,
            "torch": MagicMock() if torch_installed else None,
        },
    ):
        assert setup_seed(42) == expected_result


if __name__ == "__main__":
    unittest.main()
