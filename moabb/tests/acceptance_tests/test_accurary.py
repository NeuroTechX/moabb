from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mne.utils.config import _open_lock
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state

from moabb.datasets import BNCI2014_001, BNCI2015_001
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import MotorImagery


@pytest.mark.parametrize("dataset_class", [BNCI2014_001, BNCI2015_001])
def test_decoding_performance_stable(dataset_class):
    dataset_name = dataset_class.__name__
    random_state = check_random_state(42)

    dataset_cls = dataset_class
    dataset = dataset_cls()
    paradigm = MotorImagery()

    # Simple pipeline
    pipeline = make_pipeline(XdawnCovariances(nfilter=4), MDM(n_jobs=4))

    # Evaluate
    evaluation = CrossSessionEvaluation(
        paradigm=paradigm, datasets=[dataset], overwrite=True, random_state=random_state
    )
    results = evaluation.process({"mdm": pipeline})
    results.drop(columns=["time"], inplace=True)
    results["score"] = results["score"].astype(np.float32)
    results["samples"] = results["samples"].astype(int)
    results["subject"] = results["subject"].astype(int)

    folder_path = Path(__file__).parent / "reference_results_dataset_{}.csv".format(
        dataset_name
    )
    # Serialize access if filelock is available; otherwise just open normally.
    with _open_lock(folder_path, "r", encoding="utf-8") as f:
        reference_performance = pd.read_csv(f)

    reference_performance.drop(columns=["time", "Unnamed: 0"], inplace=True)
    reference_performance["score"] = reference_performance["score"].astype(np.float32)
    reference_performance["samples"] = reference_performance["samples"].astype(int)

    pd.testing.assert_frame_equal(results, reference_performance)
