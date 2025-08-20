import io

import numpy as np
import pandas as pd
import pytest
from pyriemann.classification import MDM
from pyriemann.estimation import XdawnCovariances
from sklearn.pipeline import make_pipeline
from sklearn.utils import check_random_state

from moabb.datasets import BNCI2014_001, BNCI2015_001
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import MotorImagery


BNCI2014_001_results = """
,score,time,samples,subject,session,channels,n_sessions,dataset,pipeline
0,0.7430556,0.28345227,288.0,1,0train,22,2,BNCI2014-001,mdm
1,0.6944444,0.2819698,288.0,1,1test,22,2,BNCI2014-001,mdm
2,0.5486111,0.28295708,288.0,2,0train,22,2,BNCI2014-001,mdm
3,0.5555556,0.28221202,288.0,2,1test,22,2,BNCI2014-001,mdm
4,0.6527778,0.27323103,288.0,3,0train,22,2,BNCI2014-001,mdm
5,0.6319444,0.28558397,288.0,3,1test,22,2,BNCI2014-001,mdm
6,0.4652778,0.28424382,288.0,4,0train,22,2,BNCI2014-001,mdm
7,0.6076389,0.28512216,288.0,4,1test,22,2,BNCI2014-001,mdm
8,0.4340278,0.26603198,288.0,5,0train,22,2,BNCI2014-001,mdm
9,0.47569445,0.2672441,288.0,5,1test,22,2,BNCI2014-001,mdm
10,0.38194445,0.28032613,288.0,6,0train,22,2,BNCI2014-001,mdm
11,0.4652778,0.29096103,288.0,6,1test,22,2,BNCI2014-001,mdm
12,0.5625,0.26360798,288.0,7,0train,22,2,BNCI2014-001,mdm
13,0.46875,0.26497293,288.0,7,1test,22,2,BNCI2014-001,mdm
14,0.6041667,0.27954388,288.0,8,0train,22,2,BNCI2014-001,mdm
15,0.6111111,0.29071403,288.0,8,1test,22,2,BNCI2014-001,mdm
16,0.5451389,0.27546215,288.0,9,0train,22,2,BNCI2014-001,mdm
17,0.7326389,0.2862649,288.0,9,1test,22,2,BNCI2014-001,mdm
"""

BNCI2015_001_results = """
,score,time,samples,subject,session,channels,n_sessions,dataset,pipeline
0,0.9898,0.104274035,200.0,1,0A,13,2,BNCI2015-001,mdm
1,0.996,0.109023094,200.0,1,1B,13,2,BNCI2015-001,mdm
2,0.9822,0.11902189,200.0,2,0A,13,2,BNCI2015-001,mdm
3,0.9817,0.10449815,200.0,2,1B,13,2,BNCI2015-001,mdm
4,0.9411,0.10515785,200.0,3,0A,13,2,BNCI2015-001,mdm
5,0.9713,0.10190797,200.0,3,1B,13,2,BNCI2015-001,mdm
6,0.8777,0.107106924,200.0,4,0A,13,2,BNCI2015-001,mdm
7,0.9653,0.10397911,200.0,4,1B,13,2,BNCI2015-001,mdm
8,0.8416,0.105483055,200.0,5,0A,13,2,BNCI2015-001,mdm
9,0.8118,0.10831189,200.0,5,1B,13,2,BNCI2015-001,mdm
10,0.6624,0.12765789,200.0,6,0A,13,2,BNCI2015-001,mdm
11,0.6314,0.10389686,200.0,6,1B,13,2,BNCI2015-001,mdm
12,0.8948,0.10865617,200.0,7,0A,13,2,BNCI2015-001,mdm
13,0.8931,0.09851694,200.0,7,1B,13,2,BNCI2015-001,mdm
14,0.6032,0.18366313,400.0,8,0A,13,2,BNCI2015-001,mdm
15,0.7523,0.19959378,400.0,8,1B,13,2,BNCI2015-001,mdm
16,0.8488,0.18477702,400.0,8,2C,13,2,BNCI2015-001,mdm
17,0.7601,0.1761918,400.0,9,0A,13,2,BNCI2015-001,mdm
18,0.8687,0.17262912,400.0,9,1B,13,2,BNCI2015-001,mdm
19,0.9154,0.17855692,400.0,9,2C,13,2,BNCI2015-001,mdm
20,0.6787,0.21773195,400.0,10,0A,13,2,BNCI2015-001,mdm
21,0.6402,0.20742917,400.0,10,1B,13,2,BNCI2015-001,mdm
22,0.6116,0.19268918,400.0,10,2C,13,2,BNCI2015-001,mdm
23,0.7974,0.20285797,400.0,11,0A,13,2,BNCI2015-001,mdm
24,0.7403,0.20020509,400.0,11,1B,13,2,BNCI2015-001,mdm
25,0.7949,0.18860793,400.0,11,2C,13,2,BNCI2015-001,mdm
26,0.6574,0.10171008,200.0,12,0A,13,2,BNCI2015-001,mdm
27,0.6693,0.10934806,200.0,12,1B,13,2,BNCI2015-001,mdm
"""


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

    # reading the score reference in put into the an dataframe
    csv_map = {
        "BNCI2014_001": BNCI2014_001_results,
        "BNCI2015_001": BNCI2015_001_results,
    }
    reference_performance = pd.read_csv(io.StringIO(csv_map[dataset_name]))
    reference_performance = reference_performance.drop(columns=["time", "Unnamed: 0"])
    reference_performance["score"] = reference_performance["score"].astype(np.float32)
    reference_performance["samples"] = reference_performance["samples"].astype(int)
    reference_performance["subject"] = reference_performance["subject"].astype(int)
    reference_performance["channels"] = reference_performance["channels"].astype(int)
    reference_performance["n_sessions"] = reference_performance["n_sessions"].astype(int)

    # Sort rows for a stable, order-invariant compare
    sort_cols = ["dataset", "subject", "session", "pipeline"]
    results = results.sort_values(sort_cols).reset_index(drop=True)
    reference_performance = reference_performance.sort_values(sort_cols).reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(results, reference_performance)
