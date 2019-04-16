from pyriemann.estimation import Covariances
from moabb.pipelines.csp import TRCSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline


pipe = make_pipeline(Covariances('scm'), TRCSP(
    nfilter=6), LinearDiscriminantAnalysis())

# this is what will be loaded
PIPELINE = {'name': 'TRCSP + LDA',
            'paradigms': ['MotorImagery'],
            'pipeline': pipe}
