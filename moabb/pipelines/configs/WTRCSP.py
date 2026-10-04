from pyriemann.estimation import Covariances
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

from moabb.pipelines.csp import TRCSP


pipe = make_pipeline(Covariances("scm"), TRCSP(nfilter=6), LinearDiscriminantAnalysis())

# this is what will be loaded
PIPELINE = {"name": "TRCSP + LDA", "paradigms": ["LeftRightImagery"], "pipeline": pipe}
