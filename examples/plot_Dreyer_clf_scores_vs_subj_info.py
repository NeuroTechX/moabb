import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from mne.decoding import Vectorizer
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

import moabb
from moabb.datasets import Dreyer2023A, Dreyer2023B, Dreyer2023C
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery


dreyer2023 = Dreyer2023A()
paradigm = MotorImagery()
pipelines = {}
pipelines["CSP+LDA"] = make_pipeline(
    Covariances(estimator="oas"), CSP(nfilter=6), LDA(solver="lsqr", shrinkage="auto")
)

evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=[dreyer2023], suffix="examples", overwrite=False
)
results = evaluation.process(pipelines)


##############################################################################
info = dreyer2023.get_subject_info(infos=["Demo_Bio"])
results_info = pd.concat([info, results], axis=1)
fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])
plt.figure(1)
sb.stripplot(
    data=results_info,
    y="score",
    x="SUJ_gender",
    ax=ax,
    jitter=True,
    alpha=0.5,
    zorder=1,
    palette="Set1",
)

sb.pointplot(data=results_info, y="score", x="SUJ_gender", ax=ax, palette="Set1")
ax.set_xticklabels(["Man", "Woman"])
ax.set_ylabel("ROC AUC")
ax.set_ylim(0.5, 1)

fig, ax2 = plt.subplots(facecolor="white", figsize=[8, 4])

sb.regplot(
    data=results_info[["score", "Birth_year"]].astype("float32"),
    y="score",
    x="Birth_year",
    ax=ax2,
    scatter_kws={"color": "black", "alpha": 0.5},
    line_kws={"color": "red"},
)
plt.show()
