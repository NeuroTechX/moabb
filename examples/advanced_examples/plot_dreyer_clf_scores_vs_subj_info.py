"""
===============================================
Examples of analysis of a Dreyer2023 A dataset.
===============================================

This example shows how to plot Dreyer2023A Left-Right Imagery ROC AUC scores
obtained with CSP+LDA pipeline versus demographic information of the examined
subjects (gender and age) and experimenters (gender).

To reduce computational time, the example is provided for four subjects.

"""

# Authors: Sara Sedlar <sara.sedlar@gmail.com>
#          Sylvain Chevallier <sylvain.chevallier@universite-paris-saclay.fr>
# License: BSD (3-clause)

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sb
from pyriemann.estimation import Covariances
from pyriemann.spatialfilters import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from moabb.datasets import Dreyer2023A
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery


########################################################################################
# 1. Defining dataset, selecting subject for analysis and getting data
dreyer2023 = Dreyer2023A()
dreyer2023.subject_list = [1, 5, 7, 35]
dreyer2023.get_data()
########################################################################################
# 2. Defining MotorImagery paradigm and CSP+LDA pipeline
paradigm = MotorImagery()
pipelines = {}
pipelines["CSP+LDA"] = make_pipeline(
    Covariances(estimator="oas"), CSP(nfilter=6), LDA(solver="lsqr", shrinkage="auto")
)
########################################################################################
# 3. Within session evaluation of the pipeline
evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=[dreyer2023], suffix="examples", overwrite=False
)
results = evaluation.process(pipelines)

########################################################################################
# 4. Loading dataset info and concatenation with the obtained results
info = dreyer2023.get_subject_info().rename(columns={"score": "score_MR"})
# Creating a new column with subject's age
info["Age"] = 2019 - info["Birth_year"]
# Casting to int for merging
info["subject"] = info["SUJ_ID"].astype(int)
results["subject"] = results["subject"].astype(int)

results_info = results.merge(info, on="subject", how="left")

########################################################################################
# 5.1 Plotting subject AUC ROC scores vs subject's gender
fig, ax = plt.subplots(nrows=2, ncols=2, facecolor="white", figsize=[16, 8], sharey=True)
fig.subplots_adjust(wspace=0.0, hspace=0.5)
sb.boxplot(
    data=results_info, y="score", x="SUJ_gender", ax=ax[0, 0], palette="Set1", width=0.3
)
sb.stripplot(
    data=results_info,
    y="score",
    x="SUJ_gender",
    ax=ax[0, 0],
    palette="Set1",
    linewidth=1,
    edgecolor="k",
    size=3,
    alpha=0.3,
    zorder=1,
)
ax[0, 0].set_title("AUC ROC scores vs. subject gender")
ax[0, 0].set_xticklabels(["Man", "Woman"])
ax[0, 0].set_ylabel("ROC AUC")
ax[0, 0].set_xlabel(None)
ax[0, 0].set_ylim(0.3, 1)
########################################################################################
# 5.2 Plotting subject AUC ROC scores vs subjects's age per gender
sb.regplot(
    data=results_info[results_info["SUJ_gender"] == 1][["score", "Age"]].astype(
        "float32"
    ),
    y="score",
    x="Age",
    ax=ax[0, 1],
    scatter_kws={"color": "#e41a1c", "alpha": 0.5},
    line_kws={"color": "#e41a1c"},
)
sb.regplot(
    data=results_info[results_info["SUJ_gender"] == 2][["score", "Age"]].astype(
        "float32"
    ),
    y="score",
    x="Age",
    ax=ax[0, 1],
    scatter_kws={"color": "#377eb8", "alpha": 0.5},
    line_kws={"color": "#377eb8"},
)
ax[0, 1].set_title("AUC ROC scores vs. subject age per gender")
ax[0, 1].set_ylabel(None)
ax[0, 1].set_xlabel(None)
ax[0, 1].legend(
    handles=[
        mpatches.Patch(color="#e41a1c", label="Man"),
        mpatches.Patch(color="#377eb8", label="Woman"),
    ]
)
########################################################################################
# 5.3 Plotting subject AUC ROC scores vs experimenter's gender
sb.boxplot(
    data=results_info, y="score", x="EXP_gender", ax=ax[1, 0], palette="Set1", width=0.3
)
sb.stripplot(
    data=results_info,
    y="score",
    x="EXP_gender",
    ax=ax[1, 0],
    palette="Set1",
    linewidth=1,
    edgecolor="k",
    size=3,
    alpha=0.3,
    zorder=1,
)
ax[1, 0].set_title("AUC ROC scores vs. experimenter gender")
ax[1, 0].set_xticklabels(["Man", "Woman"])
ax[1, 0].set_ylabel("ROC AUC")
ax[1, 0].set_xlabel(None)
ax[1, 0].set_ylim(0.3, 1)
########################################################################################
# 5.4 Plotting subject AUC ROC scores vs subject's age
sb.regplot(
    data=results_info[["score", "Age"]].astype("float32"),
    y="score",
    x="Age",
    ax=ax[1, 1],
    scatter_kws={"color": "black", "alpha": 0.5},
    line_kws={"color": "black"},
)
ax[1, 1].set_title("AUC ROC scores vs. subject age")
ax[1, 1].set_ylabel(None)
plt.show()
########################################################################################
# 5.5 Obtained results for four selected subjects correspond to the following figure.
# --------------------------------------------------------------------------------------
#
# .. image:: ../../images/Dreyer_clf_scores_vs_subj_info/4_selected_subjects.png
#    :align: center
#    :alt: 4_selected_subjects

########################################################################################
# Obtained results for all subjects correspond to the following figure.
#
# .. image:: ../../images/Dreyer_clf_scores_vs_subj_info/all_subjects.png
#    :align: center
#    :alt: all_subjects
