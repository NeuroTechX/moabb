"""
=====================================================
Tutorial: Within-Session Splitting on Real MI Dataset
=====================================================
# Authors: Thomas, Kooiman, Radovan Vodila, Jorge Sanmartin Martinez, and Paul Verhoeven
# License: BSD (3-clause)
"""

###############################################################################
#  Why would I want to split my data within a session?
###############################################################################
# In short, because we want to prevent the model from recognizing the subject
# and learning subject-specific representations instead of focusing on the task at hand.
#
# In brain-computer interface (BCI) research, careful data splitting is critical.
# A naive train_test_split can easily lead to misleading results, especially in small EEG datasets,
# where models may accidentally learn to recognize subjects instead of decoding the actual brain task.
# Each brain produces unique signals, and unless we're careful, the model can exploit these as shortcuts —
# leading to artificially high test accuracy that doesn’t generalize in practice.
#
# To avoid this, we use within-session splitting, where training and testing are done
# on different trials from the same session. This ensures the model is evaluated under realistic,
# consistent conditions while still preventing overfitting to trial-specific noise.
#
# This approach forms a critical foundation in the MOABB evaluation framework,
# which supports three levels of model generalization:
#
#     - Within-session: test generalization across trials within a single session
#     - Cross-session: test generalization across different recording sessions
#     - Cross-subject: test generalization across different brains
#
# Each level decreases in specialization, moving from highly subject-specific models
# to those that can generalize across individuals.
#
# This tutorial focuses on within-session evaluation to establish a reliable
# baseline for model performance before attempting more challenging generalization tasks.

import warnings

import matplotlib.pyplot as plt

# Standard imports
import pandas as pd
import seaborn as sns

# MNE + sklearn for pipeline
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

import moabb

# MOABB components
from moabb.datasets import BNCI2014_001
from moabb.evaluations.splitters import WithinSessionSplitter
from moabb.paradigms import LeftRightImagery


# Suppress warnings and enable informative logging
warnings.filterwarnings("ignore")
moabb.set_log_level("info")

###############################################################################
#  Load the dataset and paradigm
###############################################################################
# We use the BNCI2014_001 dataset: BCI Comp IV dataset 2a (motor imagery)
dataset = BNCI2014_001()
# Restrict to a few subjects to keep runtime reasonable for demonstration
dataset.subject_list = [1, 2, 3]

# Define the paradigm: here, left vs right hand imagery, filtered 8–35 Hz
paradigm = LeftRightImagery(fmin=8, fmax=35)

###############################################################################
#  Extract data: epochs (X), labels (y), and trial metadata (meta)
###############################################################################
# This call downloads (if needed), preprocesses, epochs, and labels the data
X, y, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list)

# Inspect the shapes: X is trials × channels × timepoints; y is labels; meta is info
print("X shape (trials, channels, timepoints):", X.shape)
print("y shape (trials,):", y.shape)
print("meta shape (trials, info columns):", meta.shape)
print(meta.head())  # shows subject/session for each trial

###############################################################################
#  Build a classification pipeline: CSP to LDA
###############################################################################
# CSP finds spatial filters that maximize variance difference between classes
# LDA is a simple linear classifier on the CSP features
pipe = make_pipeline(
    CSP(n_components=6, reg=None),  # reduce to 6 CSP components
    LDA(),  # classify based on these features
)
print("Pipeline steps:", pipe.named_steps)

###############################################################################
#  Instantiate WithinSessionSplitter
###############################################################################
# We want 5-fold CV _within_ each subject × session grouping
wss = WithinSessionSplitter(n_folds=5, shuffle=True, random_state=404)
print(f"Splitter config: folds={wss.n_folds}, shuffle={wss.shuffle}")

# How many total splits? equals n_folds × (num_subjects × sessions per subject)
total_folds = wss.get_n_splits(meta)
print("Total folds (num_subjects × sessions × n_folds):", total_folds)
# If wss is applied to a dataset where a subject has only one session,
# the splitter will skip that subject silently. Therefore, we raise an error.
if wss.get_n_splits(meta) == 0:
    raise RuntimeError("No splits generated: check that each subject has ≥2 sessions.")

###############################################################################
#  Manual evaluation loop: train/test each fold
###############################################################################
# We'll collect one row per fold: which subject/session was held out and its score
records = []
for fold_id, (train_idx, test_idx) in enumerate(wss.split(y, meta)):
    # Slice our epoch array and labels
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit the CSP+LDA pipeline on the training fold
    pipe.fit(X_train, y_train)
    # Evaluate on the held-out trials
    score = pipe.score(X_test, y_test)

    # Identify which subject & session these test trials come from
    # (all test_idx in one fold share the same subject/session)
    subject_held = meta.iloc[test_idx]["subject"].iat[0]
    session_held = meta.iloc[test_idx]["session"].iat[0]

    # Record information for later analysis
    records.append(
        {
            "fold": fold_id,
            "subject": subject_held,
            "session": session_held,
            "score": score,
        }
    )

# Create a DataFrame of fold results
df = pd.DataFrame(records)

# Show the first few rows: one entry per fold
print(df.head())

###############################################################################
#  Summary of results
###############################################################################
# We can quickly see per-subject, per-session performance:
summary = df.groupby(["subject", "session"])["score"].agg(["mean", "std"]).reset_index()
print("\nSummary of within-session fold scores (mean ± std):")
print(summary)
# We see subject 2’s Session 1 has lower mean accuracy, suggesting session variability.
# Note: you could plot these numbers to visually compare sessions,
# but here we print them to focus on the splitting logic itself.

##########################################################################
#  Plot results
##########################################################################


df["subject"] = df["subject"].astype(str)
plt.figure(figsize=(8, 6))
sns.barplot(x="score", y="subject", hue="session", data=df, orient="h", palette="viridis")
plt.xlabel("Classification accuracy")
plt.ylabel("Subject")
plt.title("Within-session CSP+LDA performance")
plt.tight_layout()
plt.show()
