"""
:meta public:
=====================================================
Tutorial: Within-Session Splitting on Real MI Dataset
=====================================================

In this notebook, we demonstrate how to:
  1. Load a real motor imagery dataset (BNCI2014_001)
  2. Extract epochs, labels, and metadata via a paradigm
  3. Build a CSP+LDA pipeline for classification
  4. Use WithinSessionSplitter to create train/test splits _within_ each session
  5. Manually run a training/testing loop and collect fold-wise scores
  6. Visualize 

# Authors: Thomas, Kooiman, Radovan Vodila, Jorge Sanmartin Martinez, and Paul Verhoeven
# License: BSD (3-clause)
"""
import warnings
import moabb
import matplotlib.pyplot as plt

#: Standard imports
import pandas as pd
import seaborn as sns

#: MNE + sklearn for pipeline
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

#: MOABB components
from moabb.datasets import BNCI2014_001
from moabb.evaluations.splitters import WithinSessionSplitter
from moabb.paradigms import LeftRightImagery

#: Suppress warnings and enable informative logging
warnings.filterwarnings("ignore")
moabb.set_log_level("info")

###############################################################################
#: 1. Load the dataset and paradigm
###############################################################################
#: We use the BNCI2014_001 dataset: BCI Comp IV dataset 2a (motor imagery)
dataset = BNCI2014_001()
#: Restrict to a few subjects to keep runtime reasonable for demonstration
dataset.subject_list = [1, 2, 3]

#: Define the paradigm: here, left vs right hand imagery, filtered 8–35 Hz
paradigm = LeftRightImagery(fmin=8, fmax=35)

###############################################################################
#: 2. Extract data: epochs (X), labels (y), and trial metadata (meta)
###############################################################################
#: This call downloads (if needed), preprocesses, epochs, and labels the data
X, y, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list)

# Inspect the shapes: X is trials × channels × timepoints; y is labels; meta is info
print("X shape (trials, channels, timepoints):", X.shape)
print("y shape (trials,):", y.shape)
print("meta shape (trials, info columns):", meta.shape)
print(meta.head())  # shows subject/session for each trial

###############################################################################
#: 3. Build a classification pipeline: CSP to LDA
###############################################################################
#: CSP finds spatial filters that maximize variance difference between classes
#: LDA is a simple linear classifier on the CSP features
pipe = make_pipeline(
    CSP(n_components=6, reg=None),  # reduce to 6 CSP components
    LDA(),  # classify based on these features
)
print("Pipeline steps:", pipe.named_steps)

###############################################################################
#: 4. Instantiate WithinSessionSplitter
###############################################################################
#: We want 5-fold CV _within_ each subject × session grouping
wss = WithinSessionSplitter(n_folds=5, shuffle=True, random_state=404)
print(f"Splitter config: folds={wss.n_folds}, shuffle={wss.shuffle}")

#: How many total splits? equals n_folds × (num_subjects × sessions per subject)
total_folds = wss.get_n_splits(meta)
print("Total folds (num_subjects × sessions × n_folds):", total_folds)
#: If wss is applied to a dataset where a subject has only one session,
#: the splitter will skip that subject silently. Therefore, we raise an error.
if wss.get_n_splits(meta) == 0:
    raise RuntimeError("No splits generated: check that each subject has ≥2 sessions.")

###############################################################################
#: 5. Manual evaluation loop: train/test each fold
###############################################################################
#: We'll collect one row per fold: which subject/session was held out and its score
records = []
for fold_id, (train_idx, test_idx) in enumerate(wss.split(y, meta)):
    #: Slice our epoch array and labels
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    #: Fit the CSP+LDA pipeline on the training fold
    pipe.fit(X_train, y_train)
    #: Evaluate on the held-out trials
    score = pipe.score(X_test, y_test)

    #: Identify which subject & session these test trials come from
    #: (all test_idx in one fold share the same subject/session)
    subject_held = meta.iloc[test_idx]["subject"].iat[0]
    session_held = meta.iloc[test_idx]["session"].iat[0]

    #: Record information for later analysis
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
#: 6. Summary of results
###############################################################################
#: We can quickly see per-subject, per-session performance:
summary = df.groupby(["subject", "session"])["score"].agg(["mean", "std"]).reset_index()
print("\nSummary of within-session fold scores (mean ± std):")
print(summary)
#: We see subject 2’s Session 1 has lower mean accuracy, suggesting session variability.
#: Note: you could plot these numbers to visually compare sessions,
#: but here we print them to focus on the splitting logic itself.


#: 6. Plot results
##########################################################################


df["subject"] = df["subject"].astype(str)
plt.figure(figsize=(8, 6))
sns.barplot(x="score", y="subject", hue="session", data=df, orient="h", palette="viridis")
plt.xlabel("Classification accuracy")
plt.ylabel("Subject")
plt.title("Within-session CSP+LDA performance")
plt.tight_layout()
plt.show()
