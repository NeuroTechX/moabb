"""
An evaluation defines how we go from trials per subject and session to a
generalization statistic (AUC score, f-score, accuracy, etc) -- it can be either
within-recording-session accuracy, across-session within-subject accuracy,
across-subject accuracy, or other transfer learning settings.
"""
from moabb.evaluations.evaluations import *
