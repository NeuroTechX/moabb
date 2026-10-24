"""
======================================
Check paradigm metadata before loading
======================================

Use :meth:`moabb.paradigms.MotorImagery.is_valid` on an existing MOABB dataset
instance to check its declared paradigm and events without loading recordings.
This is a **paradigm-only** check, not a guarantee that an evaluation can run.
The constructors used below only initialize metadata; that is not a guarantee
about every dataset constructor, especially custom ones.

A remote catalogue record is not a :class:`moabb.datasets.base.BaseDataset`.
Resolve its identity to an existing loader and verify its metadata first. Do not
invent a dataset or substitute a guessed session count for missing information.
"""

# License: BSD (3-clause)

import json

import moabb
from moabb.datasets import BNCI2014_001, AlexMI
from moabb.paradigms import MotorImagery


###############################################################################
# Require two overlapping classes explicitly
# ------------------------------------------
# AlexMI declares right-hand, feet and rest events, but not left-hand events.
# BNCI2014_001 declares both requested hand events. Neither constructor below
# downloads data. The check does not call ``get_data``, ``data_path``,
# ``used_events`` or ``paradigm.datasets`` (which enumerates datasets).

datasets = [AlexMI(), BNCI2014_001()]
paradigm = MotorImagery(events=["left_hand", "right_hand"], n_classes=2)
report = [
    {
        "dataset": dataset.code,
        "moabb_version": moabb.__version__,
        "paradigm_compatible": paradigm.is_valid(dataset),
        "declared_sessions": dataset.n_sessions,
        "evaluation_compatible": None,
    }
    for dataset in datasets
]
print(json.dumps(report, indent=2))

###############################################################################
# Preserve the native n_classes semantics
# ---------------------------------------
# With ``n_classes=None`` (the default), naming events does NOT require two
# overlapping classes. Thus this predicate accepts AlexMI. It still rejects a
# non-imagery dataset. Use ``n_classes=2`` when two overlapping classes are your
# intent; use :class:`moabb.paradigms.LeftRightImagery` for the fixed hand pair.
# Do not call ``used_events`` as a preflight: it can update ``n_classes``.

unspecified_classes = MotorImagery(events=["left_hand", "right_hand"])
print(unspecified_classes.is_valid(datasets[0]))  # True
print(paradigm.is_valid(datasets[0]))  # False

###############################################################################
# Evaluation compatibility stays unknown
# --------------------------------------
# ``evaluation_compatible`` above is JSON null, not false. Evaluation predicates
# are instance methods. Constructing an evaluation is NOT a pure preflight:
# it creates Results storage and can remove incompatible entries from the
# supplied dataset list. Do not construct an uninitialized evaluation or call
# an instance method with a dummy receiver to avoid those effects.
#
# Cross-session evaluation requires multiple sessions; AlexMI declares one,
# while BNCI2014_001 declares two. These are loader declarations, not evidence
# that a particular subject or selected session subset has sufficient data.
# We report the count without reimplementing evaluation predicates. Subject
# counts, folds, actual trials, channels, pipeline compatibility and runtime
# feasibility remain unchecked. A true paradigm result is not a runnable
# benchmark, a verified licence, or approval for data use.
#
# A synthetic discovery record illustrates missingness only. It is deliberately
# NOT passed to ``is_valid`` or converted into a MOABB dataset. Its unknown
# sessions remain null; missing metadata is not an incompatibility verdict.

remote_record = {"id": "synthetic-unresolved-record", "n_sessions": None}
print(json.dumps(remote_record))

###############################################################################
# References and provenance
# -------------------------
# See the linked API pages above and :class:`moabb.evaluations.CrossSessionEvaluation`
# for evaluation usage. For reproducibility, record ``moabb.__version__`` plus
# the source revision for a development installation. This recipe's metadata
# and predicate semantics were checked against `MOABB source at 3888687e0
# <https://github.com/NeuroTechX/moabb/tree/3888687e0781a81adff6b2903e3568407839b6c9>`_.
# Version labels alone do not identify a particular development checkout.
