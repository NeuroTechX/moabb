"""BCI Competition 2020 Track 4 — Upper-limb single-arm grasping MI.

2020 International BCI Competition, Track 4. Session-to-session transfer
for three grasping tasks from a single right arm: cylindrical, spherical,
and lumbrical grasps. Each subject was recorded on three days separated
by 7 days — day 1 (training), day 2 (validation), day 3 (test).

DOI: 10.3389/fnhum.2022.898300
Data: OSF https://osf.io/pq7vb/
"""

import numpy as np
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)
from .utils import build_raw_from_epochs


_SIGN = "BCIComp2020UpperLimb"
_SFREQ = 250.0

# fmt: off
_CH_NAMES = [
    "Fp1", "AF7", "AF3", "AFz", "F7", "F5", "F3", "F1", "Fz", "FT7",
    "FC5", "FC3", "FC1", "T7", "C5", "C3", "C1", "Cz", "TP7", "CP5",
    "CP3", "CP1", "CPz", "P7", "P5", "P3", "P1", "Pz", "PO7", "PO3",
    "POz", "Fp2", "AF4", "AF8", "F2", "F4", "F6", "F8", "FC2", "FC4",
    "FC6", "FT8", "C2", "C4", "C6", "T8", "CP2", "CP4", "CP6", "TP8",
    "P2", "P4", "P6", "P8", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
]
# fmt: on

# Class names in the order used by epo.y rows (see Data Description PDF).
_CLASS_NAMES = ["cylindrical", "spherical", "lumbrical"]
_EVENTS = {name: code for code, name in enumerate(_CLASS_NAMES, start=1)}

# Stable OSF file guids for each (split, subject) pair.
# fmt: off
_OSF_URLS: dict[tuple[str, int], str] = {
    ("training", 1): "https://osf.io/download/yzmut/",
    ("training", 2): "https://osf.io/download/9n8zv/",
    ("training", 3): "https://osf.io/download/h6ry4/",
    ("training", 4): "https://osf.io/download/atnwe/",
    ("training", 5): "https://osf.io/download/f6c5a/",
    ("training", 6): "https://osf.io/download/2wu68/",
    ("training", 7): "https://osf.io/download/dwx36/",
    ("training", 8): "https://osf.io/download/h6nv7/",
    ("training", 9): "https://osf.io/download/n52yz/",
    ("training", 10): "https://osf.io/download/tv9rs/",
    ("training", 11): "https://osf.io/download/hjx5z/",
    ("training", 12): "https://osf.io/download/whpdu/",
    ("training", 13): "https://osf.io/download/d4js3/",
    ("training", 14): "https://osf.io/download/xz6ju/",
    ("training", 15): "https://osf.io/download/9sy8r/",
    ("validation", 1): "https://osf.io/download/amnzq/",
    ("validation", 2): "https://osf.io/download/g947y/",
    ("validation", 3): "https://osf.io/download/rhme9/",
    ("validation", 4): "https://osf.io/download/sz4p3/",
    ("validation", 5): "https://osf.io/download/qv9c2/",
    ("validation", 6): "https://osf.io/download/p4am3/",
    ("validation", 7): "https://osf.io/download/49byt/",
    ("validation", 8): "https://osf.io/download/b4x8q/",
    ("validation", 9): "https://osf.io/download/cpfm6/",
    ("validation", 10): "https://osf.io/download/wjpdy/",
    ("validation", 11): "https://osf.io/download/6vrph/",
    ("validation", 12): "https://osf.io/download/mq9sx/",
    ("validation", 13): "https://osf.io/download/67rnu/",
    ("validation", 14): "https://osf.io/download/dkpf8/",
    ("validation", 15): "https://osf.io/download/bq53w/",
    ("test", 1): "https://osf.io/download/72vy5/",
    ("test", 2): "https://osf.io/download/u6epr/",
    ("test", 3): "https://osf.io/download/kbfqn/",
    ("test", 4): "https://osf.io/download/kqre9/",
    ("test", 5): "https://osf.io/download/mduc6/",
    ("test", 6): "https://osf.io/download/rctjz/",
    ("test", 7): "https://osf.io/download/emcqh/",
    ("test", 8): "https://osf.io/download/tpeqj/",
    ("test", 9): "https://osf.io/download/tnezm/",
    ("test", 10): "https://osf.io/download/ud6g8/",
    ("test", 11): "https://osf.io/download/vhy7x/",
    ("test", 12): "https://osf.io/download/eznc8/",
    ("test", 13): "https://osf.io/download/mtsxu/",
    ("test", 14): "https://osf.io/download/56qb2/",
    ("test", 15): "https://osf.io/download/mqycb/",
}
# fmt: on

# Session layout: one session per recording day. Session "0" is day 1
# (training), "1" is day 2 (validation), "2" is day 3 (test). The
# competition goal is session-to-session transfer, so users should
# typically train on session "0" and evaluate on sessions "1"/"2".
_SESSIONS: list[tuple[str, str]] = [("training", "0"), ("validation", "1"), ("test", "2")]

# epo.x holds a full 10-second trial (relaxation 0-3s, cue 3-6s,
# motor imagery 6-10s) at 250 Hz for all three splits. The
# organizer-released test files have the same (2501, 60, 150) shape
# as train/val but with the cue section (samples 750-1500, i.e.
# seconds 3-6) zeroed out in-place to prevent the visual-cue response
# from inflating test accuracy. The MI window at samples 1500-2500
# is unchanged across splits, so we use the same slice everywhere
# and return only the 4-second motor imagery segment.
_MI_DURATION_S = 4.0
_MI_N_SAMPLES = int(_MI_DURATION_S * _SFREQ)  # 1000 samples
_MI_SLICE = slice(1500, 1500 + _MI_N_SAMPLES)  # samples 1500-2500 = 6-10 s

# Day-3 test labels, extracted from the OSF answer sheet
# Track4_Answer_Sheet_Test.xlsx. Values are 0-indexed class IDs
# matching the _CLASS_NAMES order. The loader converts them to
# 1-indexed events to match epo.y semantics.
#
# Some subjects share identical trial orders (S2==S6==S8, S4==S7==S9);
# this is a property of the organizer-provided answer sheet, not a
# parsing error.
# fmt: off
_TEST_LABELS_DAY3: dict[int, tuple[int, ...]] = {
    1: (0, 0, 2, 2, 2, 0, 2, 2, 0, 2, 1, 1, 0, 0, 2, 2, 2, 1, 1, 2, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 2, 2, 2, 1, 1, 1, 2, 1, 0, 1, 2, 1, 0, 2, 2, 0, 2, 1, 2, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 1, 2, 1, 2, 0, 1, 0, 1, 1, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2, 0, 1, 0, 0, 0, 0, 2, 1, 2, 1, 1, 1, 2, 1, 0, 1, 2, 0, 0, 1, 0, 1, 1, 1, 2, 0, 1, 2, 1, 0, 1, 1, 2, 2, 0, 1, 2, 1, 0, 0, 0, 2, 2, 1, 1, 1, 2, 2, 0, 2, 0, 1, 1, 1, 0, 2, 2),
    2: (2, 0, 2, 0, 0, 0, 1, 1, 2, 2, 0, 2, 1, 2, 0, 0, 2, 0, 2, 0, 0, 1, 0, 2, 0, 1, 2, 0, 2, 2, 1, 1, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0, 2, 1, 2, 2, 0, 2, 1, 1, 2, 0, 1, 0, 2, 0, 1, 1, 0, 1, 2, 2, 1, 1, 0, 1, 2, 0, 1, 0, 1, 0, 0, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 0, 2, 1, 1, 1, 2, 0, 0, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0, 2, 2, 2, 0, 1, 0, 0, 2, 0, 0, 1, 1, 0, 1, 1, 0, 0, 2, 1, 1, 0, 1, 1, 2, 0, 2, 2, 0, 1, 2, 2, 2, 1, 0, 1, 0, 1, 1, 0, 1, 2, 0, 1, 2, 2, 0, 1),
    3: (2, 1, 1, 2, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 2, 1, 2, 1, 2, 1, 0, 2, 2, 0, 1, 0, 1, 0, 2, 2, 2, 0, 0, 2, 0, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 0, 0, 1, 2, 0, 2, 1, 0, 1, 2, 0, 1, 0, 2, 0, 2, 1, 0, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 2, 2, 1, 1, 0, 2, 0, 0, 1, 0, 1, 1, 0, 2, 1, 1, 2, 1, 0, 0, 1, 0, 1, 2, 2, 2, 1, 0, 0, 1, 0, 0, 2, 0, 1, 0, 2, 1, 1, 2, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 0, 2, 2, 1, 1, 0, 2, 1, 1, 0, 2, 2, 1, 0, 1, 0, 1, 1, 2, 0, 2, 0),
    4: (0, 0, 0, 0, 2, 0, 2, 2, 1, 2, 2, 1, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 2, 2, 1, 0, 1, 1, 0, 1, 1, 1, 2, 0, 0, 0, 2, 2, 2, 1, 1, 2, 1, 1, 0, 0, 2, 0, 2, 1, 2, 1, 1, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 2, 1, 1, 1, 2, 2, 1, 0, 0, 0, 0, 2, 2, 0, 0, 1, 0, 1, 2, 0, 0, 0, 2, 1, 0, 2, 1, 0, 0, 1, 1, 2, 2, 0, 0, 2, 2, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 2, 2, 1, 1, 2, 0, 1, 0, 2, 2, 2, 1, 1, 0, 1, 0, 1, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 1, 2, 0, 0, 1, 1, 2, 0, 1, 2),
    5: (1, 1, 2, 0, 2, 1, 1, 2, 1, 0, 1, 0, 0, 0, 2, 0, 1, 0, 1, 1, 2, 2, 2, 1, 0, 0, 1, 1, 0, 0, 1, 2, 0, 2, 0, 1, 2, 2, 0, 2, 1, 0, 2, 0, 1, 1, 0, 2, 1, 0, 1, 1, 1, 1, 0, 0, 1, 2, 2, 2, 2, 0, 1, 0, 0, 1, 1, 2, 1, 1, 0, 0, 0, 0, 1, 2, 0, 0, 2, 0, 2, 1, 1, 2, 0, 2, 0, 0, 2, 2, 1, 2, 1, 2, 0, 1, 0, 2, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 2, 0, 2, 0, 2, 0, 2, 1, 1, 2, 2, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1, 2, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1, 0, 1, 2, 0, 2, 2),
    6: (2, 0, 2, 0, 0, 0, 1, 1, 2, 2, 0, 2, 1, 2, 0, 0, 2, 0, 2, 0, 0, 1, 0, 2, 0, 1, 2, 0, 2, 2, 1, 1, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0, 2, 1, 2, 2, 0, 2, 1, 1, 2, 0, 1, 0, 2, 0, 1, 1, 0, 1, 2, 2, 1, 1, 0, 1, 2, 0, 1, 0, 1, 0, 0, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 0, 2, 1, 1, 1, 2, 0, 0, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0, 2, 2, 2, 0, 1, 0, 0, 2, 0, 0, 1, 1, 0, 1, 1, 0, 0, 2, 1, 1, 0, 1, 1, 2, 0, 2, 2, 0, 1, 2, 2, 2, 1, 0, 1, 0, 1, 1, 0, 1, 2, 0, 1, 2, 2, 0, 1),
    7: (0, 0, 0, 0, 2, 0, 2, 2, 1, 2, 2, 1, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 2, 2, 1, 0, 1, 1, 0, 1, 1, 1, 2, 0, 0, 0, 2, 2, 2, 1, 1, 2, 1, 1, 0, 0, 2, 0, 2, 1, 2, 1, 1, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 2, 1, 1, 1, 2, 2, 1, 0, 0, 0, 0, 2, 2, 0, 0, 1, 0, 1, 2, 0, 0, 0, 2, 1, 0, 2, 1, 0, 0, 1, 1, 2, 2, 0, 0, 2, 2, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 2, 2, 1, 1, 2, 0, 1, 0, 2, 2, 2, 1, 1, 0, 1, 0, 1, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 1, 2, 0, 0, 1, 1, 2, 0, 1, 2),
    8: (2, 0, 2, 0, 0, 0, 1, 1, 2, 2, 0, 2, 1, 2, 0, 0, 2, 0, 2, 0, 0, 1, 0, 2, 0, 1, 2, 0, 2, 2, 1, 1, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0, 2, 1, 2, 2, 0, 2, 1, 1, 2, 0, 1, 0, 2, 0, 1, 1, 0, 1, 2, 2, 1, 1, 0, 1, 2, 0, 1, 0, 1, 0, 0, 1, 2, 1, 2, 2, 1, 1, 2, 1, 2, 2, 0, 2, 1, 1, 1, 2, 0, 0, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0, 2, 2, 2, 0, 1, 0, 0, 2, 0, 0, 1, 1, 0, 1, 1, 0, 0, 2, 1, 1, 0, 1, 1, 2, 0, 2, 2, 0, 1, 2, 2, 2, 1, 0, 1, 0, 1, 1, 0, 1, 2, 0, 1, 2, 2, 0, 1),
    9: (0, 0, 0, 0, 2, 0, 2, 2, 1, 2, 2, 1, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 1, 1, 0, 2, 2, 1, 0, 1, 1, 0, 1, 1, 1, 2, 0, 0, 0, 2, 2, 2, 1, 1, 2, 1, 1, 0, 0, 2, 0, 2, 1, 2, 1, 1, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 2, 1, 1, 1, 2, 2, 1, 0, 0, 0, 0, 2, 2, 0, 0, 1, 0, 1, 2, 0, 0, 0, 2, 1, 0, 2, 1, 0, 0, 1, 1, 2, 2, 0, 0, 2, 2, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 2, 2, 1, 1, 2, 0, 1, 0, 2, 2, 2, 1, 1, 0, 1, 0, 1, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 1, 2, 0, 0, 1, 1, 2, 0, 1, 2),
    10: (2, 0, 1, 2, 1, 1, 1, 0, 0, 0, 1, 2, 0, 0, 2, 2, 2, 1, 2, 0, 1, 0, 2, 0, 1, 1, 0, 2, 0, 1, 1, 1, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 1, 1, 2, 0, 0, 0, 0, 2, 2, 2, 1, 1, 0, 1, 2, 0, 2, 2, 2, 1, 1, 1, 0, 1, 1, 2, 2, 0, 0, 2, 1, 2, 2, 1, 2, 0, 0, 0, 2, 1, 2, 0, 1, 0, 1, 2, 2, 2, 1, 0, 1, 1, 0, 2, 1, 0, 2, 0, 2, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 2, 2, 1, 1, 0, 0, 1, 0, 0, 2, 2, 1, 1, 2, 2, 0, 1, 2, 2, 0, 0, 0, 2, 2, 0, 2, 2, 1, 0, 1, 0, 2, 0),
    11: (0, 0, 1, 1, 1, 0, 2, 2, 1, 2, 1, 2, 2, 0, 0, 1, 0, 0, 2, 1, 1, 2, 2, 0, 1, 1, 1, 2, 2, 2, 0, 1, 1, 1, 1, 1, 2, 0, 2, 2, 0, 2, 2, 0, 0, 2, 2, 0, 2, 0, 0, 2, 2, 1, 0, 0, 1, 2, 2, 0, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 2, 2, 0, 0, 0, 2, 1, 0, 2, 0, 2, 2, 1, 0, 1, 1, 2, 1, 1, 1, 0, 2, 1, 1, 1, 2, 1, 0, 0, 0, 2, 1, 2, 1, 0, 2, 1, 2, 2, 2, 2, 2, 1, 0, 2, 2, 1, 1, 0, 0, 0, 1, 0, 2, 0, 1, 1, 0, 1, 1, 2, 0, 1, 1, 0, 2, 2, 0, 1, 0, 2, 0, 2, 1, 1, 2, 1),
    12: (1, 0, 0, 2, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 0, 1, 0, 0, 2, 0, 1, 0, 2, 0, 2, 0, 2, 1, 0, 1, 2, 2, 0, 0, 1, 1, 1, 0, 1, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 0, 2, 1, 1, 0, 2, 0, 0, 2, 2, 1, 1, 0, 2, 2, 2, 1, 2, 0, 0, 2, 1, 2, 0, 1, 1, 0, 2, 0, 2, 1, 0, 1, 2, 1, 1, 2, 0, 1, 1, 2, 2, 0, 0, 1, 1, 0, 2, 2, 0, 0, 1, 2, 0, 2, 0, 2, 1, 2, 0, 0, 0, 0, 2, 1, 1, 2, 0, 1, 2, 2, 2, 0, 2, 0, 0, 2, 0, 0, 2, 0, 1, 0, 1, 2, 2, 0, 1, 2),
    13: (0, 2, 1, 2, 2, 0, 0, 0, 0, 1, 0, 2, 2, 1, 0, 1, 0, 2, 2, 0, 1, 2, 0, 1, 0, 1, 2, 1, 1, 1, 1, 1, 2, 0, 1, 2, 0, 1, 2, 1, 1, 0, 1, 1, 0, 1, 1, 2, 2, 0, 1, 0, 2, 0, 0, 1, 0, 2, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 2, 0, 2, 2, 2, 1, 0, 0, 2, 1, 2, 0, 1, 0, 2, 1, 1, 1, 1, 2, 2, 1, 2, 0, 1, 0, 0, 2, 2, 2, 0, 2, 1, 2, 2, 2, 1, 0, 2, 0, 0, 2, 0, 0, 0, 2, 1, 2, 1, 0, 0, 1, 2, 1, 2, 2, 2, 0, 0, 2, 2, 1, 2, 1, 1, 0, 1, 0, 0, 0, 2, 2, 1, 2),
    14: (0, 1, 0, 1, 2, 1, 0, 2, 1, 0, 2, 0, 0, 0, 0, 2, 2, 1, 0, 2, 2, 2, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 2, 1, 0, 2, 1, 0, 0, 2, 1, 1, 0, 1, 1, 2, 0, 1, 0, 2, 1, 1, 1, 2, 2, 1, 0, 1, 0, 2, 2, 2, 0, 0, 1, 0, 1, 2, 0, 2, 2, 0, 0, 0, 1, 1, 0, 0, 2, 0, 1, 2, 0, 0, 0, 2, 1, 0, 2, 1, 2, 2, 1, 2, 2, 0, 2, 1, 1, 2, 2, 1, 0, 2, 0, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 0, 1, 0, 1, 2, 2, 2, 0, 2, 1, 2, 1, 2, 1, 0, 1, 2, 2, 1, 2, 0, 2, 1),
    15: (2, 2, 1, 2, 0, 1, 2, 0, 0, 0, 2, 2, 0, 1, 2, 1, 1, 1, 0, 1, 2, 0, 2, 2, 2, 1, 0, 0, 1, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 1, 2, 1, 0, 0, 1, 1, 2, 2, 0, 0, 1, 2, 1, 2, 1, 1, 2, 0, 0, 2, 0, 2, 1, 2, 1, 1, 1, 0, 0, 0, 2, 2, 0, 0, 1, 1, 1, 0, 1, 1, 2, 0, 1, 2, 1, 0, 1, 0, 0, 1, 1, 1, 0, 2, 0, 0, 1, 0, 2, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 2, 1, 1, 2, 0, 0, 2, 0, 1, 1, 0, 0, 1, 1, 2, 0, 1, 1, 2, 2, 1, 1, 1, 2, 1, 0, 2, 2, 0, 1),
}
# fmt: on


class BCIComp2020UpperLimb(BaseDataset):
    """BCI Competition 2020 Track 4 — Upper-limb grasping MI (session-to-session).

    Dataset from the 2020 International BCI Competition [1]_.

    **Dataset Description**

    Fifteen right-handed subjects (S1-S15, aged 20-34) performed motor
    imagery of three grasping tasks on a single right arm: cylindrical
    grasp (holding a cup), spherical grasp (holding a ball), and
    lumbrical grasp (holding a card). EEG was recorded at 250 Hz using
    60 channels in a 10-20 configuration with a BrainAmp amplifier
    (BrainProducts GmbH), FCz reference, Fpz ground, and a 60 Hz notch.

    Each subject was recorded on three separate days (7 days apart) to
    pose a session-to-session transfer problem. Each session contains
    150 trials (50 per class). A single trial lasts 10 s: 0-3 s
    relaxation, 3-6 s visual cue (flashing green circle around the
    targeted object), 6-10 s motor imagery. Only the 4-second motor
    imagery window is exposed by this loader, yielding consistent
    (150, 60, 1000) arrays per session.

    **Session layout**

    - Session "0": day 1, intended for training
    - Session "1": day 2, intended for validation
    - Session "2": day 3, intended as the held-out transfer test

    The test-day .mat files released by the organizers have the cue
    section (samples 750-1500, i.e. seconds 3-6 of each trial)
    zeroed in-place to prevent the visual-cue response from
    inflating results. The array shape stays at (2501, 60, 150),
    so the same sample slice (1500-2500) extracts the motor imagery
    window consistently across train / val / test.

    Test-day labels are published as a separate answer-sheet XLSX on
    OSF rather than inside the .mat files. They are embedded in this
    module as :data:`_TEST_LABELS_DAY3` so the loader can return
    labelled data for all three sessions without a second download.

    References
    ----------
    .. [1] Jeong, J.-H. et al. (2022). 2020 International brain-computer
           interface competition: A review. Frontiers in Human Neuroscience,
           16, 898300. https://doi.org/10.3389/fnhum.2022.898300
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=60,
            channel_types={"eeg": 60},
            montage="standard_1005",
            hardware="BrainAmp (BrainProducts GmbH)",
            software="BrainVision with MATLAB 2019a",
            reference="FCz",
            ground="Fpz",
            sensors=list(_CH_NAMES),
            filters={"notch_hz": 60},
            line_freq=60.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=15,
            health_status="healthy",
            age_min=20,
            age_max=34,
            # Data description PDF has garbled text around handedness
            # ("all right-handed - S1, 4, and 5; all right-handed"),
            # so the per-subject split is unclear. The task is
            # performed on a single right arm, but we leave
            # handedness unset rather than over-claim.
            species="human",
        ),
        experiment=ExperimentMetadata(
            events=_EVENTS,
            paradigm="imagery",
            n_classes=len(_EVENTS),
            class_labels=list(_CLASS_NAMES),
            trial_duration=_MI_DURATION_S,
            study_design=(
                "Three-session cue-based motor imagery of three grasping "
                "tasks on a single right arm (cylindrical, spherical, "
                "lumbrical). Each original trial is 10 s long (3 s rest, "
                "3 s cue, 4 s motor imagery); this loader exposes only "
                "the 4 s motor imagery window. Sessions are 7 days "
                "apart to pose a session-to-session transfer problem."
            ),
            stimulus_type="visual cue",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnhum.2022.898300",
            investigators=[
                "Ji-Hoon Jeong",
                "Jeong-Hyun Cho",
                "Young-Eun Lee",
                "Seo-Hyun Lee",
                "Gi-Hwan Shin",
                "Young-Seok Kweon",
                "Jose del R. Millan",
                "Klaus-Robert Muller",
                "Seong-Whan Lee",
            ],
            institution="Korea University",
            country="KR",
            publication_year=2022,
            license="CC-BY-4.0",
            data_url="https://osf.io/pq7vb/",
            repository="OSF",
            contact_info=["bcicompetition2020@gmail.com"],
            associated_paper_doi="10.3389/fnhum.2022.898300",
            keywords=[
                "motor imagery",
                "upper-limb",
                "grasping",
                "session-to-session transfer",
                "BCI competition",
                "EEG",
            ],
            description=(
                "BCI Competition 2020 Track 4: session-to-session motor "
                "imagery decoding of three grasping tasks (cylindrical, "
                "spherical, lumbrical) from a single right arm. 15 "
                "subjects, 60 channels, 250 Hz, three sessions on "
                "separate days."
            ),
        ),
        sessions_per_subject=3,
        runs_per_session=1,
        tags=Tags(
            pathology=["Healthy"],
            modality=["MotorImagery"],
            type=["Research", "Competition"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "60 Hz notch filter",
                "cue-aligned epoching",
                "motor imagery window (6-10 s of each trial) extracted by loader",
            ],
            notch_hz=60.0,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=list(_CLASS_NAMES),
            imagery_duration_s=_MI_DURATION_S,
        ),
        data_structure=DataStructureMetadata(
            n_trials=450,
            trials_context=(
                "450 trials per subject (150 trials/session x 3 sessions, "
                "50 per class). 15 subjects, 6750 trials total."
            ),
        ),
        data_processed=True,
        file_format="MAT",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=3,
            events=_EVENTS,
            code="BCIComp2020UpperLimb",
            interval=[0, _MI_DURATION_S],
            paradigm="imagery",
            doi="10.3389/fnhum.2022.898300",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    @staticmethod
    def _load_epoch_mat(fpath, split, subject):
        """Load a Track 4 .mat file and return (data, labels, ch_names).

        Extracts the 4-second motor imagery window (samples 1500-2500
        of each 10-second trial) from ``epo.x``, and uses test-set
        labels from :data:`_TEST_LABELS_DAY3` when the split is
        ``"test"`` (those files do not store ``epo.y``).
        """
        mat = loadmat(fpath, squeeze_me=False, variable_names=["epo"])
        epo = mat["epo"]

        # x is (n_times, n_channels, n_trials); slice to the MI window
        # then transpose to (n_trials, n_channels, n_samples).
        x = epo["x"][0, 0]
        data = np.transpose(x[_MI_SLICE, :, :], (2, 1, 0))

        if split == "test":
            labels = np.array(_TEST_LABELS_DAY3[subject], dtype=int) + 1
        else:
            # y is (n_classes, n_trials) one-hot; argmax gives 0-indexed label.
            labels = np.argmax(epo["y"][0, 0], axis=0) + 1

        ch_names = [str(c[0]) for c in epo["clab"][0, 0][0]]
        return data, labels, ch_names

    def _download_all_splits(self, subject, path, force_update, verbose):
        """Download every split file for ``subject`` and return the paths.

        Kept separate from :meth:`data_path` so the per-subject hot path
        in :meth:`_get_single_subject_data` downloads each file once,
        instead of re-running the loop per split.
        """
        return {
            split_name: dl.data_dl(
                _OSF_URLS[(split_name, subject)],
                _SIGN,
                path=path,
                force_update=force_update,
                verbose=verbose,
            )
            for split_name, _ in _SESSIONS
        }

    def _get_single_subject_data(self, subject):
        """Return data for a single subject, one MOABB session per recording day."""
        paths = self._download_all_splits(
            subject, path=None, force_update=False, verbose=None
        )
        sessions = {}
        for split, session_key in _SESSIONS:
            data, labels, ch_names = self._load_epoch_mat(paths[split], split, subject)
            sessions[session_key] = {
                "0": build_raw_from_epochs(
                    data, ch_names, _SFREQ, labels, montage_name="standard_1005"
                )
            }
        return sessions

    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
        *,
        split=None,
    ):
        """Return local paths for a subject's split files.

        Downloads training + validation + test files for ``subject``
        via :func:`moabb.datasets.download.data_dl`. Returns the path
        for the requested ``split`` (defaults to ``"training"``).
        """
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")

        paths = self._download_all_splits(subject, path, force_update, verbose)
        return paths[split] if split else paths["training"]
