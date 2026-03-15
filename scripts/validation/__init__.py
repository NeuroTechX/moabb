"""Shared registry and utilities for dataset validation scripts."""

import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "results"

DEFAULT_WORKERS = max(1, os.cpu_count() // 4)

# ── Dataset class names grouped by paradigm (strings for pickling) ───────────

DATASET_NAMES = {
    "imagery": [
        "Brandl2020",
        "Chang2025",
        "Forenzo2023",
        "Gao2026",
        "GuttmannFlury2025_MI",
        "HefmiIch2025",
        "Jeong2020",
        "Kaya2018",
        "Kumar2024",
        "Liu2025",
        "Ma2020",
        "Rozado2015",
        "Tavakolan2017",
        "TrianaGuzman2024",
        "Wairagkar2018",
        "Wu2020",
        "Yang2025",
        "Yi2025",
        "Zhang2017",
        "Zhou2020",
        "Zuo2025",
    ],
    "p300": [
        "Chailloux2020",
        "GuttmannFlury2025_P300",
        "Lee2021Mobile_ERP",
        "Lee2024_TV",
        "Lee2024_DL",
        "Lee2024_EL",
        "Lee2024_BS",
        "Lee2024_AC",
        "Mainsah2025_A",
        "Mainsah2025_B",
        "Mainsah2025_C",
        "Mainsah2025_D",
        "Mainsah2025_E",
        "Mainsah2025_F",
        "Mainsah2025_G",
        "Mainsah2025_H",
        "Mainsah2025_I",
        "Mainsah2025_J",
        "Mainsah2025_K",
        "Mainsah2025_L",
        "Mainsah2025_M",
        "Mainsah2025_N",
        "Mainsah2025_O",
        "Mainsah2025_P",
        "Mainsah2025_Q",
        "Mainsah2025_R",
        "Mainsah2025_S1",
        "Mainsah2025_S2",
        "Simoes2020",
        "Speier2017",
        "Zhang2025",
        "Zheng2020",
    ],
    "ssvep": [
        "Chen2017SingleFlicker",
        "Dong2023",
        "GuttmannFlury2025_SSVEP",
        "Han2024Fatigue",
        "Kim2025BetaRange",
        "Lee2021Mobile_SSVEP",
        "Liu2020BETA",
        "Liu2022EldBETA",
        "Wang2021Combined",
    ],
}


def all_work_items():
    """Return list of (dataset_name, paradigm_key) tuples."""
    work = []
    for pkey, names in DATASET_NAMES.items():
        for name in names:
            work.append((name, pkey))
    return work


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
