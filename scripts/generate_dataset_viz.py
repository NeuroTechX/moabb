#!/usr/bin/env python
"""Generate all dataset visualisation SVGs for documentation.

Produces three types of SVGs per dataset:
  1. Stimulus protocol timeline  -> _static/timelines/<Name>.svg
  2. Class balance bar chart     -> _static/viz/<Name>_classes.svg
  3. Session structure diagram   -> _static/viz/<Name>_sessions.svg

Usage (from repo root):
    PYTHONPATH=. python scripts/generate_dataset_viz.py
"""

import os
import sys
import traceback


# Ensure repo root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from moabb.analysis.timeline import (
    class_balance_svg,
    session_structure_svg,
    stimulus_timeline_svg,
)
from moabb.datasets.utils import dataset_list


TIMELINE_DIR = os.path.join("docs", "source", "_static", "timelines")
VIZ_DIR = os.path.join("docs", "source", "_static", "viz")


def main():
    os.makedirs(TIMELINE_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    n_ok = 0
    n_fail = 0

    for ds_cls in dataset_list:
        name = ds_cls.__name__
        print(f"  {name} ...", end=" ", flush=True)
        try:
            ds = ds_cls()
        except Exception:
            print("SKIP (cannot instantiate)")
            n_fail += 1
            continue

        generated = []

        # 1. Timeline
        try:
            svg = stimulus_timeline_svg(ds)
            path = os.path.join(TIMELINE_DIR, f"{name}.svg")
            with open(path, "w", encoding="utf-8") as f:
                f.write(svg)
            generated.append("timeline")
        except Exception:
            traceback.print_exc()

        # 2. Class balance
        try:
            svg = class_balance_svg(ds)
            if svg:
                path = os.path.join(VIZ_DIR, f"{name}_classes.svg")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(svg)
                generated.append("classes")
        except Exception:
            traceback.print_exc()

        # 3. Session structure
        try:
            svg = session_structure_svg(ds)
            if svg:
                path = os.path.join(VIZ_DIR, f"{name}_sessions.svg")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(svg)
                generated.append("sessions")
        except Exception:
            traceback.print_exc()

        if generated:
            print(", ".join(generated))
            n_ok += 1
        else:
            print("no output")

    print(f"\nDone: {n_ok} datasets with SVGs, {n_fail} skipped")


if __name__ == "__main__":
    main()
