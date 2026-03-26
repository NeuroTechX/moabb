#!/usr/bin/env python3
"""Convert Motor Imagery MOABB datasets to BIDS (BDF) for NEMAR upload.

End-to-end pipeline per dataset:
    convert → validate (nemar) → upload (nemar) → delete local copy

Processes one dataset at a time to minimize disk usage.

Skips:
  - PhysionetMI     (already re-hosted on NEMAR)
  - Zhou2016        (already on NEMAR)
  - Dreyer2023A/B/C (subsets already contained in Dreyer2023)
  - MunichMI        (deprecated alias of GrosseWentrup2009)

Requires:
  - mne-bids with BDF support (PR #1539):
        pip install "git+https://github.com/mne-tools/mne-bids.git@refs/pull/1539/head"
  - nemar-cli (https://nemar-cli.pages.dev/):
        bun install -g nemar-cli
  - Authenticated nemar session:
        nemar auth login

Usage
-----
    python scripts/convert_mi_to_nemar_bids.py [--output-dir DIR] [--overwrite]
        [--only DATASET ...] [--skip-uploaded] [--no-delete]
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import moabb.datasets as ds
from moabb.datasets.bids_interface import camel_to_kebab_case


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Motor Imagery datasets ordered smallest → largest by subject count.
# Excludes: PhysionetMI (already on NEMAR), Zhou2016 (already on NEMAR),
#           Dreyer2023A/B/C (subsets contained in full Dreyer2023),
#           MunichMI (deprecated alias of GrosseWentrup2009).
# Total: 47 datasets, 1087 subjects.
MI_DATASETS_TO_CONVERT = [
    "Beetl2021_B",          #    2 subjects
    "Beetl2021_A",          #    3 subjects
    "BNCI2003_004",         #    5 subjects
    "Wu2020",               #    6 subjects
    "Kaya2018",             #    7 subjects
    "AlexMI",               #    8 subjects
    "BNCI2014_001",         #    9 subjects
    "BNCI2014_004",         #    9 subjects
    "BNCI2015_004",         #    9 subjects
    "BNCI2019_001",         #   10 subjects
    "BNCI2025_002",         #   10 subjects
    "GrosseWentrup2009",    #   10 subjects
    "Weibo2014",            #   10 subjects
    "BNCI2015_001",         #   12 subjects
    "Tavakolan2017",        #   12 subjects
    "Zhang2017",            #   12 subjects
    "BNCI2022_001",         #   13 subjects
    "BNCI2014_002",         #   14 subjects
    "Schirrmeister2017",    #   14 subjects
    "Wairagkar2018",        #   14 subjects
    "Ofner2017",            #   15 subjects
    "Brandl2020",           #   16 subjects
    "Kumar2024",            #   18 subjects
    "Yi2025",               #   18 subjects
    "BNCI2024_001",         #   20 subjects
    "BNCI2025_001",         #   20 subjects
    "Zhou2020",             #   20 subjects
    "Gao2026",              #   22 subjects
    "Forenzo2023",          #   25 subjects
    "Jeong2020",            #   25 subjects
    "Ma2020",               #   25 subjects
    "Liu2025",              #   27 subjects
    "Chang2025",            #   28 subjects
    "Shin2017A",            #   29 subjects
    "Shin2017B",            #   29 subjects
    "Rozado2015",           #   30 subjects
    "Zuo2025",              #   30 subjects
    "GuttmannFlury2025_MI", #   31 subjects
    "TrianaGuzman2024",     #   32 subjects
    "HefmiIch2025",         #   37 subjects
    "BNCI2020_001",         #   45 subjects
    "Liu2024",              #   50 subjects
    "Yang2025",             #   51 subjects
    "Cho2017",              #   52 subjects
    "Lee2019_MI",           #   54 subjects
    "Stieger2021",          #   62 subjects
    "Dreyer2023",           #   87 subjects
]

# Datasets with extra constructor kwargs to expose all sessions
EXTRA_KWARGS = {
    "Shin2017A": dict(accept=True),
    "Shin2017B": dict(accept=True),
}

# Progress file to track completed datasets
PROGRESS_FILE = "mi_conversion_progress.json"

# Datasets that already have a NEMAR ID (to avoid creating duplicates).
# Use clone → copy → push instead of upload for these.
EXISTING_NEMAR_IDS = {
    "AlexMI": "nm000138",
    "BNCI2003_004": "nm000143",
    "BNCI2014_001": "nm000139",
    "BNCI2014_004": "nm000135",
    "BNCI2015_001": "nm000140",
    "BNCI2015_004": "nm000144",
    "BNCI2019_001": "nm000147",
    "GrosseWentrup2009": "nm000145",
    "Kaya2018": "nm000137",
    "Rozado2015": "nm000148",
    "Tavakolan2017": "nm000150",
    "Wairagkar2018": "nm000141",
    "Weibo2014": "nm000146",
    "Wu2020": "nm000142",
    "Zhang2017": "nm000152",
}


def load_progress(output_dir):
    """Load progress from previous runs."""
    p = Path(output_dir) / PROGRESS_FILE
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "uploaded": []}


def save_progress(output_dir, progress):
    """Save progress to disk."""
    p = Path(output_dir) / PROGRESS_FILE
    with open(p, "w") as f:
        json.dump(progress, f, indent=2)


def get_disk_free_gb():
    """Return free disk space in GB."""
    import os
    stat = os.statvfs("/")
    return (stat.f_bavail * stat.f_frsize) / (1024 ** 3)


def validate_bids(bids_root):
    """Run ``nemar dataset validate``. Returns (valid, report_str).

    Uses ``--ignore-warnings`` so only actual BIDS errors block upload.
    Warnings (missing recommended fields like CogPOID, CogAtlasID,
    HeadCircumference, etc.) are logged but don't prevent upload.
    """
    try:
        result = subprocess.run(
            ["nemar", "dataset", "validate", "--ignore-warnings", str(bids_root)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = result.stdout + result.stderr
        if result.returncode == 0:
            log.info("  VALID: %s", bids_root.name)
            return True, "valid"
        else:
            # Errors that are false positives from BIDS validator with BDF
            IGNORABLE_ERRORS = {
                "ALL_FILENAME_RULES_HAVE_ISSUES",  # channels.json false positive
                "JSON_SCHEMA_VALIDATION_ERROR",  # FiducialsCoordinates schema mismatch
            }
            real_errors = []
            for line in output.splitlines():
                stripped = line.strip()
                if stripped.startswith("[ERROR]"):
                    err_key = stripped.split("]")[1].strip().split(" ")[0]
                    if err_key not in IGNORABLE_ERRORS:
                        real_errors.append(err_key)
            if not real_errors:
                log.info("  VALID (ignorable errors only): %s", bids_root.name)
                return True, "valid (ignorable errors)"
            log.error("  INVALID: %s\n%s", bids_root.name, output[:500])
            return False, f"invalid (exit {result.returncode})"
    except FileNotFoundError:
        log.error("nemar-cli not found; install with: bun install -g nemar-cli")
        return False, "nemar-cli not installed"
    except subprocess.TimeoutExpired:
        log.warning("  Validation timed out for %s", bids_root.name)
        return False, "timeout"


def upload_to_nemar(bids_root, name):
    """Upload to NEMAR. Returns (success, message)."""
    try:
        result = subprocess.run(
            [
                "nemar", "dataset", "upload", str(bids_root),
                "--skip-orcid",
                "--skip-validation",
                "--yes",
            ],
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hours max per dataset
        )
        output = result.stdout + result.stderr
        if result.returncode == 0:
            log.info("  UPLOADED: %s", name)
            return True, "uploaded"
        else:
            # Check for overload/rate limit
            if "overload" in output.lower() or "rate" in output.lower() or "429" in output:
                log.warning("  OVERLOADED uploading %s, will retry later", name)
                return False, "overloaded"
            log.error("  UPLOAD FAILED: %s\n%s", name, output[:500])
            return False, f"upload failed (exit {result.returncode})"
    except FileNotFoundError:
        log.error("nemar-cli not found")
        return False, "nemar-cli not installed"
    except subprocess.TimeoutExpired:
        log.warning("  Upload timed out for %s", name)
        return False, "upload timeout"


def nemar_clone(dataset_id, output_dir):
    """Clone an existing NEMAR dataset. Returns clone path or None."""
    clone_path = Path(output_dir) / dataset_id
    if clone_path.exists():
        log.info("  Already cloned: %s", clone_path)
        return clone_path

    result = subprocess.run(
        ["nemar", "dataset", "clone", dataset_id, "-o", str(clone_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        log.error("  Clone failed: %s", result.stderr[:500])
        return None
    log.info("  Cloned %s -> %s", dataset_id, clone_path)
    return clone_path


def prepare_clone_for_bids(clone_path):
    """Remove all files from clone except .git/, .github/, .nemar/, .gitignore."""
    keep = {".git", ".github", ".nemar", ".gitignore"}
    for item in Path(clone_path).iterdir():
        if item.name in keep:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    log.info("  Cleared clone for fresh BIDS data")


def copy_bids_to_clone(bids_root, clone_path):
    """Copy converted BIDS tree into the clone directory."""
    for item in Path(bids_root).iterdir():
        dest = Path(clone_path) / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    log.info("  Copied BIDS data into clone")


def nemar_save_and_push(clone_path, name):
    """Save and push clone to NEMAR. Returns (success, message)."""
    try:
        # nemar dataset save (handles git-annex add + commit)
        result = subprocess.run(
            ["nemar", "dataset", "save"],
            cwd=str(clone_path),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            output = result.stdout + result.stderr
            log.error("  Save failed for %s: %s", name, output[:500])
            return False, f"save failed (exit {result.returncode})"
        log.info("  Saved %s", name)

        # nemar dataset push
        result = subprocess.run(
            ["nemar", "dataset", "push"],
            cwd=str(clone_path),
            capture_output=True,
            text=True,
            timeout=7200,
        )
        if result.returncode != 0:
            output = result.stdout + result.stderr
            if "overload" in output.lower() or "rate" in output.lower() or "429" in output:
                log.warning("  OVERLOADED pushing %s, will retry later", name)
                return False, "overloaded"
            log.error("  Push failed for %s: %s", name, output[:500])
            return False, f"push failed (exit {result.returncode})"
        log.info("  Pushed %s to NEMAR", name)
        return True, "pushed"
    except FileNotFoundError:
        log.error("nemar-cli not found")
        return False, "nemar-cli not installed"
    except subprocess.TimeoutExpired:
        log.warning("  Save/push timed out for %s", name)
        return False, "save/push timeout"


def process_one(name, output_dir, overwrite=False, delete_after=True):
    """Full pipeline for one dataset: convert → validate → upload → delete.

    Returns (name, status, elapsed, message, info).
    Status is one of: "uploaded", "valid_not_uploaded", "overloaded", "failed".
    info is a dict with optional keys like "nemar_id".
    """
    t0 = time.time()

    try:
        cls = getattr(ds, name)
        kwargs = EXTRA_KWARGS.get(name, {})
        dataset = cls(**kwargs)
        n_subj = len(dataset.subject_list)
        code = dataset.code
        folder_name = camel_to_kebab_case(code)

        log.info(
            "━━━ [%s] %s (%d subjects) ━━━",
            name, code, n_subj,
        )

        free_gb = get_disk_free_gb()
        log.info("  Disk free: %.1f GB", free_gb)
        if free_gb < 5:
            log.error("  LOW DISK SPACE (%.1f GB), skipping %s", free_gb, name)
            return name, "failed", time.time() - t0, "low disk space", {}

        # For NEMAR deposits, include ALL available data — many datasets
        # exclude runs/conditions by default for benchmarking convenience.
        # --- Lee2019: include test runs and resting state ---
        if hasattr(dataset, "test_run") and not dataset.test_run:
            dataset.test_run = True
        if hasattr(dataset, "resting_state") and not dataset.resting_state:
            dataset.resting_state = True
        # --- BrainInvaders: include online/adaptive phases ---
        if hasattr(dataset, "online") and not dataset.online:
            dataset.online = True
        if hasattr(dataset, "adaptive") and not dataset.adaptive:
            dataset.adaptive = True
        # --- PhysionetMI / GuttmannFlury: include motor execution ---
        if hasattr(dataset, "executed") and not dataset.executed:
            dataset.executed = True
        # --- Speier2017: include online runs ---
        if hasattr(dataset, "include_online") and not dataset.include_online:
            dataset.include_online = True
        # --- RomaniBF2025ERP: include inference + failed sessions ---
        if hasattr(dataset, "include_inference") and not dataset.include_inference:
            dataset.include_inference = True
        if hasattr(dataset, "load_failed") and not dataset.load_failed:
            dataset.load_failed = True
        # --- Sosulski2019: include SOA 60ms ---
        if hasattr(dataset, "load_soa_60") and not dataset.load_soa_60:
            dataset.load_soa_60 = True
        # --- Chailloux2020: include all tasks ---
        if hasattr(dataset, "task") and dataset.task != "all":
            dataset.task = "all"
        # --- Shin2017A/B: re-instantiate with both MI and MA ---
        if name in ("Shin2017A", "Shin2017B"):
            from moabb.datasets import Shin2017A, Shin2017B
            cls = Shin2017A if name == "Shin2017A" else Shin2017B
            dataset = cls(
                accept=True,
                motor_imagery=True,
                mental_arithmetic=True,
                return_all_modalities=True,
            )
        # --- All datasets: keep EOG/EMG/misc channels ---
        if hasattr(dataset, "return_all_modalities"):
            dataset.return_all_modalities = True

        # Step 1: Convert
        bids_format = os.environ.get("BIDS_FORMAT", "BDF")
        log.info("  [1/4] Converting to BIDS (%s)...", bids_format)
        bids_root = dataset.convert_to_bids(
            path=str(output_dir),
            subjects=None,
            overwrite=overwrite,
            format=bids_format,
            generate_figures=False,
        )

        # Rename: remove "MNE-BIDS-" prefix
        if bids_root.name.startswith("MNE-BIDS-"):
            new_name = bids_root.name[len("MNE-BIDS-"):]
            new_root = bids_root.parent / new_name
            if new_root.exists() and new_root != bids_root:
                shutil.rmtree(new_root)
            bids_root.rename(new_root)
            bids_root = new_root

        log.info("  Converted -> %s", bids_root)

        # Step 2: Validate
        log.info("  [2/4] Validating with NEMAR...")
        valid, val_report = validate_bids(bids_root)
        if not valid:
            elapsed = time.time() - t0
            return name, "failed", elapsed, f"validation: {val_report}", {}

        # Step 3: Upload (or clone→push for existing NEMAR datasets)
        nemar_id = EXISTING_NEMAR_IDS.get(name)
        clone_path = None

        if nemar_id:
            log.info("  [3/4] Pushing to existing NEMAR dataset %s...", nemar_id)
            clone_path = nemar_clone(nemar_id, output_dir)
            if clone_path is None:
                elapsed = time.time() - t0
                return name, "failed", elapsed, f"clone failed for {nemar_id}", {}

            prepare_clone_for_bids(clone_path)
            copy_bids_to_clone(bids_root, clone_path)
            uploaded, upl_report = nemar_save_and_push(clone_path, name)
        else:
            log.info("  [3/4] Uploading to NEMAR...")
            uploaded, upl_report = upload_to_nemar(bids_root, name)

        if not uploaded:
            elapsed = time.time() - t0
            if "overloaded" in upl_report:
                return name, "overloaded", elapsed, upl_report, {}
            return name, "failed", elapsed, f"upload: {upl_report}", {}

        # Step 4: Delete local copies
        if delete_after:
            log.info("  [4/4] Deleting local copies to free space...")
            # git-annex objects are read-only; fix permissions before removing
            def _force_remove(func, path, exc_info):
                try:
                    os.chmod(path, 0o755)
                except OSError:
                    pass
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    elif os.path.exists(path):
                        os.unlink(path)
                except OSError:
                    pass
            shutil.rmtree(bids_root, onexc=_force_remove)
            if clone_path and clone_path.exists():
                shutil.rmtree(clone_path, onexc=_force_remove)
            log.info("  Freed space: %s", folder_name)

        elapsed = time.time() - t0
        log.info("  DONE %s (%.1fs)", name, elapsed)
        result_info = {"nemar_id": nemar_id} if nemar_id else {}
        return name, "uploaded", elapsed, "success", result_info

    except Exception:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        log.error("  EXCEPTION %s after %.1fs:\n%s", name, elapsed, tb)
        # Clean up partial conversions to free disk space
        def _force_remove(func, path, exc_info):
            os.chmod(path, 0o755)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)
        for pattern in [f"MNE-BIDS-{folder_name}", folder_name]:
            p = Path(output_dir) / pattern
            if p.exists():
                log.info("  Cleaning up partial conversion: %s", p)
                shutil.rmtree(p, onexc=_force_remove)
        return name, "failed", elapsed, f"exception: {tb[:200]}", {}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="nemar_dataset_upload",
        help="Output directory (default: nemar_dataset_upload)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing BIDS data",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Only convert these datasets (by class name)",
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Keep local copies after upload",
    )
    parser.add_argument(
        "--skip-uploaded",
        action="store_true",
        help="Skip datasets already uploaded (from progress file)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = load_progress(output_dir)
    datasets = args.only if args.only else MI_DATASETS_TO_CONVERT

    # Filter
    to_process = []
    for name in datasets:
        if args.skip_uploaded and name in progress.get("uploaded", []):
            log.info("Skipping %s (already uploaded)", name)
            continue
        to_process.append(name)

    total = len(to_process)
    log.info(
        "Processing %d MI datasets (upload→delete, %d already done)",
        total,
        len(progress.get("uploaded", [])),
    )
    log.info("Disk free: %.1f GB", get_disk_free_gb())

    for i, name in enumerate(to_process, 1):
        log.info("")
        log.info("═══ Dataset %d/%d ═══", i, total)

        name, status, elapsed, msg, info = process_one(
            name, output_dir, args.overwrite, delete_after=not args.no_delete
        )

        if status == "uploaded":
            if name not in progress["uploaded"]:
                progress["uploaded"].append(name)
            if info.get("nemar_id"):
                progress.setdefault("nemar_ids", {})[name] = info["nemar_id"]
        elif status == "overloaded":
            log.warning("NEMAR overloaded at %s, stopping. Re-run with --skip-uploaded to resume.", name)
            save_progress(output_dir, progress)
            sys.exit(2)  # Special exit code for overload
        else:
            if name not in progress["failed"]:
                progress["failed"].append(name)

        save_progress(output_dir, progress)

        log.info(
            "Progress: %d/%d uploaded, %d failed",
            len(progress["uploaded"]),
            total,
            len(progress["failed"]),
        )

    # Final summary
    log.info("")
    log.info("═" * 70)
    log.info("ALL DONE: %d uploaded, %d failed", len(progress["uploaded"]), len(progress["failed"]))
    for name in progress["uploaded"]:
        log.info("  ✓ %s", name)
    for name in progress["failed"]:
        log.info("  ✗ %s", name)


if __name__ == "__main__":
    main()
