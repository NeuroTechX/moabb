#!/usr/bin/env python3
"""Convert all downloaded MOABB datasets to BIDS format for NEMAR upload.

Converts datasets smallest-first using ``dataset.convert_to_bids()``.
Skips PhysionetMI (already on PhysioNet/NEMAR), Zhou2016 (per user request),
and Dreyer2023A/B/C subsets (included in the full Dreyer2023 dataset).

Usage
-----
    python scripts/convert_to_nemar_bids.py [--output-dir DIR] [--overwrite] [--only DATASET] [--workers N]
"""

import argparse
import logging
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(process)d] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Ordered smallest → largest
DATASETS_TO_CONVERT = [
    "Beetl2021_B",  #   2 subjects
    "Beetl2021_A",  #   3 subjects
    "BNCI2003_004",  #   5 subjects
    "BNCI2015_013",  #   6 subjects
    "AlexMI",  #   8 subjects
    "BNCI2014_008",  #   8 subjects
    "EPFLP300",  #   8 subjects
    "Wang2021Combined",  #   8 subjects
    "BNCI2014_001",  #   9 subjects
    "BNCI2014_004",  #   9 subjects
    "BNCI2015_004",  #   9 subjects
    "Nakanishi2015",  #   9 subjects
    "BNCI2014_009",  #  10 subjects
    "BNCI2015_003",  #  10 subjects
    "BNCI2015_012",  #  10 subjects
    "BNCI2019_001",  #  10 subjects
    "BNCI2025_002",  #  10 subjects
    "GrosseWentrup2009",  #  10 subjects
    "Weibo2014",  #  10 subjects
    "BNCI2015_006",  #  11 subjects
    "Kojima2024A",  #  11 subjects
    "MAMEM1",  #  11 subjects
    "MAMEM2",  #  11 subjects
    "MAMEM3",  #  11 subjects
    "BNCI2015_001",  #  12 subjects
    "BNCI2015_010",  #  12 subjects
    "CastillosBurstVEP100",  #  12 subjects
    "CastillosBurstVEP40",  #  12 subjects
    "CastillosCVEP100",  #  12 subjects
    "CastillosCVEP40",  #  12 subjects
    "Cattan2019_PHMD",  #  12 subjects
    "Chen2017SingleFlicker",  #  12 subjects
    "Huebner2018",  #  12 subjects
    "Kalunga2016",  #  12 subjects
    "Thielen2015",  #  12 subjects
    "BNCI2015_008",  #  13 subjects
    "BNCI2022_001",  #  13 subjects
    "Huebner2017",  #  13 subjects
    "Sosulski2019",  #  13 subjects
    "BNCI2014_002",  #  14 subjects
    "Schirrmeister2017",  #  14 subjects
    "BNCI2016_002",  #  15 subjects
    "Hinss2021",  #  15 subjects
    "Kojima2024B",  #  15 subjects
    "BNCI2015_007",  #  16 subjects
    "MartinezCagigal2023Checker",  # 16 subjects
    "MartinezCagigal2023Pary",  # 16 subjects
    "BNCI2020_002",  #  18 subjects
    "Rodrigues2017",  #  19 subjects
    "BNCI2024_001",  #  20 subjects
    "BNCI2025_001",  #  20 subjects
    "BNCI2015_009",  #  21 subjects
    "Cattan2019_VR",  #  21 subjects
    "RomaniBF2025ERP",  #  22 subjects
    "BI2013a",  #  24 subjects
    "Han2024Fatigue",  #  24 subjects
    "Lee2021Mobile_ERP",  #  24 subjects
    "Lee2021Mobile_SSVEP",  #  24 subjects
    "BI2012",  #  25 subjects
    "Shin2017A",  #  29 subjects
    "Shin2017B",  #  29 subjects
    "Thielen2021",  #  30 subjects
    "Wang2016",  #  34 subjects
    "BI2014b",  #  38 subjects
    "ErpCore2021_ERN",  #  40 subjects
    "ErpCore2021_LRP",  #  40 subjects
    "ErpCore2021_MMN",  #  40 subjects
    "ErpCore2021_N170",  #  40 subjects
    "ErpCore2021_N2pc",  #  40 subjects
    "ErpCore2021_N400",  #  40 subjects
    "ErpCore2021_P3",  #  40 subjects
    "Kim2025BetaRange",  #  40 subjects
    "BI2015a",  #  43 subjects
    "BI2015b",  #  44 subjects
    "BNCI2020_001",  #  45 subjects
    "Liu2024",  #  50 subjects
    "Cho2017",  #  52 subjects
    "Dong2023",  #  59 subjects
    "BI2014a",  #  64 subjects
    "Liu2020BETA",  #  70 subjects
    "Dreyer2023",  #  87 subjects
]

# Datasets with extra flags that unlock additional sessions/conditions
EXTRA_KWARGS = {
    "BI2012": dict(training=True, online=True),
    "BI2013a": dict(non_adaptive=True, adaptive=True, training=True, online=True),
    "Shin2017A": dict(accept=True),
    "Shin2017B": dict(accept=True),
    "Sosulski2019": dict(load_soa_60=True),
    "RomaniBF2025ERP": dict(extra_runs=True, include_inference=True),
}

# Datasets that must run in the main process (not in ProcessPoolExecutor).
# Wang2021Combined uses mne.io.read_raw_ant() which relies on a C library
# (libEep) that is not fork-safe on macOS.
IN_PROCESS_DATASETS = {"Wang2021Combined"}


def _expected_folder_name(name):
    """Return the expected BIDS folder name for a dataset class name.

    Uses the dataset's ``code`` property (not the class name) to match the
    folder name that ``convert_to_bids()`` actually creates.
    """
    import moabb.datasets as ds
    from moabb.datasets.bids_interface import camel_to_kebab_case

    cls = getattr(ds, name)
    kwargs = EXTRA_KWARGS.get(name, {})
    dataset = cls(**kwargs)
    return camel_to_kebab_case(dataset.code)


def convert_one(name, output_dir, overwrite=False):
    """Convert a single dataset to BIDS. Returns (name, success, elapsed_seconds).

    This function runs in a worker process.
    """
    import shutil

    # Re-configure logging in worker process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(process)d] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    wlog = logging.getLogger(name)

    t0 = time.time()
    try:
        import moabb.datasets as ds

        cls = getattr(ds, name)
        kwargs = EXTRA_KWARGS.get(name, {})
        dataset = cls(**kwargs)

        n_subj = len(dataset.subject_list)
        wlog.info("Converting %s (%d subjects)...", name, n_subj)
        bids_root = dataset.convert_to_bids(
            path=str(output_dir),
            subjects=None,  # all subjects
            overwrite=overwrite,
            format="EEGLAB",
        )
        # Rename folder: remove "MNE-BIDS-" prefix
        if bids_root.name.startswith("MNE-BIDS-"):
            new_name = bids_root.name[len("MNE-BIDS-") :]
            new_root = bids_root.parent / new_name
            if new_root.exists() and new_root != bids_root:
                shutil.rmtree(new_root)
            bids_root.rename(new_root)
            bids_root = new_root
        elapsed = time.time() - t0
        wlog.info("Done %s -> %s (%.1fs)", name, bids_root, elapsed)
        return name, True, elapsed
    except Exception:
        elapsed = time.time() - t0
        wlog.error("FAILED %s after %.1fs:\n%s", name, elapsed, traceback.format_exc())
        return name, False, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="nemar_dataset_upload",
        help="Output directory for BIDS datasets (default: nemar_dataset_upload)",
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
        "--skip-existing",
        action="store_true",
        help="Skip datasets whose output folder already exists",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of parallel workers (default: 3)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.only if args.only else DATASETS_TO_CONVERT

    # Filter out skipped datasets
    to_convert = []
    skipped = []
    for name in datasets:
        if name not in DATASETS_TO_CONVERT and not args.only:
            log.warning("Skipping unknown dataset: %s", name)
            continue
        if args.skip_existing:
            folder = _expected_folder_name(name)
            if (output_dir / folder).exists():
                log.info("Skipping %s (folder %s already exists)", name, folder)
                skipped.append(name)
                continue
        to_convert.append(name)

    log.info(
        "Converting %d datasets with %d workers (%d skipped)",
        len(to_convert),
        args.workers,
        len(skipped),
    )

    succeeded = []
    failed = []

    # Run fork-unsafe datasets in the main process first
    in_process = [n for n in to_convert if n in IN_PROCESS_DATASETS]
    via_executor = [n for n in to_convert if n not in IN_PROCESS_DATASETS]

    for name in in_process:
        name, ok, elapsed = convert_one(name, output_dir, args.overwrite)
        (succeeded if ok else failed).append((name, elapsed))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(convert_one, name, output_dir, args.overwrite): name
            for name in via_executor
        }
        for future in as_completed(futures):
            name, ok, elapsed = future.result()
            if ok:
                succeeded.append((name, elapsed))
            else:
                failed.append((name, elapsed))

    log.info("=" * 60)
    log.info(
        "CONVERSION COMPLETE: %d succeeded, %d failed, %d skipped",
        len(succeeded),
        len(failed),
        len(skipped),
    )
    for name, elapsed in succeeded:
        log.info("  OK   %s (%.1fs)", name, elapsed)
    for name, elapsed in failed:
        log.info("  FAIL %s (%.1fs)", name, elapsed)
    for name in skipped:
        log.info("  SKIP %s", name)


if __name__ == "__main__":
    main()
