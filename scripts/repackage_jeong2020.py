#!/usr/bin/env python3
"""Download, resample, and repackage Jeong2020 for Zenodo re-hosting.

Uses s5cmd for fast parallel S3 downloads (no tarball needed).

Pipeline:
  1. Download raw BrainVision files per subject via s5cmd from Wasabi S3
  2. Load with MNE, resample 2500 -> 1000 Hz (keep all channels)
  3. Save as new BrainVision, ZIP per subject
  4. Clean up raw files

Disk requirements (in WORK_DIR):
  - ~13 GB per subject raw (downloaded, processed one at a time)
  Output: ~2 GB per subject ZIP = ~50 GB total

Usage:
  python scripts/repackage_jeong2020.py \\
      --work-dir /data/tau/iceberg_1/shared/jeong2020_tmp \\
      --output-dir /home/tau/baristim/mne_data/jeong2020_zenodo
"""

import argparse
import logging
import shutil
import subprocess
import zipfile
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

S3_BUCKET = "s3://gigadb-datasets/live/pub/10.5524/100001_101000/100788/RawData"
S3_ENDPOINT = "https://s3.ap-northeast-1.wasabisys.com"
S5CMD = "/home/tau/baristim/s5cmd"

N_SUBJECTS = 25
TARGET_SFREQ = 1000.0

# File naming in S3: session{N}_sub{NN}_{task}_{condition}.{eeg,vhdr,vmrk}
SESSIONS = [1, 2, 3]
TASKS = ["reaching", "multigrasp", "twist"]
CONDITIONS = ["MI", "realMove"]


def download_subject(subject, raw_dir):
    """Download all files for one subject using s5cmd."""
    subj_str = f"sub{subject}"
    subj_dir = raw_dir / subj_str
    subj_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_vhdr = list(subj_dir.glob("*.vhdr"))
    if len(existing_vhdr) >= len(SESSIONS) * len(TASKS) * len(CONDITIONS):
        log.info("  Already downloaded %s (%d .vhdr files)", subj_str, len(existing_vhdr))
        return subj_dir

    log.info("  Downloading %s via s5cmd ...", subj_str)

    # Build list of files to download
    for session in SESSIONS:
        for task in TASKS:
            for condition in CONDITIONS:
                base = f"session{session}_{subj_str}_{task}_{condition}"
                for ext in [".vhdr", ".vmrk", ".eeg"]:
                    s3_path = f"{S3_BUCKET}/{base}{ext}"
                    local_path = subj_dir / f"{base}{ext}"
                    if local_path.exists():
                        continue
                    try:
                        subprocess.run(
                            [
                                S5CMD,
                                "--no-sign-request",
                                "--endpoint-url",
                                S3_ENDPOINT,
                                "cp",
                                s3_path,
                                str(local_path),
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                    except subprocess.CalledProcessError as e:
                        log.warning("    Failed: %s (%s)", base + ext, e.stderr.strip())

    downloaded = list(subj_dir.glob("*.vhdr"))
    total_mb = sum(f.stat().st_size for f in subj_dir.iterdir()) / 1e6
    log.info("  Downloaded %d recordings (%.0f MB)", len(downloaded), total_mb)
    return subj_dir


def download_subject_parallel(subject, raw_dir):
    """Download all files for one subject using s5cmd bulk copy."""
    subj_str = f"sub{subject}"
    subj_dir = raw_dir / subj_str
    subj_dir.mkdir(parents=True, exist_ok=True)

    existing_vhdr = list(subj_dir.glob("*.vhdr"))
    if len(existing_vhdr) >= len(SESSIONS) * len(TASKS) * len(CONDITIONS):
        log.info("  Already downloaded %s (%d .vhdr files)", subj_str, len(existing_vhdr))
        return subj_dir

    # Download all 3 sessions × 3 tasks × 2 conditions × 3 extensions = 54 files
    # Use individual s5cmd calls per file to avoid wildcard matching sub1 vs sub10-19
    log.info("  Downloading %s via s5cmd ...", subj_str)
    for session in SESSIONS:
        for task in TASKS:
            for condition in CONDITIONS:
                base = f"session{session}_{subj_str}_{task}_{condition}"
                for ext in [".vhdr", ".vmrk", ".eeg"]:
                    local_path = subj_dir / f"{base}{ext}"
                    if local_path.exists() and local_path.stat().st_size > 0:
                        continue
                    s3_path = f"{S3_BUCKET}/{base}{ext}"
                    try:
                        subprocess.run(
                            [
                                S5CMD,
                                "--no-sign-request",
                                "--endpoint-url",
                                S3_ENDPOINT,
                                "cp",
                                s3_path,
                                str(local_path),
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                    except subprocess.CalledProcessError as e:
                        log.warning("    Failed: %s (%s)", base + ext, e.stderr.strip())

    downloaded = list(subj_dir.glob("*.vhdr"))
    total_mb = sum(f.stat().st_size for f in subj_dir.iterdir()) / 1e6
    log.info("  Downloaded %d recordings (%.0f MB)", len(downloaded), total_mb)
    return subj_dir


def resample_subject(subj_dir, resampled_dir, subject):
    """Load all channels, resample 2500->1000 Hz, save as BrainVision."""
    import mne

    subj_str = f"sub{subject}"
    subj_out = resampled_dir / subj_str
    subj_out.mkdir(parents=True, exist_ok=True)

    vhdr_files = sorted(subj_dir.glob("*.vhdr"))
    if not vhdr_files:
        log.warning("  No .vhdr files for %s", subj_str)
        return None

    for vhdr in vhdr_files:
        out_path = subj_out / vhdr.name
        if out_path.exists():
            log.info("  Already resampled %s", vhdr.name)
            continue

        log.info("  Resampling %s ...", vhdr.name)
        try:
            raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose=False)

            if raw.info["sfreq"] > TARGET_SFREQ:
                raw.resample(TARGET_SFREQ, verbose=False)

            mne.export.export_raw(str(out_path), raw, overwrite=True, verbose=False)

            log.info(
                "    -> %d ch, %.0f Hz, %.1f s",
                len(raw.ch_names),
                raw.info["sfreq"],
                raw.times[-1],
            )
        except Exception:
            log.error("    FAILED: %s", vhdr.name, exc_info=True)

    return subj_out


def zip_subject(subj_out, zip_dir):
    """Create a ZIP archive for one subject."""
    subj_name = subj_out.name
    zip_path = zip_dir / f"{subj_name}.zip"

    files = sorted(subj_out.iterdir())
    if not files:
        log.warning("  No files to zip for %s", subj_name)
        return None

    log.info("  Zipping %s (%d files) ...", subj_name, len(files))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.name)

    size_mb = zip_path.stat().st_size / 1e6
    log.info("  -> %s (%.1f MB)", zip_path.name, size_mb)
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Repackage Jeong2020 for Zenodo")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/data/tau/iceberg_1/shared/jeong2020_tmp"),
        help="Temporary directory for raw downloads + resampled files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "mne_data" / "jeong2020_zenodo",
        help="Output directory for per-subject ZIPs",
    )
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="*",
        default=None,
        help="Process specific subjects (default: all 1-25)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove raw + resampled files after zipping each subject",
    )
    args = parser.parse_args()

    work_dir = args.work_dir
    output_dir = args.output_dir
    raw_dir = work_dir / "raw"
    resampled_dir = work_dir / "resampled"
    zip_dir = output_dir

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or list(range(1, N_SUBJECTS + 1))

    for subject in subjects:
        subj_str = f"sub{subject}"
        zip_path = zip_dir / f"{subj_str}.zip"

        if zip_path.exists():
            log.info("Skipping %s (ZIP exists: %s)", subj_str, zip_path)
            continue

        log.info("=" * 60)
        log.info("Subject %d / %d (%s)", subject, N_SUBJECTS, subj_str)
        log.info("=" * 60)

        # Download
        subj_raw = download_subject_parallel(subject, raw_dir)

        # Resample
        subj_out = resample_subject(subj_raw, resampled_dir, subject)

        # Zip
        if subj_out and list(subj_out.iterdir()):
            zip_subject(subj_out, zip_dir)

        # Cleanup
        if args.cleanup:
            shutil.rmtree(subj_raw, ignore_errors=True)
            if subj_out:
                shutil.rmtree(subj_out, ignore_errors=True)

    # Summary
    log.info("=" * 60)
    log.info("DONE")
    log.info("=" * 60)
    zips = sorted(zip_dir.glob("sub*.zip"))
    total_mb = sum(z.stat().st_size for z in zips) / 1e6
    log.info("Output: %d ZIPs, %.1f MB total in %s", len(zips), total_mb, zip_dir)
    for z in zips:
        log.info("  %s (%.1f MB)", z.name, z.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
