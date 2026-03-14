#!/usr/bin/env python3
"""Upload Jeong2020 per-subject ZIPs to Zenodo in 5 batched records.

Each record holds 5 subjects (~35-45 GB), under Zenodo's 50 GB limit.

Requires ZENODO_TOKEN environment variable.

Usage:
  export ZENODO_TOKEN=your_token
  python scripts/upload_jeong2020_zenodo.py --part 1       # upload part 1 only
  python scripts/upload_jeong2020_zenodo.py --part 1 2 3   # parts 1-3
  python scripts/upload_jeong2020_zenodo.py                # all 5 parts
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


ZENODO_API = "https://zenodo.org/api"
TOKEN = os.environ.get("ZENODO_TOKEN")
ZIP_DIR = Path.home() / "mne_data" / "jeong2020_zenodo"
RESULTS_FILE = ZIP_DIR / "zenodo_batch_records.json"

PARTS = {
    1: list(range(1, 6)),
    2: list(range(6, 11)),
    3: list(range(11, 16)),
    4: list(range(16, 21)),
    5: list(range(21, 26)),
}

DESCRIPTION = (
    "<p>Repackaged EEG data from Jeong et al. (2020), originally hosted on "
    "GigaDB (DOI: 10.5524/100788). Raw BrainVision recordings resampled from "
    "2500 Hz to 1000 Hz for practical download sizes. All channels preserved "
    "(EEG + EMG + EOG).</p>"
    "<p>25 subjects total (split across 5 Zenodo records), 3 sessions each, "
    "3 tasks (reaching, multigrasp, twist) x 2 conditions (MI, realMove) = "
    "up to 18 recordings per subject per session.</p>"
    "<p>71 channels, 1000 Hz, BrainVision format (.vhdr/.vmrk/.eeg).</p>"
    "<p>Paper: Jeong, J. H. et al. (2020). Multimodal signal dataset for "
    "11 intuitive movement tasks from single upper extremity during multiple "
    "recording sessions. GigaScience, 9(10), giaa098. "
    "DOI: 10.1093/gigascience/giaa098</p>"
)

CREATORS = [
    {"name": "Jeong, Jeong-Hyun"},
    {"name": "Cho, Jeong-Hyun"},
    {"name": "Shim, Kyung-Hwan"},
    {"name": "Kwon, Byoung-Hee"},
    {"name": "Lee, Byeong-Hoo"},
    {"name": "Lee, Do-Yeun"},
    {"name": "Lee, Dae-Hyeok"},
    {"name": "Lee, Seong-Whan"},
]

MAX_RETRIES = 3
BACKOFF_BASE = 10


def _retry_request(method, url, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = method(url, **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"  Retry {attempt}/{MAX_RETRIES} in {wait}s: {e}")
            time.sleep(wait)


def create_deposition(part_num, subjects):
    subj_str = ", ".join(f"sub{s}" for s in subjects)
    metadata = {
        "metadata": {
            "title": (
                f"Jeong2020 — Multimodal EEG for 11 intuitive movement "
                f"tasks (resampled 1000 Hz, part {part_num}/5: "
                f"sub{subjects[0]}-sub{subjects[-1]})"
            ),
            "upload_type": "dataset",
            "description": (
                f"<p><strong>Part {part_num} of 5</strong>: "
                f"{subj_str}.</p>" + DESCRIPTION
            ),
            "creators": CREATORS,
            "license": "cc-by-4.0",
            "related_identifiers": [
                {
                    "identifier": "10.1093/gigascience/giaa098",
                    "relation": "isSupplementTo",
                    "scheme": "doi",
                },
                {
                    "identifier": "10.5524/100788",
                    "relation": "isDerivedFrom",
                    "scheme": "doi",
                },
                {
                    "identifier": "https://github.com/NeuroTechX/moabb",
                    "relation": "isPartOf",
                    "scheme": "url",
                },
            ],
            "keywords": [
                "EEG",
                "motor imagery",
                "BCI",
                "brain-computer interface",
                "upper limb",
                "multimodal",
            ],
            "communities": [{"identifier": "moabb"}],
            "access_right": "open",
            "language": "eng",
        }
    }

    r = _retry_request(
        requests.post,
        f"{ZENODO_API}/deposit/depositions",
        json={},
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
        },
    )
    dep = r.json()
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]

    _retry_request(
        requests.put,
        f"{ZENODO_API}/deposit/depositions/{dep_id}",
        json=metadata,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}",
        },
    )

    print(f"  Created deposition {dep_id}")
    return dep_id, bucket


def upload_file(bucket, zip_path):
    size_gb = zip_path.stat().st_size / (1024**3)
    print(f"  Uploading {zip_path.name} ({size_gb:.1f} GB)...", end=" ", flush=True)
    with open(zip_path, "rb") as fp:
        r = _retry_request(
            requests.put,
            f"{bucket}/{zip_path.name}",
            data=fp,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
    checksum = r.json().get("checksum", "?")
    print(f"OK ({checksum})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part",
        type=int,
        nargs="+",
        default=list(range(1, 6)),
        help="Parts to upload (1-5). Default: all.",
    )
    args = parser.parse_args()

    if not TOKEN:
        print("Error: set ZENODO_TOKEN")
        sys.exit(1)

    for part_num in args.part:
        subjects = PARTS[part_num]
        print(f"\n=== Part {part_num}/5: sub{subjects[0]}-sub{subjects[-1]} ===")

        dep_id, bucket = create_deposition(part_num, subjects)

        for s in subjects:
            zp = ZIP_DIR / f"sub{s}.zip"
            if not zp.exists():
                print(f"  ERROR: {zp} not found, skipping")
                continue
            upload_file(bucket, zp)

        print(f"  Draft: https://zenodo.org/deposit/{dep_id}")

        # Save result
        results = {}
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                results = json.load(f)
        results[str(part_num)] = {"dep_id": dep_id, "subjects": subjects}
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== Done ===")
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        for p in sorted(results, key=int):
            r = results[p]
            print(
                f"  Part {p}: dep {r['dep_id']} (sub{r['subjects'][0]}-sub{r['subjects'][-1]})"
            )


if __name__ == "__main__":
    main()
