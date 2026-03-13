#!/usr/bin/env python3
"""Upload Simoes2020 (BCIAUT-P300) per-subject ZIPs to Zenodo.

Requires ZENODO_TOKEN environment variable.

Usage:
  export ZENODO_TOKEN=your_token
  python scripts/upload_simoes2020_zenodo.py
"""

import os
import sys
from pathlib import Path

import requests


ZENODO_API = "https://zenodo.org/api"
TOKEN = os.environ.get("ZENODO_TOKEN")
ZIP_DIR = Path.home() / "mne_data" / "simoes2020_zenodo"

METADATA = {
    "title": (
        "BCIAUT-P300: A Multi-Session and Multi-Subject Benchmark Dataset "
        "on Autism for P300-Based Brain-Computer-Interfaces (per-subject ZIPs)"
    ),
    "upload_type": "dataset",
    "description": (
        "<p>Re-hosted per-subject ZIP archives of the BCIAUT-P300 dataset "
        "from Simoes et al. (2020) for programmatic access in MOABB.</p>"
        "<p>Original data: "
        '<a href="https://www.kaggle.com/datasets/disbeat/bciaut-p300">'
        "Kaggle</a></p>"
        "<p>15 ASD subjects, 7 sessions each, 8 EEG channels at 250 Hz. "
        "Pre-epoched MATLAB files with Target/NonTarget labels.</p>"
        "<p>Paper: Simoes et al. (2020). BCIAUT-P300. "
        "Frontiers in Neuroscience, 14, 568104. "
        "DOI: 10.3389/fnins.2020.568104</p>"
    ),
    "creators": [
        {"name": "Simoes, Marco"},
        {"name": "Borra, Davide"},
        {"name": "Santamaria-Vazquez, Eduardo"},
        {"name": "Castelo-Branco, Miguel"},
    ],
    "license": "cc-by-4.0",
    "related_identifiers": [
        {
            "identifier": "10.3389/fnins.2020.568104",
            "relation": "isSupplementTo",
            "scheme": "doi",
        },
    ],
    "keywords": ["EEG", "P300", "BCI", "autism", "ASD", "MOABB"],
}


def main():
    if not TOKEN:
        print("Set ZENODO_TOKEN environment variable")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {TOKEN}"}

    # Create deposition
    print("Creating Zenodo deposition...")
    r = requests.post(
        f"{ZENODO_API}/deposit/depositions",
        json={"metadata": METADATA},
        headers=headers,
    )
    r.raise_for_status()
    dep = r.json()
    dep_id = dep["id"]
    bucket_url = dep["links"]["bucket"]
    print(f"  Deposition ID: {dep_id}")
    print(f"  Bucket URL: {bucket_url}")

    # Upload files
    zips = sorted(ZIP_DIR.glob("SBJ*.zip"))
    print(f"\nUploading {len(zips)} files...")
    for zp in zips:
        print(f"  Uploading {zp.name} ({zp.stat().st_size / 1e6:.0f} MB)...")
        with open(zp, "rb") as f:
            r = requests.put(
                f"{bucket_url}/{zp.name}",
                data=f,
                headers=headers,
            )
            r.raise_for_status()
        print(f"    Done: {r.json().get('checksum', 'ok')}")

    print(f"\nAll files uploaded to deposition {dep_id}")
    print(f"Review and publish at: https://zenodo.org/deposit/{dep_id}")
    print("\nAfter publishing, set in simoes2020.py:")
    print(f"  _ZENODO_RECORD = {dep_id}")
    print(f'  _ZENODO_BASE = "https://zenodo.org/records/{dep_id}/files"')


if __name__ == "__main__":
    main()
