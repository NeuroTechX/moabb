#!/usr/bin/env python
"""Upload Nguyen2017 condition ZIPs to Zenodo.

Creates a draft Zenodo deposition with rich metadata, uploads all four
condition ZIPs (Vowels / Short_words / Long_words / Short_Long_words)
+ the top-level README.md, sets metadata, and optionally publishes.

Usage
-----
    # Sandbox dry-run (recommended first step):
    python scripts/upload_nguyen2017_zenodo.py --token YOUR_SANDBOX_TOKEN --sandbox

    # Production draft:
    python scripts/upload_nguyen2017_zenodo.py --token YOUR_ZENODO_TOKEN

    # Publish an existing draft once reviewed:
    python scripts/upload_nguyen2017_zenodo.py --token YOUR_TOKEN --publish DEPOSITION_ID
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests


SANDBOX_URL = "https://sandbox.zenodo.org"
PRODUCTION_URL = "https://zenodo.org"

MAX_RETRIES = 3
BACKOFF_BASE = 5

SRC = Path.home() / "mne_data" / "nguyen2017_zenodo"

# LICENSE DECISION — read this carefully.
#
# The JNE paper (DOI 10.1088/1741-2552/aa8235) declares copyright as
# "© 2017 IOP Publishing Ltd" and is a paid-access article (no CC-BY
# badge). The dataset was separately published by the authors on
# Dropbox as a companion to the paper.
#
# Each condition zip inside the authors' dataset.zip contains a
# Read_me.txt that says:
#     "This data is provided by the Human-Oriented Robotics and
#      Controls (HORC) lab, ASU. [URL]
#      When using this dataset, please cite the following paper:
#      C. Nguyen, G. Karavas, P. Artemiadis, 'Inferring imagined
#      speech using EEG signals: a new approach using Riemannian
#      Manifold features', Journal of Neural Engineering, July 2017"
#
# There is NO explicit Creative Commons license, only a citation
# request. Because of this, we use Zenodo's "other-open" category
# with full provenance in the notes field. The mirror carries the
# authors' Read_me.txt files verbatim so provenance is preserved.
# Takedown on author request is promised in the notes.
LICENSE = "other-open"

METADATA = {
    "metadata": {
        "title": (
            "Nguyen2017 — Imagined speech EEG for Riemannian manifold "
            "classification (re-hosted)"
        ),
        "upload_type": "dataset",
        "description": (
            "<p>Re-hosted EEG data from Nguyen, Karavas & Artemiadis "
            "(2017), originally distributed through a Dropbox archive "
            "(<code>dataset.zip</code>) as a companion to the Journal "
            "of Neural Engineering paper. Repackaged into four "
            "normalized condition ZIPs for programmatic access from "
            "the MOABB benchmarking framework.</p>"
            ""
            "<h3>Paper</h3>"
            "<p>C. H. Nguyen, G. K. Karavas, and P. Artemiadis, "
            "&ldquo;Inferring imagined speech using EEG signals: a new "
            "approach using Riemannian manifold features,&rdquo; "
            "<em>Journal of Neural Engineering</em>, vol. 15, no. 1, "
            "2017.<br>"
            "DOI: <a href='https://doi.org/10.1088/1741-2552/aa8235'>"
            "10.1088/1741-2552/aa8235</a></p>"
            ""
            "<h3>Paradigm</h3>"
            "<p>Imagined speech in four experimental conditions:</p>"
            "<table>"
            "<thead><tr>"
            "<th>Condition</th><th>Classes</th><th>Subjects</th>"
            "</tr></thead>"
            "<tbody>"
            "<tr><td><strong>Vowels</strong></td>"
            "<td>/a/, /i/, /u/ (3 classes)</td><td>8</td></tr>"
            "<tr><td><strong>Short words</strong></td>"
            "<td>out, in, up (3 classes)</td><td>6</td></tr>"
            "<tr><td><strong>Long words</strong></td>"
            "<td>cooperate, independent (2 classes)</td><td>6</td></tr>"
            "<tr><td><strong>Short vs Long</strong></td>"
            "<td>cooperate, in (2 classes)</td><td>6</td></tr>"
            "</tbody></table>"
            "<p>Each trial uses a 5-second imagined-speech window "
            "(the <em>last_beep</em> interval in the authors' "
            "terminology).</p>"
            ""
            "<h3>Participants</h3>"
            "<ul>"
            "<li>Subjects counts vary per condition (6&ndash;8); see "
            "table above and the per-condition README inside each zip. "
            "Some participants contributed to multiple conditions on "
            "different recording days.</li>"
            "<li>Institution: Arizona State University</li>"
            "<li>IRB: ASU Protocols 1309009601, STUDY00001345</li>"
            "</ul>"
            ""
            "<h3>Recording setup</h3>"
            "<table>"
            "<thead><tr><th>Parameter</th><th>Value</th></tr></thead>"
            "<tbody>"
            "<tr><td>Channels</td>"
            "<td>64 or 80 (only first 64 are EEG for 80-ch recordings)</td></tr>"
            "<tr><td>Sampling rate</td><td>256 Hz (downsampled)</td></tr>"
            "<tr><td>EOG channel indices</td>"
            "<td>[0, 9, 32, 63] (0-based)</td></tr>"
            "<tr><td>Preprocessing</td>"
            "<td>EOG artifacts removed by the authors before release</td></tr>"
            "<tr><td>File format</td><td>MATLAB (.mat)</td></tr>"
            "</tbody></table>"
            ""
            "<h3>File structure</h3>"
            "<p>Four condition ZIPs at the top level of this record. "
            "Each contains normalized per-subject .mat files and an "
            "embedded README with the original &rarr; clean filename "
            "mapping:</p>"
            "<pre>"
            "Vowels.zip\n"
            "  README.md\n"
            "  sub-01.mat   ... sub-08.mat\n"
            "Short_words.zip\n"
            "  README.md\n"
            "  sub-01.mat   ... sub-06.mat\n"
            "Long_words.zip\n"
            "  README.md\n"
            "  sub-01.mat   ... sub-06.mat\n"
            "Short_Long_words.zip\n"
            "  README.md\n"
            "  sub-01.mat   ... sub-06.mat\n"
            "README.md           &larr; top-level record readme\n"
            "</pre>"
            "<p>The .mat payloads are bit-identical to the authors' "
            "originals; only filenames inside the zips were normalized "
            "(from e.g. <code>sub_4b_ch80_v_eog_removed_256Hz.mat</code> "
            "to <code>sub-01.mat</code>). See each inner README for "
            "the full filename mapping.</p>"
            ""
            "<h3>Data format</h3>"
            "<p>Each .mat file contains a MATLAB variable "
            "<code>eeg_data_wrt_task_rep_no_eog_256Hz_last_beep</code>, "
            "a <code>(n_classes, n_trials)</code> object array whose "
            "cells are <code>(n_channels, 1280)</code> float matrices "
            "(5 s &times; 256 Hz).</p>"
            ""
            "<h3>Loading with MOABB</h3>"
            "<pre><code>"
            "from moabb.datasets import (\n"
            "    Nguyen2017_V, Nguyen2017_S, Nguyen2017_L, Nguyen2017_SL,\n"
            ")\n"
            "from moabb.paradigms import MotorImagery\n"
            "\n"
            "ds = Nguyen2017_V()\n"
            "paradigm = MotorImagery(\n"
            '    events=["vowel_a", "vowel_i", "vowel_u"],\n'
            "    n_classes=3,\n"
            ")\n"
            "X, y, metadata = paradigm.get_data(dataset=ds, subjects=[1])\n"
            "</code></pre>"
            ""
            "<h3>Re-hosting rationale</h3>"
            "<p>The original Dropbox share relies on an "
            "<code>rlkey</code> token that can rate-limit and has no "
            "persistent DOI. Additionally, the outer <code>dataset.zip"
            "</code> wrapper has a 4 GB prefix that breaks Python's "
            "built-in <code>zipfile</code> module (it requires the "
            "system <code>unzip</code> tool to extract). This Zenodo "
            "mirror provides clean, DOI-addressed URLs for the four "
            "condition zips so MOABB's automated benchmarking pipeline "
            "can fetch the data without a workaround.</p>"
            "<p>The signal data is unchanged &mdash; only the outer "
            "wrapper archive has been removed and filenames inside "
            "each condition zip were normalized.</p>"
            ""
            "<h3>License and attribution</h3>"
            "<p>Each of the authors&rsquo; original "
            "<code>Read_me.txt</code> files (included verbatim in this "
            "mirror) states: <em>&ldquo;This data is provided by the "
            "Human-Oriented Robotics and Controls (HORC) lab, ASU. "
            "When using this dataset, please cite the following paper: "
            "C. Nguyen, G. Karavas, P. Artemiadis, &lsquo;Inferring "
            "imagined speech using EEG signals&hellip;&rsquo;, Journal "
            "of Neural Engineering, July 2017.&rdquo;</em></p>"
            "<p>The authors grant no explicit Creative Commons "
            "license, only a citation request. This mirror is "
            "published under Zenodo&rsquo;s <code>other-open</code> "
            "category on the same implicit terms as the "
            "authors&rsquo; Dropbox distribution: open for research "
            "use with attribution to the paper "
            "(<code>10.1088/1741-2552/aa8235</code>). If you are one "
            "of the dataset authors and wish to clarify the license "
            "or request removal of this mirror, please contact "
            "<a href='mailto:chuong.h.nguyen@asu.edu'>"
            "chuong.h.nguyen@asu.edu</a>, "
            "<a href='mailto:panagiotis.artemiadis@asu.edu'>"
            "panagiotis.artemiadis@asu.edu</a>, or file an issue at "
            "<a href='https://github.com/NeuroTechX/moabb/issues'>"
            "NeuroTechX/moabb</a>.</p>"
        ),
        "creators": [
            {"name": "Nguyen, Chuong H.", "affiliation": "Arizona State University"},
            {"name": "Karavas, George K.", "affiliation": "Arizona State University"},
            {"name": "Artemiadis, Panagiotis", "affiliation": "Arizona State University"},
        ],
        "communities": [{"identifier": "moabb"}],
        "access_right": "open",
        "license": LICENSE,
        "keywords": [
            "EEG",
            "BCI",
            "imagined speech",
            "Riemannian manifold",
            "brain-computer interface",
            "vowels",
            "words",
            "MOABB",
        ],
        "related_identifiers": [
            {
                "identifier": "10.1088/1741-2552/aa8235",
                "relation": "isSupplementTo",
                "scheme": "doi",
            },
            {
                "identifier": (
                    "https://www.dropbox.com/scl/fi/20j120qae7c2rlmr5lfwr/"
                    "dataset.zip?rlkey=0xjdairhprrakmw27d2fnesj7&dl=1"
                ),
                "relation": "isDerivedFrom",
                "scheme": "url",
            },
            {
                "identifier": "https://github.com/NeuroTechX/moabb",
                "relation": "isPartOf",
                "scheme": "url",
            },
        ],
        "language": "eng",
        "notes": (
            "Re-hosted copy of the imagined-speech EEG dataset "
            "distributed on Dropbox by the HORC lab at Arizona State "
            "University as a companion to DOI 10.1088/1741-2552/aa8235 "
            "(Nguyen, Karavas, and Artemiadis, Journal of Neural "
            "Engineering, 2017). Signal data is bit-identical to the "
            "authors' originals. Changes: (1) the outer dataset.zip "
            "wrapper was removed because it has a 4 GB prefix that "
            "breaks Python's built-in zipfile module; the four "
            "condition zips (Vowels, Short_words, Long_words, "
            "Short_Long_words) are now top-level files on this record; "
            "(2) files inside each condition zip were renamed from "
            "e.g. sub_4b_ch80_v_eog_removed_256Hz.mat to sub-01.mat "
            "matching the MOABB adapter's integer subject IDs; (3) "
            "the authors' original Read_me.txt is kept verbatim inside "
            "each condition zip, and a supplementary README.md was "
            "added mapping clean to original filenames. Extra "
            "analysis files (sub_8_..._time_correlation_effect.mat "
            "in Vowels.zip, sub_14_..._bw20_8s.mat in "
            "Short_Long_words.zip) are preserved under each zip's "
            "supplementary/ directory, unchanged. Authors' license: "
            "implicit (citation request only, no explicit CC grant). "
            "This mirror is provided under Zenodo's 'other-open' "
            "category on the same terms. Takedown on author request. "
            "Contact: chuong.h.nguyen@asu.edu, "
            "panagiotis.artemiadis@asu.edu, or "
            "https://github.com/NeuroTechX/moabb."
        ),
    }
}


def _request_with_retry(method, url, **kwargs):
    """Make an HTTP request with exponential backoff retry."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = method(url, **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"  Attempt {attempt} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)


def create_deposition(base_url, token):
    r = _request_with_retry(
        requests.post,
        f"{base_url}/api/deposit/depositions",
        json={},
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    data = r.json()
    print(f"Created deposition: {data['id']}")
    print(f"  Draft URL: {data['links']['html']}")
    return data


def upload_file(bucket_url, file_path, token):
    size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"  Uploading {file_path.name} ({size_mb:.1f} MB)...", end=" ", flush=True)
    with open(file_path, "rb") as fp:
        r = _request_with_retry(
            requests.put,
            f"{bucket_url}/{file_path.name}",
            data=fp,
            headers={"Authorization": f"Bearer {token}"},
        )
    print(f"OK ({r.json()['checksum']})")


def set_metadata(base_url, deposition_id, metadata, token):
    r = _request_with_retry(
        requests.put,
        f"{base_url}/api/deposit/depositions/{deposition_id}",
        json=metadata,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    print(f"Metadata set: {r.json()['title']}")


def publish(base_url, deposition_id, token):
    r = _request_with_retry(
        requests.post,
        f"{base_url}/api/deposit/depositions/{deposition_id}/actions/publish",
        headers={"Authorization": f"Bearer {token}"},
    )
    data = r.json()
    print("Published!")
    print(f"  DOI: {data['doi']}")
    print(f"  URL: {data['links']['html']}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Upload Nguyen2017 dataset to Zenodo")
    parser.add_argument("--token", required=True, help="Zenodo API token")
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org")
    parser.add_argument(
        "--publish", metavar="ID", help="Publish existing deposition by ID"
    )
    args = parser.parse_args()

    base_url = SANDBOX_URL if args.sandbox else PRODUCTION_URL

    if args.publish:
        data = publish(base_url, args.publish, args.token)
        record_id = data["id"]
        print(f"\nZenodo record: {base_url}/records/{record_id}")
        return

    expected = ["Vowels.zip", "Short_words.zip", "Long_words.zip", "Short_Long_words.zip"]
    condition_files = [SRC / name for name in expected]
    missing = [f for f in condition_files if not f.exists()]
    if missing:
        print(f"Error: missing condition zips in {SRC}:")
        for f in missing:
            print(f"  - {f.name}")
        print("Run: python scripts/repackage_nguyen2017.py")
        sys.exit(1)

    readme_file = SRC / "README.md"
    upload_files = list(condition_files)
    if readme_file.exists():
        upload_files.append(readme_file)

    print(
        f"Found {len(condition_files)} condition ZIPs"
        + (" + README.md" if readme_file.exists() else "")
    )

    deposition = create_deposition(base_url, args.token)
    deposition_id = deposition["id"]
    bucket_url = deposition["links"]["bucket"]

    for f in upload_files:
        upload_file(bucket_url, f, args.token)

    set_metadata(base_url, deposition_id, METADATA, args.token)

    print(f"\nDraft ready for review: {deposition['links']['html']}")
    print(f"Deposition ID: {deposition_id}")
    print("\nTo publish:")
    print(
        f"  python scripts/upload_nguyen2017_zenodo.py "
        f"--token <TOKEN> "
        f"{'--sandbox ' if args.sandbox else ''}"
        f"--publish {deposition_id}"
    )


if __name__ == "__main__":
    main()
