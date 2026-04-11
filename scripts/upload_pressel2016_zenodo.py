#!/usr/bin/env python
"""Upload Pressel2016 per-subject ZIPs to Zenodo.

Creates a draft Zenodo deposition with rich metadata, uploads all 15
subject ZIPs + the top-level README.md, sets metadata, and optionally
publishes.

Usage
-----
    # Sandbox dry-run (recommended first step):
    python scripts/upload_pressel2016_zenodo.py --token YOUR_SANDBOX_TOKEN --sandbox

    # Production draft:
    python scripts/upload_pressel2016_zenodo.py --token YOUR_ZENODO_TOKEN

    # Publish an existing draft once reviewed:
    python scripts/upload_pressel2016_zenodo.py --token YOUR_TOKEN --publish DEPOSITION_ID
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests


SANDBOX_URL = "https://sandbox.zenodo.org"
PRODUCTION_URL = "https://zenodo.org"

MAX_RETRIES = 3
BACKOFF_BASE = 5

# TODO: Zenodo is migrating to InvenioRDM. The legacy /api/deposit/depositions
# endpoint used below is on borrowed time. When it sunsets, switch to
# /api/records + /api/records/{id}/draft. The version header lets Zenodo pin
# us to the legacy contract in the interim.
_ZENODO_HEADERS = {"Accept": "application/vnd.zenodo.v1+json"}

SRC = Path.home() / "mne_data" / "pressel2016_zenodo"

# LICENSE DECISION — read this carefully.
#
# The SPIE paper (DOI 10.1117/12.2255697) declares copyright as
# "© 2017 Society of Photo-Optical Instrumentation Engineers (SPIE)"
# and describes the dataset as "open access" and "publicly available",
# but grants NO explicit Creative Commons license.
#
# The authoritative hosting URL at the authors' institution
# (fich.unl.edu.ar/sinc/downloads/imagined_speech/) just redirects to
# the same Google Drive file we are mirroring.
#
# The Protocolo.txt embedded in imagined_speech.zip describes the
# protocol and recording setup but contains NO license or
# redistribution terms. The only author-provided contact is:
#     germanpressel@gmail.com
#
# Because there is no explicit CC license, we use Zenodo's
# "other-open" category with full provenance in the notes field.
# This accurately represents what we know: the data is author-published
# for research use, attribution is expected, and the mirror exists to
# provide a reliable DOI-addressed source for automated benchmarking.
# Takedown on author request is promised in the notes.
LICENSE = "other-open"

METADATA = {
    "metadata": {
        "title": (
            "Pressel2016 — Imagined speech EEG (Spanish vowels and "
            "directional commands, re-hosted)"
        ),
        "upload_type": "dataset",
        "description": (
            "<p>Re-hosted EEG data from Pressel Coretto, Gareis, and "
            "Rufiner (2017), originally distributed through Google "
            "Drive (file id "
            "<code>0By7apHbIp8ENZVBLRFVlSFhzbHc</code>) as an open-access "
            "companion to the SIPAIM/SPIE proceedings paper. Repackaged "
            "into per-subject ZIP archives for easier programmatic "
            "access from the MOABB benchmarking framework.</p>"
            ""
            "<h3>Paper</h3>"
            "<p>G. A. Pressel Coretto, I. E. Gareis, and H. L. Rufiner, "
            "&ldquo;Open access database of EEG signals recorded during "
            "imagined speech,&rdquo; in <em>Proc. SPIE 10160, 12th "
            "International Symposium on Medical Information Processing "
            "and Analysis</em>, 2017.<br>"
            "DOI: <a href='https://doi.org/10.1117/12.2255697'>"
            "10.1117/12.2255697</a></p>"
            ""
            "<h3>Paradigm</h3>"
            "<p>11-class imagined speech of Spanish phonemes and "
            "directional commands:</p>"
            "<ul>"
            "<li><strong>Vowels</strong>: /a/, /e/, /i/, /o/, /u/ "
            "(stim codes 1-5)</li>"
            "<li><strong>Directional commands</strong>: arriba (up), "
            "abajo (down), adelante (forward), atr&aacute;s (back), "
            "derecha (right), izquierda (left) (stim codes 6-11)</li>"
            "</ul>"
            "<p>Two modalities were recorded per stimulus: imagined "
            "speech (modality code 1) and pronounced speech (modality "
            "code 2). The MOABB adapter loads the imagined condition by "
            "default.</p>"
            "<p>Each trial is <strong>4 seconds</strong> at "
            "<strong>1024 Hz</strong> (4096 samples).</p>"
            ""
            "<h3>Participants</h3>"
            "<ul>"
            "<li>15 Argentinian volunteers (7 female, 8 male, ages "
            "24-28)</li>"
            "<li>Institution: Universidad Nacional de Entre R&iacute;os "
            "(Argentina)</li>"
            "</ul>"
            ""
            "<h3>Recording setup</h3>"
            "<table>"
            "<thead><tr><th>Parameter</th><th>Value</th></tr></thead>"
            "<tbody>"
            "<tr><td>Amplifier</td>"
            "<td>Grass 8-18-36</td></tr>"
            "<tr><td>ADC</td>"
            "<td>DataTranslation DT9816</td></tr>"
            "<tr><td>Channels</td>"
            "<td>6 (F3, F4, C3, C4, P3, P4)</td></tr>"
            "<tr><td>Sampling rate</td><td>1024 Hz</td></tr>"
            "<tr><td>Bandpass</td><td>2&ndash;45 Hz</td></tr>"
            "<tr><td>File format</td><td>MATLAB (.mat)</td></tr>"
            "</tbody></table>"
            ""
            "<h3>File structure</h3>"
            "<p>15 per-subject ZIP files "
            "(<code>S01.zip</code>&ndash;<code>S15.zip</code>). Each "
            "contains a single MATLAB file renamed from the original "
            "<code>S{NN}_EEG.mat</code> to "
            "<code>sub-{NN}_eeg.mat</code>:</p>"
            "<pre>"
            "S01.zip\n"
            "  sub-01_eeg.mat\n"
            "S02.zip\n"
            "  sub-02_eeg.mat\n"
            "...\n"
            "S15.zip\n"
            "  sub-15_eeg.mat\n"
            "README.md   &larr; top-level, stimulus codes and channels\n"
            "</pre>"
            "<p>The <code>.mat</code> files are bit-identical to the "
            "originals; only the filename was changed to adopt the "
            "MOABB subject-ID convention.</p>"
            ""
            "<h3>Data format</h3>"
            "<p>Each <code>.mat</code> file contains an "
            "<code>EEG</code> variable of shape "
            "<code>(n_trials, 6 * 4096 + 3)</code>. The first 24576 "
            "columns are flattened channel-first EEG samples; the last "
            "three columns are labels: "
            "<code>[modality, stimulus_code, artifact_flag]</code>.</p>"
            ""
            "<h3>Loading with MOABB</h3>"
            "<pre><code>"
            "from moabb.datasets import Pressel2016\n"
            "from moabb.paradigms import MotorImagery\n"
            "\n"
            "dataset = Pressel2016()\n"
            "paradigm = MotorImagery(\n"
            '    events=["vowel_a", "vowel_e", "vowel_i"],\n'
            ")\n"
            "X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])\n"
            "</code></pre>"
            ""
            "<h3>Re-hosting rationale</h3>"
            "<p>The original Google Drive share relies on the "
            "<code>gdown</code> library and a legacy "
            "<code>resourcekey</code> token. Both are fragile: Google "
            "Drive can quota-block downloads, change its confirm-token "
            "protocol, or revoke resource keys (we observed all three "
            "modes of failure). This Zenodo mirror provides direct, "
            "DOI-addressed URLs so the data can be fetched from any "
            "CI environment without credentials and without depending "
            "on consumer cloud storage.</p>"
            "<p>The signal data is unchanged &mdash; only filename "
            "conventions and archive layout have been normalized.</p>"
            ""
            "<h3>License and attribution</h3>"
            "<p>The original SPIE publication describes this dataset "
            "as &ldquo;open access&rdquo; and &ldquo;publicly "
            "available&rdquo; but does not declare an explicit "
            "Creative Commons license. This mirror is published under "
            "Zenodo&rsquo;s <code>other-open</code> category on the "
            "same terms as the authors&rsquo; original distribution: "
            "open for research use with attribution to the paper "
            "(<code>10.1117/12.2255697</code>). If you are one of the "
            "dataset authors and wish to clarify the license or "
            "request removal of this mirror, please contact "
            "<a href='mailto:germanpressel@gmail.com'>"
            "germanpressel@gmail.com</a> (the corresponding author "
            "listed in the paper) or file an issue at "
            "<a href='https://github.com/NeuroTechX/moabb/issues'>"
            "NeuroTechX/moabb</a>.</p>"
        ),
        "creators": [
            {
                "name": "Pressel Coretto, Germ\u00e1n A.",
                "affiliation": "Universidad Nacional de Entre R\u00edos",
            },
            {
                "name": "Gareis, Iv\u00e1n E.",
                "affiliation": "Universidad Nacional de Entre R\u00edos",
            },
            {
                "name": "Rufiner, Hugo Leonardo",
                "affiliation": "Universidad Nacional de Entre R\u00edos",
            },
        ],
        "communities": [{"identifier": "moabb"}],
        "access_right": "open",
        "license": LICENSE,
        "keywords": [
            "EEG",
            "BCI",
            "imagined speech",
            "Spanish",
            "vowels",
            "directional commands",
            "brain-computer interface",
            "MOABB",
        ],
        "related_identifiers": [
            {
                "identifier": "10.1117/12.2255697",
                "relation": "isSupplementTo",
                "scheme": "doi",
            },
            {
                "identifier": (
                    "https://drive.google.com/uc?export=download"
                    "&id=0By7apHbIp8ENZVBLRFVlSFhzbHc"
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
        "language": "spa",
        "notes": (
            "Re-hosted copy of the imagined-speech EEG database "
            "distributed by Pressel Coretto, Gareis, and Rufiner as a "
            "companion to DOI 10.1117/12.2255697 (SPIE / SIPAIM 2016). "
            "The authoritative upstream source is the authors' Google "
            "Drive file 0By7apHbIp8ENZVBLRFVlSFhzbHc, which is also "
            "what their institutional page at "
            "fich.unl.edu.ar/sinc/downloads/imagined_speech/ redirects "
            "to. The EEG signal data is bit-identical to that source; "
            "only filenames were normalized to a sub-NN convention "
            "(sub-01_eeg.mat was S01_EEG.mat) and the Base de Datos "
            "Habla Imaginada/ prefix was dropped for cleaner extraction. "
            "The authors' original Protocolo.txt (Spanish) and "
            "Registros_sujetos.xlsx are included unchanged. The "
            "authors' paper calls the dataset 'open access' but does "
            "not grant an explicit Creative Commons license. This "
            "mirror is provided under Zenodo's 'other-open' category "
            "on the same implicit terms: open for research with "
            "attribution to the paper. Takedown on author request. "
            "Authors of this dataset may contact germanpressel@gmail.com "
            "(corresponding author) or file an issue at "
            "https://github.com/NeuroTechX/moabb to clarify the "
            "license or request removal."
        ),
    }
}


def _request_with_retry(method, url, **kwargs):
    """Make an HTTP request with exponential backoff retry.

    Retries transient failures (connection drops, timeouts, 5xx) but
    NOT 4xx — a 401/403/404 will never become a 200 on its own, so
    retrying just delays the failure and burns rate-limit budget.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = method(url, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"  Attempt {attempt} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
            continue
        if r.status_code < 500:
            r.raise_for_status()  # 4xx raises immediately, 2xx/3xx returns
            return r
        if attempt == MAX_RETRIES:
            r.raise_for_status()
        wait = BACKOFF_BASE * (2 ** (attempt - 1))
        print(f"  Attempt {attempt} failed: HTTP {r.status_code}. Retrying in {wait}s...")
        time.sleep(wait)
    raise RuntimeError("unreachable: retry loop exited without returning")


def _auth(token, *, content_type=None):
    headers = {**_ZENODO_HEADERS, "Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def create_deposition(base_url, token):
    r = _request_with_retry(
        requests.post,
        f"{base_url}/api/deposit/depositions",
        json={},
        headers=_auth(token, content_type="application/json"),
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
            requests.put, f"{bucket_url}/{file_path.name}", data=fp, headers=_auth(token)
        )
    print(f"OK ({r.json()['checksum']})")


def set_metadata(base_url, deposition_id, metadata, token):
    r = _request_with_retry(
        requests.put,
        f"{base_url}/api/deposit/depositions/{deposition_id}",
        json=metadata,
        headers=_auth(token, content_type="application/json"),
    )
    print(f"Metadata set: {r.json()['title']}")


def publish(base_url, deposition_id, token):
    r = _request_with_retry(
        requests.post,
        f"{base_url}/api/deposit/depositions/{deposition_id}/actions/publish",
        headers=_auth(token),
    )
    data = r.json()
    print("Published!")
    print(f"  DOI: {data['doi']}")
    print(f"  URL: {data['links']['html']}")
    return data


def _resolve_token(cli_token):
    token = cli_token or os.environ.get("ZENODO_TOKEN")
    if not token:
        print(
            "Error: no Zenodo API token. Pass --token <value> or set "
            "ZENODO_TOKEN in the environment."
        )
        sys.exit(2)
    return token


def main():
    parser = argparse.ArgumentParser(description="Upload Pressel2016 dataset to Zenodo")
    parser.add_argument(
        "--token", help="Zenodo API token (falls back to ZENODO_TOKEN env var)"
    )
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org")
    parser.add_argument(
        "--publish", metavar="ID", help="Publish existing deposition by ID"
    )
    args = parser.parse_args()

    token = _resolve_token(args.token)
    base_url = SANDBOX_URL if args.sandbox else PRODUCTION_URL

    if args.publish:
        data = publish(base_url, args.publish, token)
        record_id = data["id"]
        print(f"\nZenodo record: {base_url}/records/{record_id}")
        return

    zip_files = sorted(SRC.glob("S*.zip"))
    if not zip_files:
        print(f"Error: No S*.zip files in {SRC}")
        print(
            "Stage the per-subject ZIPs in this directory before running "
            "the upload. This script was used once to produce Zenodo "
            "record 19502780; see that record for the canonical layout."
        )
        sys.exit(1)

    readme_file = SRC / "README.md"
    upload_files = list(zip_files)
    if readme_file.exists():
        upload_files.append(readme_file)

    print(
        f"Found {len(zip_files)} subject ZIPs"
        + (" + README.md" if readme_file.exists() else "")
    )

    deposition = create_deposition(base_url, token)
    deposition_id = deposition["id"]
    bucket_url = deposition["links"]["bucket"]

    for f in upload_files:
        upload_file(bucket_url, f, token)

    set_metadata(base_url, deposition_id, METADATA, token)

    print(f"\nDraft ready for review: {deposition['links']['html']}")
    print(f"Deposition ID: {deposition_id}")
    print("\nTo publish:")
    print(
        f"  python scripts/upload_pressel2016_zenodo.py "
        f"--token <TOKEN> "
        f"{'--sandbox ' if args.sandbox else ''}"
        f"--publish {deposition_id}"
    )


if __name__ == "__main__":
    main()
