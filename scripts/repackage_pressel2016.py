#!/usr/bin/env python
"""Repackage Pressel2016 Google Drive data into per-subject ZIPs for Zenodo.

Input
-----
The original ``imagined_speech.zip`` as downloaded by the current
``moabb.datasets.Pressel2016`` adapter via ``gdown`` (Google Drive file
ID ``0By7apHbIp8ENZVBLRFVlSFhzbHc``). The archive lives at either:

    ~/mne_data/MNE-pressel2016-data/imagined_speech.zip
    ~/mne_data/pressel2016/imagined_speech.zip

and unpacks to::

    Base de Datos Habla Imaginada/
        S01/S01_EEG.mat
        S02/S02_EEG.mat
        ...
        S15/S15_EEG.mat

Output
------
``~/mne_data/pressel2016_zenodo/``:

    S01.zip  -> sub-01_eeg.mat
    S02.zip  -> sub-02_eeg.mat
    ...
    S15.zip  -> sub-15_eeg.mat
    README.md  (uploaded to the Zenodo record, NOT inside the per-subject zips)

Notes
-----
- ``.mat`` files are already binary-packed so we use ``ZIP_STORED`` (no
  deflate) to keep repackaging fast and avoid wasting CPU.
- The inner directory structure is flattened: each per-subject ZIP holds
  a single file at the top level named ``sub-{NN:02d}_eeg.mat``.
- The original filename mapping is recorded in the top-level README so
  the repackaging remains fully auditable.

Usage
-----
    # If the original zip is already downloaded somewhere:
    python scripts/repackage_pressel2016.py

    # Or point at an explicit source zip:
    python scripts/repackage_pressel2016.py --src /path/to/imagined_speech.zip

    # Force re-creation of output ZIPs even if they already exist:
    python scripts/repackage_pressel2016.py --force
"""

from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

N_SUBJECTS = 15
DEFAULT_SRC_CANDIDATES = [
    Path.home() / "mne_data" / "MNE-pressel2016-data" / "imagined_speech.zip",
    Path.home() / "mne_data" / "pressel2016" / "imagined_speech.zip",
    Path.home() / "mne_data" / "MNE-pressel2016-data" / "imagined_speech",
    Path.home() / "mne_data" / "pressel2016",
]
OUT_DIR = Path.home() / "mne_data" / "pressel2016_zenodo"

README_BODY = """\
# Pressel2016 — Imagined Speech EEG (re-hosted)

Spanish vowels and directional commands. Re-hosted copy of the data
originally distributed by Pressel Coretto, Gareis & Rufiner through
Google Drive (file id `0By7apHbIp8ENZVBLRFVlSFhzbHc`).

**Paper:** G. A. Pressel Coretto, I. E. Gareis, H. L. Rufiner, "Open
access database of EEG signals recorded during imagined speech",
*SIPAIM/SPIE Proceedings* (2017). DOI: `10.1117/12.2255697`.

## Contents

15 subject archives, each containing a single MATLAB file:

| File       | Inner file             | Subject |
|------------|------------------------|---------|
| S01.zip    | sub-01_eeg.mat         | S01     |
| S02.zip    | sub-02_eeg.mat         | S02     |
| ...        | ...                    | ...     |
| S15.zip    | sub-15_eeg.mat         | S15     |

Each `sub-{NN}_eeg.mat` is **bit-identical** to the original
`Base de Datos Habla Imaginada/S{NN}/S{NN}_EEG.mat`. Only the filename
was changed to adopt a BIDS-like `sub-{NN}` convention that lines up
with the integer subject IDs used by MOABB.

## Format

Each `.mat` file contains an `EEG` variable of shape
`(n_trials, 6*4096 + 3)`. The last three columns are labels:

| Column | Meaning                                  |
|--------|------------------------------------------|
| -3     | Modality: `1` = imagined, `2` = pronounced |
| -2     | Stimulus code (1-11, see table below)    |
| -1     | Artifact flag: `1` = clean, `2` = artifact |

The first `6 * 4096 = 24576` columns are the EEG samples, flattened
channel-first across 6 channels (F3, F4, C3, C4, P3, P4) at 1024 Hz,
4 seconds per trial.

### Stimulus codes

| Code | Label     | Translation |
|------|-----------|-------------|
| 1    | vowel_a   | /a/         |
| 2    | vowel_e   | /e/         |
| 3    | vowel_i   | /i/         |
| 4    | vowel_o   | /o/         |
| 5    | vowel_u   | /u/         |
| 6    | arriba    | up          |
| 7    | abajo     | down        |
| 8    | adelante  | forward     |
| 9    | atras     | back        |
| 10   | derecha   | right       |
| 11   | izquierda | left        |

## Loading with MOABB

```python
from moabb.datasets import Pressel2016
from moabb.paradigms import MotorImagery

dataset = Pressel2016()
paradigm = MotorImagery(events=["vowel_a", "vowel_e", "vowel_i"])
X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1])
```

## Re-hosting rationale

The original Google Drive share relies on `gdown` and a legacy
`resourcekey`; both are fragile (quota blocking, changing confirm
token protocol). This Zenodo mirror provides direct DOI-addressed URLs
so the data can be fetched from any CI environment without credentials.

The data is unchanged — only filename conventions and archive layout
have been normalized.
"""


def _find_source(cli_src: str | None) -> Path:
    """Locate either the original imagined_speech.zip or an extracted tree."""
    if cli_src:
        p = Path(cli_src).expanduser()
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    for cand in DEFAULT_SRC_CANDIDATES:
        if cand.exists():
            log.info("Using source: %s", cand)
            return cand
    raise FileNotFoundError(
        "Could not locate Pressel2016 source data. Either:\n"
        "  1. Run `python -c 'from moabb.datasets import Pressel2016; "
        "Pressel2016().data_path(1)'` to fetch from Google Drive, or\n"
        "  2. Pass --src <path-to-imagined_speech.zip-or-dir>.\n"
        f"Looked in: {', '.join(str(p) for p in DEFAULT_SRC_CANDIDATES)}"
    )


def _open_subject_reader(src: Path, subject: int):
    """Return (reader, cleanup) that yields bytes for one subject's .mat.

    Supports both a source zip and an already-extracted directory tree.
    """
    target_inside_zip = (
        f"Base de Datos Habla Imaginada/S{subject:02d}/S{subject:02d}_EEG.mat"
    )

    if src.is_file() and src.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(str(src))
        try:
            info = zf.getinfo(target_inside_zip)
        except KeyError:
            # Fall back: scan for any member ending with the subject filename.
            candidates = [
                n for n in zf.namelist() if n.endswith(f"S{subject:02d}_EEG.mat")
            ]
            if not candidates:
                zf.close()
                raise FileNotFoundError(
                    f"S{subject:02d}_EEG.mat not found in {src}"
                ) from None
            info = zf.getinfo(candidates[0])
        return zf.open(info), zf.close

    # Extracted directory tree.
    if src.is_dir():
        # Try common layouts.
        for candidate in (
            src
            / "Base de Datos Habla Imaginada"
            / f"S{subject:02d}"
            / f"S{subject:02d}_EEG.mat",
            src / f"S{subject:02d}" / f"S{subject:02d}_EEG.mat",
            src / f"S{subject:02d}_EEG.mat",
        ):
            if candidate.exists():
                return candidate.open("rb"), lambda: None
        raise FileNotFoundError(f"S{subject:02d}_EEG.mat not found under {src}")

    raise ValueError(f"Unsupported source path: {src}")


def repackage(src: Path, force: bool) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = 0

    for subject in range(1, N_SUBJECTS + 1):
        out_zip = OUT_DIR / f"S{subject:02d}.zip"
        if out_zip.exists() and not force:
            log.info("S%02d.zip already exists (skip; use --force to rebuild)", subject)
            continue

        reader, cleanup = _open_subject_reader(src, subject)
        try:
            payload = reader.read()
        finally:
            reader.close()
            cleanup()

        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr(f"sub-{subject:02d}_eeg.mat", payload)

        size_mb = out_zip.stat().st_size / (1024 * 1024)
        log.info("wrote S%02d.zip (%.1f MB)", subject, size_mb)
        created += 1

    readme_path = OUT_DIR / "README.md"
    if force or not readme_path.exists():
        readme_path.write_text(README_BODY, encoding="utf-8")
        log.info("wrote README.md")

    log.info("Created %d subject ZIPs in %s", created, OUT_DIR)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        help=(
            "Source path: either an imagined_speech.zip file or the "
            "extracted 'Base de Datos Habla Imaginada' directory."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild output ZIPs even if they already exist.",
    )
    args = parser.parse_args()

    src = _find_source(args.src)
    repackage(src, args.force)


if __name__ == "__main__":
    main()
