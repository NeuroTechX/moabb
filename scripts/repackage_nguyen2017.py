#!/usr/bin/env python
"""Repackage Nguyen2017 Dropbox data into cleaned per-condition ZIPs for Zenodo.

Input
-----
The outer ``dataset.zip`` as downloaded by the current
``moabb.datasets._Nguyen2017Base`` adapter from Dropbox. It contains
four inner condition ZIPs:

    Vowels.zip            (8 subjects)
    Short_words.zip       (6 subjects)
    Long_words.zip        (6 subjects)
    Short_Long_words.zip  (6 subjects)

Each inner zip holds per-subject ``.mat`` files with opaque names like
``sub_4b_ch80_v_eog_removed_256Hz.mat``.

Output
------
``~/mne_data/nguyen2017_zenodo/``:

    Vowels.zip             (10 entries: 8 sub-NN.mat + Read_me.txt
                            + README.md + 1 supplementary file)
    Short_words.zip        (8 entries:  6 sub-NN.mat + Read_me.txt + README.md)
    Long_words.zip         (8 entries:  6 sub-NN.mat + Read_me.txt + README.md)
    Short_Long_words.zip   (9 entries:  6 sub-NN.mat + Read_me.txt + README.md
                            + 1 supplementary file)
    README.md              (top-level record readme)

The 4 outer zip filenames match the existing ``_CONDITIONS[...]['zip_name']``
mapping so the MOABB loader only needs to swap the base URL. The .mat
payloads themselves are **bit-identical** to the originals; only the
filenames inside each condition zip are normalized to ``sub-{NN}.mat``.

Each condition zip includes:
- The authors' original ``Read_me.txt`` verbatim (for provenance and
  because it carries the only licensing/attribution statement the
  authors provided).
- A new ``README.md`` documenting the clean → original filename
  mapping, the 80-ch vs 64-ch channel count split, class labels,
  sampling rate, and EOG indices.
- Any supplementary analysis files (``sub_8_..._time_correlation_effect.mat``
  for Vowels, ``sub_14_..._bw20_8s.mat`` for Short_Long_words) preserved
  under a ``supplementary/`` directory with their original filenames.

Usage
-----
    python scripts/repackage_nguyen2017.py
    python scripts/repackage_nguyen2017.py --src /path/to/dataset.zip
    python scripts/repackage_nguyen2017.py --force
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

# Condition -> (inner zip filename, ordered original basenames,
#               inner folder name, supplementary file list).
#
# Order matches moabb/datasets/nguyen2017.py:50-115 so MOABB integer
# subject IDs line up one-to-one with the new sub-NN.mat filenames.
#
# The "folder" is the top-level directory inside each inner zip
# (e.g. Vowels.zip contains Vowels/*.mat).
#
# supplementary = extra analysis files not in the subject list but
# referenced in the paper (Vowels time_correlation_effect file for
# Section VI.A, ShortLongWords 8s file for Section V.A). These are
# carried forward under supplementary/ with their original names.
CONDITIONS: dict[str, dict] = {
    "Vowels": {
        "zip_name": "Vowels.zip",
        "folder": "Vowels",
        "originals": (
            "sub_4b_ch80_v_eog_removed_256Hz",
            "sub_5b_ch80_v_eog_removed_256Hz",
            "sub_8e_ch64_v_eog_removed_256Hz",
            "sub_9_ch64_v_eog_removed_256Hz",
            "sub_11_ch64_v_eog_removed_256Hz",
            "sub_12_ch64_v_eog_removed_256Hz",
            "sub_13_ch64_v_eog_removed_256Hz",
            "sub_15_ch64_v_eog_removed_256Hz",
        ),
        "supplementary": ("sub_8_ch64_v_eog_removed_256Hz_time_correlation_effect",),
    },
    "ShortWords": {
        "zip_name": "Short_words.zip",
        "folder": "Short_words",
        "originals": (
            "sub_1_ch64_s_eog_removed_256Hz",
            "sub_3_ch64_s_eog_removed_256Hz",
            "sub_5_ch64_s_eog_removed_256Hz",
            "sub_6b_ch80_s_eog_removed_256Hz",
            "sub_8g_ch64_s_eog_removed_256Hz",
            "sub_12b_ch64_s_eog_removed_256Hz",
        ),
        "supplementary": (),
    },
    "LongWords": {
        "zip_name": "Long_words.zip",
        "folder": "Long_words",
        "originals": (
            "sub_2b_ch64_l_eog_removed_256Hz",
            "sub_3b_ch80_l_eog_removed_256Hz",
            "sub_6_ch64_l_eog_removed_256Hz",
            "sub_7_ch64_l_eog_removed_256Hz",
            "sub_9c_ch64_l_eog_removed_256Hz",
            "sub_11b_ch64_l_eog_removed_256Hz",
        ),
        "supplementary": (),
    },
    "ShortLongWords": {
        "zip_name": "Short_Long_words.zip",
        "folder": "Short_Long_words",
        "originals": (
            "sub_1c_ch64_sl_eog_removed_256Hz_bw20",
            "sub_5c_ch64_sl_eog_removed_256Hz_bw20",
            "sub_8d_ch64_sl_eog_removed_256Hz",
            "sub_9b_ch64_sl_eog_removed_256Hz_bw20",
            "sub_10_ch64_sl_eog_removed_256Hz_bw20",
            "sub_14_ch64_sl_eog_removed_256Hz_bw20",
        ),
        "supplementary": ("sub_14_ch64_sl_eog_removed_256Hz_bw20_8s",),
    },
}

CLASS_LABELS: dict[str, tuple[str, ...]] = {
    "Vowels": ("vowel_a", "vowel_i", "vowel_u"),
    "ShortWords": ("out", "in", "up"),
    "LongWords": ("cooperate", "independent"),
    "ShortLongWords": ("cooperate", "in"),
}

DEFAULT_SRC_CANDIDATES = [
    # Directory with the 4 extracted condition zips (preferred — the
    # outer dataset.zip has a 4 GB prefix that breaks Python's
    # built-in zipfile module, so callers should extract it first
    # with the system `unzip` tool).
    Path.home() / "mne_data" / "nguyen2017_extracted",
    Path.home() / "mne_data_test" / "nguyen2017_extracted",
    Path.home() / "mne_data" / "bci_speech_imagery",
    # Legacy paths: the outer dataset.zip (only works if not
    # corrupted by the 4 GB prefix).
    Path.home()
    / "mne_data"
    / "MNE-nguyen2017-data"
    / "scl"
    / "fi"
    / "20j120qae7c2rlmr5lfwr"
    / "dataset.zip",
    Path.home()
    / "mne_data_test"
    / "MNE-nguyen2017-data"
    / "scl"
    / "fi"
    / "20j120qae7c2rlmr5lfwr"
    / "dataset.zip",
    Path.home() / "mne_data" / "nguyen2017v" / "dataset.zip",
    Path.home() / "mne_data" / "nguyen2017" / "dataset.zip",
]
OUT_DIR = Path.home() / "mne_data" / "nguyen2017_zenodo"

TOP_README = """\
# Nguyen2017 — Imagined Speech EEG (re-hosted)

Re-hosted copy of the dataset from:

> C. H. Nguyen, G. K. Karavas, P. Artemiadis, "Inferring imagined
> speech using EEG signals: a new approach using Riemannian manifold
> features", *Journal of Neural Engineering*, 15(1), 2017.
> DOI: `10.1088/1741-2552/aa8235`

Originally distributed by the authors as a single Dropbox archive
(`dataset.zip`) wrapping four condition zips. This Zenodo mirror
provides the four condition zips directly, with normalized filenames
and an embedded README in each zip.

## Files in this record

| File                   | Subjects | Classes |
|------------------------|----------|---------|
| Vowels.zip             | 8        | 3 (vowel_a, vowel_i, vowel_u) |
| Short_words.zip        | 6        | 3 (out, in, up)               |
| Long_words.zip         | 6        | 2 (cooperate, independent)    |
| Short_Long_words.zip   | 6        | 2 (cooperate, in)             |

All recordings are 64 or 80 channels at 256 Hz, EOG-removed, 5 s
imagined-speech window. See the embedded `README.md` inside each
condition zip for the original → normalized filename mapping and the
per-subject channel count.

## Loading with MOABB

```python
from moabb.datasets import Nguyen2017_V, Nguyen2017_S, Nguyen2017_L, Nguyen2017_SL
from moabb.paradigms import MotorImagery

ds = Nguyen2017_V()
paradigm = MotorImagery(events=["vowel_a", "vowel_i", "vowel_u"], n_classes=3)
X, y, metadata = paradigm.get_data(dataset=ds, subjects=[1])
```

## Re-hosting rationale

The original Dropbox share relies on an `rlkey` token that can
rate-limit and has no persistent DOI. This Zenodo mirror provides
DOI-addressed URLs for each of the four condition zips, so MOABB's
automated benchmarking pipeline can fetch the data without relying on
consumer cloud storage.

No signal data was modified. Only:

1. The outer `dataset.zip` wrapper was removed (the four condition
   zips are now top-level files on the record).
2. Files inside each condition zip were renamed from e.g.
   `sub_4b_ch80_v_eog_removed_256Hz.mat` to `sub-01.mat`, following
   the order used by the MOABB adapter.
3. A `README.md` was added inside each condition zip describing the
   renaming and the 64/80-channel split.
"""


def _condition_readme(condition: str) -> str:
    cfg = CONDITIONS[condition]
    originals = cfg["originals"]
    supplementary = cfg["supplementary"]
    labels = CLASS_LABELS[condition]

    lines = [
        f"# Nguyen2017 — {condition} (MOABB re-host notes)",
        "",
        "Re-hosted from <https://doi.org/10.1088/1741-2552/aa8235> (Dropbox share).",
        "See `Read_me.txt` in this same archive for the authors'",
        "original notes and attribution request (verbatim copy from",
        "the upstream distribution).",
        "",
        "## Filename mapping",
        "",
        "| New name      | Original filename | EEG channels |",
        "|---------------|-------------------|--------------|",
    ]
    for idx, original in enumerate(originals, start=1):
        if "_ch80_" in original:
            n_ch = "80 (only first 64 are EEG)"
        elif "_ch64_" in original:
            n_ch = "64"
        else:
            n_ch = "unknown"
        lines.append(f"| sub-{idx:02d}.mat | {original}.mat | {n_ch} |")

    if supplementary:
        lines += [
            "",
            "## Supplementary files",
            "",
            "These files are carried forward unchanged under",
            "`supplementary/` with their original filenames. They are",
            "*not* part of the main MOABB subject set but are referenced",
            "in the original paper's analysis sections:",
            "",
        ]
        for name in supplementary:
            lines.append(f"- `supplementary/{name}.mat`")

    lines += [
        "",
        "## Paradigm",
        "",
        f"- Classes: {', '.join(labels)}",
        "- Sampling rate: 256 Hz (downsampled from 1000 Hz)",
        "- Bandpass: 8-70 Hz (5th order Butterworth) + 60 Hz notch",
        "- EOG artifacts removed by authors",
        "- Imagined-speech window: 5 s per trial (`_last_beep` variable)",
        "- Resting baseline: 2 s at end of trial (`_end_trial` variable)",
        "- EOG channel indices (0-based, relative to first 64): [0, 9, 32, 63]",
        "  (channels [1, 10, 33, 64] in 1-based indexing per authors' readme)",
        "",
        "## Data format",
        "",
        "Each `.mat` file contains **two** variables:",
        "",
        "1. `eeg_data_wrt_task_rep_no_eog_256Hz_last_beep` — a",
        "   `(n_classes, n_trials)` object array where each cell is a",
        "   `(n_channels, 1280)` float matrix (5 s * 256 Hz, speech imagery).",
        "2. `eeg_data_wrt_task_rep_no_eog_256Hz_end_trial` — a",
        "   `(n_classes, n_trials)` object array where each cell is a",
        "   `(n_channels, 512)` float matrix (2 s * 256 Hz, resting baseline).",
        "",
        "The MOABB adapter currently only uses `_last_beep`; the resting",
        "baseline is preserved in the files for future use.",
        "",
        "The payloads in this zip are **bit-identical** to the authors'",
        "original Dropbox files; only the filenames were normalized to",
        "`sub-NN.mat` to match MOABB's integer subject IDs.",
        "",
    ]
    return "\n".join(lines)


def _locate_inner_zip(src: Path, inner_zip_name: str) -> Path:
    """Return the path to an inner condition zip.

    The source may be a directory containing the 4 extracted condition
    zips, or (legacy) the outer ``dataset.zip``. The outer zip has a
    4 GB prefix bug that breaks Python's zipfile module, so we do NOT
    use ``zipfile`` to read from it — callers must have extracted the
    inner zips first using the system ``unzip`` tool.
    """
    if src.is_dir():
        # Look for the inner zip at the top or under common subdirs.
        for cand in (
            src / inner_zip_name,
            src / "dataset" / inner_zip_name,
            src / "nguyen2017_extracted" / inner_zip_name,
        ):
            if cand.exists():
                return cand
        raise FileNotFoundError(
            f"{inner_zip_name} not found under {src}. If you only have "
            f"the outer dataset.zip, extract it first with:\n"
            f"  mkdir -p ~/mne_data/nguyen2017_extracted\n"
            f"  cd ~/mne_data/nguyen2017_extracted\n"
            f"  unzip /path/to/dataset.zip\n"
            f"then re-run this script."
        )

    if src.is_file() and src.suffix.lower() == ".zip":
        raise RuntimeError(
            f"Pointing at the outer dataset.zip ({src}) is not "
            f"supported: it has a 4 GB prefix that Python's zipfile "
            f"module cannot parse. Extract it first with the system "
            f"`unzip` tool, then re-run with --src pointing at the "
            f"extraction directory. Example:\n"
            f"  mkdir -p ~/mne_data/nguyen2017_extracted\n"
            f"  cd ~/mne_data/nguyen2017_extracted\n"
            f"  unzip {src}\n"
            f"  python scripts/repackage_nguyen2017.py "
            f"--src ~/mne_data/nguyen2017_extracted"
        )

    raise ValueError(f"Unsupported source path: {src}")


def _find_member(zf: zipfile.ZipFile, folder: str, basename: str) -> str | None:
    """Locate a member by folder + basename, tolerating layout variations."""
    target_exact = f"{folder}/{basename}.mat"
    for name in zf.namelist():
        if name == target_exact:
            return name
    # Fallback: scan for any member whose basename matches.
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        if name.endswith(f"{basename}.mat"):
            return name
    return None


def repackage_condition(src: Path, condition: str, force: bool) -> bool:
    cfg = CONDITIONS[condition]
    inner_zip_name = cfg["zip_name"]
    folder = cfg["folder"]
    originals = cfg["originals"]
    supplementary = cfg["supplementary"]

    out_zip = OUT_DIR / inner_zip_name
    if out_zip.exists() and not force:
        log.info("%s already exists (skip; use --force to rebuild)", inner_zip_name)
        return False

    inner_zip_path = _locate_inner_zip(src, inner_zip_name)
    log.info("Reading %s from %s ...", inner_zip_name, inner_zip_path)

    log.info(
        "Rebuilding %s with %d renamed subjects + Read_me.txt + %d supplementary ...",
        inner_zip_name,
        len(originals),
        len(supplementary),
    )

    with (
        zipfile.ZipFile(str(inner_zip_path)) as inner,
        zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as out,
    ):
        # 1. Carry forward the authors' original Read_me.txt verbatim.
        # Read_me.txt isn't a .mat, so scan the namelist directly rather
        # than using _find_member (which only matches .mat files).
        readme_candidates = [
            n
            for n in inner.namelist()
            if n.endswith("Read_me.txt") and not n.endswith("/")
        ]
        if readme_candidates:
            with inner.open(readme_candidates[0]) as fh:
                out.writestr("Read_me.txt", fh.read())
            log.info("  + Read_me.txt (verbatim from authors)")
        else:
            log.warning("  ! Read_me.txt not found in %s", inner_zip_name)

        # 2. Renamed subject files: sub-01.mat .. sub-NN.mat
        for idx, basename in enumerate(originals, start=1):
            member = _find_member(inner, folder, basename)
            if member is None:
                raise FileNotFoundError(f"{basename}.mat not found in {inner_zip_path}")
            with inner.open(member) as fh:
                out.writestr(f"sub-{idx:02d}.mat", fh.read())
            log.info("  + sub-%02d.mat (was %s.mat)", idx, basename)

        # 3. Supplementary analysis files (preserve original names).
        for basename in supplementary:
            member = _find_member(inner, folder, basename)
            if member is None:
                log.warning("  ! supplementary %s.mat not found, skipping", basename)
                continue
            with inner.open(member) as fh:
                out.writestr(f"supplementary/{basename}.mat", fh.read())
            log.info("  + supplementary/%s.mat", basename)

        # 4. Our own README.md with the rehost notes.
        out.writestr("README.md", _condition_readme(condition))
        log.info("  + README.md (rehost notes)")

    size_mb = out_zip.stat().st_size / (1024 * 1024)
    log.info("wrote %s (%.1f MB)", inner_zip_name, size_mb)
    return True


def _find_source(cli_src: str | None) -> Path:
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
        "Could not locate Nguyen2017 source data. Either:\n"
        "  1. Run `python -c 'from moabb.datasets import Nguyen2017_V; "
        "Nguyen2017_V().data_path(1)'` to fetch from Dropbox, or\n"
        "  2. Pass --src <path-to-dataset.zip-or-extracted-dir>.\n"
        f"Looked in: {', '.join(str(p) for p in DEFAULT_SRC_CANDIDATES)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--src",
        help=(
            "Source path: either the outer dataset.zip (Dropbox archive) "
            "or a directory containing the four condition zips."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild output ZIPs even if they already exist.",
    )
    args = parser.parse_args()

    src = _find_source(args.src)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rebuilt = 0
    for condition in CONDITIONS:
        if repackage_condition(src, condition, args.force):
            rebuilt += 1

    readme_path = OUT_DIR / "README.md"
    if args.force or not readme_path.exists():
        readme_path.write_text(TOP_README, encoding="utf-8")
        log.info("wrote top-level README.md")

    log.info("Rebuilt %d/%d condition ZIPs in %s", rebuilt, len(CONDITIONS), OUT_DIR)


if __name__ == "__main__":
    main()
