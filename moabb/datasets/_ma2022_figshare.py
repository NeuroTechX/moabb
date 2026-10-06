"""Figshare fallback for :class:`moabb.datasets.Ma2022`.

NEMAR deposit ``nm000288`` is the loader's primary source, but as long as it
is awaiting NEMAR publication, users still need a public mirror. The original
authors' deposit on figshare (DOI 10.6084/m9.figshare.19228725, CC BY 4.0) is
already BIDS-shaped — same ``task-motorimagery`` sidecars, same subject/
session labels, same EDF representation (32 channels at 250 Hz, four-second
imagery windows). This module rebuilds that BIDS tree locally on first use so
that :meth:`moabb.datasets.Ma2022._get_single_subject_data` reads it with the
exact same ``mne_bids.read_raw_bids`` pipeline it uses for NEMAR.

Figshare bundles the 125 EDFs in a single 737 MB archive (``edf_files.zip``),
with no per-file md5 outside of the zip's own CRC-32. Loose sidecars (one
events TSV per session + eight top-level BIDS sidecars) ship as standalone
files with per-file md5 served by the figshare API. We follow that split:

* Loose sidecars go through :func:`pooch.retrieve` with the figshare-reported
  md5 (``_FIGSHARE_SIDECARS``, ``_FIGSHARE_EVENTS``).
* The EDF archive is fetched once with its whole-zip md5
  (``_FIGSHARE_EDF_ZIP``) and extracted lazily per subject. Each stored EDF
  is additionally checked against the zip local-header CRC-32 before being
  written to the BIDS tree, so a corrupted single file is caught even though
  figshare only publishes an aggregate hash.

The result is a BIDS root ``MNE-ma-edf2022-figshare/`` under the usual MOABB
dataset path, structurally identical to what ``nemar_dl(nm000288, ...)``
materialises when the deposit is public.
"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import pooch

from .bids_interface import get_bids_root
from .download import get_dataset_path


log = logging.getLogger(__name__)


# Figshare DOI 10.6084/m9.figshare.19228725, version 5 (frozen 2022-03-10),
# ``SHU Multi-session Dataset``. The ``ndownloader`` URLs resolve to AWS S3
# with a short-lived signed URL; the file IDs themselves are permanent.
_FIGSHARE_ARTICLE_ID = 19228725

_FIGSHARE_URL = "https://ndownloader.figshare.com/files/{file_id}"


def _url(file_id: int) -> str:
    return _FIGSHARE_URL.format(file_id=file_id)


# One per-subject bundle (737 MB), extracted on demand.
_FIGSHARE_EDF_ZIP: dict[str, object] = {
    "file_id": 36728991,
    "md5": "8585aaa9c458f5e64863e71ec3bb83ad",
    "size": 737904661,
    "name": "edf_files.zip",
}


# Top-level BIDS sidecars; each one carries a figshare-reported md5.
_FIGSHARE_SIDECARS: dict[str, tuple[int, str]] = {
    "dataset_description.json": (34166145, "0a03f39dbbc53b0c952b466c20fe30a8"),
    "participants.json": (34166148, "dcdb2209e66d5b2201a8b460f627f14a"),
    "participants.tsv": (34166151, "09b5a14272892ae2c0c849492849ec31"),
    "task-motorimagery_channels.tsv": (34166154, "44a48de8ddb3e0f911912cff7a5c7ccc"),
    "task-motorimagery_coordsystem.json": (34166157, "c0ddd76983c871f75724747797f62fa8"),
    "task-motorimagery_eeg.json": (34166160, "1e224d2c6d7cba3b19ae8948f38c8ae7"),
    "task-motorimagery_electrodes.tsv": (34166163, "3ce4896e85c0344feb4230700890eeec"),
    "task-motorimagery_events.json": (34166166, "d962891d1f011161ba1e8534c0a25dcc"),
    "README.txt": (36729006, "29eeed4fbc37e111a9bb2392ccaa4700"),
}


# Per-subject events tsv (file_id, md5), one tuple per session 1..5.
# Generated from the figshare API; order matches session index.
# fmt: off
_FIGSHARE_EVENTS: dict[int, list[tuple[int, str]]] = {
    1: [(34166184, "c7c4bcd536c3ff27a86658515f714023"), (34166187, "40d6af735fd346c62760fe21120b64f9"), (34166190, "f4a99343a091d3f54f1a080bcd2c56a5"), (34166193, "e9b67f93dad0bdfde23f71b8cc03756c"), (34166196, "e2c61153252adbff2accc4a721cab987")],
    2: [(34166211, "b0f2dc71f4985f25f77ab671963d983b"), (34166199, "658b3aeff3bdd6369c085fd0661dfc55"), (34166202, "fd382a5756e7f05ac4285359e9882cd9"), (34166205, "1597f5ddee301f957f1def23578f8e2b"), (34166208, "df97eba928d65fd6447779660bfdee6b")],
    3: [(34166214, "73d181cd57bddf54d90abb3c179ffb04"), (34166217, "59e7a942aa83f14cba8f65993d2a5977"), (34166220, "58dcc98c8ffacda7f769dee2e9858f12"), (34166223, "031326504df7af153c8d03ec3e0bc266"), (34166226, "3cea49a68f3ed9a0aab728fb638de5e5")],
    4: [(34166229, "a7e42dc1d72c299d3771b30d3bbcf966"), (34166232, "a49a0b3aca0899e89f5b118d6dec1ec9"), (34166235, "25ee545720a230d9d9318252793f5548"), (34166238, "3e4c7b2f074400accdd7b45bd2cc6632"), (34166241, "bd603dc39cf9c7b2a2ff3f5fdd13fabf")],
    5: [(34166244, "ab019ab28b6eea68189f2956dd81d902"), (34166247, "3820aa3b6c0e65a4dea024577d20c7ea"), (34166250, "7788de0ec45be60571457cf0f5462cb6"), (34166253, "ac04b9d586798bb8b17965fb75ef5415"), (34166256, "60cc449bd29337d653e4ba61dea153a2")],
    6: [(34166259, "694e3dec105e456217c1981992f87e42"), (34166262, "82f7b4d2887280503220f09f001137c1"), (34166265, "5648f34e6e34a859a78c645cd4cef74a"), (34166268, "cd6d46d91b7de88f70ca914dae850423"), (34166271, "7279db2f0727960967c2f69438e6acca")],
    7: [(34166274, "1cac55505fd0e83a12ae2ee1610212ef"), (34166277, "2fcba7e334e06bba9c296da4e28ac906"), (34166280, "48b315877359dd418e2af5769cf4b82c"), (34166283, "8b53c12caf769bee726f1dfd76fc099d"), (34166286, "578d4618fb1a8b7cdd79865b2c90c508")],
    8: [(34166289, "529e07ea0262a31b9888b79f7c746ce8"), (34166292, "2a7994d9747fea0c199bb34e1591a4c4"), (34166295, "99f21dbeef606ec61f16df1837cdb0aa"), (34166298, "d64180cb6214b9a728e341c71d770dc0"), (34166301, "47747f48c292f28918afa63e55b49d7d")],
    9: [(34166304, "952a562bba987aed613a64aea9664ab7"), (34166307, "df39c9900e56d2543877b5de36fdae3a"), (34166310, "55a7bb75f17cd79606dbaac4ed6e3c81"), (34166313, "219064d92e9177d9f5960dda0825f665"), (34166316, "e6fddcf8e8317700b01a6b7d679e8008")],
    10: [(34166319, "3b681b8a8071a9d67ecff921db424758"), (34166322, "e054442e37b4812d04f7c42274975a47"), (34166325, "a062f9a69a9e223ad38ab4864b397714"), (34166328, "c88eafc3e0f1af39699d8ad66088152b"), (34166331, "2067e589e4c8d32e120a47566645d4f0")],
    11: [(34166334, "2e7bb3e2c3ce25836b2a65cdf209fb9c"), (34166337, "2f245e5b6724d9c83e203f8c499d39f3"), (34166340, "1d8fbce8471342c6f23d8cff96f07729"), (34166343, "a95da37295cc080f2fdc10305e8c81b5"), (34166346, "56714185e30c25b0259f422c29a2751e")],
    12: [(34166349, "ffcb7178eea6903dfd8ce0c8dd693ce7"), (34166364, "a3d130cd9fe844a26ae6a3d178d7d755"), (34166352, "78c3cfa73842ac219bd1a9c06b1f8be9"), (34166355, "8b2df0f7e3be0188debce7aa91921714"), (34166358, "9eb4d480af88bb87b3c9b59a9f2f7439")],
    13: [(34166361, "705a8581697b116e7e70286d3bd1c3ad"), (34166367, "9351e7217022270b6da178aed7d8232c"), (34166370, "757639089f3e10343412ba0b82252cc8"), (34166373, "bb8f714b6aba695d18f9abb8418c8050"), (34166376, "0d74aebc5db33d6ac924621c73f25a7b")],
    14: [(34166379, "a5a36611c6946cd32b1b5ec566c75f90"), (34166382, "7a0ec0b1dd83fab2d2095369922ce138"), (34166385, "b32d005120f1aa312188e4c872c97ada"), (34166388, "f1011e13ef798c8e83cd990abab23d79"), (34166391, "9c852bd446ca37dce7ae83c937f7c2b2")],
    15: [(34166394, "7c8590fe656f403ecfe225689fc06fea"), (34166397, "a8b651be51b03cd2db036b7c14c16477"), (34166400, "0a83d320993db802710b8998dd7b764a"), (34166403, "a8b651be51b03cd2db036b7c14c16477"), (34166406, "e45ef597ad4312943e2e3fee409b876b")],
    16: [(34166409, "beb88094e7a0a70d4bc3598ceb946897"), (34166412, "69d42338ede5579b812aac3976b65b1a"), (34166415, "1c77d64caccaae1341e34c3864b6b439"), (34166418, "e9a1ad7e7c4bc34a0dd201cd13323e25"), (34166421, "c02e3e4b49d7a9a4ec54698e1849f759")],
    17: [(34166424, "1c77d64caccaae1341e34c3864b6b439"), (34166427, "6664edfebc588831407c8928f8dc38c1"), (34166430, "e45ef597ad4312943e2e3fee409b876b"), (34166433, "beb88094e7a0a70d4bc3598ceb946897"), (34166436, "7952ecaf749214abc3d2b3c2f8d6851f")],
    18: [(34166439, "6b4421f2f35fe5bf2f584b6795aab721"), (34166442, "808f1ad4874e6d87757d6d583231b1d6"), (34166445, "9c852bd446ca37dce7ae83c937f7c2b2"), (34166448, "69d42338ede5579b812aac3976b65b1a"), (34166451, "1c77d64caccaae1341e34c3864b6b439")],
    19: [(34166454, "c82cedf5300b34010b32c187bb215895"), (34166457, "e9a1ad7e7c4bc34a0dd201cd13323e25"), (34166460, "25e7e558889d8ffb0af1fc2cfb4e20f2"), (34166463, "f3afc1f7d588267407e83e802b25b20e"), (34166466, "4659e5b5b0be925f83e73c185346989a")],
    20: [(34166469, "5c2c4a01dc795df9f27476032f0c75de"), (34166472, "6664edfebc588831407c8928f8dc38c1"), (34166475, "bc933bfe17fb27a083cf091c2a23ddbc"), (34166478, "7c8590fe656f403ecfe225689fc06fea"), (34166481, "6664edfebc588831407c8928f8dc38c1")],
    21: [(34166484, "41d024f27380bce57469ef6766824cef"), (34166487, "3791197717023c5237e74afb65e5d0a2"), (34166490, "32be1c3d2e2433edac5ba03567d6787c"), (34166493, "0d0c606ab92e92ca43449327fca1708f"), (34166496, "8faac26ec40fccf1a1702fb1e9ce67c3")],
    22: [(34166499, "5d6be3c59ecf055dc457ffd9cacfdb75"), (34166502, "7c8590fe656f403ecfe225689fc06fea"), (34166505, "69d42338ede5579b812aac3976b65b1a"), (34166508, "0d0c606ab92e92ca43449327fca1708f"), (34166511, "a18b5f02762b4e87ec1c4e3111181f5d")],
    23: [(34166514, "d829af9022361a767d0bf40b4f218f0e"), (34166517, "c6607a0cd649a9b9267f869a0b559820"), (34166520, "a8c5cd1a8aec0a1df6ca901d1e68722a"), (34166523, "d6efe0b81f0147b6fffc8254bb7d7132"), (34166526, "6e45d6636750191a91c3c9f0f1b0492c")],
    24: [(34166529, "d9b44cbd5056ced84acfc2850fd98a5c"), (34166532, "a116d968b3804804e9024d6d0d119afa"), (34166535, "08270db0d2e5747712266ee5a56262fc"), (34166538, "e6c498ef2ae307279d952cb91b29ccb9"), (34166541, "69d42338ede5579b812aac3976b65b1a")],
    25: [(34166544, "d8db63b06fe88f2be2081f088e9a4efc"), (34166547, "e2842f5a573cebcb18e93ea2aaa937f8"), (34166550, "85dc1d6b3518887ee8baead58db163cc"), (34166553, "bd33feff8311aa59a5d46fef6cdd0727"), (34166556, "69d42338ede5579b812aac3976b65b1a")],
}
# fmt: on


# Local MOABB cache code for the figshare mirror; kept distinct from the
# NEMAR code so the two caches never cross-populate.
_FIGSHARE_CODE = "Ma-edf2022-figshare"


def figshare_bids_root(code: str, path: str | None) -> Path:
    """Resolve the figshare BIDS root under the usual MOABB cache."""
    base = Path(get_dataset_path(code, path))
    return get_bids_root(code=_FIGSHARE_CODE, path=base)


def _edf_entry_name(subject: int, session: int) -> str:
    """Entry path inside edf_files.zip for one session recording."""
    return f"edf/sub-{subject:03d}_ses-{session:02d}_task_motorimagery_eeg.edf"


def _session_dir(bids_root: Path, subject: int, session: int) -> Path:
    return bids_root / f"sub-{subject:03d}" / f"ses-{session:02d}" / "eeg"


def _session_stem(subject: int, session: int) -> str:
    return f"sub-{subject:03d}_ses-{session:02d}_task-motorimagery"


def _subject_has_bids_data(bids_root: Path, subject: int) -> bool:
    """True when every expected session file is present for this subject."""
    for session in range(1, 6):
        stem = _session_stem(subject, session)
        session_dir = _session_dir(bids_root, subject, session)
        for suffix in ("_eeg.edf", "_events.tsv", "_channels.tsv"):
            if not (session_dir / f"{stem}{suffix}").is_file():
                return False
    return True


def _retrieve(file_id: int, md5: str, dest: Path, name: str, verbose) -> Path:
    """Fetch one small file through pooch, writing it under ``dest/name``.

    Pooch handles atomic rename + md5 verification; a stale corrupted file is
    silently redownloaded.
    """
    dest.mkdir(parents=True, exist_ok=True)
    local = pooch.retrieve(
        url=_url(file_id),
        known_hash=f"md5:{md5}",
        fname=name,
        path=str(dest),
        progressbar=bool(verbose),
    )
    return Path(local)


def _ensure_sidecars(bids_root: Path, verbose) -> None:
    """Download the top-level BIDS sidecars once per cache."""
    for name, (file_id, md5) in _FIGSHARE_SIDECARS.items():
        if (bids_root / name).is_file():
            continue
        _retrieve(file_id, md5, bids_root, name, verbose)


def _ensure_events(bids_root: Path, subject: int, verbose) -> None:
    """Download the five per-session events tsvs and copy under each session dir."""
    staging = bids_root / ".figshare" / "events" / f"sub-{subject:03d}"
    for session, (file_id, md5) in enumerate(_FIGSHARE_EVENTS[subject], start=1):
        session_dir = _session_dir(bids_root, subject, session)
        target = session_dir / f"{_session_stem(subject, session)}_events.tsv"
        if target.is_file():
            continue
        src = _retrieve(
            file_id,
            md5,
            staging,
            f"sub-{subject:03d}_ses-{session:02d}_task_motorimagery_events.tsv",
            verbose,
        )
        session_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)


def _write_channels_sidecar(bids_root: Path, subject: int) -> None:
    """Fan the top-level ``task-motorimagery_channels.tsv`` into each session.

    mne-bids expects per-recording ``_channels.tsv`` sidecars for raw reads;
    the figshare deposit only ships one at the dataset level.
    """
    source = bids_root / "task-motorimagery_channels.tsv"
    text = source.read_text()
    for session in range(1, 6):
        session_dir = _session_dir(bids_root, subject, session)
        target = session_dir / f"{_session_stem(subject, session)}_channels.tsv"
        if target.is_file():
            continue
        session_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


def _ensure_edf_zip(bids_root: Path, verbose) -> Path:
    """Download the one monolithic edf_files.zip (md5-verified) once."""
    staging = bids_root / ".figshare"
    local = _retrieve(
        _FIGSHARE_EDF_ZIP["file_id"],
        _FIGSHARE_EDF_ZIP["md5"],
        staging,
        _FIGSHARE_EDF_ZIP["name"],
        verbose,
    )
    return local


def _extract_subject_edfs(zip_path: Path, bids_root: Path, subject: int) -> None:
    """Pull the five session EDFs for one subject out of edf_files.zip.

    Figshare's zip embeds all 125 EDFs with upstream names that use the BIDS
    key-value separator incorrectly (``task_motorimagery`` instead of
    ``task-motorimagery``). Rename on extraction so the output matches the
    NEMAR BIDS layout the loader already reads.
    """
    with zipfile.ZipFile(zip_path, "r") as archive:
        for session in range(1, 6):
            session_dir = _session_dir(bids_root, subject, session)
            stem = _session_stem(subject, session)
            target = session_dir / f"{stem}_eeg.edf"
            if target.is_file():
                continue
            session_dir.mkdir(parents=True, exist_ok=True)
            entry = _edf_entry_name(subject, session)
            # ``extract`` verifies the stored CRC-32 against the decompressed
            # bytes; a corruption inside the archive aborts the write.
            tmp = session_dir / f".{target.name}.part"
            with archive.open(entry, "r") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            # ZipFile does CRC checking on close, but mirror it here too so
            # we never leave a half-good EDF visible under its final name.
            info = archive.getinfo(entry)
            if tmp.stat().st_size != info.file_size:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Short read extracting {entry} ({tmp.stat().st_size} of "
                    f"{info.file_size} bytes) from figshare edf_files.zip."
                )
            tmp.replace(target)


def ensure_bids_mirror(
    bids_root: Path, subjects: Iterable[int], *, force_update: bool = False, verbose=None
) -> None:
    """Materialise the figshare mirror on disk for the requested subjects.

    After this call, the BIDS root exposes exactly the file names
    :meth:`Ma2022.bids_paths` expects (one EDF per subject per session, plus
    the matching events/channels sidecars and the top-level dataset sidecars).
    """
    bids_root.mkdir(parents=True, exist_ok=True)
    _ensure_sidecars(bids_root, verbose)

    requested = sorted({int(s) for s in subjects})
    if force_update:
        for subject in requested:
            for session in range(1, 6):
                session_dir = _session_dir(bids_root, subject, session)
                stem = _session_stem(subject, session)
                for suffix in ("_eeg.edf", "_events.tsv", "_channels.tsv"):
                    target = session_dir / f"{stem}{suffix}"
                    if target.is_file():
                        target.unlink()

    pending_edf = [s for s in requested if not _subject_has_bids_data(bids_root, s)]
    for subject in pending_edf:
        _ensure_events(bids_root, subject, verbose)
        _write_channels_sidecar(bids_root, subject)

    pending_edf = [
        s
        for s in requested
        if any(
            not (
                _session_dir(bids_root, s, session)
                / f"{_session_stem(s, session)}_eeg.edf"
            ).is_file()
            for session in range(1, 6)
        )
    ]
    if pending_edf:
        zip_path = _ensure_edf_zip(bids_root, verbose)
        for subject in pending_edf:
            _extract_subject_edfs(zip_path, bids_root, subject)
