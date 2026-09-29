"""NETBCI longitudinal right-hand motor imagery dataset (Corsi 2026).

Corsi, M.-C., Gitton, C., Gonzalez-Astudillo, J., Kahn, A. E., Hugueville, L.,
Schwartz, D., George, N., Chavez, M., Dupont, S., Bassett, D. S. and
De Vico Fallani, F. Understanding Brain-Computer Interfaces training: a
longitudinal and multimodal dataset. Sci Data (2026).
DOI: 10.1038/s41597-026-08237-5
Data DOI: 10.57745/RBJRC7 (Recherche Data Gouv, CC-BY 4.0)
"""

import hashlib
import io
import logging
import struct
import time
import zipfile
import zlib
from pathlib import Path

import mne
import pandas as pd
import requests

from .base import BaseBIDSDataset
from .download import get_dataset_path
from .metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    Tags,
)
from .utils import stim_channels_with_selected_ids


log = logging.getLogger(__name__)

_DATAVERSE = "https://entrepot.recherche.data.gouv.fr"
_PERSISTENT_ID = "doi:10.57745/RBJRC7"
# Datafile ids and MD5 checksums of dataset version 2.2 (2026-08-24). The
# archive (id 744565, same MD5) is unchanged since v1.0; only participants.*
# were replaced in v2.0. Pinning keeps every download on the same bytes.
# The whole BIDS tree (EEG + MEG, 19 subjects, 49.0 GB) is one zip64 archive.
_ARCHIVE = ("Archive.zip", 744565, "e7324cd60c711713dc6bf62ac70aedf8", 49016733269)
# Small root-level BIDS files published next to the archive.
_ROOT_FILES = {
    "dataset_description.json": (743365, "c4619efa3892c437825ef5a86dcf5eed"),
    "participants.tsv": (758600, "b9aa1c07015820e80bc79be6a61f30fc"),
    "participants.json": (758595, "41b33bc709e7b36a14faa91da128aaba"),
}
# The access endpoint answers with a 303 to a presigned S3 URL valid 4 h;
# re-resolve it well before it expires.
_SIGNED_URL_TTL_S = 3 * 3600
_SFREQ = 250.0

_TASK = "MotorImageryRest"
# ``value`` column of the BIDS events.tsv, as written by the authors'
# BIDS conversion (github.com/mccorsi/NETBCI_data, BIDSify.py:
# ``event_dict = {"MI": 1, "Rest": 2}``): 1 = "up" target, sustained
# right-hand grasping motor imagery; 2 = "down" target, rest.
_EVENTS = {"right_hand": 1, "rest": 2}

_N_SUBJECTS = 19
_N_SESSIONS = 4
_N_RUNS = 6

# Every EEG BIDS file of one motor-imagery run.
_RUN_SUFFIXES = (
    "eeg.vhdr",
    "eeg.vmrk",
    "eeg.eeg",
    "eeg.json",
    "channels.tsv",
    "events.tsv",
)

_DOWNLOAD_ATTEMPTS = 4
# Local file header: signature, version, flags, method, time, date, CRC-32,
# compressed size, uncompressed size, name length, extra length.
_LOCAL_HEADER = struct.Struct("<4s5H3L2H")
_LOCAL_HEADER_SIG = b"PK\x03\x04"
# Slack after the central-directory extra field: a local extra field can be
# longer (zip64 sizes); a short read triggers one follow-up request.
_HEADER_SLACK = 1024


class _HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable, *unbuffered* view of a remote file via HTTP ranges.

    Used by :class:`zipfile.ZipFile` only to read the archive's end records and
    central directory (about 1 MB for 7,449 members); members are then fetched
    one exact byte range each (:meth:`read_range`). The access URL redirects
    to a presigned S3 URL, which is resolved once and refreshed on expiry or
    on a 401/403, so the Dataverse API is not hit on every request.
    """

    def __init__(self, url, session=None, timeout=300, size=None):
        super().__init__()
        self._url = url
        self._session = session or requests.Session()
        self._timeout = timeout
        self._pos = 0
        self._signed = None
        self._signed_at = 0.0
        self.bytes_fetched = 0
        self.n_requests = 0
        response = self._get(0, 0)
        content_range = response.headers.get("Content-Range", "")
        if response.status_code != 206 or "/" not in content_range:
            raise OSError(f"{url} does not support HTTP range requests")
        self._size = int(content_range.rsplit("/", 1)[1])
        if size is not None and self._size != size:
            raise OSError(
                f"{url} is {self._size} bytes, expected {size}: the deposit changed"
            )

    def _resolve(self, force=False):
        """Return the presigned storage URL behind the access endpoint."""
        fresh = time.monotonic() - self._signed_at < _SIGNED_URL_TTL_S
        if self._signed is not None and fresh and not force:
            return self._signed
        response = self._session.get(
            self._url,
            headers={"Range": "bytes=0-0"},
            allow_redirects=False,
            timeout=self._timeout,
        )
        location = response.headers.get("Location")
        if response.is_redirect and location:
            self._signed = location
        else:  # served directly, no storage redirect
            response.raise_for_status()
            self._signed = self._url
        self._signed_at = time.monotonic()
        return self._signed

    def _get(self, start, end):
        last_exc = None
        for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
            try:
                url = self._resolve(force=attempt > 1)
                response = self._session.get(
                    url, headers={"Range": f"bytes={start}-{end}"}, timeout=self._timeout
                )
                response.raise_for_status()
                expected = end - start + 1
                if response.status_code == 206 and len(response.content) != expected:
                    raise requests.RequestException(
                        f"short read: {len(response.content)} of {expected} bytes"
                    )
                self.n_requests += 1
                self.bytes_fetched += len(response.content)
                return response
            except requests.RequestException as exc:
                last_exc = exc
                log.warning(
                    "Range request %d-%d failed (attempt %d/%d): %s",
                    start,
                    end,
                    attempt,
                    _DOWNLOAD_ATTEMPTS,
                    exc,
                )
                time.sleep(min(2**attempt, 30))
        raise OSError(f"Could not read bytes {start}-{end} of {self._url}") from last_exc

    def read_range(self, start, length):
        """Bytes ``[start, start + length)`` of the remote file, one request."""
        end = min(start + length, self._size) - 1
        return self._get(start, end).content

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"invalid whence {whence!r}")
        return self._pos

    def readinto(self, buffer):
        if self._pos >= self._size:
            return 0
        end = min(self._pos + len(buffer), self._size) - 1
        data = self._get(self._pos, end).content
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)


class Corsi2026(BaseBIDSDataset):
    """NETBCI: longitudinal right-hand motor imagery vs rest BCI training [1]_.

    .. admonition:: Dataset summary

        ============  =======  =======  ========  ===============  ============  =========
        Name          #Subj    #Chan    #Classes  #Trials / class  Trial length  #Sessions
        ============  =======  =======  ========  ===============  ============  =========
        Corsi2026     19       74       2         357-384          5 s           4
        ============  =======  =======  ========  ===============  ============  =========

    Networks for BCI (NETBCI) was recorded at the Paris Brain Institute (ICM,
    CENIR MEG-EEG platform). 19 healthy right-handed adults (7 female, aged
    19-35) trained over four sessions on four different days (two sessions a
    week for two weeks). EEG (74 Ag/AgCl passive electrodes, Easycap, 10-10
    montage, mastoid reference, ground on the left scapula) was recorded
    simultaneously with 306-channel Elekta MEG; both were downsampled to
    250 Hz. This loader exposes the EEG only.

    The task is a one-dimensional, two-target box task: a cursor moves
    rightward at constant speed and the participant controls its vertical
    position. To hit the *up* target the participant performs sustained
    kinesthetic motor imagery of right-hand grasping (``right_hand``); to hit
    the *down* target they stay at rest with eyes open (``rest``). Each trial
    has a 1 s inter-stimulus interval followed by a 5 s target presentation,
    with online cursor feedback from 3 to 6 s after trial start. Event onsets
    in ``events.tsv`` mark the target presentation, and the epoch window here
    covers the 5 s target period.

    Each session has six feedback runs (the released ``task-MotorImageryRest``
    runs), each with a balanced mix of the two targets. The protocol specifies
    32 trials per run (16 per class). The released event files hold the
    authors' checked trials: 375 of the 456 runs keep all 32, the rest keep
    29-31, giving 717-768 trials per subject (357-384 per class) and 14,431
    trials in total (7,224 ``right_hand``, 7,207 ``rest``).
    One of them cannot be epoched: the last ``rest`` cue of sub-03 ses-01
    run-03 is at 224.03 s but the recording ends at 227.69 s, before the 5 s
    window closes, so :class:`~moabb.paradigms.MotorImagery` returns 14,430
    epochs (7,206 ``rest``).

    Events are read from the BIDS ``events.tsv``. The BrainVision marker file
    carries the same two codes; the loader *replaces* those annotations with
    the events.tsv ones rather than adding to them, so no trial is counted
    twice.

    The EEG of 440 of the 456 runs is stored at 250 Hz. Ten runs are stored at
    249.9 Hz (sub-01 ses-01 run-04, sub-03 ses-03 run-05, all of sub-06 ses-04,
    sub-17 ses-01 run-04, sub-19 ses-02 run-02) and the six runs of sub-09
    ses-02 at 1000 Hz (not downsampled). Their events.tsv samples follow the
    stored rate, so the headers are genuine. The loader resamples these runs
    to 250 Hz after setting the annotations (in seconds), so that every epoch
    has the same length and event timing is kept.

    The data are distributed on Recherche Data Gouv (doi:10.57745/RBJRC7,
    version 2.2) as one 49.0 GB zip64 archive holding the whole BIDS tree
    (EEG and MEG), plus small root files. The archive has not changed since
    version 1.0; the loader pins its datafile id, size and MD5, and checks the
    MD5 of each root file. It reads the archive's central directory, then
    fetches only the requested subject's EEG members, each with one HTTP
    range request, inflates them and checks each member's CRC-32. That is
    about 386 MB per subject; neither the full archive nor the MEG is
    downloaded.

    References
    ----------
    .. [1] Corsi, M.-C., Gitton, C., Gonzalez-Astudillo, J., Kahn, A. E.,
           Hugueville, L., Schwartz, D., George, N., Chavez, M., Dupont, S.,
           Bassett, D. S., & De Vico Fallani, F. (2026). Understanding
           Brain-Computer Interfaces training: a longitudinal and multimodal
           dataset. Scientific Data.
           https://doi.org/10.1038/s41597-026-08237-5
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            channel_types={"eeg": 74},
            montage="standard_1005",
            hardware="Easycap 74-channel passive Ag/AgCl EEG with Elekta Neuromag MEG",
            cap_manufacturer="Easycap",
            electrode_type="passive",
            electrode_material="Ag/AgCl",
            reference="mastoids",
            ground="left scapula",
            impedance_threshold_kohm=20.0,
            line_freq=50.0,
            software="BCI2000",
        ),
        participants=ParticipantMetadata(
            n_subjects=_N_SUBJECTS,
            health_status="healthy",
            species="human",
            gender={"female": 7, "male": 12},
            handedness="right-handed",
            age_mean=27.47,
            age_std=4.07,
            age_min=19.0,
            age_max=35.0,
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=list(_EVENTS.keys()),
            trial_duration=6.0,
            study_design=(
                "Longitudinal BCI training: 4 sessions on 4 days. One-dimensional "
                "two-target box task; up target = sustained right-hand grasping "
                "motor imagery, down target = rest. Trial: 1 s ISI then 5 s target "
                "presentation with cursor feedback from 3 to 6 s. 6 feedback runs "
                "per session, 32 trials per run in the protocol."
            ),
            feedback_type="visual cursor",
            stimulus_type="visual target",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-026-08237-5",
            investigators=[
                "Marie-Constance Corsi",
                "Christophe Gitton",
                "Juliana Gonzalez-Astudillo",
                "Ari E. Kahn",
                "Laurent Hugueville",
                "Denis Schwartz",
                "Nathalie George",
                "Mario Chavez",
                "Sophie Dupont",
                "Danielle S. Bassett",
                "Fabrizio De Vico Fallani",
            ],
            senior_author="Fabrizio De Vico Fallani",
            institution="Paris Brain Institute (ICM)",
            country="FR",
            repository="Recherche Data Gouv",
            data_url="https://doi.org/10.57745/RBJRC7",
            publication_year=2026,
            license="CC-BY-4.0",
            ethics_approval=["CPP-IDF-VI of Paris, 2016-A00626-45"],
            related_paper_dois=["10.1142/S0129065718500144"],
        ),
        sessions_per_subject=_N_SESSIONS,
        runs_per_session=_N_RUNS,
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["right_hand", "rest"],
            cue_duration_s=5.0,
            imagery_duration_s=5.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=14431,
            n_trials_per_class={"right_hand": 7224, "rest": 7207},
            trials_context=(
                "6 online feedback runs per session x 4 sessions; the protocol has "
                "32 trials per run (16 per class); the released events hold the "
                "authors' checked trials: 375/456 runs keep 32, the rest 29-31 "
                "(357-384 trials per class per subject, 14,431 in total)."
            ),
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="cross-session", evaluation_type=["within_subject", "cross_session"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["motor_control", "neurofeedback"],
            environment="laboratory",
            online_feedback=True,
        ),
        file_format="BrainVision (BIDS)",
        data_processed=False,
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, _N_SUBJECTS + 1)),
            sessions_per_subject=_N_SESSIONS,
            events=dict(_EVENTS),
            code="Corsi2026",
            interval=[0, 5],
            paradigm="imagery",
            doi="10.1038/s41597-026-08237-5",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_path_search_params(self, subject):
        """Two-digit subject labels; EEG motor-imagery runs only (no MEG, no rest)."""
        out = {"datatypes": "eeg", "tasks": _TASK, "extensions": [".vhdr"]}
        if subject is not None:
            out["subjects"] = f"{subject:02d}"
        return out

    def _get_single_subject_data(self, subject):
        """Read each run's BrainVision file and set the events.tsv annotations."""
        bids_paths = self.bids_paths(subject)
        inv_events = {code: label for label, code in self.event_id.items()}
        montage = mne.channels.make_standard_montage("standard_1005")

        sessions = {}
        for bids_path in bids_paths:
            vhdr = Path(bids_path.fpath)
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
            events_tsv = vhdr.with_name(vhdr.name.replace("_eeg.vhdr", "_events.tsv"))
            # set_annotations *replaces* the BrainVision markers (same codes),
            # so every trial appears exactly once.
            raw.set_annotations(self._build_annotations(events_tsv, inv_events))
            # 16 of the 456 runs are not at 250 Hz: ten at 249.9 Hz and the
            # six of sub-09 ses-02 at 1000 Hz; their events.tsv samples follow
            # the stored rate. Their epochs would differ in length from the
            # rest, so resample them to the nominal rate. Annotations are in
            # seconds, so event timing is kept.
            if abs(raw.info["sfreq"] - _SFREQ) > 1e-6:
                log.info(
                    "%s is at %.4f Hz; resampling to %.0f Hz",
                    vhdr.name,
                    raw.info["sfreq"],
                    _SFREQ,
                )
                raw.resample(_SFREQ, verbose=False)
            # O9 and O10 have no standard_1005 position; they stay unplaced.
            raw.set_montage(montage, match_case=False, on_missing="ignore")
            raw = stim_channels_with_selected_ids(raw, self.event_id)

            session = bids_path.session if bids_path.session is not None else "0"
            run = bids_path.run if bids_path.run is not None else "0"
            sessions.setdefault(session, {})[run] = raw
        return sessions

    @staticmethod
    def _build_annotations(events_tsv, inv_events):
        """Annotations from a BIDS events.tsv (onsets already in seconds)."""
        # The released events.tsv files start with a UTF-8 byte-order mark.
        df = pd.read_csv(events_tsv, sep="\t", encoding="utf-8-sig")
        df = df[df["value"].isin(inv_events.keys())]
        return mne.Annotations(
            onset=df["onset"].astype(float).to_numpy(),
            duration=df["duration"].astype(float).to_numpy(),
            description=df["value"].map(inv_events).to_numpy(),
        )

    # ------------------------------------------------------------------ download --
    def _download_subject(self, subject, path, force_update, update_path, verbose) -> str:
        """Fetch one subject's EEG runs from the archive and return the BIDS root."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        bids_root = Path(get_dataset_path("Corsi2026", path)) / "MNE-corsi2026-data"
        bids_root.mkdir(parents=True, exist_ok=True)
        if not force_update and self._staged_subject_is_complete(bids_root, subject):
            log.info("Using locally staged Corsi2026 subject %02d", subject)
            return str(bids_root)

        session = requests.Session()
        for name, (file_id, md5) in _ROOT_FILES.items():
            target = bids_root / name
            if force_update or not target.is_file():
                url = f"{_DATAVERSE}/api/access/datafile/{file_id}"
                response = session.get(url, timeout=300)
                response.raise_for_status()
                got = hashlib.md5(response.content).hexdigest()
                if got != md5:
                    raise OSError(f"{name}: MD5 {got} != published {md5}")
                self._atomic_write(target, response.content)

        _, archive_id, _, archive_size = _ARCHIVE
        remote = _HTTPRangeFile(
            f"{_DATAVERSE}/api/access/datafile/{archive_id}", session, size=archive_size
        )
        with zipfile.ZipFile(remote) as archive:
            infos = self._subject_members(archive.infolist(), subject)
        if not infos:
            raise OSError(f"{_ARCHIVE[0]} holds no EEG files for sub-{subject:02d}")
        for info in infos:
            target = bids_root / info.filename
            if target.is_file() and not force_update:
                continue
            log.info("Extracting %s ...", info.filename)
            self._atomic_write(target, self._read_member(remote, info))
        log.info(
            "Corsi2026 subject %02d: %d members, %.0f MB in %d range requests",
            subject,
            len(infos),
            remote.bytes_fetched / 1e6,
            remote.n_requests,
        )
        return str(bids_root)

    @staticmethod
    def _read_member(remote, info):
        """Fetch, inflate and CRC-check one member with a single range request."""
        head = _LOCAL_HEADER.size + len(info.filename.encode()) + len(info.extra)
        blob = remote.read_range(
            info.header_offset, head + _HEADER_SLACK + info.compress_size
        )
        sig, *_, name_len, extra_len = _LOCAL_HEADER.unpack_from(blob)
        if sig != _LOCAL_HEADER_SIG:
            raise OSError(f"{info.filename}: bad local header at {info.header_offset}")
        start = _LOCAL_HEADER.size + name_len + extra_len
        data = blob[start : start + info.compress_size]
        if len(data) < info.compress_size:  # longer local extra than the slack
            data += remote.read_range(
                info.header_offset + start + len(data), info.compress_size - len(data)
            )
        if info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(data, -zlib.MAX_WBITS)
        elif info.compress_type != zipfile.ZIP_STORED:
            raise OSError(
                f"{info.filename}: unsupported compression {info.compress_type}"
            )
        if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise OSError(f"{info.filename}: size or CRC-32 mismatch after download")
        return data

    @staticmethod
    def _subject_members(infos, subject):
        """EEG members of one subject (MI runs + session sidecars), in archive order.

        Accepts :class:`zipfile.ZipInfo` objects or plain names; names are
        returned sorted, infos in ``header_offset`` order.
        """
        prefix = f"sub-{subject:02d}/"
        members = []
        for info in infos:
            name = getattr(info, "filename", info)
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            parts = name.split("/")
            if len(parts) != 4 or parts[2] != "eeg":
                continue
            fname = parts[3]
            if f"_task-{_TASK}_" in fname or fname.endswith(
                ("_electrodes.tsv", "_coordsystem.json")
            ):
                members.append(info)
        if members and isinstance(members[0], str):
            return sorted(members)
        return sorted(members, key=lambda i: i.header_offset)

    @staticmethod
    def _staged_subject_is_complete(bids_root, subject):
        """True when all 4 x 6 runs of the subject have every EEG sidecar."""
        subject_dir = Path(bids_root) / f"sub-{subject:02d}"
        vhdrs = sorted(subject_dir.glob(f"ses-*/eeg/*_task-{_TASK}_run-*_eeg.vhdr"))
        if len(vhdrs) != _N_SESSIONS * _N_RUNS:
            return False
        for vhdr in vhdrs:
            prefix = vhdr.name.removesuffix("eeg.vhdr")
            if not all((vhdr.parent / f"{prefix}{s}").is_file() for s in _RUN_SUFFIXES):
                return False
        return True

    @staticmethod
    def _atomic_write(target, data):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".part")
        temp.write_bytes(data)
        temp.replace(target)
