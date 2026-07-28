"""Song2026 multi-class motor-attempt dataset (chronic stroke, raw Neuracle NDF)."""

import re
import struct
import warnings
from collections import Counter
from pathlib import Path

import mne

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParticipantMetadata,
    Tags,
)

from ._neuracle_ndf.NDFSysMNE import mneNDF


# The dataset is distributed only through Baidu Netdisk (no direct HTTP mirror);
# the analysis code and a data pointer live at https://github.com/sjx66606/MVCMGNet.
SONG2026_BAIDU_URL = "https://pan.baidu.com/s/1ZNHmlcpRD9yDFok6NOv1mg"
SONG2026_BAIDU_CODE = "ma93"

# The 32 scalp-EEG channels in the Neuracle recording order (verified against the
# raw .nsf/.ndf headers). A1/A2 are the mastoids used as the reference in the
# authors' pop_reref (channels 17-18). The recording also carries 40 auxiliary
# channels (per-limb EMG + gyro/accel/magneto), which this loader drops.
SONG2026_CHANNELS = [
    "Fp1",
    "Fp2",
    "Fz",
    "F3",
    "F4",
    "F7",
    "F8",
    "FC1",
    "FC2",
    "FC5",
    "FC6",
    "Cz",
    "C3",
    "C4",
    "T7",
    "T8",
    "A1",
    "A2",
    "CP1",
    "CP2",
    "CP5",
    "CP6",
    "Pz",
    "P3",
    "P4",
    "P7",
    "P8",
    "PO3",
    "PO4",
    "Oz",
    "O1",
    "O2",
]

# The tens digit of each Neuracle marker identifies the motor-attempt task.
SONG2026_EVENTS = {"fist_clench": 2, "pinch_grip": 3, "wrist_lift": 4, "elbow_flexion": 5}
_TENS_TO_LABEL = {v: k for k, v in SONG2026_EVENTS.items()}

# Every raw recording contains three consecutive 40-trial blocks. The public
# preprocessing script relabels the middle block from *1 to *2 when it is
# followed by fixed-rate ``20`` flicker markers. The final *3 block is the one
# matching the paper's published 40-trial motor-attempt protocol: its ``20``
# markers follow the movement-specific looping image sequences. Loading all
# eight markers from the preprocessing script would silently mix the three
# experimental conditions and create 120 trials rather than the reported 40.
_CLASS_MARKERS = {"23", "33", "43", "53"}
_EXPECTED_TRIALS_PER_CLASS = 10


def _label_from_marker(marker):
    """Return the motor-attempt class for a published-protocol marker."""
    if marker not in _CLASS_MARKERS:
        return None
    return _TENS_TO_LABEL[int(marker[0])]


def _parse_nef(nef_path):
    """Parse a Neuracle ``.nef`` event file into ``[(onset_sample, marker), ...]``.

    The vendored reader's ``FileNEF`` miscomputes the frame stride (it reads
    ``event-count * 1004`` bytes where the on-disk stride is 1005) and silently
    fails on real event files, so parse it directly. Layout: a 64-byte header
    (0xFEFF BOM, record date/time, ``uint32`` event count at byte 29), then
    ``event-count`` frames of 1005 bytes: a 1-byte valid flag, ``uint32`` id /
    onset-sample / duration / colour (each padded to 6 bytes) and a
    null-delimited annotation string (the marker, e.g. ``"21"``).
    """
    raw = Path(nef_path).read_bytes()
    if len(raw) < 64:
        raise ValueError(f"Truncated Neuracle event header in {nef_path}")
    is_le = len(raw) > 1 and raw[0] == 0xFE and raw[1] == 0xFF
    u = "<I" if is_le else ">I"
    n_events = struct.unpack(u, raw[29:33])[0]
    expected_size = 64 + n_events * 1005
    if len(raw) < expected_size:
        raise ValueError(
            f"Truncated Neuracle event data in {nef_path}: expected at least "
            f"{expected_size} bytes, found {len(raw)}"
        )
    body, stride, out = raw[64:], 1005, []
    for i in range(n_events):
        p = i * stride
        if p + 23 > len(body) or not body[p]:
            continue
        onset = struct.unpack(u, body[p + 5 : p + 9])[0]
        ann = body[p + 23 : p + 23 + 982]
        if b"\x16" in ann:
            ann = ann[: ann.index(b"\x16")]
        parts = [x for x in ann.split(b"\x00") if x]
        if parts:
            out.append((onset, parts[0].decode("latin1", "ignore")))
    return out


class Song2026(BaseDataset):
    """Motor-attempt dataset from chronic stroke patients [1]_.

    **Dataset description**

    EEG (and synchronous 4-channel surface EMG) was recorded from 50 chronic
    stroke patients while they attempted four unilateral upper-limb movements
    of the affected side: fist clenching (Fc), pinch grip (Pg), wrist lift up
    (Wlu), and elbow flexion (Ef). The paper retained 45 participants after
    quality control but does not identify the five exclusions; this raw-data
    loader therefore exposes all 50 released participants. The task is framed
    both as a quad-task (4-class) problem and, pairwise, as a set of dual-task
    (2-class) problems. Signals were acquired with a 32-channel Neuracle system
    referenced to the bilateral mastoids at 1000 Hz. Each paradigm consists of
    10 trials of 10 s (5 s motor attempt followed by 5 s rest), preceded by a
    3-min baseline.

    Notes
    -----
    The data are distributed only through Baidu Netdisk
    (``pan.baidu.com/s/1ZNHmlcpRD9yDFok6NOv1mg``, code ``ma93``), which cannot be
    fetched programmatically. Download and extract the archive once under the
    dataset folder; see :meth:`data_path`. The raw recordings are in Neuracle
    NDF format (per-subject ``data/<session>/<condition>/`` folders holding an
    ``M_*`` EEG block plus per-limb EMG/IMU folders, with events in a
    session-level ``neuracle.nef``). They are read here via the vendored official
    Neuracle reader
    (``moabb/datasets/_neuracle_ndf``, from
    https://github.com/neuracle/neuracle-ndffile-reader): the 32 scalp-EEG
    channels are kept and scaled to volts. Only the final ``23/33/43/53`` block
    is exposed because it matches the paper's 40-trial motor-attempt protocol;
    the two auxiliary 40-trial blocks in the raw recording are intentionally
    excluded.

    References
    ----------

    .. [1] Song, J., Wang, N., Li, Z., Zhang, X., Lv, Z., Shan, X., Yang, Y.,
       Liu, J., & Chai, X. (2026). Decoding multi-class motor attempt from the
       affected unilateral limbs in chronic stroke patients. Journal of
       NeuroEngineering and Rehabilitation, 23(1), 109.
       DOI: https://doi.org/10.1186/s12984-026-01920-z

    .. versionadded:: 1.1.1

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=32,
            channel_types={"eeg": 32, "emg": 4},
            montage="10-10",
            hardware="Neuracle 32-channel EEG amplifier (with synchronous 4-channel EMG)",
            cap_manufacturer="Neuracle",
            reference="bilateral mastoids (A1, A2)",
            ground=None,
            sensors=SONG2026_CHANNELS,
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_emg=True,
                emg_channels=4,
                other_physiological=[
                    "EMG brachioradialis",
                    "EMG wrist flexors",
                    "EMG wrist extensors",
                    "EMG biceps brachii",
                ],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=50,
            health_status="chronic stroke patients",
            clinical_population="chronic stroke patients with affected unilateral upper limb",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=4,
            class_labels=["fist_clench", "pinch_grip", "wrist_lift", "elbow_flexion"],
            trials_per_class={
                "fist_clench": 10,
                "pinch_grip": 10,
                "wrist_lift": 10,
                "elbow_flexion": 10,
            },
            trial_duration=10.0,
            study_design="Motor attempt (attempted movement) of the affected unilateral upper limb: fist clenching, pinch grip, wrist lift up, elbow flexion. Each paradigm: 10 trials of 10 s (5 s attempt + 5 s rest), preceded by 3-min baseline.",
            feedback_type="none",
            stimulus_type="cue",
            synchronicity="cue-based",
            mode="offline",
            events=SONG2026_EVENTS,
        ),
        documentation=DocumentationMetadata(
            doi="10.1186/s12984-026-01920-z",
            description=(
                "EEG + EMG motor-attempt recordings from 50 chronic stroke "
                "patients performing four cued unilateral upper-limb tasks with "
                "the affected side; the paper retained 45 after quality control."
            ),
            investigators=[
                "Jiuxiang Song",
                "Nan Wang",
                "Zhaolin Li",
                "Xuemin Zhang",
                "Zeping Lv",
                "Xinying Shan",
                "Yi Yang",
                "Jizhong Liu",
                "Xiaoke Chai",
            ],
            senior_author="Jizhong Liu",
            institution="Beijing Tiantan Hospital, Capital Medical University",
            country="CN",
            data_url=SONG2026_BAIDU_URL,
            repository="Baidu Netdisk (code ma93); code at https://github.com/sjx66606/MVCMGNet",
            license="CC-BY-NC-ND-4.0",
            publication_year=2026,
            keywords=[
                "motor attempt",
                "attempted movement",
                "BCI",
                "brain-computer interface",
                "chronic stroke",
                "EEG",
                "EMG",
                "upper limb",
                "rehabilitation",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        file_format="NDF (Neuracle raw)",
        tags=Tags(pathology=["Stroke"], modality=["Motor"], type=["Motor Attempt"]),
        abstract=(
            "Multi-class motor-attempt decoding from the affected unilateral "
            "limbs of chronic stroke patients, using synchronous EEG and EMG. "
            "The release contains 50 participants; the paper analyzed 45 after "
            "quality control and reports 78.52% accuracy for dual-task (2-class) "
            "and 52.79% for quad-task (4-class) scenarios with MVCMGNet."
        ),
    )

    def __init__(self, subjects=None, sessions=None, **kwargs):
        super().__init__(
            subjects=list(range(1, 50 + 1)),
            sessions_per_subject=1,
            events=SONG2026_EVENTS,
            code="Song2026",
            interval=(0, 5),
            paradigm="imagery",
            doi="10.1186/s12984-026-01920-z",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the local path to the extracted ``data/`` tree.

        The recordings are distributed only through Baidu Netdisk
        (``pan.baidu.com/s/1ZNHmlcpRD9yDFok6NOv1mg``, code ``ma93``), which cannot
        be downloaded automatically. Download the archive manually and place it as
        ``data.zip`` under the following directory and extract it there exactly
        once::

            <mne_data>/MNE-song2026-data/

        The result must be ``<mne_data>/MNE-song2026-data/data/``. Extraction is
        deliberately not performed inside this subject-level method because
        parallel data loading could otherwise race and unpack the 4.6 GB archive
        more than once.

        Returns a one-element list with the path to the extracted ``data/``
        directory.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        root = Path(dl.get_dataset_path("Song2026", path)) / "MNE-song2026-data"
        data_dir = root / "data"
        if not data_dir.is_dir():
            archive = root / "data.zip"
            archive_note = (
                f" Archive {archive} exists but must be extracted once before loading."
                if archive.is_file()
                else ""
            )
            raise RuntimeError(
                f"Song2026 raw Neuracle data not found under {data_dir}. The dataset "
                f"is distributed only on Baidu Netdisk ({SONG2026_BAIDU_URL}, code "
                f"{SONG2026_BAIDU_CODE}); download it and place the archive as "
                f"{archive}, then extract it once in {root}.{archive_note}"
            )
        return [str(data_dir)]

    def _get_single_subject_data(self, subject):
        """Return ``{session: {run: mne.io.Raw}}`` for a single subject.

        Subjects map to the sorted per-session folders (``<14-digit>_<n>``). The
        primary condition folder (the one holding the ``M_*`` EEG block) is read
        via the official Neuracle reader, the 32 scalp channels are kept and
        scaled to volts, and the task cues from the session ``neuracle.nef`` are
        annotated with their class label.
        """
        data_dir = Path(self.data_path(subject)[0])
        sessions = {}
        for candidate in data_dir.iterdir():
            match = re.fullmatch(r"\d{14}_(\d+)", candidate.name)
            if candidate.is_dir() and match:
                sessions[int(match.group(1))] = candidate
        if subject not in sessions:
            raise FileNotFoundError(
                f"Song2026: no session folder for subject {subject} under {data_dir}"
            )
        sdir = sessions[subject]
        nef = sdir / "neuracle.nef"
        if not nef.is_file():
            raise FileNotFoundError(f"Song2026: missing event file {nef}")
        events = _parse_nef(nef)

        conds = sorted(
            c
            for c in sdir.iterdir()
            if c.is_dir() and any(m.name.startswith("M_") for m in c.iterdir())
        )
        if not conds:
            raise FileNotFoundError(f"Song2026: no EEG (M_*) block under {sdir}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mneNDF(str(conds[0]) + "/").read2MneRaw()
            missing = sorted(set(SONG2026_CHANNELS).difference(raw.ch_names))
            if missing:
                raise RuntimeError(
                    f"Song2026 subject {subject}: missing EEG channels {missing}"
                )
            raw.pick(SONG2026_CHANNELS)
            # The Neuracle reader returns physical values in microvolts; MNE uses
            # volts. Scale the full EEG matrix in one vectorized operation.
            raw.apply_function(lambda data: data * 1e-6, picks="eeg", channel_wise=False)
            raw.set_montage("standard_1005", on_missing="ignore", verbose=False)

        sfreq = raw.info["sfreq"]
        onsets, descriptions = [], []
        for onset, marker in events:
            label = _label_from_marker(marker)
            if label is not None and onset + 5 * sfreq <= raw.n_times:
                onsets.append(onset / sfreq)
                descriptions.append(label)

        counts = Counter(descriptions)
        expected = dict.fromkeys(SONG2026_EVENTS, _EXPECTED_TRIALS_PER_CLASS)
        if counts != expected:
            raise RuntimeError(
                f"Song2026 subject {subject}: expected published-protocol trial "
                f"counts {expected}, found {dict(counts)}"
            )
        raw.set_annotations(mne.Annotations(onsets, [5.0] * len(onsets), descriptions))
        return {"0": {"0": raw}}
