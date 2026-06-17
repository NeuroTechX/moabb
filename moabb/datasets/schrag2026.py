"""Pediatric SSVEP-based BCI dataset (Schrag et al. 2026).

Preprint DOI: 10.21203/rs.3.rs-9347306/v1
Data DOI:     10.5281/zenodo.19440997
"""

from __future__ import annotations

import csv
import logging
import re
import zipfile
from pathlib import Path

import mne
import numpy as np
from mne.utils import _soft_import

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)
from .utils import safe_extract_zip


log = logging.getLogger(__name__)

_SIGN = "Schrag2026"
_ZENODO_URL = "https://zenodo.org/api/records/19440997/files/DatasetData.zip/content"
_DOI = "10.5281/zenodo.19440997"
_PREPRINT_DOI = "10.21203/rs.3.rs-9347306/v1"

# 16 channels in the order recorded by g.USBamp (Schrag 2026, sec. EEG Acquisition).
_CH_NAMES = [
    "Fz",
    "F4",
    "F8",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "P4",
    "P8",
    "PO7",
    "PO8",
    "O1",
    "Oz",
    "O2",
]
_SFREQ = 256.0
_N_SUBJECTS = 47
_TRIAL_DURATION_S = 5.0

# 4-target SSVEP game frequencies, as strings (MOABB uses the keys as event names).
_GAME_EVENTS = {"6.25": 1, "10": 2, "11.11": 3, "14.28": 4}
_PERSONALIZATION_FREQ = "10"  # all 12 personalization stimuli flicker at 10 Hz
_PERSONALIZATION_LABELS = [
    f"Contrast{c}Size{s}" for c in range(1, 5) for s in range(1, 4)
]

# Run keys must start with an integer (BaseDataset.check_run_names).
_R_STD, _R_PERS, _R_PERSO = "0standard", "1personal", "2personalization"

# Game XDF filename: sub-P###_ses-S001_task-T{2,3}_acq-{BW|CXSX}_M{1,2}_run-...
_GAME_FILE_RE = re.compile(r"sub-P(\d+)_ses-S\d+_task-T[23]_acq-(BW|C\dS\d)_M([12])_run-")

# Per-subject demographics (47 rows from Participant_Demographic_Info.csv).
# fmt: off
_AGES = [12, 10, 12, 13, 16, 14,  9, 17, 16,  8, 11,  9, 10, 17, 17,  6,
         16, 17, 15, 10, 12,  8, 11, 17, 17, 17,  9, 17, 17, 17, 17, 18,
         10, 12, 11,  9, 17, 14,  5, 15, 15, 16,  5,  8,  6,  7, 10]
_M, _F = "male", "female"
_SEXES = [_M, _F, _F, _M, _M, _M, _M, _M, _M, _F, _M, _M, _M, _F, _F, _F,
          _F, _F, _M, _F, _M, _M, _M, _M, _F, _F, _M, _F, _F, _F, _F, _F,
          _M, _M, _M, _M, _M, _M, _M, _M, _M, _F, _M, _F, _M, _F, _M]
# fmt: on


class Schrag2026Pediatric(BaseDataset):
    """SSVEP-based BCI dataset in children and adolescents (Schrag et al. 2026).

    Dataset from [1]_, hosted on Zenodo [2]_.

    Forty-seven neurotypical children and adolescents (ages 5-18, mean
    12.6 +/- 3.9 yr; 40.4% female) recorded with a g.tec g.GAMMAsys gel-based
    system (16 scalp channels at 256 Hz, ground Fpz, earlobe reference)
    completed a two-stage SSVEP-BCI session: (a) a *personalization
    pipeline* presenting 12 visual stimuli (4 contrasts x 3 sizes) all
    flickering at 10 Hz, and (b) an *online 4-target SSVEP game* at
    6.25 / 10 / 11.11 / 14.28 Hz, played twice (once with the personal
    stimulus, once with a high-contrast standard) across two themed maps.

    By default this class exposes the SSVEP game runs only -- a single
    session per subject holding two runs (``"0standard"``, ``"1personal"``),
    5 s trials at four target frequencies. Set
    ``include_personalization=True`` to also load the 12-stimulus
    personalization recording as a third run ``"2personalization"``; all
    its trials carry the ``"10"`` event since every personalization
    stimulus flickers at 10 Hz, conflating with the game's 10 Hz target if
    both are loaded.

    .. warning::
        Trial labels for the game runs come from the recorded fbCCA
        classifier output (the ``Selected SPO`` column of the per-game
        movement CSV) -- the frequency *the system identified* during the
        live game, which then drove avatar movement. They are **not**
        ground-truth target frequencies; treating ``y`` as such biases
        benchmarks toward fbCCA's behaviour. For ground-truth labels parse
        ``Intended Movement Direction`` together with the per-trial
        corner-to-frequency mapping (randomised across the game; not
        currently exposed by this loader).

    .. note::
        The dataset ships as a single ~1.2 GB ``DatasetData.zip`` on
        Zenodo. Subjects are extracted on demand; ``pyxdf`` is required
        (``pip install moabb[xdf]``).

    References
    ----------
    .. [1] E. Schrag, D. Comaduran Marquez, A. Kirton, and E. Kinney-Lang,
       "A steady-state visual evoked potential-based brain-computer
       interface dataset in children and adolescents," Research Square
       preprint, 2026. DOI: 10.21203/rs.3.rs-9347306/v1
    .. [2] Schrag et al., 2026 SSVEP Pediatric Dataset.
       Zenodo. DOI: 10.5281/zenodo.19440997
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=_SFREQ,
            n_channels=16,
            channel_types={"eeg": 16},
            montage="standard_1020",
            hardware="g.tec g.GAMMAsys + g.USBamp + g.GAMMAcap",
            sensors=list(_CH_NAMES),
            reference="earlobe",
            ground="Fpz",
            line_freq=60.0,
            sensor_type="active",
            electrode_type="wet",
            electrode_material="Ag/AgCl gel",
            cap_manufacturer="g.tec",
            software="Unity3D + BCI-Essentials",
        ),
        participants=ParticipantMetadata(
            n_subjects=_N_SUBJECTS,
            health_status="healthy",
            age_mean=12.6,
            age_std=3.9,
            age_min=5,
            age_max=18,
            ages=list(_AGES),
            sexes=list(_SEXES),
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="ssvep",
            events=dict(_GAME_EVENTS),
            n_classes=4,
            class_labels=list(_GAME_EVENTS),
            trial_duration=_TRIAL_DURATION_S,
            stimulus_type="flickering visual targets (4-target game)",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
            task_type="SSVEP-controlled videogame (4-target navigation)",
            feedback_type="visual",
            study_design=(
                "Per-subject pipeline: (1) personalization (12 stimuli at "
                "10 Hz, 5 s on / 5 s baseline / pairwise comfort, ~20 sets), "
                "(2) online 4-target SSVEP game played twice -- personal "
                "stimulus and standard high-contrast stimulus across two "
                "themed maps."
            ),
        ),
        documentation=DocumentationMetadata(
            doi=_DOI,
            associated_paper_doi=_PREPRINT_DOI,
            data_url=f"https://doi.org/{_DOI}",
            investigators=[
                "Emily Schrag",
                "Daniel Comaduran Marquez",
                "Adam Kirton",
                "Eli Kinney-Lang",
            ],
            senior_author="Eli Kinney-Lang",
            institution="University of Calgary",
            country="CA",
            repository="Zenodo",
            # Zenodo deposit registers cc-by-nd-4.0; the preprint PDF says
            # CC-BY-4.0. Zenodo metadata is authoritative for the data.
            license="CC-BY-ND-4.0",
            publication_year=2026,
            ethics_approval=[
                "University of Calgary Conjoint Health Research Ethics Board, REB25-0723"
            ],
            keywords=[
                "SSVEP",
                "BCI",
                "pediatric",
                "children",
                "adolescents",
                "stimulus personalization",
                "comfort",
                "EEG",
            ],
            description=(
                "Open-access pediatric SSVEP-BCI dataset: 47 "
                "children aged 5-18 performing a personalization "
                "pipeline and an online 4-target SSVEP game with "
                "both personal and standard stimuli."
            ),
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw", preprocessing_applied=False
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep", stimulus_frequencies_hz=[6.25, 10.0, 11.11, 14.28]
        ),
        data_structure=DataStructureMetadata(
            n_trials=12,
            trials_context=(
                "Each game session contains ~30-160 movement trials (one "
                "per 5 s SSVEP stimulation period). Of these, exactly 12 "
                "are ground-truth target events (4 frequencies x 3 "
                "predefined target positions, minus skipped events on "
                "certain map layouts; see Notes.pdf in the Zenodo "
                "deposit). The remaining trials are user-driven movements "
                "whose labels are fbCCA classifier outputs, not ground "
                "truth -- this loader exposes all classifier-labelled "
                "trials with non-empty Selected SPO."
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["fbCCA"],
            feature_extraction=["fbCCA"],
            frequency_bands={"analysis": [3.0, 29.0]},
        ),
        bci_application=BCIApplicationMetadata(
            environment="lab", online_feedback=True, applications=["navigation game"]
        ),
        tags=Tags(pathology=["healthy"], modality=["visual"], type=["perception"]),
        sessions_per_subject=1,
        runs_per_session=2,
        file_format="XDF",
    )

    def __init__(self, subjects=None, sessions=None, *, include_personalization=False):
        self.include_personalization = bool(include_personalization)
        super().__init__(
            subjects=list(range(1, _N_SUBJECTS + 1)),
            sessions_per_subject=1,
            events=dict(_GAME_EVENTS),
            code="Schrag2026Pediatric",
            interval=[0.0, _TRIAL_DURATION_S],
            paradigm="ssvep",
            doi=_PREPRINT_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    # ----- Subject loading ----------------------------------------------

    def _get_single_subject_data(self, subject):
        eeg_dir = Path(self.data_path(subject)) / "EEG"
        if not eeg_dir.is_dir():
            raise FileNotFoundError(f"EEG dir missing for subject {subject}: {eeg_dir}")

        # One visit per subject = one session; the game is played twice (standard
        # and personal stimulus) so those are two runs, matched by acq tag.
        runs = {}
        for path in eeg_dir.glob(f"sub-P{subject:03d}_*task-T[23]*.xdf"):
            m = _GAME_FILE_RE.match(path.stem)
            if m is None:
                continue
            key = _R_STD if m.group(2) == "BW" else _R_PERS
            runs[key] = _load_game_run(path)

        # Personalization (T1) is a single XDF per subject, opt-in, third run.
        if self.include_personalization:
            t1_path = next(eeg_dir.glob(f"sub-P{subject:03d}_*task-T1*_eeg.xdf"), None)
            if t1_path is None:
                log.warning(
                    "Subject %d: personalization (T1) XDF missing in %s", subject, eeg_dir
                )
            else:
                runs[_R_PERSO] = _load_personalization_run(t1_path)

        if not runs:
            raise FileNotFoundError(f"No XDF files matched expected pattern in {eeg_dir}")
        return {"0": runs}

    # ----- Download / extract -------------------------------------------

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number: {subject}")

        zip_path = Path(dl.data_dl(_ZENODO_URL, _SIGN, path, force_update, verbose))
        # data_dl strips file extensions; rename so zipfile opens by name.
        if zip_path.suffix != ".zip":
            target = zip_path.with_suffix(".zip")
            if not target.exists():
                zip_path.rename(target)
            zip_path = target

        # The single ~1.2 GB archive holds every subject; extract just the one
        # requested on demand instead of unpacking all 47.
        subject_dir = zip_path.parent / "DatasetData" / f"P{subject:03d}"
        if force_update or not (subject_dir / "EEG").is_dir():
            prefix = f"DatasetData/P{subject:03d}/"
            # ponytail: plain extract, no temp-dir staging. Two workers
            # unpacking the same subject concurrently could race; restore
            # atomic os.replace staging if data_path ever runs in parallel.
            with zipfile.ZipFile(zip_path) as zf:
                members = [i for i in zf.infolist() if i.filename.startswith(prefix)]
                if not members:
                    raise FileNotFoundError(
                        f"No entries under {prefix!r} in {zip_path.name}"
                    )
                safe_extract_zip(zf, zip_path.parent, members=members)
        return str(subject_dir)


# ----- Helpers (module level so they're easy to read top-to-bottom) -----


def _load_xdf_streams(fpath):
    """Return ``(eeg_stream, marker_stream)`` from an XDF.

    The marker stream is picked by name: each recording also ships an
    empty ``gUSBamp-1Markers`` stream that wins a type-based ``"Markers"``
    match in some files (it appears first in the XDF stream order).
    """
    pyxdf = _soft_import("pyxdf", "loading XDF data for Schrag2026Pediatric")
    streams, _ = pyxdf.load_xdf(
        str(fpath),
        select_streams=[{"type": "EEG"}, {"name": "UnityMarkerStream"}],
        verbose=False,
    )
    eeg_stream = marker_stream = None
    for s in streams:
        if s["info"]["type"][0] == "EEG":
            eeg_stream = s
        if s["info"]["name"][0] == "UnityMarkerStream":
            marker_stream = s
    if eeg_stream is None or marker_stream is None:
        raise RuntimeError(f"Missing EEG or UnityMarkerStream in {fpath.name}")
    return eeg_stream, marker_stream


def _read_unity_markers(marker_stream):
    """Return ``(marker_ts, markers)`` arrays from a UnityMarkerStream payload."""
    marker_ts = np.asarray(marker_stream["time_stamps"], dtype=float)
    markers = [str(p[0]) if p else "" for p in marker_stream["time_series"]]
    return marker_ts, markers


def _build_raw(eeg_stream, annotations):
    """Build an :class:`mne.io.RawArray` from an XDF EEG stream + annotations.

    Reorders the XDF channels to match :data:`_CH_NAMES`, scales microvolts
    to volts, sets the standard 10-20 montage, and shifts annotation onsets
    so they are relative to the recording start (XDF stamps are absolute LSL
    seconds; MNE wants offsets from t0).
    """
    chans = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
    labels = [c["label"][0] for c in chans]
    try:
        idx = [labels.index(name) for name in _CH_NAMES]
    except ValueError as exc:
        raise RuntimeError(f"Channel not found in XDF stream (have: {labels})") from exc

    # gUSBamp samples are stored in microvolts; MNE expects volts.
    data = np.asarray(eeg_stream["time_series"]).T[idx, :] * 1e-6
    info = mne.create_info(_CH_NAMES, _SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(
        mne.channels.make_standard_montage("standard_1020"), on_missing="ignore"
    )

    if len(annotations):
        t0 = float(eeg_stream["time_stamps"][0])
        raw.set_annotations(
            mne.Annotations(
                onset=np.asarray(annotations.onset) - t0,
                duration=annotations.duration,
                description=annotations.description,
            )
        )
    return raw


def _load_game_run(fpath):
    """SSVEP-game XDF -> Raw with one annotation per labelled trial.

    Trial onsets come from ``"Trial Started"`` Unity markers; labels come
    from the matching ``movements_*.csv``'s ``Selected SPO`` column (the
    live fbCCA classifier output that drove the avatar). Trials with empty
    SPO (blocked moves) are dropped.

    Trials are paired with CSV rows by index. Some sessions have a few
    extra trailing CSV rows from end-of-game bookkeeping; if the count
    drift is large (>10 percent) we drop the run's labels entirely rather
    than emit silently-shifted ones.
    """
    eeg_stream, marker_stream = _load_xdf_streams(fpath)
    marker_ts, markers = _read_unity_markers(marker_stream)
    trial_starts = [i for i, m in enumerate(markers) if m == "Trial Started"]

    # Find the sibling movements CSV using the EEG filename.
    m = _GAME_FILE_RE.match(fpath.stem)
    csv_path = (
        (
            fpath.parent.parent
            / "Movements"
            / f"movements_P{m.group(1)}_{m.group(2)}_M{m.group(3)}.csv"
        )
        if m
        else None
    )

    spos = []
    if csv_path is None or not csv_path.is_file():
        log.warning("No movement CSV alongside %s; trials unlabelled", fpath.name)
    else:
        with csv_path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                spos.append(_match_freq(row.get("Selected SPO", "")))

    # Decide how many trial<->row pairs to trust.
    n_trials, n_rows = len(trial_starts), len(spos)
    if not (n_trials and n_rows):
        n = 0
    else:
        drift = abs(n_rows - n_trials) / n_trials
        if n_rows != n_trials:
            (log.warning if drift <= 0.10 else log.error)(
                "Trial/CSV mismatch in %s: %d 'Trial Started' markers vs "
                "%d rows (%.0f%% drift)%s",
                fpath.name,
                n_trials,
                n_rows,
                drift * 100,
                "; truncating" if drift <= 0.10 else "; dropping labels",
            )
        n = min(n_trials, n_rows) if drift <= 0.10 else 0

    onsets, descs = [], []
    for k in range(n):
        if spos[k] is not None:
            onsets.append(float(marker_ts[trial_starts[k]]))
            descs.append(spos[k])

    return _build_raw(
        eeg_stream,
        mne.Annotations(
            onset=np.asarray(onsets, dtype=float),
            duration=np.full(len(onsets), _TRIAL_DURATION_S, dtype=float),
            description=np.asarray(descs, dtype="U16"),
        ),
    )


def _load_personalization_run(fpath):
    """T1 XDF -> Raw with one ``"10"``-labelled annotation per stimulus presentation.

    Each ``ssvep,...,ContrastXSizeY`` Unity marker opens a trial; the
    duration runs to the next ``"getting score"`` marker (which signals
    the start of the rating phase). All 12 personalization stimuli flicker
    at 10 Hz, so the visual condition is not preserved as an event label
    -- consume the raw Unity marker stream directly for per-stimulus access.
    """
    eeg_stream, marker_stream = _load_xdf_streams(fpath)
    marker_ts, markers = _read_unity_markers(marker_stream)

    # End-of-stim markers, in chronological order (pyxdf emits sorted timestamps).
    end_times = np.asarray(
        [t for t, m in zip(marker_ts, markers) if m == "getting score"], dtype=float
    )

    onsets, durations = [], []
    for ts, marker in zip(marker_ts, markers):
        # Markers look like "ssvep,2,1,10, ContrastXSizeY"; we only care about
        # the trailing stimulus token.
        token = marker.rsplit(",", 1)[-1].strip()
        if token not in _PERSONALIZATION_LABELS:
            continue
        j = np.searchsorted(end_times, ts, side="right")
        if j < len(end_times):
            onsets.append(float(ts))
            durations.append(float(end_times[j] - ts))

    return _build_raw(
        eeg_stream,
        mne.Annotations(
            onset=np.asarray(onsets, dtype=float),
            duration=np.asarray(durations, dtype=float),
            description=np.asarray([_PERSONALIZATION_FREQ] * len(onsets), dtype="U16"),
        ),
    )


def _match_freq(text):
    """Map a ``Selected SPO`` cell (e.g. ``"6.25"``, ``"11.11"``) to a game-event key."""
    try:
        f = float((text or "").strip())
    except ValueError:
        return None
    for key in _GAME_EVENTS:
        if abs(float(key) - f) < 0.05:
            return key
    return None
