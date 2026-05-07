"""Pediatric SSVEP-based BCI dataset (Schrag et al. 2026).

Preprint DOI: 10.21203/rs.3.rs-9347306/v1
Data DOI:     10.5281/zenodo.19440997
"""

from __future__ import annotations

import csv
import logging
import os
import re
import shutil
import tempfile
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

# Keys must be strings; MOABB's SSVEP paradigm reads them as event names.
_GAME_EVENTS = {"6.25": 1, "10": 2, "11.11": 3, "14.28": 4}
_PERSONALIZATION_FREQ_KEY = "10"
_PERSONALIZATION_LABELS = frozenset(
    f"Contrast{c}Size{s}" for c in range(1, 5) for s in range(1, 4)
)

# Session keys; the leading integer encodes recording order
# (BaseDataset.check_session_names enforces the format).
_SESSION_STANDARD, _SESSION_PERSONAL, _SESSION_PERSONALIZATION = (
    "0standard",
    "1personal",
    "2personalization",
)

# Game XDF filename: sub-P###_ses-S001_task-T{2,3}_acq-{BW|CXSX}_M{1,2}_run-...
_GAME_FILENAME_RE = re.compile(
    r"sub-P(?P<subj>\d+)_ses-S\d+_task-T[23]_acq-(?P<acq>BW|C\dS\d)_M(?P<map>[12])_run-"
)
_TRIAL_START_MARKER = "Trial Started"
_BASELINE_END_MARKER = "getting score"

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


def _normalize_spo(value):
    """Map a movement-CSV ``Selected SPO`` cell to a ``_GAME_EVENTS`` key."""
    try:
        f = float((value or "").strip())
    except ValueError:
        return None
    return next((k for k in _GAME_EVENTS if abs(float(k) - f) < 0.05), None)


def _personalization_label(text):
    """Tail token of a T1 marker like ``ssvep,2,1,10, ContrastXSizeY``."""
    candidate = (text or "").rsplit(",", 1)[-1].strip()
    return candidate if candidate in _PERSONALIZATION_LABELS else None


def _movement_csv_for_eeg(eeg_path):
    """Sibling ``Movements/movements_P{N}_{acq}_M{m}.csv`` for a game XDF."""
    m = _GAME_FILENAME_RE.match(eeg_path.stem)
    if m is None:
        return None
    csv_path = (
        eeg_path.parent.parent
        / "Movements"
        / f"movements_P{m['subj']}_{m['acq']}_M{m['map']}.csv"
    )
    return csv_path if csv_path.is_file() else None


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

    By default this class exposes the SSVEP game runs only -- two sessions
    per subject (``"0standard"``, ``"1personal"``), 5 s trials at four
    target frequencies. Set ``include_personalization=True`` to also load
    the 12-stimulus personalization recording as session
    ``"2personalization"``; all its trials carry the ``"10"`` event since
    every personalization stimulus flickers at 10 Hz, conflating with the
    game's 10 Hz target if both are loaded.

    .. warning::
        Trial labels for the game sessions come from the recorded fbCCA
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
                "Each game session contains ~30-90 movement trials (one "
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
        sessions_per_subject=2,
        runs_per_session=1,
        file_format="XDF",
    )

    def __init__(self, subjects=None, sessions=None, *, include_personalization=False):
        self.include_personalization = bool(include_personalization)
        super().__init__(
            subjects=list(range(1, _N_SUBJECTS + 1)),
            sessions_per_subject=3 if include_personalization else 2,
            events=dict(_GAME_EVENTS),
            code="Schrag2026Pediatric",
            interval=[0.0, _TRIAL_DURATION_S],
            paradigm="ssvep",
            doi=_PREPRINT_DOI,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    # ------------------------------------------------------------------ #
    # Subject-level orchestration
    # ------------------------------------------------------------------ #

    def _get_single_subject_data(self, subject):
        eeg_dir = Path(self.data_path(subject)) / "EEG"
        if not eeg_dir.is_dir():
            raise FileNotFoundError(
                f"EEG directory missing for subject {subject}: {eeg_dir}"
            )

        wanted = self._wanted_session_keys()
        sessions = {}
        if wanted & {_SESSION_STANDARD, _SESSION_PERSONAL}:
            game_files = self._find_game_files(eeg_dir, subject)
            for sess_key, role in (
                (_SESSION_STANDARD, "standard"),
                (_SESSION_PERSONAL, "personal"),
            ):
                if sess_key not in wanted:
                    continue
                if role not in game_files:
                    log.warning(
                        "Subject %d: %s game XDF missing in %s", subject, role, eeg_dir
                    )
                    continue
                sessions[sess_key] = {"0": self._load_game_run(game_files[role])}

        if self.include_personalization and _SESSION_PERSONALIZATION in wanted:
            t1_path = next(eeg_dir.glob(f"sub-P{subject:03d}_*task-T1*_eeg.xdf"), None)
            if t1_path is None:
                log.warning(
                    "Subject %d: personalization (T1) XDF missing in %s", subject, eeg_dir
                )
            else:
                sessions[_SESSION_PERSONALIZATION] = {
                    "0": self._load_personalization_run(t1_path)
                }

        if not sessions:
            raise FileNotFoundError(f"No XDF files matched expected pattern in {eeg_dir}")
        return sessions

    def _wanted_session_keys(self):
        """Session keys to load, honouring ``selected_sessions`` (full or
        description-only form, mirroring ``BaseDataset.get_data``)."""
        all_keys = {_SESSION_STANDARD, _SESSION_PERSONAL}
        if self.include_personalization:
            all_keys.add(_SESSION_PERSONALIZATION)
        if self._selected_sessions is None:
            return all_keys
        wanted = {str(s) for s in self._selected_sessions}
        return {k for k in all_keys if k in wanted or k.lstrip("0123456789") in wanted}

    @staticmethod
    def _find_game_files(eeg_dir, subject):
        """Map ``"standard"``/``"personal"`` to T2/T3 XDFs."""
        out = {}
        for path in eeg_dir.glob(f"sub-P{subject:03d}_*task-T[23]*.xdf"):
            m = _GAME_FILENAME_RE.match(path.stem)
            if m is not None:
                # Regex constrains acq to BW or CXSX; else-branch is exhaustive.
                out["standard" if m["acq"] == "BW" else "personal"] = path
        return out

    # ------------------------------------------------------------------ #
    # XDF loaders
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_xdf_streams(fpath):
        """Return ``(eeg_stream, marker_stream)`` from an XDF.

        The marker stream is picked by name: each recording also ships an
        empty ``gUSBamp-1Markers`` stream that wins a type-based match in
        some files (it appears first in the XDF stream order).
        """
        pyxdf = _soft_import("pyxdf", "loading XDF data for Schrag2026Pediatric")
        streams, _ = pyxdf.load_xdf(
            str(fpath),
            select_streams=[{"type": "EEG"}, {"name": "UnityMarkerStream"}],
            verbose=False,
        )
        eeg = next((s for s in streams if s["info"]["type"][0] == "EEG"), None)
        markers = next(
            (s for s in streams if s["info"]["name"][0] == "UnityMarkerStream"), None
        )
        if eeg is None or markers is None:
            raise RuntimeError(f"Missing EEG or UnityMarkerStream in {fpath.name}")
        return eeg, markers

    @staticmethod
    def _unity_markers(marker_stream):
        ts = np.asarray(marker_stream["time_stamps"], dtype=float)
        texts = [str(p[0]) if p else "" for p in marker_stream["time_series"]]
        return ts, texts

    @classmethod
    def _build_raw(cls, eeg_stream, annotations):
        chans = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
        labels = [c["label"][0] for c in chans]
        try:
            idx = [labels.index(name) for name in _CH_NAMES]
        except ValueError as exc:
            raise RuntimeError(
                f"Channel not found in XDF stream (have: {labels})"
            ) from exc

        # gUSBamp samples are stored in microvolts; MNE expects volts.
        data = np.asarray(eeg_stream["time_series"]).T[idx, :] * 1e-6
        info = mne.create_info(_CH_NAMES, _SFREQ, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.set_montage(
            mne.channels.make_standard_montage("standard_1020"), on_missing="ignore"
        )

        if len(annotations):
            # XDF timestamps are absolute LSL seconds; MNE wants offsets from t0.
            t0 = float(eeg_stream["time_stamps"][0])
            raw.set_annotations(
                mne.Annotations(
                    onset=np.asarray(annotations.onset) - t0,
                    duration=annotations.duration,
                    description=annotations.description,
                )
            )
        return raw

    @classmethod
    def _load_game_run(cls, fpath):
        """SSVEP-game XDF -> Raw with one annotation per labelled trial.

        Onsets come from ``"Trial Started"`` markers, labels from the
        sibling ``movements_*.csv``'s ``Selected SPO`` column (live fbCCA
        output). Trials with empty SPO are dropped. Trials are paired with
        CSV rows by index; if the count drift exceeds 10 percent we drop
        the run's labels entirely rather than emit silently-shifted ones.
        """
        eeg_stream, marker_stream = cls._load_xdf_streams(fpath)
        marker_ts, marker_text = cls._unity_markers(marker_stream)
        starts = [i for i, t in enumerate(marker_text) if t == _TRIAL_START_MARKER]

        csv_path = _movement_csv_for_eeg(fpath)
        if csv_path is None:
            log.warning("No movement CSV alongside %s; trials unlabelled", fpath.name)
            spos = []
        else:
            with csv_path.open(newline="", encoding="utf-8-sig") as fh:
                spos = [
                    _normalize_spo(r.get("Selected SPO", "")) for r in csv.DictReader(fh)
                ]

        n = cls._safe_pair_count(fpath, len(starts), len(spos))
        onsets, descs = [], []
        for k in range(n):
            if spos[k] is not None:
                onsets.append(float(marker_ts[starts[k]]))
                descs.append(spos[k])

        return cls._build_raw(
            eeg_stream,
            mne.Annotations(
                onset=np.asarray(onsets, dtype=float),
                duration=np.full(len(onsets), _TRIAL_DURATION_S, dtype=float),
                description=np.asarray(descs, dtype="U16"),
            ),
        )

    @staticmethod
    def _safe_pair_count(fpath, n_trials, n_rows):
        """Min-truncate when CSV/marker drift is small; drop labels otherwise."""
        if not n_rows or not n_trials:
            return 0
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
        return min(n_trials, n_rows) if drift <= 0.10 else 0

    @classmethod
    def _load_personalization_run(cls, fpath):
        """T1 trials labelled with :data:`_PERSONALIZATION_FREQ_KEY`.

        Each ``ssvep,...,ContrastXSizeY`` marker opens a trial; duration
        runs to the next ``"getting score"`` marker. All 12 personalization
        stimuli flicker at 10 Hz, so the visual condition is not preserved
        as an event label -- consume the raw Unity marker stream directly
        for per-stimulus access.
        """
        eeg_stream, marker_stream = cls._load_xdf_streams(fpath)
        marker_ts, marker_text = cls._unity_markers(marker_stream)
        # Marker timestamps from pyxdf are sorted -> end_times stays sorted.
        end_mask = np.fromiter(
            (m == _BASELINE_END_MARKER for m in marker_text),
            dtype=bool,
            count=len(marker_text),
        )
        end_times = marker_ts[end_mask]

        onsets, durations = [], []
        for ts, text in zip(marker_ts, marker_text):
            if _personalization_label(text) is None:
                continue
            j = np.searchsorted(end_times, ts, side="right")
            if j < len(end_times):
                onsets.append(float(ts))
                durations.append(float(end_times[j] - ts))

        return cls._build_raw(
            eeg_stream,
            mne.Annotations(
                onset=np.asarray(onsets, dtype=float),
                duration=np.asarray(durations, dtype=float),
                description=np.asarray(
                    [_PERSONALIZATION_FREQ_KEY] * len(onsets), dtype="U16"
                ),
            ),
        )

    # ------------------------------------------------------------------ #
    # Download / extract
    # ------------------------------------------------------------------ #

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

        subject_dir = zip_path.parent / "DatasetData" / f"P{subject:03d}"
        if force_update or not (subject_dir / "EEG").is_dir():
            self._extract_subject(zip_path, subject)
        return str(subject_dir)

    @staticmethod
    def _extract_subject(zip_path, subject):
        """Atomically extract one subject's folder from the 1.2 GB archive.

        Stages into a sibling temp dir then ``os.replace``-s into place,
        so a concurrent worker observing ``P{nnn}/EEG/`` always sees a
        complete tree. A losing ``os.replace`` race is recovered by
        verifying the winner's tree exists.
        """
        prefix = f"DatasetData/P{subject:03d}/"
        final_root = zip_path.parent / "DatasetData"
        final_dir = final_root / f"P{subject:03d}"
        if (final_dir / "EEG").is_dir():
            return

        final_root.mkdir(parents=True, exist_ok=True)
        log.info("Extracting %s from %s", prefix, zip_path.name)
        # ``dir=`` keeps tmp on the same filesystem so os.replace is atomic.
        with tempfile.TemporaryDirectory(
            dir=zip_path.parent, prefix=f".extract_P{subject:03d}_"
        ) as tmp:
            tmp_path = Path(tmp)
            with zipfile.ZipFile(zip_path) as zf:
                members = [
                    info for info in zf.infolist() if info.filename.startswith(prefix)
                ]
                if not members:
                    raise FileNotFoundError(
                        f"No entries under {prefix!r} in {zip_path.name}"
                    )
                safe_extract_zip(zf, tmp_path, members=members)
            staged = tmp_path / "DatasetData" / f"P{subject:03d}"
            try:
                os.replace(staged, final_dir)
            except OSError:
                if not (final_dir / "EEG").is_dir():
                    shutil.copytree(staged, final_dir, dirs_exist_ok=True)
