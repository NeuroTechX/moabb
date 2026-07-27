"""Perdikis2018 CNBI EPFL Cybathlon BCI race motor-imagery dataset."""

import tarfile
import warnings
from pathlib import Path

import mne
from mne.channels import make_standard_montage

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
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

from .utils import safe_extract_tar


# Zenodo record 841764 (DOI 10.5281/zenodo.841764). Each pilot is a single
# gzip-compressed tar archive. The /records/<id>/files/<name> endpoint yields a
# distinct local filename per pilot, unlike the API /content endpoint which
# would collide on the name "content".
PERDIKIS2018_BASE = "https://zenodo.org/records/841764/files/{name}?download=1"

# The two Cybathlon 2016 BCI-race pilots (both tetraplegic). Subject index ->
# pilot code used both in the archive name and in the top-level folder.
_PILOTS = {1: "MA25VE", 2: "AN14VE"}

# 16 EEG electrodes over the sensorimotor cortex, in acquisition order. The GDF
# header stores only generic labels ("eeg:1" .. "eeg:16"); the electrode
# identities are taken from the recording's Laplacian montage grid (a 4x5 grid
# with one central frontal site plus full FC/C/CP rows) documented in the
# per-pilot classifier .mat and named in the paper (S3A Fig).
_EEG_CHANNELS = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
]

# Data-borne GDF event-type (TYP) codes for the class cues, mapped to labels.
# These are the standard BioSig/CNBI motor-imagery cue codes, verified from the
# real recordings: the 2-class online/race "bhbf" runs carry exactly {771, 773}
# and drop 783, which identifies 783 as the rest class; 771 is the standard
# "feet" code and 773 the remaining (both-hands) class.
_CLASS_CODES = {"771": "both_feet", "773": "both_hands", "783": "rest"}

# Single-limb / tongue MI cue codes (left hand, right hand, tongue). Their
# presence marks an exploratory taskset (e.g. mi_rlsf) outside the both-hands /
# both-feet / rest label space, so such offline runs are skipped.
_OTHER_MI_CODES = {"769", "770", "772"}


class Perdikis2018(BaseDataset):
    """CNBI EPFL Cybathlon BCI-race motor-imagery dataset [1]_.

    .. admonition:: Dataset summary

        ============ ======= ======= ========== ================= ============ ============ ===========
        Name         #Subj   #Chan   #Classes   #Trials/class     Trials len   Sampling     #Sessions
        ============ ======= ======= ========== ================= ============ ============ ===========
        Perdikis2018 2       16      3          ~15 per run       4 s          512 Hz       1
        ============ ======= ======= ========== ================= ============ ============ ===========

    **Dataset description**

    Longitudinal brain-computer interface (BCI) training and competition data
    from the two tetraplegic pilots (codes ``MA25VE`` and ``AN14VE``) of team
    Brain Tweakers (Defitech Chair in Brain-Machine Interface, CNBI, EPFL), who
    took part in the BCI-race discipline of the first Cybathlon (Zurich, October
    2016) [1]_. The pilots drove an avatar in the "BrainRunners" game by
    delivering sustained kinesthetic motor-imagery commands.

    EEG was recorded with a g.USBamp amplifier (g.tec, Austria) at 512 Hz from 16
    electrodes covering the sensorimotor cortex, stored in GDF format following
    the CNBI recording convention.

    This loader exposes the **offline calibration recordings** of the
    both-hands / both-feet / rest motor-imagery family. Runs are selected by
    their data-borne cue codes rather than by file name (the taskset strings are
    spelled inconsistently across pilots, e.g. ``mi_bhbfrst`` vs
    ``mi_bhbfrest``): a run is kept when its GDF cue codes fall within
    ``{771, 773, 783}`` (both-feet, both-hands, rest) and contain at least two of
    these classes. Runs whose cues include single-limb or tongue imagery
    (exploratory tasksets such as ``mi_rlsf``) are skipped, as are the
    ``incomplete`` and ``corrupted`` sub-folders.

    Each offline trial consists of a fixation cross (GDF code 786, ~3 s), a
    1-second class cue (771 / 773 / 783), and a continuous-feedback period (GDF
    code 781) of about 4 s during which the pilot performed the cued motor
    imagery. The class labels live in the GDF event table (annotations); events
    are anchored at the class-cue onset and the imagery interval is set to the
    1-5 s window following the cue (the calibration classifier used a 1-6 s
    analysis window).

    .. warning::

        The online closed-loop and race recordings of the same archives are
        continuous-control (BrainRunners game) recordings and are intentionally
        not exposed by this loader, which serves only the cue-based offline
        calibration runs with discrete, data-borne class labels.

    References
    ----------

    .. [1] Perdikis, S., Tonin, L., Saeedi, S., Schneider, C., & Millan, J. del
       R. (2018). The Cybathlon BCI race: Successful longitudinal mutual
       learning with two tetraplegic users. PLoS Biology, 16(5), e2003787.
       DOI: https://doi.org/10.1371/journal.pbio.2003787
       Data: https://doi.org/10.5281/zenodo.841764

    Notes
    -----

    .. versionadded:: 1.5.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=16,
            channel_types={"eeg": 16},
            montage="standard_1005",
            hardware="g.USBamp (g.tec medical engineering, Austria)",
            sensor_type="Ag/AgCl",
            reference=None,
            ground=None,
            sensors=list(_EEG_CHANNELS),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=2,
            health_status="spinal cord injury",
            clinical_population="tetraplegia (chronic spinal cord injury)",
            bci_experience="experienced",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=3,
            class_labels=["both_hands", "both_feet", "rest"],
            trial_duration=4.0,
            study_design="Longitudinal motor-imagery BCI training for the Cybathlon 2016 BCI race; two tetraplegic pilots delivered sustained kinesthetic motor-imagery commands (both hands, both feet, rest) to drive an avatar in the BrainRunners game.",
            feedback_type="visual",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="online",
            has_training_test_split=True,
            events={"both_feet": 771, "both_hands": 773, "rest": 783},
            instructions="Perform the cued kinesthetic motor imagery (both-hands, both-feet, or rest) to control the BrainRunners avatar.",
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.841764",
            description="EEG recordings and application logs from the two tetraplegic pilots of team Brain Tweakers (CNBI, EPFL) during longitudinal motor-imagery BCI training and the Cybathlon 2016 BCI race; this loader exposes the cue-based offline calibration runs (both-hands / both-feet / rest).",
            investigators=[
                "Serafeim Perdikis",
                "Luca Tonin",
                "Sareh Saeedi",
                "Christoph Schneider",
                "Jose del R. Millan",
            ],
            senior_author="Jose del R. Millan",
            institution="Defitech Chair in Brain-Machine Interface (CNBI), Center for Neuroprosthetics, Ecole Polytechnique Federale de Lausanne (EPFL)",
            country="CH",
            data_url="https://doi.org/10.5281/zenodo.841764",
            publication_year=2018,
            keywords=[
                "motor imagery",
                "brain-computer interface",
                "EEG",
                "Cybathlon",
                "BCI race",
                "spinal cord injury",
                "tetraplegia",
            ],
            license="CC-BY-4.0",
            repository="Zenodo",
            associated_paper_doi="10.1371/journal.pbio.2003787",
        ),
        sessions_per_subject=1,
        tags=Tags(
            pathology=["spinal cord injury"], modality=["motor"], type=["Motor Imagery"]
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw", preprocessing_applied=False
        ),
        signal_processing=SignalProcessingMetadata(
            frequency_bands={"mu": [8.0, 12.0], "beta": [12.0, 30.0]}
        ),
        cross_validation=CrossValidationMetadata(evaluation_type=["within_subject"]),
        bci_application=BCIApplicationMetadata(
            applications=["avatar control", "BCI game (BrainRunners)", "Cybathlon"],
            environment="lab and competition arena",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["both_hands", "both_feet", "rest"],
            cue_duration_s=1.0,
            imagery_duration_s=4.0,
        ),
        data_structure=DataStructureMetadata(
            trials_context="Offline calibration runs of the both-hands / both-feet / rest family, each with about 15 cued trials per class present; runs span several recording days per pilot and are pooled as runs of a single session."
        ),
        file_format="GDF",
        data_processed=False,
    )

    def __init__(self):
        super().__init__(
            subjects=[1, 2],
            sessions_per_subject=1,
            events={"both_feet": 771, "both_hands": 773, "rest": 783},
            code="Perdikis2018",
            interval=[1, 5],
            paradigm="imagery",
            doi="10.5281/zenodo.841764",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the path to the extracted per-pilot data folder.

        Downloads the pilot's tar.gz archive from Zenodo and extracts it if
        needed.

        Parameters
        ----------
        subject : int
            Subject number (1 for pilot MA25VE, 2 for pilot AN14VE).
        path : None | str
            Storage location override.
        force_update : bool
            Re-download even if a local copy exists.
        update_path : bool | None
            Unused, kept for API compatibility.
        verbose : bool, str, int, or None
            Verbosity level.

        Returns
        -------
        list of str
            Single-element list with the path to the extracted pilot folder.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        pilot = _PILOTS[subject]
        url = PERDIKIS2018_BASE.format(name=f"{pilot}.tar.gz")
        tar_path = Path(dl.data_dl(url, self.code, path, force_update, verbose))

        pilot_dir = tar_path.parent / pilot
        if not pilot_dir.is_dir():
            with tarfile.open(tar_path, "r:gz") as tf:
                safe_extract_tar(tf, tar_path.parent)

        return [str(pilot_dir)]

    def _get_single_subject_data(self, subject):
        """Return the offline motor-imagery data of a single subject.

        Parameters
        ----------
        subject : int
            Subject number (1 or 2).

        Returns
        -------
        dict
            ``{"0": {run_str: Raw}}`` -- one session whose runs are the pilot's
            cue-based offline calibration recordings (both-hands / both-feet /
            rest family), ordered chronologically.
        """
        pilot_dir = Path(self.data_path(subject)[0])

        # Offline calibration GDFs, excluding the incomplete/corrupted folders.
        offline_files = sorted(
            p
            for p in pilot_dir.rglob("*.offline.mi.*.gdf")
            if "incomplete" not in p.parts and "corrupted" not in p.parts
        )

        montage = make_standard_montage("standard_1005")
        runs = {}
        run_idx = 0
        for gdf_path in offline_files:
            raw = self._read_calibration_run(gdf_path, montage)
            if raw is not None:
                runs[str(run_idx)] = raw
                run_idx += 1

        if not runs:
            raise FileNotFoundError(
                f"No both-hands/both-feet/rest offline calibration runs found "
                f"for subject {subject} under {pilot_dir}"
            )

        return {"0": runs}

    def _read_calibration_run(self, gdf_path, montage):
        """Read one offline GDF run, or return ``None`` if it is out of scope.

        A run is kept only when its data-borne cue codes lie within the
        both-hands / both-feet / rest label space (``{771, 773, 783}``) and at
        least two of those classes are present.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_gdf(str(gdf_path), preload=False, verbose="ERROR")

        codes = set(raw.annotations.description)
        # Skip exploratory tasksets that use single-limb or tongue cues.
        if codes & _OTHER_MI_CODES:
            return None
        cue_present = codes & set(_CLASS_CODES)
        if len(cue_present) < 2:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.load_data(verbose="ERROR")

        # The GDF stores 16 EEG channels ("eeg:1".."eeg:16") plus a trigger
        # channel; the trigger's events have already been parsed into
        # annotations. Rename the EEG channels positionally to the sensorimotor
        # montage and drop everything else.
        eeg_chs = [ch for ch in raw.ch_names if ch.lower().startswith("eeg")]
        if len(eeg_chs) != len(_EEG_CHANNELS):
            raise ValueError(
                f"Expected {len(_EEG_CHANNELS)} EEG channels in {gdf_path.name}, "
                f"found {len(eeg_chs)}: {eeg_chs}"
            )
        raw.rename_channels(dict(zip(eeg_chs, _EEG_CHANNELS)))
        drop = [ch for ch in raw.ch_names if ch not in _EEG_CHANNELS]
        if drop:
            raw.drop_channels(drop)

        # Keep only the class-cue events, relabelled to class names.
        onset, duration, desc = [], [], []
        for ann in raw.annotations:
            label = _CLASS_CODES.get(str(ann["description"]))
            if label is not None:
                onset.append(ann["onset"])
                duration.append(ann["duration"])
                desc.append(label)
        raw.set_annotations(
            mne.Annotations(onset=onset, duration=duration, description=desc)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_montage(montage, match_case=False, on_missing="ignore")

        return raw
