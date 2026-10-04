"""Perdikis2018 CNBI EPFL Cybathlon BCI race motor-imagery dataset."""

import tarfile
import warnings
from pathlib import Path

import mne
import numpy as np
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


# Zenodo record 841764: one tar.gz per pilot. The /files/<name> endpoint gives a
# distinct local filename per pilot (the API /content endpoint would collide).
PERDIKIS2018_BASE = "https://zenodo.org/records/841764/files/{name}?download=1"

# Subject -> pilot code (archive name and top-level folder).
_PILOTS = {1: "MA25VE", 2: "AN14VE"}

# The GDF header only has "eeg:1".."eeg:16"; electrode names follow the
# Laplacian grid of the per-pilot classifier .mat and the paper (S3A Fig).
_EEG_CHANNELS = "Fz FC3 FC1 FCz FC2 FC4 C3 C1 Cz C2 C4 CP3 CP1 CPz CP2 CP4".split()

# GDF cue codes (BioSig/CNBI). The 2-class "bhbf" race runs carry only
# {771, 773}, identifying 783 as rest; 771 is the standard feet code.
_CLASS_CODES = {"771": "both_feet", "773": "both_hands", "783": "rest"}

# Left hand / right hand / tongue cues mark exploratory tasksets (e.g. mi_rlsf).
_OTHER_MI_CODES = {"769", "770", "772"}

# The GDF 2.x files declare ``uV`` only in the legacy dimension fields (numeric
# unit codes are zero), so MNE leaves the payload in microvolts.
PERDIKIS2018_EEG_SCALE_TO_VOLTS = 1e-6


class Perdikis2018(BaseDataset):
    """CNBI EPFL Cybathlon BCI-race motor-imagery dataset [1]_.

    **Dataset description**

    BCI training and competition data of the two tetraplegic pilots (``MA25VE``,
    ``AN14VE``) of team Brain Tweakers (CNBI, EPFL) in the BCI race of the first
    Cybathlon (2016) [1]_; 16 sensorimotor electrodes (g.USBamp, 512 Hz, GDF).

    Only the **offline calibration runs** of the both-hands / both-feet / rest
    family are exposed. Runs are selected by their data-borne cue codes, not by
    file name (taskset names are spelled inconsistently, e.g. ``mi_bhbfrst`` vs
    ``mi_bhbfrest``): a run is kept when its cues lie within ``{771, 773, 783}``
    and include at least two of them. Runs with single-limb or tongue cues (e.g.
    ``mi_rlsf``) and the ``incomplete`` / ``corrupted`` folders are skipped.
    Events are the class cues (after a ~3 s fixation, code 786), followed by ~4 s
    of feedback (781); the interval is 1-5 s after the cue. The GDF header only
    has generic ``eeg:<n>`` labels, so channels are renamed positionally, and
    the microvolt payload (unscaled by MNE) is converted to volts.

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

    .. versionadded:: 1.8

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
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
            class_labels=["both_feet", "both_hands", "rest"],
            trial_duration=4.0,
            study_design="Longitudinal motor-imagery BCI training for the Cybathlon 2016 BCI race; two tetraplegic pilots delivered sustained kinesthetic motor-imagery commands (both hands, both feet, rest) to drive an avatar in the BrainRunners game.",
            feedback_type="visual",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            synchronicity="cue-based",
            mode="offline",
            has_training_test_split=False,
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
        """Download/extract the pilot's archive; return ``[pilot_folder]``."""
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        pilot = _PILOTS[subject]
        url = PERDIKIS2018_BASE.format(name=f"{pilot}.tar.gz")
        tar_path = Path(dl.data_dl(url, self.code, path, force_update, verbose))

        pilot_dir = tar_path.parent / pilot
        if force_update or not pilot_dir.is_dir():
            with tarfile.open(tar_path, "r:gz") as tf:
                safe_extract_tar(tf, tar_path.parent)

        return [str(pilot_dir)]

    def _get_single_subject_data(self, subject):
        """Return ``{"0": {run: Raw}}`` with the in-scope offline runs, in order."""
        pilot_dir = Path(self.data_path(subject)[0])

        # Offline calibration GDFs, excluding the incomplete/corrupted folders.
        offline_files = sorted(
            p
            for p in pilot_dir.rglob("*.offline.mi.*.gdf")
            if "incomplete" not in p.parts and "corrupted" not in p.parts
        )

        montage = make_standard_montage("colin27_1005")
        raws = [self._read_calibration_run(p, montage) for p in offline_files]
        runs = {str(i): raw for i, raw in enumerate(r for r in raws if r is not None)}
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
        raw.apply_function(
            lambda data: data * PERDIKIS2018_EEG_SCALE_TO_VOLTS,
            picks=eeg_chs,
            channel_wise=False,
        )
        raw.rename_channels(dict(zip(eeg_chs, _EEG_CHANNELS)))
        raw.drop_channels([ch for ch in raw.ch_names if ch not in _EEG_CHANNELS])
        _keep_class_cues(raw, _CLASS_CODES)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw.set_montage(montage, match_case=False, on_missing="ignore")

        return raw


def _keep_class_cues(raw, class_codes):
    """Keep only the GDF class-cue annotations, relabelled to class names."""
    ann = raw.annotations
    keep = np.array([str(d) in class_codes for d in ann.description], dtype=bool)
    raw.set_annotations(
        mne.Annotations(
            onset=ann.onset[keep],
            duration=ann.duration[keep],
            description=[class_codes[str(d)] for d in ann.description[keep]],
        )
    )
    return raw
