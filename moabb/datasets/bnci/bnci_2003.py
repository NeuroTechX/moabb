"""BNCI 2003 datasets."""

from mne.utils import verbose

from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
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

from .base import MNEBNCI, _convert_bbci2003, _finalize_raw, data_path
from .utils import validate_subject


@verbose
def _load_data_iva_2003(
    subject,
    path=None,
    force_update=False,
    update_path=None,
    base_url=None,
    only_filenames=False,
    verbose=None,
):
    """Loads data for the BNCI2003-IVa dataset."""
    validate_subject(subject, 5, "BNCI2003-004")

    subject_names = ["aa", "al", "av", "aw", "ay"]

    # fmt: off
    ch_names = ['Fp1', 'AFp1', 'Fpz', 'AFp2', 'Fp2', 'AF7', 'AF3',
                'AF4', 'AF8', 'FAF5', 'FAF1', 'FAF2', 'FAF6', 'F7',
                'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FFC7',
                'FFC5', 'FFC3', 'FFC1', 'FFC2', 'FFC4', 'FFC6', 'FFC8',
                'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
                'FC6', 'FT8', 'FT10', 'CFC7', 'CFC5', 'CFC3', 'CFC1',
                'CFC2', 'CFC4', 'CFC6', 'CFC8', 'T7', 'C5', 'C3', 'C1',
                'Cz','C2', 'C4', 'C6', 'T8', 'CCP7', 'CCP5', 'CCP3', 'CCP1',
                'CCP2', 'CCP4', 'CCP6', 'CCP8', 'TP9', 'TP7', 'CP5', 'CP3',
                'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'PCP7',
                'PCP5', 'PCP3', 'PCP1', 'PCP2', 'PCP4', 'PCP6', 'PCP8', 'P9',
                'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10',
                'PPO7', 'PPO5', 'PPO1', 'PPO2', 'PPO6', 'PPO8', 'PO7', 'PO3',
                'PO1', 'POz', 'PO2', 'PO4', 'PO8', 'OPO1', 'OPO2', 'O1',
                'Oz', 'O2', 'OI1', 'OI2', 'I1', 'I2']
    # fmt: on
    ch_type = ["eeg"] * 118

    url = "{u}download/competition_iii/berlin/100Hz/data_set_IVa_{r}_mat.zip".format(
        u=base_url, r=subject_names[subject - 1]
    )

    filename = data_path(url, path, force_update, update_path)

    if only_filenames:
        return filename

    runs, ev = _convert_bbci2003(filename[0], ch_names, ch_type)
    _finalize_raw(runs, "BNCI2003-004", subject)

    session = {"0train": {"0": runs}}
    return session


class BNCI2003_004(MNEBNCI):
    """
    BNCI2003_IVa Motor Imagery dataset.

    Dataset IVa from BCI Competition III [1]_.

    **Dataset Description**

    This data set was recorded from five healthy subjects. Subjects sat in
    a comfortable chair with arms resting on armrests. This data set
    contains only data from the 4 initial sessions without feedback.
    Visual cues indicated for 3.5 s which of the following 3 motor
    imageries the subject should perform: (L) left hand, (R) right hand,
    (F) right foot. The presentation of target cues were intermitted by
    periods of random length, 1.75 to 2.25 s, in which the subject could
    relax.

    There were two types of visual stimulation: (1) where targets were
    indicated by letters appearing behind a fixation cross (which might
    nevertheless induce little target-correlated eye movements), and (2)
    where a randomly moving object indicated targets (inducing target-
    uncorrelated eye movements). From subjects al and aw 2 sessions of
    both types were recorded, while from the other subjects 3 sessions
    of type (2) and 1 session of type (1) were recorded.

    References
    ----------
    .. [1] Guido Dornhege, Benjamin Blankertz, Gabriel Curio, and Klaus-Robert
           Muller. Boosting bit rates in non-invasive EEG single-trial
           classifications by feature combination and multi-class paradigms.
           IEEE Trans. Biomed. Eng., 51(6):993-1002, June 2004.

    Notes
    -----
    .. versionadded:: 0.4.0

    This is one of the earliest and most influential motor imagery BCI datasets,
    used extensively for benchmarking classification algorithms. The dataset
    was part of BCI Competition III and has been cited in hundreds of papers.

    See Also
    --------
    BNCI2014_001 : BCI Competition IV 4-class motor imagery dataset
    BNCI2014_004 : BCI Competition 2008 2-class motor imagery dataset
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=100.0,
            n_channels=118,
            channel_types={"eeg": 118},
            hardware="multichannel EEG amplifiers",
            reference="Car",
            filters={"bandpass": [0.05, 200]},
            sensors=[
                "AF3",
                "AF4",
                "AF7",
                "AF8",
                "AFp1",
                "AFp2",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "CCP1",
                "CCP2",
                "CCP3",
                "CCP4",
                "CCP5",
                "CCP6",
                "CCP7",
                "CCP8",
                "CFC1",
                "CFC2",
                "CFC3",
                "CFC4",
                "CFC5",
                "CFC6",
                "CFC7",
                "CFC8",
                "CP1",
                "CP2",
                "CP3",
                "CP4",
                "CP5",
                "CP6",
                "CPz",
                "Cz",
                "F1",
                "F2",
                "F3",
                "F4",
                "F5",
                "F6",
                "F7",
                "F8",
                "FAF1",
                "FAF2",
                "FAF5",
                "FAF6",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "FCz",
                "FFC1",
                "FFC2",
                "FFC3",
                "FFC4",
                "FFC5",
                "FFC6",
                "FFC7",
                "FFC8",
                "FT10",
                "FT7",
                "FT8",
                "FT9",
                "Fp1",
                "Fp2",
                "Fpz",
                "Fz",
                "I1",
                "I2",
                "O1",
                "O2",
                "OI1",
                "OI2",
                "OPO1",
                "OPO2",
                "Oz",
                "P1",
                "P10",
                "P2",
                "P3",
                "P4",
                "P5",
                "P6",
                "P7",
                "P8",
                "P9",
                "PCP1",
                "PCP2",
                "PCP3",
                "PCP4",
                "PCP5",
                "PCP6",
                "PCP7",
                "PCP8",
                "PO1",
                "PO2",
                "PO3",
                "PO4",
                "PO7",
                "PO8",
                "POz",
                "PPO1",
                "PPO2",
                "PPO5",
                "PPO6",
                "PPO7",
                "PPO8",
                "Pz",
                "T7",
                "T8",
                "TP10",
                "TP7",
                "TP8",
                "TP9",
            ],
            line_freq=50.0,
            sensor_type="EEG",
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_type=["horizontal", "vertical"],
                has_emg=True,
                other_physiological=["gsr"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="healthy",
        ),
        experiment=ExperimentMetadata(
            events={"right_hand": 0, "feet": 1},
            paradigm="imagery",
            n_classes=3,
            class_labels=["right_hand", "left_hand", "feet"],
            trial_duration=4.5,
            stimulus_type="cursor_feedback",
            mode="both",
            stimulus_presentation={
                "duration": "3 s",
                "interval": "4.5 s",
                "modality": "visual (letter on screen)",
            },
            instructions="subjects were instructed to imagine movement or sensation according to displayed letter",
        ),
        documentation=DocumentationMetadata(
            doi="10.1109/TBME.2004.827088",
            investigators=[
                "Guido Dornhege",
                "Benjamin Blankertz",
                "Gabriel Curio",
                "Klaus-Robert Müller",
            ],
            institution="Fraunhofer FIRST (IDA); Charité University Medicine Berlin",
            country="Germany",
            publication_year=2004,
            funding=[
                "Bundesministerium für Bildung und Forschung (BMBF) under Grants FKZ 01IBB02A and FKZ 01IBB02B"
            ],
            keywords=[
                "brain-computer interface",
                "BCI",
                "common spatial patterns",
                "electroencephalogram",
                "EEG",
                "event-related desynchronization",
                "feature combination",
                "movement related potential",
                "multiclass",
                "single-trial analysis",
            ],
            institution_address="12489 Berlin, Germany; 12203 Berlin, Germany",
            institution_department="Fraunhofer FIRST (IDA); Department of Neurology, Campus Benjamin Franklin",
            senior_author="Klaus-Robert Müller",
            contact_info=["guido.dornhege@first.fraunhofer.de"],
            license="CC-BY-4.0",
            repository="BBCI",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="downsampled to 100 Hz for offline analysis",
            preprocessing_applied=True,
            preprocessing_steps=[
                "downsampling",
                "baseline correction",
                "spatial Laplacian filtering",
                "bandpass filtering",
            ],
            highpass_hz=4.0,
            lowpass_hz=2.5,
            bandpass={"low_cutoff_hz": 0.05, "high_cutoff_hz": 200.0},
            filter_type="causal elliptic IIR",
            artifact_methods=["ICA"],
            re_reference="car",
            downsampled_to_hz=100,
            notes="surface EMG at both forearms and one leg, as well as horizontal and vertical EOG signals were recorded to check for muscle activation and eye movements, but no trial was rejected",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA", "SVM", "Shrinkage LDA"],
            feature_extraction=["CSP", "ERD", "Covariance/Riemannian", "AR"],
            frequency_bands={
                "alpha": [8, 13],
            },
            spatial_filters=["CSP", "spatial Laplacian"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="leave-one-out",
            cv_folds=10,
            evaluation_type=["10x10-fold cross validation"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=[
                "speller",
                "wheelchair/navigation",
                "prosthetic",
                "vr_ar",
                "communication",
            ],
            environment="outdoor",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=[
                "left_hand",
                "right_hand",
                "feet",
                "visual",
                "auditory",
                "tactile",
            ],
            cue_duration_s=3.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=200,
            trials_context="total",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        data_processed=True,
        file_format="gdf",
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 6)),
            sessions_per_subject=1,
            events={"right_hand": 0, "feet": 1},
            code="BNCI2003-004",
            interval=[0, 3.5],
            paradigm="imagery",
            doi="10.1109/TBME.2004.827088",
        )
