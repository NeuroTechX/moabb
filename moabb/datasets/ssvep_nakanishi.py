"""SSVEP Nakanishi dataset."""

import logging

import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

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

from . import download as dl
from .base import BaseDataset


log = logging.getLogger(__name__)

NAKAHISHI_URL = "https://github.com/mnakanishi/12JFPM_SSVEP/raw/master/data/"


class Nakanishi2015(BaseDataset):
    """SSVEP Nakanishi 2015 dataset.

    This dataset contains 12-class joint frequency-phase modulated steady-state
    visual evoked potentials (SSVEPs) acquired from 10 subjects used to
    estimate an online performance of brain-computer interface (BCI) in the
    reference study [1]_.

    references
    ----------

    .. [1] Masaki Nakanishi, Yijun Wang, Yu-Te Wang and Tzyy-Ping Jung,
           "A Comparison Study of Canonical Correlation Analysis Based Methods for
           Detecting Steady-State Visual Evoked Potentials," PLoS One, vol.10, no.10,
           e140703, 2015.
           `<http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140703>`_
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=8,
            channel_types={"eeg": 8},
            hardware="Biosemi ActiveTwo",
            reference="CMS/DRL",
            software=None,
            sensors=["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"],
            line_freq=60.0,
            electrode_type=None,
            montage="standard_1020",
            sensor_type="EEG",
        ),
        participants=ParticipantMetadata(
            n_subjects=10,
            health_status="healthy",
            gender={"male": 9, "female": 1},
            age_mean=28.0,
            bci_experience="not specified",
        ),
        experiment=ExperimentMetadata(
            events={
                "9.25": 1,
                "11.25": 2,
                "13.25": 3,
                "9.75": 4,
                "11.75": 5,
                "13.75": 6,
                "10.25": 7,
                "12.25": 8,
                "14.25": 9,
                "10.75": 10,
                "12.75": 11,
                "14.75": 12,
            },
            paradigm="ssvep",
            n_classes=12,
            class_labels=[
                "9.25Hz",
                "9.75Hz",
                "10.25Hz",
                "10.75Hz",
                "11.25Hz",
                "11.75Hz",
                "12.25Hz",
                "12.75Hz",
                "13.25Hz",
                "13.75Hz",
                "14.25Hz",
                "14.75Hz",
            ],
            trial_duration=4.0,
            study_design="12-class SSVEP target identification task with joint frequency and phase coding",
            stimulus_type="flickering",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            has_training_test_split=False,
            instructions="Subjects were asked to gaze at one of the visual stimuli indicated by the stimulus program in a random order for 4s. At the beginning of each trial, a red square appeared for 1s at the position of the target stimulus. Subjects were asked to shift their gaze to the target within the same 1s duration. After that, all stimuli started to flicker simultaneously for 4s.",
            stimulus_presentation={
                "SoftwareName": "MATLAB with Psychophysics Toolbox",
                "monitor": "ASUS VG278 27-inch LCD",
                "refresh_rate": "60Hz",
                "resolution": "1280x800 pixels",
                "stimulus_size": "6x6 cm each",
                "viewing_distance": "60cm",
                "arrangement": "4x3 matrix virtual keypad",
            },
            feedback_type="none",
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0140703",
            funding=[
                "Swartz Foundation gift fund",
                "U.S. Office of Naval Research (N00014-08-1215)",
                "Army Research Office (W911NF-09-1-0510)",
                "Army Research Laboratory (W911NF-10-2-0022)",
                "DARPA (USDI D11PC20183)",
                "UC Proof of Concept Grant Award (269228)",
                "NIH Grant (1R21EY025056-01)",
                "Recruitment Program for Young Professionals",
            ],
            investigators=[
                "Masaki Nakanishi",
                "Yijun Wang",
                "Yu-Te Wang",
                "Tzyy-Ping Jung",
            ],
            institution="University of California San Diego",
            institution_department="Swartz Center for Computational Neuroscience, Institute for Neural Computation; Center for Advanced Neurological Engineering, Institute of Engineering in Medicine",
            country="US",
            data_url="https://github.com/mnakanishi/12JFPM_SSVEP/raw/master/data/",
            publication_year=2015,
            contact_info=["wangyj@semi.ac.cn"],
            ethics_approval=[
                "Human Research Protections Program of the University of California San Diego"
            ],
            description="A comparison study of canonical correlation analysis based methods for detecting steady-state visual evoked potentials. This study performed a comparison of existing CCA-based SSVEP detection methods using a 12-class SSVEP dataset recorded from 10 subjects in a simulated online BCI experiment.",
            keywords=[
                "SSVEP",
                "BCI",
                "CCA",
                "canonical correlation analysis",
                "brain-computer interface",
                "steady-state visual evoked potentials",
            ],
            license="Unknown",
            repository="Github",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        data_processed=True,
        file_format="mat",
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Research"],
        ),
        preprocessing=PreprocessingMetadata(
            bandpass={"low_cutoff_hz": 6.0, "high_cutoff_hz": 80.0},
            filter_type="IIR",
            downsampled_to_hz=256.0,
            preprocessing_applied=True,
            preprocessing_steps=["downsampling", "bandpass filtering"],
            notes="Zero-phase forward and reverse IIR filtering was implemented using the filtfilt() function in MATLAB. Data epochs were extracted with a 135-ms latency delay considering the visual system delay.",
            epoch_window=[0.135, 4.135],
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=[
                "CCA",
                "IT-CCA",
                "MwayCCA",
                "L1-MCCA",
                "MsetCCA",
                "CACC",
                "Combination Method",
            ],
            feature_extraction=["CCA", "canonical correlation"],
            spatial_filters=["CCA"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="leave-one-block-out",
            cv_folds=15,
            evaluation_type=["cross_validation"],
        ),
        performance={
            "accuracy_percent": 92.78,
            "itr_bits_per_min": 91.68,
            "r_square": 0.87,
            "combination_method_accuracy_1s": 92.78,
            "combination_method_itr_1s": 91.68,
            "standard_cca_accuracy_1s": 55.00,
            "standard_cca_itr_2s": 50.40,
        },
        bci_application=BCIApplicationMetadata(
            applications=["communication"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep",
            stimulus_frequencies_hz=[
                9.25,
                9.75,
                10.25,
                10.75,
                11.25,
                11.75,
                12.25,
                12.75,
                13.25,
                13.75,
                14.25,
                14.75,
            ],
            frequency_resolution_hz=0.5,
            n_targets=12,
            code_type="joint frequency and phase coding",
        ),
        data_structure=DataStructureMetadata(
            n_trials=180,
            n_blocks=15,
            trials_context="15 blocks x 12 trials per block = 180 trials total per subject",
        ),
        abstract="Canonical correlation analysis (CCA) has been widely used in the detection of the steady-state visual evoked potentials (SSVEPs) in brain-computer interfaces (BCIs). The standard CCA method, which uses sinusoidal signals as reference signals, was first proposed for SSVEP detection without calibration. However, the detection performance can be deteriorated by the interference from the spontaneous EEG activities. Recently, various extended methods have been developed to incorporate individual EEG calibration data in CCA to improve the detection performance. Although advantages of the extended CCA methods have been demonstrated in separate studies, a comprehensive comparison between these methods is still missing. This study performed a comparison of the existing CCA-based SSVEP detection methods using a 12-class SSVEP dataset recorded from 10 subjects in a simulated online BCI experiment. Classification accuracy and information transfer rate (ITR) were used for performance evaluation. The results suggest that individual calibration data can significantly improve the detection performance. Furthermore, the results showed that the combination method based on the standard CCA and the individual template based CCA (IT-CCA) achieved the highest performance.",
        methodology="A simulated online BCI experiment was conducted with 10 subjects. Each subject completed 15 blocks, with each block containing 12 trials (one for each of the 12 targets). Visual stimuli were presented as a 4x3 matrix on a 27-inch LCD monitor at 60Hz refresh rate. The 12 targets used joint frequency and phase coding (frequencies: 9.25-14.75Hz with 0.5Hz intervals; phases: 0 to 5.5π with 0.5π intervals). Each trial began with a 1s cue (red square) followed by 4s of flickering stimulation. EEG was recorded from 8 occipital electrodes at 2048Hz and downsampled to 256Hz for analysis. Seven CCA-based methods were compared using leave-one-block-out cross-validation (14 blocks for training, 1 for testing). Performance was evaluated using classification accuracy and ITR.",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=1,
            events={
                "9.25": 1,
                "11.25": 2,
                "13.25": 3,
                "9.75": 4,
                "11.75": 5,
                "13.75": 6,
                "10.25": 7,
                "12.25": 8,
                "14.25": 9,
                "10.75": 10,
                "12.75": 11,
                "14.75": 12,
            },
            code="Nakanishi2015",
            interval=[0.15, 4.3],
            paradigm="ssvep",
            doi="10.1371/journal.pone.0140703",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""
        n_samples, n_channels, n_trials = 1114, 8, 15
        n_classes = len(self.event_id)

        fname = self.data_path(subject)
        mat = loadmat(fname, squeeze_me=True)
        data = np.transpose(mat["eeg"], axes=(0, 3, 1, 2))
        data = np.reshape(data, (-1, n_channels, n_samples))
        data = data - data.mean(axis=2, keepdims=True)
        raw_events = np.zeros((data.shape[0], 1, n_samples))
        raw_events[:, 0, 0] = np.array(
            [n_trials * [i + 1] for i in range(n_classes)]
        ).flatten()
        data = np.concatenate([1e-6 * data, raw_events], axis=1)
        # add buffer in between trials
        log.warning(
            "Trial data de-meaned and concatenated with a buffer"
            " to create continuous data"
        )
        buff = (data.shape[0], n_channels + 1, 50)
        data = np.concatenate([np.zeros(buff), data, np.zeros(buff)], axis=2)
        ch_names = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "stim"]
        ch_types = ["eeg"] * 8 + ["stim"]
        sfreq = 256
        info = create_info(ch_names, sfreq, ch_types)
        raw = RawArray(data=np.concatenate(list(data), axis=1), info=info, verbose=False)
        montage = make_standard_montage("standard_1005")
        raw.set_montage(montage)
        return {"0": {"0": raw}}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        url = "{:s}s{:d}.mat".format(NAKAHISHI_URL, subject)
        return dl.data_dl(url, "NAKANISHI", path, force_update, verbose)
