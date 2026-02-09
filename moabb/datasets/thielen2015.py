import mne
import numpy as np
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    FilterDetails,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PerformanceMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)
from moabb.datasets.utils import add_stim_channel_epoch, add_stim_channel_trial


Thielen2015_URL = "https://public.data.ru.nl/dcc/DSC_2018.00047_553_v3"

# Each session consisted of 3 fixed-length trials runs
NR_RUNS = 3

# Each trial contained 4 cycles of a 1.05 second code
NR_CYCLES_PER_TRIAL = 4

# Codes were presented at a 120 Hz monitor refresh rate
PRESENTATION_RATE = 120


class Thielen2015(BaseDataset):
    """c-VEP dataset from Thielen et al. (2015)

    Dataset [1]_ from the study on reconvolution for c-VEP [2]_.

    **Dataset description**

    EEG recordings were obtained with a sampling rate of 2048 Hz, using a setup comprising 64 Ag/AgCl electrodes, and
    amplified by a Biosemi ActiveTwo EEG amplifier. Electrode placement followed the international 10-10 system.

    During the experimental sessions, participants actively operated a 6 x 6 visual speller brain-computer interface
    (BCI) with real-time feedback, encompassing 36 distinct classes. Each cell within the symbol grid underwent
    luminance modulation at full contrast, achieved through the application of pseudo-random noise-codes derived from a
    set of modulated Gold codes. These binary codes have a balanced distribution of ones and zeros while adhering to a
    limited run-length pattern, with a maximum run-length of 2 bits. Codes were presented at a rate of 120 Hz. Given
    that one cycle of these modulated Gold codes comprises 126 bits, the duration of a complete cycle spans 1.05
    seconds.

    Throughout the experiment, participants underwent four distinct blocks: an initial practice block consisting of two
    runs, followed by a training block of one run. Subsequently, they engaged in a copy-spelling block comprising six
    runs, and finally, a free-spelling block consisting of one run. Between the training and copy-spelling block, a
    classifier was calibrated using data from the training block. This calibrated classifier was then applied during
    both the copy-spelling and free-spelling runs. Additionally, during calibration, the stimulation codes were
    tailored and optimized specifically for each individual participant.

    Among the six copy-spelling runs, there were three fixed-length runs. Trials in these runs started with a cueing
    phase, where the target symbol was highlighted in a green hue for 1 second. Participants maintained their gaze
    fixated on the target symbol as all symbols flashed in sync with their corresponding pseudo-random noise-codes for a
    duration of 4.2 seconds (equivalent to 4 code cycles). Immediately following this stimulation, the output of the
    classifier was shown by coloring the cell blue for 1 second. Each run consisted of 36 trials, presented in a
    randomized order.

    Here, our focus is solely on the three copy-spelling runs characterized by fixed-length trials lasting 4.2 seconds
    (equivalent to four code cycles). The other three runs utilized a dynamic stopping procedure, resulting in trials of
    varying durations, rendering them unsuitable for benchmarking purposes. Similarly, the practice and free-spelling
    runs included dynamic stopping and are ignored in this dataset. The training dataset, comprising 36 trials, used a
    different noise-code set, and is therefore also ignored in this dataset. In total, this dataset should contain 108
    trials of 4.2 seconds each, with 3 repetitions for each of the 36 codes.

    References
    ----------

    .. [1] Thielen, J. (Jordy), Jason Farquhar, Desain, P.W.M. (Peter) (2023): Broad-Band Visually Evoked Potentials:
           Re(con)volution in Brain-Computer Interfacing. Version 2. Radboud University. (dataset).
           DOI: https://doi.org/10.34973/1ecz-1232

    .. [2] Thielen, J., Van Den Broek, P., Farquhar, J., & Desain, P. (2015). Broad-Band visually evoked potentials:
           re(con)volution in brain-computer interfacing. PLOS ONE, 10(7), e0133797.
           DOI: https://doi.org/10.1371/journal.pone.0133797

    Notes
    -----

    .. versionadded:: 1.0.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="10-20",
            hardware="Biosemi ActiveTwo",
            sensor_type="active electrodes",
            reference="Car",
            sensors=[
                "Fp1",
                "Fpz",
                "Fp2",
                "AF7",
                "AF3",
                "AFz",
                "AF4",
                "AF8",
                "F7",
                "F5",
                "F3",
                "F1",
                "Fz",
                "F2",
                "F4",
                "F6",
                "F8",
                "FT7",
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "FT8",
                "T7",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "T8",
                "TP7",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "TP8",
                "P7",
                "P5",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "P6",
                "P8",
                "PO7",
                "PO3",
                "POz",
                "PO4",
                "PO8",
                "O1",
                "Oz",
                "O2",
                "Iz",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                other_physiological=["ppg"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="ALS",
            gender={"male": 4, "female": 8},
            age_mean=24.0,
        ),
        experiment=ExperimentMetadata(
            paradigm="cvep",
            n_classes=1,
            class_labels=["rest"],
            trial_duration=28.0,
            study_design="6x6 matrix speller BCI using modulated Gold codes for visual stimulation; participants focused on target symbols while cells flashed according to pseudo-random bit-sequences",
            feedback_type="visual colour feedback (cell coloured blue for classifier output, green/red edge colouring for certainty)",
            stimulus_type="rc_speller",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="asynchronous",
            mode="both",
            has_training_test_split=True,
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0133797",
            repository="GitHub",
            data_url="https://github.com/thijor/",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="preprocessed",
            preprocessing_applied=True,
            preprocessing_steps=[
                "downsampling from 2048 Hz to 360 Hz",
                "linear de-trending",
                "common average referencing",
                "spectral filtering",
            ],
            filter_details=FilterDetails(
                highpass_hz=5,
                lowpass_hz=100,
                bandpass=["5-48 Hz", "52-100 Hz"],
            ),
            artifact_methods=["ICA"],
            re_reference="car",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["SVM", "CCA"],
            feature_extraction=["Wavelet"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10-fold",
            cv_folds=10,
        ),
        performance=PerformanceMetadata(
            accuracy_percent=86.0,
        ),
        bci_application=BCIApplicationMetadata(
            applications=["speller", "prosthetic", "gaming", "vr_ar", "communication"],
            environment="outdoor",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="cvep",
        ),
        data_structure=DataStructureMetadata(
            n_trials=36,
            trials_context="total",
        ),
        data_processed=True,
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 12 + 1)),
            sessions_per_subject=1,
            events={"1.0": 101, "0.0": 100},
            code="Thielen2015",
            interval=(0, 0.3),
            paradigm="cvep",
            doi="10.34973/1ecz-1232",
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""
        file_path_list = self.data_path(subject)

        # Channels
        montage = mne.channels.read_custom_montage(file_path_list[-1])

        # There is only one session, each of 3 runs
        sessions = {"0": {}}
        for i_b in range(NR_RUNS):
            # EEG
            raw = mne.io.read_raw_gdf(
                file_path_list[2 * i_b],
                stim_channel="status",
                preload=True,
                verbose=False,
            )

            # Drop redundant ANA and EXG channels
            ana = [f"ANA{1 + i}" for i in range(32)]
            exg = [f"EXG{1 + i}" for i in range(8)]
            raw.drop_channels(ana + exg)

            # Set electrode positions
            raw.set_montage(montage)

            # Read info file
            tmp = loadmat(file_path_list[2 * i_b + 1])

            # Labels at trial level (i.e., symbols)
            trial_labels = tmp["labels"].astype("uint8").flatten() - 1

            # Codes (select optimized subset and layout, and repeat to trial length)
            subset = (
                tmp["subset"].astype("uint8").flatten() - 1
            )  # the optimized subset of 36 codes from a set of 65
            layout = (
                tmp["layout"].astype("uint8").flatten() - 1
            )  # the optimized position of the 36 codes in the grid
            codes = tmp["codes"][:, subset[layout]]
            codes = np.tile(codes, (NR_CYCLES_PER_TRIAL, 1))

            # Find onsets of trials
            events = mne.find_events(raw, verbose=False)
            trial_onsets = events[:, 0]

            # Create stim channel with trial information (i.e., symbols)
            # Specifically: 200 = symbol-0, 201 = symbol-1, 202 = symbol-2, etc.
            raw = add_stim_channel_trial(raw, trial_onsets, trial_labels, offset=200)

            # Create stim channel with epoch information (i.e., 1 / 0, or on / off)
            # Specifically: 100 = "0", 101 = "1"
            raw = add_stim_channel_epoch(
                raw, trial_onsets, trial_labels, codes, PRESENTATION_RATE, offset=100
            )

            # Add data as a new run
            run_name = str(i_b)
            sessions["0"][run_name] = raw

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the data paths of a single subject."""
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        sub = f"sub-{subject:02d}"
        subject_paths = []
        for i_b in range(NR_RUNS):
            blk = f"test_sync_{1 + i_b:d}"

            # EEG
            url = f"{Thielen2015_URL:s}/sourcedata/{sub}/{blk}/{sub}_{blk}.gdf"
            subject_paths.append(dl.data_dl(url, self.code, path, force_update, verbose))

            # Labels at trial level (i.e., symbols)
            url = f"{Thielen2015_URL:s}/sourcedata/{sub}/{blk}/{sub}_{blk}.mat"
            subject_paths.append(dl.data_dl(url, self.code, path, force_update, verbose))

        # Channel locations
        url = f"{Thielen2015_URL:s}/resources/biosemi64.loc"
        subject_paths.append(dl.data_dl(url, self.code, path, force_update, verbose))

        return subject_paths
