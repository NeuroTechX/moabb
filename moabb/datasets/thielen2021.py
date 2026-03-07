import h5py
import mne
import numpy as np
from scipy.io import loadmat

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
from moabb.datasets.utils import add_stim_channel_epoch, add_stim_channel_trial


Thielen2021_URL = "https://public.data.ru.nl/dcc/DSC_2018.00122_448_v3"

# The default electrode locations in the raw file are wrong. We used the ExG channels on the Biosemi with a custom 8
# channel set, according to an optimization as published in the following article:
# Ahmadi, S., Borhanazad, M., Tump, D., Farquhar, J., & Desain, P. (2019). Low channel count montages using sensor
# tying for VEP-based BCI. Journal of Neural Engineering, 16(6), 066038. DOI: https://doi.org/10.1088/1741-2552/ab4057
ELECTRODE_MAPPING = {
    "AF3": "Fpz",
    "F3": "T7",
    "FC5": "O1",
    "P7": "POz",
    "P8": "Oz",
    "FC6": "Iz",
    "F4": "O2",
    "AF4": "T8",
}

# Individual sessions of each of the 30 individual participants in the dataset
SESSIONS = (
    "20181128",
    "20181206",
    "20181217",
    "20181217",
    "20181217",
    "20181218",
    "20181218",
    "20181219",
    "20181219",
    "20181220",
    "20181220",
    "20181220",
    "20190107",
    "20190107",
    "20190110",
    "20190110",
    "20190110",
    "20190117",
    "20190117",
    "20190118",
    "20190118",
    "20190118",
    "20190220",
    "20190222",
    "20190225",
    "20190301",
    "20190307",
    "20190308",
    "20190311",
    "20190311",
)

# Each session consisted of 5 blocks (i.e., runs)
NR_BLOCKS = 5

# Each trial contained 15 cycles of a 2.1 second code
NR_CYCLES_PER_TRIAL = 15

# Codes were presented at a 60 Hz monitor refresh rate
PRESENTATION_RATE = 60


class Thielen2021(BaseDataset):
    """c-VEP dataset from Thielen et al. (2021)

    Dataset [1]_ from the study on zero-training c-VEP [2]_.

    **Dataset description**

    EEG recordings were acquired at a sampling rate of 512 Hz, employing 8 Ag/AgCl electrodes. The Biosemi ActiveTwo EEG
    amplifier was utilized during the experiment. The electrode array consisted of Fz, T7, O1, POz, Oz, Iz, O2, and T8,
    connected as EXG channels. This is a custom electrode montage as optimized in a previous study for c-VEP, see [3]_.

    During the experimental sessions, participants engaged in passive operation (i.e., without feedback) of a 4 x 5
    visual speller brain-computer interface (BCI) comprising 20 distinct classes. Each cell of the symbol grid
    underwent luminance modulation at full contrast, accomplished through pseudo-random noise-codes derived from a
    collection of modulated Gold codes. These codes are binary, have a balanced distribution of ones and zeros, and
    adhere to a limited run-length pattern (maximum run-length of 2 bits). The codes were presented at a presentation
    rate of 60 Hz. As one cycle of these modulated Gold codes contains 126 bits, the duration of one cycle is 2.1
    seconds.

    For each of the five blocks, a trial started with a cueing phase, during which the target symbol was highlighted in
    a green hue for a duration of 1 second. Following this, participants maintained their gaze fixated on the target
    symbol while all symbols flashed in accordance with their respective pseudo-random noise-codes for a duration of
    31.5 seconds (i.e., 15 code cycles). Each block encompassed 20 trials, presented in a randomized sequence, thereby
    ensuring that each symbol was attended to once within the span of a block.

    Note, here, we only load the offline data of this study and ignore the online phase.

    References
    ----------

    .. [1] Thielen, J. (Jordy), Pieter Marsman, Jason Farquhar, Desain, P.W.M. (Peter) (2023): From full calibration to
           zero training for a code-modulated visual evoked potentials brain computer interface. Version 3. Radboud
           University. (dataset).
           DOI: https://doi.org/10.34973/9txv-z787

    .. [2] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a
           code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5),
           056007.
           DOI: https://doi.org/10.1088/1741-2552/abecef

    .. [3] Ahmadi, S., Borhanazad, M., Tump, D., Farquhar, J., & Desain, P. (2019). Low channel count montages using
           sensor tying for VEP-based BCI. Journal of Neural Engineering, 16(6), 066038.
           DOI: https://doi.org/10.1088/1741-2552/ab4057

    Notes
    -----

    .. versionadded:: 0.6.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=8,
            channel_types={"eeg": 8},
            montage="custom",
            hardware="Biosemi ActiveTwo",
            sensor_type="sintered Ag/AgCl active electrodes",
            reference="CMS/DRL",
            sensors=[
                "Fpz",
                "Iz",
                "O1",
                "O2",
                "Oz",
                "POz",
                "T7",
                "T8",
            ],
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=30,
            health_status="healthy",
            gender={"female": 17, "male": 13},
            age_mean=25.0,
            age_min=19,
            age_max=62,
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            events={"1.0": 101, "0.0": 100},
            paradigm="cvep",
            n_classes=20,
            class_labels=None,
            trial_duration=31.5,
            study_design="Code-modulated visual evoked potentials BCI task where participants fixated on target cells in a calculator grid (offline) or keyboard layout (online) while all cells flashed with unique pseudo-random Gold code modulated bit-sequences",
            feedback_type="none",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            has_training_test_split=False,
            instructions="Participants maintained fixation at the target cell which was cued in green for 1 s before trial onset. No feedback was given after trials in the offline experiment.",
        ),
        documentation=DocumentationMetadata(
            doi="10.1088/1741-2552/abecef",
            investigators=["J Thielen", "P Marsman", "J Farquhar", "P Desain"],
            institution="Radboud University",
            institution_department="Donders Institute for Brain, Cognition and Behaviour",
            country="NL",
            data_url="https://doi.org/10.34973/9txv-z787",
            publication_year=2021,
            senior_author="P Desain",
            contact_info=["jordy.thielen@donders.ru.nl"],
            ethics_approval=[
                "Approved by the local ethical committee of the Faculty of Social Sciences of Radboud University"
            ],
            keywords=[
                "brain–computer interface (BCI)",
                "electroencephalography (EEG)",
                "code-modulated visual evoked potentials (cVEPs)",
                "reconvolution",
                "zero training",
                "spread spectrum communication",
            ],
            funding=[
                "NWO/TTW Takeoff Grant No. 14054",  # codespell:ignore nwo
                "International ALS Association and Dutch ALS Foundation Grant Nos. ATC20610 and 2017-57",
            ],
            license="CC0-1.0",
            repository="Radboud",
            associated_paper_doi="10.1088/1741-2552/ab4057",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Research"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["template-matching", "reconvolution", "CCA"],
            feature_extraction=["encoding model", "event responses", "spatio-temporal"],
            frequency_bands=None,
            spatial_filters=["CCA"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="cross-validation",
            cv_folds=5,
            evaluation_type=["within_session", "transfer_learning", "zero_training"],
        ),
        performance={
            "high_communication_rates": "achieved in online spelling task",
        },
        bci_application=BCIApplicationMetadata(
            applications=["speller"],
            environment="indoor",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="cvep",
            code_type="modulated Gold codes",
            code_length=126,
            n_targets=20,
            n_repetitions=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials=100,
            n_blocks=5,
            trials_context="per_subject (5 blocks × 20 trials each)",
        ),
        sessions_per_subject=1,
        runs_per_session=5,
        sessions=None,
        contributing_labs=["MindAffect", "Radboud University"],
        n_contributing_labs=2,
        data_processed=False,
        file_format="gdf",
        external_links={"source": "https://doi.org/10.34973/9txv-z787"},
        abstract="Objective. Typically, a brain–computer interface (BCI) is calibrated using user- and session-specific data because of the individual idiosyncrasies and the non-stationary signal properties of the electroencephalogram (EEG). Therefore, it is normal for BCIs to undergo a time-consuming passive training stage that prevents users from directly operating them. In this study, we systematically reduce the training data set in a stepwise fashion, to ultimately arrive at a calibration-free method for a code-modulated visually evoked potential (cVEP)-based BCI to fully eliminate the tedious training stage. Approach. In an extensive offline analysis, we compare our sophisticated encoding model with a traditional event-related potential (ERP) technique. We calibrate the encoding model in a standard way, with data limited to a single class while generalizing to all others and without any data. In addition, we investigate the feasibility of the zero-training cVEP BCI in an online setting. Main results. By adopting the encoding model, the training data can be reduced substantially, while maintaining both the classification performance as well as the explained variance of the ERP method. Moreover, with data from only one class or even no data at all, it still shows excellent performance. In addition, the zero-training cVEP BCI achieved high communication rates in an online spelling task, proving its feasibility for practical use. Significance. To date, this is the fastest zero-training cVEP BCI in the field, allowing high communication speeds without calibration while using only a few non-invasive water-based EEG electrodes. This allows us to skip the training stage altogether and spend all the valuable time on direct operation. This minimizes the session time and opens up new exciting directions for practical plug-and-play BCI. Fundamentally, these results validate that the adopted neural encoding model compresses data into event responses without the loss of explanatory power compared to using full ERPs as a template.",
        methodology="The study compared four training regimes: (1) e-train: traditional ERP template-matching with data from all classes, (2) n-train: encoding model (reconvolution) with data from all n classes, (3) 1-train: encoding model with data from only one class while generating templates for all sequences, (4) 0-train: zero-training encoding model requiring no calibration data. Offline experiment: 30 participants completed 5 blocks of 20 trials each (100 trials total), with 31.5 s trials using a 4×5 calculator grid (n=20 symbols). Stimuli were luminance-modulated pseudo-random Gold codes (126-bit sequences, 2.1 s duration) presented on an iPad Pro at 60 Hz. Online experiment: 11 participants (9 analyzed) used a keyboard layout (n=29 symbols) with dynamic stopping rule for spelling tasks. EEG recorded at 512 Hz from 8 electrodes, preprocessed with 2-30 Hz Butterworth filtering and downsampled to 120 Hz. Classification used template-matching with reconvolution encoding model that decomposes responses to sequences into linear sums of individual event responses.",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 30 + 1)),
            sessions_per_subject=1,
            events={"1.0": 101, "0.0": 100},
            code="Thielen2021",
            interval=(0, 0.3),
            paradigm="cvep",
            doi="10.34973/9txv-z787",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""
        file_path_list = self.data_path(subject)

        # Codes
        codes = np.tile(loadmat(file_path_list[-2])["codes"], (NR_CYCLES_PER_TRIAL, 1))

        # Channels
        montage = mne.channels.read_custom_montage(file_path_list[-1])

        # There is only one session, each of 5 blocks (i.e., runs)
        sessions = {"0": {}}
        for i_b in range(NR_BLOCKS):
            # EEG
            raw = mne.io.read_raw_gdf(
                file_path_list[2 * i_b],
                stim_channel="status",
                preload=True,
                verbose=False,
            )

            # The default electrode locations in the raw file are wrong. We used the ExG channels on the Biosemi with a
            # custom 8 channel set, according to an optimization as published in the following article:
            # Ahmadi, S., Borhanazad, M., Tump, D., Farquhar, J., & Desain, P. (2019). Low channel count montages using
            # sensor tying for VEP-based BCI. Journal of Neural Engineering, 16(6), 066038.
            # DOI: https://doi.org/10.1088/1741-2552/ab4057
            mne.rename_channels(raw.info, ELECTRODE_MAPPING)
            raw.set_montage(montage)

            # Labels at trial level (i.e., symbols)
            trial_labels = (
                np.array(h5py.File(file_path_list[2 * i_b + 1], "r")["v"])
                .astype("uint8")
                .flatten()
                - 1
            )

            # Find onsets of trials
            # Note, every 2.1 seconds an event was generated: 15 times per trial, plus one 16th "leaking epoch". This
            # "leaking epoch" is not always present, so taking epoch[::16, :] won't work.
            events = mne.find_events(raw, verbose=False)
            cond = np.logical_or(
                np.diff(events[:, 0]) < 1.8 * raw.info["sfreq"],
                np.diff(events[:, 0]) > 2.4 * raw.info["sfreq"],
            )
            idx = np.concatenate(([0], 1 + np.where(cond)[0]))
            trial_onsets = events[idx, 0]

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
        ses = SESSIONS[subject - 1]
        subject_paths = []
        for i_b in range(NR_BLOCKS):
            blk = f"block_{1 + i_b:d}"

            # EEG
            url = f"{Thielen2021_URL:s}/sourcedata/offline/{sub}/{blk}/{sub}_{ses}_{blk}_main_eeg.gdf"
            subject_paths.append(dl.data_dl(url, self.code, path, force_update, verbose))

            # Labels at trial level (i.e., symbols)
            url = f"{Thielen2021_URL:s}/sourcedata/offline/{sub}/{blk}/trainlabels.mat"
            subject_paths.append(dl.data_dl(url, self.code, path, force_update, verbose))

        # Codes
        url = f"{Thielen2021_URL:s}/resources/mgold_61_6521_flip_balanced_20.mat"
        subject_paths.append(dl.data_dl(url, self.code, path, force_update, verbose))

        # Channel locations
        url = f"{Thielen2021_URL:s}/resources/nt_cap8.loc"
        subject_paths.append(dl.data_dl(url, self.code, path, force_update, verbose))

        return subject_paths
