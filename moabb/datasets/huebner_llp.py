import logging
import re
import zipfile
from abc import ABC
from pathlib import Path

import mne
import numpy as np

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
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)


logger = logging.getLogger(__name__)

VSPELL_BASE_URL = "https://zenodo.org/record/"
VISUAL_SPELLER_LLP_URL = VSPELL_BASE_URL + "5831826/files/"
VISUAL_SPELLER_MIX_URL = VSPELL_BASE_URL + "5831879/files/"
OPTICAL_MARKER_CODE = 500


class _BaseVisualMatrixSpellerDataset(BaseDataset, ABC):
    def __init__(
        self,
        src_url,
        n_subjects,
        raw_slice_offset,
        use_blocks_as_sessions=True,
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
        **kwargs,
    ):
        self.n_channels = 31  # all channels except 5 times x_* CH and EOGvu
        if kwargs["interval"] is None:
            # "Epochs were windowed to [−200, 700] ms relative to the stimulus onset [...]."
            kwargs["interval"] = [-0.2, 0.7]

        super().__init__(
            events=dict(Target=10002, NonTarget=10001),
            paradigm="p300",
            subjects=(np.arange(n_subjects) + 1).tolist(),
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
            **kwargs,
        )

        self.raw_slice_offset = 2_000 if raw_slice_offset is None else raw_slice_offset
        self._src_url = src_url
        self.use_blocks_as_sessions = use_blocks_as_sessions
        self.description_map = {"Stimulus/S   1": "Target", "Stimulus/S   0": "NonTarget"}

    @staticmethod
    def _filename_trial_info_extraction(vhdr_file_path):
        vhdr_file_path = Path(vhdr_file_path)
        vhdr_file_name = vhdr_file_path.name
        run_file_pattern = "^matrixSpeller_Block([0-9]+)_Run([0-9]+)\\.vhdr$"
        vhdr_file_patter_match = re.match(run_file_pattern, vhdr_file_name)

        if not vhdr_file_patter_match:
            # TODO: raise a wild exception?
            logger.info(vhdr_file_path)

        session_name = "0"
        block_idx = vhdr_file_patter_match.group(1)
        block_idx = int(block_idx) - 1
        run_idx = vhdr_file_patter_match.group(2)
        run_idx = int(run_idx) - 1
        return session_name, block_idx, run_idx

    def _get_single_subject_data(self, subject):
        subject_data_vhdr_files = self.data_path(subject)
        sessions = dict()

        for _file_idx, subject_data_vhdr_file in enumerate(subject_data_vhdr_files):
            (
                session_name,
                block_idx,
                run_idx,
            ) = Huebner2017._filename_trial_info_extraction(subject_data_vhdr_file)

            raw_bvr_list = _read_raw_llp_study_data(
                vhdr_fname=subject_data_vhdr_file,
                raw_slice_offset=self.raw_slice_offset,
                verbose=None,
            )

            raw_bvr_list[0].annotations.rename(self.description_map)

            if self.use_blocks_as_sessions:
                session_name = str(block_idx)
            if session_name not in sessions.keys():
                sessions[session_name] = dict()
            run_name = str(run_idx)
            sessions[session_name][run_name] = raw_bvr_list[0]

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        url = f"{self._src_url}subject{subject:02d}.zip"
        zipfile_path = Path(dl.data_dl(url, "llp"))
        zipfile_extracted_path = zipfile_path.parent

        subject_dir_path = zipfile_extracted_path / f"subject{subject:02d}"

        if not subject_dir_path.is_dir():
            _BaseVisualMatrixSpellerDataset._extract_data(
                zipfile_extracted_path, zipfile_path
            )

        subject_paths = zipfile_extracted_path / f"subject{subject:02d}"
        subject_paths = subject_paths.glob("matrixSpeller_Block*_Run*.vhdr")
        subject_paths = [str(p) for p in subject_paths]
        return sorted(subject_paths)

    @staticmethod
    def _extract_data(data_dir_extracted_path, data_archive_path):
        zip_ref = zipfile.ZipFile(data_archive_path, "r")
        zip_ref.extractall(data_dir_extracted_path)


class Huebner2017(_BaseVisualMatrixSpellerDataset):
    """Learning from label proportions for a visual matrix speller (ERP)
    dataset from Hübner et al 2017 [1]_.

    **Dataset description**

    The subjects were asked to spell the sentence: “Franzy jagt im komplett verwahrlosten Taxi quer durch Freiburg”.
    The sentence was chosen because it contains each letter used in German at least once. Each subject spelled this
    sentence three times. The stimulus onset asynchrony (SOA) was 250 ms (corresponding to 15 frames on the LCD screen
    utilized) while the stimulus duration was 100 ms (corresponding to 6 frames on the LCD screen utilized). For each
    character, 68 highlighting events occurred and a total of 63 characters were spelled three times. This resulted in
    a total of 68 ⋅ 63 ⋅ 3 = 12852 EEG epochs per subject. Spelling one character took around 25 s including 4 s for
    cueing the current symbol, 17 s for highlighting and 4 s to provide feedback to the user. Assuming a perfect
    decoding, these timing constraints would allow for a maximum spelling speed of 2.4 characters per minute. Fig 1
    shows the complete experimental structure and how LLP is used to reconstruct average target and non-target ERP
    responses.

    Subjects were placed in a chair at 80 cm distance from a 24-inch flat screen. EEG signals from 31 passive Ag/AgCl
    electrodes (EasyCap) were recorded, which were placed approximately equidistantly according to the extended
    10–20 system, and whose impedances were kept below 20 kΩ. All channels were referenced against the nose and the
    ground was at FCz. The signals were registered by multichannel EEG amplifiers (BrainAmp DC, Brain Products) at a
    sampling rate of 1 kHz. To control for vertical ocular movements and eye blinks, we recorded with an EOG electrode
    placed below the right eye and referenced against the EEG channel Fp2 above the eye. In addition, pulse and
    breathing activity were recorded.

    Parameters
    ----------
    interval: array_like
        range/interval in milliseconds in which the brain response/activity relative to an event/stimulus onset lies in.
        Default is set to [-.2, .7].
    raw_slice_offset: int, None
        defines the crop offset in milliseconds before the first and after the last event (target or non-targeet) onset.
        Default None which crops with an offset 2,000 ms.

    References
    ----------
    .. [1] Hübner, D., Verhoeven, T., Schmid, K., Müller, K. R., Tangermann, M., & Kindermans, P. J. (2017)
           Learning from label proportions in brain-computer interfaces: Online unsupervised learning with guarantees.
           PLOS ONE 12(4): e0175856.
           https://doi.org/10.1371/journal.pone.0175856

    .. versionadded:: 0.4.5
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=31,
            channel_types={"eeg": 31, "misc": 6},
            montage="standard_1020",
            hardware="BrainAmp DC",
            sensor_type="passive Ag/AgCl",
            reference="nose",
            ground="FCz",
            software=None,
            impedance_threshold_kohm=20.0,
            sensors=[
                "C3",
                "C4",
                "CP1",
                "CP2",
                "CP5",
                "CP6",
                "Cz",
                "EOGvu",
                "F10",
                "F3",
                "F4",
                "F7",
                "F8",
                "F9",
                "FC1",
                "FC2",
                "FC5",
                "FC6",
                "Fp1",
                "Fp2",
                "Fz",
                "O1",
                "O2",
                "P10",
                "P3",
                "P4",
                "P7",
                "P8",
                "P9",
                "Pz",
                "T7",
                "T8",
                "x_EMGl",
                "x_GSR",
                "x_Optic",
                "x_Pulse",
                "x_Respi",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_channels=1,
                eog_type=["vertical"],
                other_physiological=["pulse", "respiration"],
            ),
            cap_manufacturer="EasyCap",
        ),
        participants=ParticipantMetadata(
            n_subjects=13,
            health_status="healthy",
            gender={"female": 5, "male": 8},
            age_mean=26.0,
            age_std=1.5,
            bci_experience="mostly naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"Target": 10002, "NonTarget": 10001},
            paradigm="p300",
            n_classes=2,
            class_labels=["target", "non-target"],
            trial_duration=25.0,
            study_design="Visual ERP speller copy-spelling task using a 6x7 grid with learning from label proportions (LLP) classifier. Two sequences with different target/non-target ratios: sequence 1 (3 targets/8 stimuli), sequence 2 (2 targets/18 stimuli). Unsupervised calibrationless approach.",
            feedback_type="visual",
            stimulus_type="character matrix",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="online",
            synchronicity="synchronous",
            has_training_test_split=False,
            instructions="Copy-spelling task: subjects spelled the sentence 'FRANZY JAGT IM KOMPLETT VERWAHRLOSTEN TAXI QUER DURCH FREIBURG' three times",
            stimulus_presentation={
                "soa_ms": "250",
                "stimulus_duration_ms": "100",
                "grid_size": "6x7",
                "highlighting_method": "salient (brightness enhancement, rotation, enlargement, trichromatic grid overlay)",
                "viewing_distance_cm": "80",
                "screen_size_inches": "24",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0175856",
            repository="Zenodo",
            data_url="http://doi.org/10.5281/zenodo.192684",
            publication_year=2017,
            investigators=[
                "David Hübner",
                "Thibault Verhoeven",
                "Konstantin Schmid",
                "Klaus-Robert Müller",
                "Michael Tangermann",
                "Pieter-Jan Kindermans",
            ],
            senior_author="Michael Tangermann",
            contact_info=[
                "david.huebner@blbt.uni-freiburg.de",
                "michael.tangermann@blbt.uni-freiburg.de",
                "p.kindermans@tu-berlin.de",
            ],
            institution="Albert-Ludwigs-University",
            institution_department="Brain State Decoding Lab, Cluster of Excellence BrainLinks-BrainTools, Department of Computer Science",
            institution_address="Freiburg, Germany",
            country="DE",
            funding=[
                "BrainLinks-BrainTools Cluster of Excellence funded by the German Research Foundation (DFG), grant number EXC 1086",
                "bwHPC initiative, grant INST 39/963-1 FUGG",
                "European Union's Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No 657679",
                "Special Research Fund from Ghent University",
                "BK21 program funded by Korean National Research Foundation grant No. 2012-005741",
            ],
            ethics_approval=[
                "Ethics Committee of the University Medical Center Freiburg",
                "Declaration of Helsinki",
            ],
            license="CC-BY-4.0",
            keywords=[
                "brain-computer interface",
                "BCI",
                "event-related potentials",
                "ERP",
                "P300",
                "learning from label proportions",
                "LLP",
                "unsupervised learning",
                "calibrationless",
                "visual speller",
            ],
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
            classifiers=[
                "LLP (Learning from Label Proportions)",
                "shrinkage-LDA",
                "EM-algorithm",
            ],
            feature_extraction=["mean amplitude per time interval"],
            frequency_bands={
                "analyzed_range": [0.5, 8.0],
            },
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="5-fold chronological cross-validation",
            cv_folds=5,
            evaluation_type=["within_subject"],
        ),
        performance={
            "accuracy_percent": 84.5,
            "auc": 0.975,
            "online_spelling_accuracy_percent": 84.5,
            "post_hoc_spelling_accuracy_percent": 95.0,
            "accuracy_after_rampup_percent": 90.2,
            "supervised_auc": 0.975,
            "max_spelling_speed_chars_per_min": 2.4,
        },
        bci_application=BCIApplicationMetadata(
            applications=["speller", "communication"],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=42,
            soa_ms=250.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=12852,
            trials_context="68 highlighting events per character, 63 characters per sentence, 3 sentences = 68*63*3 = 12852 EEG epochs per subject. Each epoch is a Target (10002) or NonTarget (10001) event.",
        ),
        sessions_per_subject=3,
        runs_per_session=9,
        sessions=["session_1"],
        data_processed=False,
        file_format="BrainVision",
        abstract="Using traditional approaches, a brain-computer interface (BCI) requires the collection of calibration data for new subjects prior to online use. This work introduces learning from label proportions (LLP) to the BCI community as a new unsupervised, and easy-to-implement classification approach for ERP-based BCIs. The LLP estimates the mean target and non-target responses based on known proportions of these two classes in different groups of the data. We present a visual ERP speller to meet the requirements of LLP. For evaluation, we ran simulations on artificially created data sets and conducted an online BCI study with 13 subjects performing a copy-spelling task. Theoretical considerations show that LLP is guaranteed to minimize the loss function similar to a corresponding supervised classifier. LLP performed well in simulations and in the online application, where 84.5% of characters were spelled correctly on average without prior calibration.",
        methodology="The experiment used a modified visual ERP speller with a 6×7 grid. Two distinct stimulus sequences with different target/non-target ratios were used: sequence 1 had 3 targets in 8 stimuli, sequence 2 had 2 targets in 18 stimuli. Each trial consisted of 4 sequences of length 8 and 2 sequences of length 18, totaling 68 highlighting events per character. The LLP algorithm exploited these known proportions to reconstruct mean target and non-target ERP responses without requiring labeled data. The classifier was reset at the start of each sentence and retrained after each character. Subjects spelled a German pangram sentence three times. One subject (S2) had prior EEG experience; others were naive. Sessions lasted about 3 hours including setup. Participants were compensated 8 Euros per hour.",
    )

    def __init__(
        self,
        interval=None,
        raw_slice_offset=None,
        use_blocks_as_sessions=True,
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
    ):
        llp_speller_paper_doi = "10.1371/journal.pone.0175856"
        super().__init__(
            src_url=VISUAL_SPELLER_LLP_URL,
            raw_slice_offset=raw_slice_offset,
            n_subjects=13,
            sessions_per_subject=3,
            code="Huebner2017",  # Before: "VisualSpellerLLP"
            interval=interval,
            doi=llp_speller_paper_doi,
            use_blocks_as_sessions=use_blocks_as_sessions,
            subjects=subjects,
            sessions=sessions,
            return_all_modalities=return_all_modalities,
        )


class Huebner2018(_BaseVisualMatrixSpellerDataset):
    """Mixture of LLP and EM for a visual matrix speller (ERP) dataset from
    Hübner et al 2018 [1]_.

    **Dataset description**

    Within a single session, a subject was asked to spell the beginning of a sentence in each of three blocks.The text
    consists of the 35 symbols “Franzy jagt im Taxi quer durch das ”. Each block, one of the three decoding
    algorithms (EM, LLP, MIX) was used in order to guess the attended symbol. The order of the blocks was
    pseudo-randomized over subjects, such that each possible order of the three decoding algorithms was used twice.
    This randomization should reduce systematic biases by order effects or temporal effects, e.g., due to fatigue or
    task-learning.

    A trial describes the process of spelling one character. Each of the 35 trials per block contained 68 highlighting
    events. The stimulus onset asynchrony (SOA) was 250 ms and the stimulus duration was 100 ms leading to an
    interstimulus interval (ISI) of 150 ms.

    Parameters
    ----------
    interval: array_like
        range/interval in milliseconds in which the brain response/activity relative to an event/stimulus onset lies in.
        Default is set to [-.2, .7].
    raw_slice_offset: int, None
        defines the crop offset in milliseconds before the first and after the last event (target or non-targeet) onset.
        Default None which crops with an offset 2,000 ms.

    References
    ----------
    .. [1] Huebner, D., Verhoeven, T., Mueller, K. R., Kindermans, P. J., & Tangermann, M. (2018).
           Unsupervised learning for brain-computer interfaces based on event-related potentials: Review and online comparison [research frontier].
           IEEE Computational Intelligence Magazine, 13(2), 66-77.
           https://doi.org/10.1109/MCI.2018.2807039

    .. versionadded:: 0.4.5
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=31,
            channel_types={"eeg": 31, "misc": 6},
            montage="extended 10-20",
            hardware="BrainAmp DC",
            sensor_type="Ag/AgCl",
            reference="nose",
            software="BBCI toolbox",
            impedance_threshold_kohm=20.0,
            sensors=[
                "C3",
                "C4",
                "CP1",
                "CP2",
                "CP5",
                "CP6",
                "Cz",
                "EOGvu",
                "F10",
                "F3",
                "F4",
                "F7",
                "F8",
                "F9",
                "FC1",
                "FC2",
                "FC5",
                "FC6",
                "Fp1",
                "Fp2",
                "Fz",
                "O1",
                "O2",
                "P10",
                "P3",
                "P4",
                "P7",
                "P8",
                "P9",
                "Pz",
                "T7",
                "T8",
                "x_EMGl",
                "x_GSR",
                "x_Optic",
                "x_Pulse",
                "x_Respi",
            ],
            line_freq=50.0,
            cap_manufacturer="EasyCap",
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="healthy",
            gender={"female": 8, "male": 4},
            age_mean=26,
            age_min=19,
            age_max=31,
            bci_experience="mixed",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            n_classes=2,
            class_labels=["target", "non-target"],
            trial_duration=17.0,
            study_design="Visual ERP copy-spelling task using a modified 6x6 grid extended with 10 # symbols as visual blanks, using flexible highlighting scheme with two interleaved sequences to enable unsupervised learning methods (EM, LLP, MIX)",
            feedback_type="visual",
            stimulus_type="modified matrix speller with flexible highlighting",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="online",
            instructions="copy-spelling task - spell German sentence 'Franzy jagt im Taxi quer durch das'",
            tasks=["copy-spelling"],
            events={"Target": 10002, "NonTarget": 10001},
            stimulus_presentation={
                "soa_ms": "250",
                "stimulus_duration_ms": "100",
                "isi_ms": "150",
                "highlighting_type": "combination of brightness enhancement, rotation, enlargement and trichromatic grid overlay",
                "distance_to_screen_cm": "80",
                "screen_size_inches": "24",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.192684",
            repository="Zenodo",
            data_url="https://zenodo.org/record/5831879",
            publication_year=2018,
            investigators=[
                "David Hübner",
                "Thibault Verhoeven",
                "Klaus-Robert Müller",
                "Pieter-Jan Kindermans",
                "Michael Tangermann",
            ],
            institution="University of Freiburg",
            country="DE",
            institution_department="Brain State Decoding Lab",
            contact_info=[
                "p.kindermans@tu-berlin.de",
                "michael.tangermann@blbt.uni-freiburg.de",
            ],
            funding=[
                "BrainLinks-BrainTools Cluster of Excellence funded by the German Research Foundation (DFG), grant number EXC 1086",
                "bwHPC initiative, grant INST 39/963-1 FUGG",
                "European Union's Horizon 2020 research and innovation program under the Marie Sklodowska-Curie grant agreement NO 657679",
                "Special Research Fund of Ghent University",
                "DFG (DFG SPP 1527, MU 987/14-1)",
                "Federal Ministry for Education and Research (BMBF No. 2017-0-00451)",
                "Brain Korea 21 Plus Program by the Institute for Information & Communications Technology Promotion (IITP) grant (1IS14013A) funded by the Korean government",
            ],
            ethics_approval=["University Medical Center Freiburg ethics committee"],
            institution_address="Brain State Decoding Lab, University of Freiburg, Freiburg, GERMANY",
            keywords=[
                "unsupervised learning",
                "brain-computer interface",
                "event-related potentials",
                "P300 speller",
                "expectation-maximization",
                "learning from label proportions",
                "MIX method",
                "EEG",
            ],
            license="CC-BY-4.0",
            associated_paper_doi="10.1109/MCI.2018.2807039",
        ),
        sessions_per_subject=3,
        runs_per_session=None,
        sessions=["0", "1", "2"],
        contributing_labs=None,
        n_contributing_labs=None,
        data_processed=False,
        file_format="BrainVision",
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
            classifiers=[
                "EM (Expectation-Maximization)",
                "LLP (Learning from Label Proportions)",
                "MIX (mixture of EM and LLP)",
                "shrinkage-regularized LDA (Ledoit-Wolf)",
                "Bayesian least square regression",
            ],
            feature_extraction=["mean amplitudes in six temporal intervals per channel"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="leave-one-character-out for offline analysis; online sequential testing",
            evaluation_type=["online", "within_session", "unsupervised_learning"],
        ),
        performance={
            "accuracy_percent": 80.0,
            "MIX_AUC_after_7_chars": 80.0,
            "time_to_80_percent_accuracy_seconds": 168.0,
            "epochs_to_80_percent_accuracy": 476.0,
            "characters_to_80_percent_accuracy": 7.0,
        },
        bci_application=BCIApplicationMetadata(
            applications=["speller", "communication"],
            environment="controlled laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=46,
            isi_ms=150.0,
            soa_ms=250.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=35,
            n_blocks=3,
            trials_context="35 characters per block (one trial = spelling one character), 3 blocks per session (one block per unsupervised algorithm: EM, LLP, MIX in pseudo-randomized order)",
        ),
        abstract="One of the fundamental challenges in brain-computer interfaces (BCIs) is to tune a brain signal decoder to reliably detect a user's intention. While information about the decoder can partially be transferred between subjects or sessions, optimal decoding performance can only be reached with novel data from the current session. Thus, it is preferable to learn from unlabeled data gained from the actual usage of the BCI application instead of conducting a calibration recording prior to BCI usage. We review such unsupervised machine learning methods for BCIs based on event-related potentials of the electroencephalogram. We present results of an online study with twelve healthy participants controlling a visual speller. Online performance is reported for three completely unsupervised learning methods: (1) learning from label proportions, (2) an expectation-maximization approach and (3) MIX, which combines the strengths of the two other methods. After a short ramp-up, we observed that the MIX method not only defeats its two unsupervised competitors but even performs on par with a state-of-the-art regularized linear discriminant analysis trained on the same number of data points and with full label access. With this online study, we deliver the best possible proof in BCI that an unsupervised decoding method can in practice render a supervised method unnecessary. This is possible despite skipping the calibration, without losing much performance and with the prospect of continuous improvement over a session. Thus, our findings pave the way for a transition from supervised to unsupervised learning methods in BCIs based on event-related potentials.",
        methodology="Online study comparing three unsupervised learning methods (EM, LLP, MIX) for P300 speller. Twelve healthy volunteers (8 female, 4 male, mean age 26, range 19-31 years) participated in a single session each. Subjects spelled the German sentence 'Franzy jagt im Taxi quer durch das' (35 characters) in three blocks, each using a different unsupervised algorithm in pseudo-randomized order. Each trial (spelling one character) consisted of 68 highlighting events with 250 ms SOA and 100 ms stimulus duration (ISI=150 ms). The speller used a modified 6x6 grid with 36 normal characters extended with 10 # symbols as visual blanks (total 46 symbols). Two interleaved highlighting sequences were used: S1 highlighted only normal characters, S2 highlighted both normal characters and # symbols, creating different known target-to-non-target ratios to enable learning from label proportions. Highlighting consisted of brightness enhancement, rotation, enlargement and trichromatic grid overlay. Classifiers were randomly initialized at block start and updated after each trial. No labeled data was provided during online session. Participants sat 80 cm from a 24-inch screen. EEG was recorded from 31 passive Ag/AgCl electrodes (EasyCap) placed according to extended 10-20 system, with impedances kept below 20 kOhm. Signals were recorded and amplified by BrainAmp DC at 1 kHz sampling rate using BBCI toolbox in Matlab. Data was bandpass filtered (0.5-8 Hz, 3rd order Chebyshev Type II), downsampled to 100 Hz, epoched to [-200, 700] ms relative to stimulus onset, and baseline corrected using [-200, 0] ms interval. Features were mean amplitudes of six time intervals ([50-120], [121-200], [201-280], [281-380], [381-530], [531-700] ms post-stimulus) per channel. No artifact rejection was applied; participants were instructed to avoid artifacts. Performance metrics: spelling accuracy and AUC for target vs. non-target discrimination. Results showed MIX method achieved ~80% accuracy after ~7 characters (168 seconds, 476 epochs) and performed comparably to supervised regularized LDA trained on same amount of labeled data after 10+ characters. Ethics approval was obtained from University Medical Center Freiburg. Participants were compensated 8 Euros per hour for the ~3 hour session (including EEG setup).",
    )

    def __init__(
        self,
        interval=None,
        raw_slice_offset=None,
        use_blocks_as_sessions=True,
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
    ):
        mix_speller_paper_doi = "10.1109/MCI.2018.2807039"
        super().__init__(
            src_url=VISUAL_SPELLER_MIX_URL,
            raw_slice_offset=raw_slice_offset,
            n_subjects=12,
            sessions_per_subject=3,
            code="Huebner2018",  # Before: "VisualSpellerMIX"
            interval=interval,
            doi=mix_speller_paper_doi,
            use_blocks_as_sessions=use_blocks_as_sessions,
            subjects=subjects,
            sessions=sessions,
            return_all_modalities=return_all_modalities,
        )


def _read_raw_llp_study_data(vhdr_fname, raw_slice_offset, verbose=None):
    """Read LLP BVR recordings file. Ignore the different sequence lengths.
    Just tag event as target or non-target if it contains a target or does not
    contain a target.

    Parameters
    ----------
    vhdr_fname: str
        Path to the EEG header file.
    verbose : bool, int, None
        specify the loglevel.

    Returns
    -------
    raw_object: mne.io.Raw
        the loaded BVR raw object.
    """
    non_scalp_channels = ["EOGvu", "x_EMGl", "x_GSR", "x_Respi", "x_Pulse", "x_Optic"]
    raw_bvr = mne.io.read_raw_brainvision(
        vhdr_fname=vhdr_fname,  # eog='EOGvu',
        misc=non_scalp_channels,
        preload=True,
        verbose=verbose,
    )  # type: mne.io.Raw
    raw_bvr = raw_bvr.set_montage("standard_1020")

    events = _parse_events(raw_bvr)

    onset_arr_list, marker_arr_list = _extract_target_non_target_description(events)

    def annotate_and_crop_raw(onset_arr, marker_arr):
        raw = raw_bvr

        raw_annotated = raw.set_annotations(
            _create_annotations_from(marker_arr, onset_arr, raw)
        )

        tmin = max((onset_arr[0] - raw_slice_offset) / 1e3, 0)
        tmax = min((onset_arr[-1] + raw_slice_offset) / 1e3, raw.times[-1])
        return raw_annotated.crop(tmin=tmin, tmax=tmax, include_tmax=True)

    return list(map(annotate_and_crop_raw, onset_arr_list, marker_arr_list))


def _create_annotations_from(marker_arr, onset_arr, raw_bvr):
    default_bvr_marker_duration = raw_bvr.annotations[0]["duration"]

    onset = onset_arr / 1e3  # convert onset in seconds to ms
    durations = np.repeat(default_bvr_marker_duration, len(marker_arr))
    description = list(map(lambda m: f"Stimulus/S {m:3}", marker_arr))
    orig_time = raw_bvr.annotations[0]["orig_time"]
    return mne.Annotations(
        onset=onset, duration=durations, description=description, orig_time=orig_time
    )


def _parse_events(raw_bvr):
    stimulus_pattern = re.compile("(Stimulus/S|Optic/O) *([0-9]+)")

    def parse_marker(desc):
        match = stimulus_pattern.match(desc)
        if match is None:
            return None
        if match.group(1) == "Optic/O":
            return OPTICAL_MARKER_CODE

        return int(match.group(2))

    events, _ = mne.events_from_annotations(
        raw=raw_bvr, event_id=parse_marker, verbose=None
    )
    return events


def _find_single_trial_start_end_idx(events):
    trial_start_end_markers = [21, 22, 10]
    return np.where(np.isin(events[:, 2], trial_start_end_markers))[0]


def _extract_target_non_target_description(events):
    single_trial_start_end_idx = _find_single_trial_start_end_idx(events)

    n_events = single_trial_start_end_idx.size - 1

    onset_arr = np.empty((n_events,), dtype=np.int64)
    marker_arr = np.empty((n_events,), dtype=np.int64)

    broken_events_idx = list()
    for epoch_idx in range(n_events):
        epoch_start_idx = single_trial_start_end_idx[epoch_idx]
        epoch_end_idx = single_trial_start_end_idx[epoch_idx + 1]

        epoch_events = events[epoch_start_idx:epoch_end_idx]

        onset_ms = _find_epoch_onset(epoch_events)
        if onset_ms == -1:
            broken_events_idx.append(epoch_idx)
            continue

        onset_arr[epoch_idx] = onset_ms
        marker_arr[epoch_idx] = int(
            _single_trial_contains_target(epoch_events)
        )  # 1/true if single trial has target

    return [np.delete(onset_arr, broken_events_idx)], [
        np.delete(marker_arr, broken_events_idx)
    ]


def _find_epoch_onset(epoch_events):
    optical_idx = epoch_events[:, 2] == OPTICAL_MARKER_CODE
    stimulus_onset_time = epoch_events[optical_idx, 0]

    def second_optical_is_feedback():
        if stimulus_onset_time.size != 2:
            return False

        stimulus_prior_second_optical_marker = epoch_events[
            np.where(optical_idx)[0][1] - 1, 2
        ]
        return stimulus_prior_second_optical_marker in [50, 51, 11]

    if stimulus_onset_time.size == 1 or second_optical_is_feedback():
        return stimulus_onset_time[0]

    # broken epoch: no true onset found..
    return -1


def _single_trial_contains_target(trial_events):
    trial_markers = trial_events[:, 2]
    return np.any((trial_markers > 100) & (trial_markers <= 142))
