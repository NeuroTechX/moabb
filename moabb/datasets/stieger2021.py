# Authors: Reinmar Kobler <kobler.reinmar@gmail.com>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#
# License: BSD (3-clause)

import logging
import os

import mne
import numpy as np
import pooch
from scipy.io import loadmat

import moabb.datasets.download as dl
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

from .base import BaseDataset
from .download import get_dataset_path


LOGGER = logging.getLogger(__name__)
BASE_URL = "https://ndownloader.figshare.com/files/"


class Stieger2021(BaseDataset):
    """Motor Imagery dataset from Stieger et al. 2021 [1]_.

    The main goals of our original study were to characterize how individuals
    learn to control SMR-BCIs and to test whether this learning can be improved
    through behavioral interventions such as mindfulness training. Participants
    were initially assessed for baseline BCI proficiency and then randomly
    assigned to an 8-week mindfulness intervention (Mindfulness-based stress
    reduction), or waitlist control condition where participants waited for
    the same duration as the MBSR class before starting BCI training, but
    were offered a comparable MBSR course after completing all experimental
    requirements. Following the 8-weeks, participants returned to the lab
    for 6 to 10 sessions of BCI training.

    All experiments were approved by the institutional review boards of the
    University of Minnesota and Carnegie Mellon University. Informed consents
    were obtained from all subjects. In total, 144 participants were enrolled
    in the study and 76 participants completed all experimental requirements.
    Seventy-two participants were assigned to each intervention by block
    randomization, with 42 participants completing all sessions in the
    experimental group (MBSR before BCI training; MBSR subjects) and 34
    completing experimentation in the control group. Four subjects were
    excluded from the analysis due to non-compliance with the task demands and
    one was excluded due to experimenter error. We were primarily interested
    in how individuals learn to control BCIs, therefore analysis focused on
    those that did not demonstrate ceiling performance in the baseline BCI
    assessment (accuracy above 90% in 1D control). The dataset descriptor
    presented here describes data collected from 62 participants:
    33 MBSR participants (Age=42+/-15, (F)emale=26) and 29 controls
    (Age=36+/-13, F=23). In the United States, women are twice as likely to
    practice meditation compared to men. Therefore, the gender
    imbalance in our study may result from a greater likelihood of
    women to respond to flyers offering a meditation class in exchange
    for participating in our study.

    For all BCI sessions, participants were seated comfortably in a chair and
    faced a computer monitor that was placed approximately 65cm in front of them.
    After the EEG capping procedure (see data acquisition), the BCI tasks began.
    Before each task, participants received the appropriate instructions. During
    the BCI tasks, users attempted to steer a virtual cursor from the center of
    the screen out to one of four targets. Participants initially received the
    following instructions: “Imagine your left (right) hand opening and closing
    to move the cursor left (right). Imagine both hands opening and closing to
    move the cursor up. Finally, to move the cursor down, voluntarily rest; in
    other words, clear your mind.” In separate blocks of trials, participants
    directed the cursor toward a target that required left/right (LR) movement
    only, up/down (UD) only, and combined 2D movement (2D)30. Each experimental
    block (LR, UD, 2D) consisted of 3 runs, where each run was composed of 25
    trials. After the first three blocks, participants were given a short break
    (5-10 minutes) that required rating comics by preference. The break task was
    chosen to standardize subject experience over the break interval. Following
    the break, participants competed the same 3 blocks as before. In total,
    each session consisted of 2 blocks of each task (6 runs total of LR, UD,
    and 2D control), which culminated in 450 trials performed each day.

    Online BCI control of the cursor proceeded in a series of steps.
    The first step, feature extraction, consisted of spatial filtering and
    spectrum estimation. During spatial filtering, the average signal of the 4
    electrodes surrounding the hand knob of the motor cortex was subtracted from
    electrodes C3 and C4 to reduce the spatial noise. Following spatial filtering,
    the power spectrum was estimated by fitting an autoregressive model of order 16
    to the most recent 160 ms of data using the maximum entropy method. The goal
    of this method is to find the coefficients of a linear all-pole filter that,
    when applied to white noise, reproduces the data's spectrum. The main advantage
    of this method is that it produces high frequency resolution estimates for short
    segments of data. The parameters are found by minimizing (through least squares)
    the forward and backward prediction errors on the input data subject to the
    constraint that the filter used for estimation shares the same autocorrelation
    sequence as the input data. Thus, the estimated power spectrum directly
    corresponds to this filter's transfer function divided by the signal's total power.
    Numerical integration was then used to find the power within a 3 Hz bin
    centered within the alpha rhythm (12 Hz).
    The translation algorithm, the next step in the pipeline, then translated the
    user's alpha power into cursor movement. Horizontal motion was controlled by
    lateralized alpha power (C4 - C3) and vertical motion was controlled by up
    and down regulating total alpha power (C4 + C3). These control signals were
    normalized to zero mean and unit variance across time by subtracting the
    signals' mean and dividing by its standard deviation. A balanced estimate
    of the mean and standard deviation of the horizontal and vertical control
    signals was calcu- lated by estimating these values across time from data
    derived from 30 s buffers of individual trial type (e.g., the normalized
    control signal should be positive for right trials and negative for left
    trials, but the average of left and right trials should be zero). Finally,
    the normalized control signals were used to update the position of
    the cursor every 40 ms.

    References
    ----------

    .. [1] Stieger, J. R., Engel, S. A., & He, B. (2021).
           Continuous sensorimotor rhythm based brain computer interface
           learning in a large population. Scientific Data, 8(1), 98.
           https://doi.org/10.1038/s41597-021-00883-1

    Notes
    -----
    .. versionadded:: 1.1.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=1000.0,
            n_channels=60,
            channel_types={"eeg": 60},
            montage="10-10",
            hardware="Neuroscan SynAmps RT amplifiers",
            sensor_type="Neuroscan Quik-Cap",
            reference="CAR",
            software="BCI2000",
            filters="0.1 to 200 Hz with 60 Hz notch filter",
            sensors=[
                "AF3",
                "AF4",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
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
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "FCz",
                "FT7",
                "FT8",
                "Fp1",
                "Fp2",
                "Fpz",
                "Fz",
                "O1",
                "O2",
                "Oz",
                "P1",
                "P2",
                "P3",
                "P4",
                "P5",
                "P6",
                "P7",
                "P8",
                "PO3",
                "PO4",
                "PO5",
                "PO6",
                "PO7",
                "PO8",
                "POz",
                "Pz",
                "T7",
                "T8",
                "TP7",
                "TP8",
            ],
            line_freq=60.0,
            impedance_threshold_kohm=5.0,
            cap_manufacturer="Neuroscan",
            cap_model="Quik-Cap",
        ),
        participants=ParticipantMetadata(
            n_subjects=62,
            health_status="healthy",
            gender={"male": 13, "female": 49},
            age_mean=39.4,
            age_std=14.2,
            age_min=18,
            age_max=63,
            handedness={"right": 62},
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"right_hand": 1, "left_hand": 2, "both_hand": 3, "rest": 4},
            paradigm="imagery",
            n_classes=4,
            class_labels=["right_hand", "left_hand", "both_hands", "rest"],
            feedback_type="continuous online visual feedback (cursor movement)",
            stimulus_type="cursor_feedback",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="online",
            has_training_test_split=True,
            trial_duration=10.0,
            instructions="Imagine your left (right) hand opening and closing to move the cursor left (right). Imagine both hands opening and closing to move the cursor up. Finally, to move the cursor down, voluntarily rest; in other words, clear your mind.",
            study_design="longitudinal training study with intervention",
            tasks=["LR", "UD", "2D"],
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-021-00883-1",
            publication_year=2021,
            investigators=["James R. Stieger", "Stephen A. Engel", "Bin He"],
            senior_author="Bin He",
            contact_info=["bhe1@andrew.cmu.edu"],
            institution="Carnegie Mellon University, University of Minnesota",
            country="USA",
            institution_address="Pittsburgh, PA, USA; Minneapolis, MN, USA",
            institution_department="Carnegie Mellon University, Pittsburgh, PA, USA; University of Minnesota, Minneapolis, MN, USA",
            ethics_approval=[
                "University of Minnesota IRB",
                "Carnegie Mellon University IRB",
            ],
            description="Continuous sensorimotor rhythm based brain computer interface learning in a large population",
            keywords=[
                "BCI",
                "sensorimotor rhythm",
                "motor imagery",
                "EEG",
                "longitudinal",
                "learning",
            ],
            repository="GitHub",
            data_url="https://github.com/bfinl/BCI_Data_Paper",
            funding=["NIH under"],
            license="CC-BY-NC-4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Active"],
        ),
        preprocessing=PreprocessingMetadata(
            highpass_hz=0.1,
            lowpass_hz=200.0,
            notch_hz=[60],
            re_reference="CAR",
            data_state="raw",
            preprocessing_applied=False,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CNN", "RNN", "Neural Network", "EEGNet"],
            feature_extraction=["ERD", "ERS", "autoregressive model", "power spectrum"],
            frequency_bands={
                "alpha": [8, 13],
                "mu": [8, 12],
            },
            spatial_filters=["Laplacian (C3/C4 with 4 surrounding electrodes)"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["cross_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=[
                "cursor_control",
                "wheelchair/navigation",
                "robotic_arm",
                "drone",
            ],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand", "both_hands", "rest"],
            cue_duration_s=2.0,
            imagery_duration_s=6.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=450,
            n_blocks=6,
            trials_context="per_session",
        ),
        file_format="MAT",
        sessions_per_subject=10,
        runs_per_session=1,
        data_processed=False,
        abstract="Brain computer interfaces (BCIs) are valuable tools that expand the nature of communication through bypassing traditional neuromuscular pathways. The non-invasive, intuitive, and continuous nature of sensorimotor rhythm (SMR) based BCIs enables individuals to control computers, robotic arms, wheelchairs, and even drones by decoding motor imagination from electroencephalography (EEG). Large and uniform datasets are needed to design, evaluate, and improve the BCI algorithms. In this work, we release a large and longitudinal dataset collected during a study that examined how individuals learn to control SMR-BCIs. The dataset contains over 600 hours of EEG recordings collected during online and continuous BCI control from 62 healthy adults, (mostly) right hand dominant participants, across (up to) 11 training sessions per participant. The data record consists of 598 recording sessions, and over 250,000 trials of 4 different motor-imagery-based BCI tasks.",
        methodology="Participants completed 7-11 online BCI training sessions. Each session consisted of 450 trials across 3 tasks (LR, UD, 2D) with 6 runs total. Each trial: 2s inter-trial interval, 2s target presentation, up to 6s feedback control. Online control used spatial filtering (Laplacian around C3/C4), autoregressive model (order 16) for spectrum estimation, alpha power (12 Hz ± 1.5 Hz) for control signal. Horizontal motion controlled by lateralized alpha power (C4-C3), vertical motion by total alpha power (C4+C3). Control signals normalized to zero mean and unit variance. Cursor position updated every 40 ms.",
        performance={
            "accuracy_percent": 70.0,
            "PVC_1D_threshold": 70.0,
            "PVC_2D_threshold": 40.0,
        },
    )

    def __init__(self, interval=[0, 3], sessions=None, fix_bads=True):
        super().__init__(
            subjects=list(range(1, 63)),
            sessions_per_subject=11,
            events=dict(right_hand=1, left_hand=2, both_hand=3, rest=4),
            code="Stieger2021",
            interval=interval,
            paradigm="imagery",
            doi="10.1038/s41597-021-00883-1",
        )
        self.fix_bads = fix_bads
        self.sessions = sessions
        self.figshare_id = 13123148  # id on figshare

        assert interval[0] >= -2.00, (
            "The epoch interval has to start earlierst at "
            + f" -2.0 s (specified: {interval[0]:.1f})."
        )
        assert interval[1] < 6.00, (
            "The epoch interval has to end latest at "
            + f"6.0 s (specified: {interval[1]:.1f})."
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError(
                f"Invalid subject number, must be in the range {self.subject_list}"
            )

        path = get_dataset_path(self.code, path)
        basepath = os.path.join(path, f"MNE-{self.code:s}-data")

        file_list = dl.fs_get_file_list(self.figshare_id)
        hash_file_list = dl.fs_get_file_hash(file_list)
        id_file_list = dl.fs_get_file_id(file_list)

        spath = []
        for file_name in id_file_list.keys():
            if ".mat" not in file_name:
                continue
            # Parse session and subject from the  file name
            sub = int(file_name.split("_")[0][1:])
            ses = int(file_name.split("_")[-1].split(".")[0])

            if sub == subject:
                if self.sessions is not None and ses not in self.sessions:
                    continue
                fpath = os.path.join(basepath, file_name)
                if not os.path.exists(fpath):
                    pooch.retrieve(
                        url=BASE_URL + id_file_list[file_name],
                        known_hash=hash_file_list[id_file_list[file_name]],
                        fname=file_name,
                        path=basepath,
                        downloader=pooch.HTTPDownloader(progressbar=True),
                    )
                spath.append(fpath)
        return spath

    def _get_single_subject_data(self, subject):
        file_path = self.data_path(subject)

        subject_data = {}

        for file in file_path:

            session = int(os.path.basename(file).split("_")[2].split(".")[0])

            if self.sessions is not None:
                if session not in set(self.sessions):
                    continue

            container = loadmat(
                file_name=file,
                squeeze_me=True,
                struct_as_record=False,
                verify_compressed_data_integrity=False,
            )["BCI"]

            srate = container.SRATE

            eeg_ch_names = container.chaninfo.label.tolist()
            # adjust naming convention
            eeg_ch_names = [
                ch.replace("Z", "z").replace("FP", "Fp") for ch in eeg_ch_names
            ]
            # extract all standard EEG channels
            montage = mne.channels.make_standard_montage("standard_1005")
            channel_mask = np.isin(eeg_ch_names, montage.ch_names)
            ch_names = [ch for ch, found in zip(eeg_ch_names, channel_mask) if found] + [
                "stim"
            ]
            ch_types = ["eeg"] * channel_mask.sum() + ["stim"]

            X_flat = []
            stim_flat = []
            for i in range(container.data.shape[0]):
                x = container.data[i][channel_mask, :]
                y = container.TrialData[i].targetnumber
                stim = np.zeros_like(container.time[i])
                if (  # check if the trial is artifact-free and long enough
                    container.TrialData[i].artifact == 0
                    and (container.TrialData[i].triallength + 2) > self.interval[1]
                ):
                    # this should be the cue time-point
                    assert (
                        container.time[i][2 * srate] == 0
                    ), "This should be the cue time-point"
                    stim[2 * srate] = y
                X_flat.append(x)
                stim_flat.append(stim[None, :])

            X_flat = np.concatenate(X_flat, axis=1)
            stim_flat = np.concatenate(stim_flat, axis=1)

            p_keep = np.flatnonzero(stim_flat).shape[0] / container.data.shape[0]

            message = (
                "The trial length of this dataset is dynamic."
                f"For the specified interval [{self.interval[0]}, {self.interval[1]}], "
                f"{(1 - p_keep) * 100:.0f}% of the epochs of record {subject}/"
                f"{session} (subject/session) were rejected (artifact or"
                " too short) ."
            )
            if p_keep < 0.5:
                LOGGER.warning(message)
            else:
                LOGGER.info(message)

            eeg_data = np.concatenate([X_flat * 1e-6, stim_flat], axis=0)

            info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=srate)
            raw = mne.io.RawArray(data=eeg_data, info=info, verbose=False)
            raw.set_montage(montage)
            if isinstance(container.chaninfo.noisechan, int):
                badchanidxs = [container.chaninfo.noisechan]
            elif isinstance(container.chaninfo.noisechan, np.ndarray):
                badchanidxs = container.chaninfo.noisechan
            else:
                badchanidxs = []

            for idx in badchanidxs:
                used_channels = (
                    ch_names
                    if (not hasattr(self, "channels") or self.channels is None)
                    else self.channels
                )
                if eeg_ch_names[idx - 1] in used_channels:
                    raw.info["bads"].append(eeg_ch_names[idx - 1])

            if len(raw.info["bads"]) > 0:
                LOGGER.info(
                    "Record %s/%s (subject/session) contains bad channels: %s",
                    subject,
                    session,
                    raw.info["bads"],
                )
                if self.fix_bads:
                    raw = raw.interpolate_bads()
                    LOGGER.info(
                        "Bad channels were interpolated for record %s/%s (subject/session)",
                        subject,
                        session,
                    )

            subject_data[str(session)] = {"0": raw}
        return subject_data
