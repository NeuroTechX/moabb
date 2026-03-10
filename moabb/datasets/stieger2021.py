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
from moabb.utils import _handle_deprecated_kwargs

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
            n_channels=62,
            channel_types={"eeg": 62},
            montage="10-10",
            hardware="Neuroscan SynAmps RT amplifiers",
            sensor_type="EEG",
            reference=None,
            software="Neuroscan",
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
            age_mean=None,
            age_std=None,
            age_min=18,
            age_max=63,
            handedness="mostly right-handed",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"right_hand": 1, "left_hand": 2, "both_hand": 3, "rest": 4},
            paradigm="imagery",
            n_classes=4,
            class_labels=["right_hand", "left_hand", "both_hands", "rest"],
            feedback_type="visual",
            stimulus_type="target_bar",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="online",
            has_training_test_split=None,
            trial_duration=None,
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
            country="US",
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
            data_url="https://doi.org/10.6084/m9.figshare.13123148.v1",
            funding=[
                "NIH AT009263",
                "NIH EB021027",
                "NIH NS096761",
                "NIH MH114233",
                "NIH EB029354",
            ],
            license="CC-BY-NC-4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Active"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=False,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=None,
            feature_extraction=["ERD", "ERS", "autoregressive model", "power spectrum"],
            frequency_bands={
                "alpha": [10.5, 13.5],
                "mu": [8, 14],
            },
            spatial_filters=["Laplacian (C3/C4 with 4 surrounding electrodes)"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["cross_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["cursor_control"],
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
            n_blocks=18,
            trials_context="per_session",
        ),
        file_format="MAT",
        sessions_per_subject=11,
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

    def __init__(
        self,
        interval=[0, 3],
        sessions=None,
        fix_bads=True,
        subjects=None,
        **kwargs,  # noqa: B006
    ):
        """Initialize Stieger2021 dataset.

        Parameters
        ----------
        interval : list of float, default=[0, 3]
            Epoch interval ``[tmin, tmax]`` in seconds relative to stimulus
            onset.  Because trials in this dataset have variable lengths
            (roughly 0.04 s to 6 s), epochs whose trial is shorter than
            ``tmax`` are automatically rejected.  Use
            :meth:`get_trial_info` to inspect per-subject trial-length
            distributions and :meth:`suggest_interval` to pick an interval
            that retains a desired fraction of trials.
        sessions : list of int or None
            Sessions to load.
        fix_bads : bool
            If True, bad channels are interpolated.
        subjects : list of int or None
            Subjects to load.
        """
        deprecated_renames = {
            "Interval": "interval",
            "Sessions": "sessions",
            "FixBads": "fix_bads",
            "Subjects": "subjects",
        }
        resolved = _handle_deprecated_kwargs(kwargs, deprecated_renames, "Stieger2021")
        interval = resolved.get("interval", interval)
        sessions = resolved.get("sessions", sessions)
        fix_bads = resolved.get("fix_bads", fix_bads)
        subjects = resolved.get("subjects", subjects)

        super().__init__(
            subjects=list(range(1, 63)),
            sessions_per_subject=11,
            events=dict(right_hand=1, left_hand=2, both_hand=3, rest=4),
            code="Stieger2021",
            interval=interval,
            paradigm="imagery",
            doi="10.1038/s41597-021-00883-1",
            selected_subjects=subjects,
            selected_sessions=sessions,
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

    @staticmethod
    def _parse_session(filepath):
        """Extract the integer session number from a Stieger2021 filename."""
        return int(os.path.basename(filepath).split("_")[2].split(".")[0])

    @staticmethod
    def _load_container(filepath):
        """Load the BCI container from a Stieger2021 .mat file."""
        return loadmat(
            file_name=filepath,
            squeeze_me=True,
            struct_as_record=False,
            verify_compressed_data_integrity=False,
        )["BCI"]

    def _get_single_subject_data(self, subject):
        file_path = self.data_path(subject)

        subject_data = {}

        for file in file_path:

            session = self._parse_session(file)

            if self.sessions is not None:
                if session not in set(self.sessions):
                    continue

            container = self._load_container(file)

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
            accepted_triallengths = []
            rejected_lengths = []
            for i in range(container.data.shape[0]):
                x = container.data[i][channel_mask, :]
                td = container.TrialData[i]
                stim = np.zeros_like(container.time[i])
                if td.artifact == 0:
                    if td.triallength >= self.interval[1]:
                        # this should be the cue time-point
                        assert (
                            container.time[i][2 * srate] == 0
                        ), "This should be the cue time-point"
                        stim[2 * srate] = td.targetnumber
                        accepted_triallengths.append(float(td.triallength))
                    else:
                        rejected_lengths.append(td.triallength)
                X_flat.append(x)
                stim_flat.append(stim[None, :])

            X_flat = np.concatenate(X_flat, axis=1)
            stim_flat = np.concatenate(stim_flat, axis=1)

            n_total = container.data.shape[0]
            n_accepted = len(accepted_triallengths)
            p_keep = n_accepted / n_total

            message = (
                "The trial length of this dataset is dynamic. "
                f"For the specified interval [{self.interval[0]}, {self.interval[1]}], "
                f"{(1 - p_keep) * 100:.0f}% of the epochs of record {subject}/"
                f"{session} (subject/session) were rejected (artifact or"
                " too short)."
            )
            if rejected_lengths:
                message += (
                    f" Rejected-for-length trials had durations in "
                    f"[{min(rejected_lengths):.2f}, {max(rejected_lengths):.2f}] s."
                )
            if p_keep < 0.5:
                message += (
                    " Consider using suggest_interval() to find an interval"
                    " that retains more trials."
                )
                LOGGER.warning(message)
            else:
                LOGGER.info(message)

            eeg_data = np.concatenate([X_flat * 1e-6, stim_flat], axis=0)

            info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=srate)
            raw = mne.io.RawArray(data=eeg_data, info=info, verbose=False)
            raw.set_montage(montage)

            # Attach triallength as annotation extras so it flows through
            # to BIDS events.tsv during conversion.
            events_arr = mne.find_events(raw, shortest_event=0, verbose=False)
            if len(events_arr) > 0:
                event_desc = {v: k for k, v in self.event_id.items()}
                annotations = mne.annotations_from_events(
                    events_arr,
                    raw.info["sfreq"],
                    event_desc,
                    first_samp=raw.first_samp,
                    verbose=False,
                )
                assert len(annotations) == len(accepted_triallengths), (
                    f"Mismatch: {len(annotations)} annotations but "
                    f"{len(accepted_triallengths)} trial lengths."
                )
                annotations.extras = [{"triallength": tl} for tl in accepted_triallengths]
                raw.set_annotations(annotations)
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

    def get_trial_info(self, subjects=None):
        """Return trial-length metadata for the requested subjects.

        Loads only the ``TrialData`` metadata from the ``.mat`` files
        (without building full MNE Raw objects) and summarises trial
        durations for artifact-free trials.

        Parameters
        ----------
        subjects : list of int or None
            Subjects to query.  Defaults to all selected subjects
            (``self.subject_list``).

        Returns
        -------
        info : dict
            Nested dict ``{subject_id: {session_id: {...}}}`` where each
            innermost dict contains:

            - ``triallengths`` : np.ndarray – lengths of artifact-free trials
            - ``n_total`` : int – total number of trials
            - ``n_artifact_free`` : int – trials without artifacts
            - ``min`` : float – shortest artifact-free trial
            - ``max`` : float – longest artifact-free trial
            - ``median`` : float – median artifact-free trial length
        """
        if subjects is None:
            subjects = self.subject_list

        info = {}
        for subject in subjects:
            file_paths = self.data_path(subject)
            subject_info = {}
            for file in file_paths:
                session = self._parse_session(file)
                if self.sessions is not None and session not in set(self.sessions):
                    continue

                container = self._load_container(file)

                n_total = container.data.shape[0]
                lengths = []
                for i in range(n_total):
                    td = container.TrialData[i]
                    if td.artifact == 0:
                        lengths.append(float(td.triallength))

                lengths = np.array(lengths)
                subject_info[session] = {
                    "triallengths": lengths,
                    "n_total": n_total,
                    "n_artifact_free": len(lengths),
                    "min": float(lengths.min()) if len(lengths) > 0 else np.nan,
                    "max": float(lengths.max()) if len(lengths) > 0 else np.nan,
                    "median": float(np.median(lengths)) if len(lengths) > 0 else np.nan,
                }
            info[subject] = subject_info
        return info

    def suggest_interval(self, subjects=None, keep_ratio=1.0):
        """Suggest an epoch interval that retains a given fraction of trials.

        Parameters
        ----------
        subjects : list of int or None
            Subjects to consider.  Defaults to all selected subjects.
        keep_ratio : float
            Fraction of artifact-free trials to retain (between 0 and 1).
            For example, ``keep_ratio=0.95`` returns an interval whose
            ``tmax`` equals the 5th percentile of trial lengths, so that
            at least 95 % of artifact-free trials are long enough.

        Returns
        -------
        interval : list of float
            ``[tmin, tmax]`` where ``tmin`` is the current
            ``self.interval[0]`` and ``tmax`` is chosen to satisfy
            ``keep_ratio``.

        Examples
        --------
        >>> ds = Stieger2021(subjects=[1, 2, 3])
        >>> ds.suggest_interval(keep_ratio=0.95)  # doctest: +SKIP
        [0, 2.1]
        """
        if not 0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1].")

        info = self.get_trial_info(subjects=subjects)

        lengths_list = [
            sess_info["triallengths"]
            for subj_info in info.values()
            for sess_info in subj_info.values()
            if len(sess_info["triallengths"]) > 0
        ]

        if len(lengths_list) == 0:
            raise ValueError("No artifact-free trials found for the requested subjects.")

        all_lengths = np.concatenate(lengths_list)
        # The quantile at (1 - keep_ratio) gives the tmax that keeps
        # at least keep_ratio fraction of trials.
        tmax = float(np.quantile(all_lengths, 1 - keep_ratio))

        tmin = self.interval[0]
        if tmax <= tmin:
            raise ValueError(
                f"Cannot satisfy keep_ratio={keep_ratio}: the resulting tmax "
                f"({tmax:.3f} s) is not greater than tmin ({tmin} s). "
                "Try a smaller keep_ratio."
            )

        return [tmin, tmax]
