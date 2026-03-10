import os.path as osp
import zipfile as z
from collections import OrderedDict

import mne
import numpy as np

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
from moabb.utils import _handle_deprecated_kwargs


Castillos2023_URL = "https://zenodo.org/records/8255618"

TRIAL_PRESENTATION_TIME = 2.2


class BaseCastillos2023(BaseDataset):

    def __init__(
        self,
        events,
        sessions_per_subject,
        code,
        paradigm,
        paradigm_type,
        window_size=0.25,
        subjects=None,
        sessions=None,
        **kwargs,
    ):
        deprecated_renames = {"WindowSize": "window_size"}
        resolved = _handle_deprecated_kwargs(
            kwargs, deprecated_renames, "BaseCastillos2023"
        )
        window_size = resolved.get("window_size", window_size)

        super().__init__(
            subjects=list(range(1, 12 + 1)),
            sessions_per_subject=sessions_per_subject,
            events=events,
            code=code,
            interval=(0, window_size),
            paradigm=paradigm,
            doi="https://doi.org/10.1016/j.neuroimage.2023.120446",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )
        self.paradigm_type = paradigm_type
        self.sfreq = 500
        self.fps = 60
        self.n_channels = 32
        self.window_size = window_size

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""
        file_path_list = self.data_path(subject, self.paradigm_type)
        raw = mne.io.read_raw_eeglab(file_path_list[0], preload=True, verbose=False)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw = raw.set_montage(montage)
        # Strip the annotations that were script to make them easier to process
        events, event_id = mne.events_from_annotations(
            raw, event_id="auto", verbose=False
        )
        to_remove = []
        for idx in range(len(raw.annotations.description)):
            if (
                ("collects" in raw.annotations.description[idx])
                or ("iti" in raw.annotations.description[idx])
                or (raw.annotations.description[idx] == "[]")
            ):
                to_remove.append(idx)
            else:
                code = raw.annotations.description[idx].split("_")[0]
                lab = raw.annotations.description[idx].split("_")[1]
                code = code.replace("\n", "")
                code = code.replace("[", "")
                code = code.replace("]", "")
                code = code.replace(" ", "")
                raw.annotations.description[idx] = code + "_" + lab
        to_remove = np.array(to_remove)
        if len(to_remove) > 0:
            raw.annotations.delete(to_remove)

        # Get the labels and data
        events, event_id = mne.events_from_annotations(
            raw, event_id="auto", verbose=False
        )
        shift = 0.0
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=shift,
            tmax=2.2 + shift,
            baseline=(None, None),
            preload=False,
            verbose=False,
        )
        labels = epochs.events[..., -1]
        onset_code = epochs.events[..., 0]
        labels -= np.min(labels)
        data = epochs.get_data()
        self.codes = self._code2array(event_id)

        n_samples_windows = int(self.window_size * self.sfreq)

        # Get the windows epoch of each frame, the label of each frame and the onset for each frame in sample time
        raw_window, y_window, frame_taken = self._to_window_by_frame(
            data, labels, n_samples_windows, self.codes
        )
        onset, onset_0 = self._onset_annotations(frame_taken, y_window, onset_code, 1, 60)

        # Add trial-level annotations so trial identity survives BIDS export.
        # SetRawAnnotations transfers extras by sample position to the first
        # bit event of each trial, which then appear as columns in events.tsv.
        trial_onsets_sec = onset_code / raw.info["sfreq"]
        trial_annotations = mne.Annotations(
            onset=trial_onsets_sec,
            duration=[0.0] * len(trial_onsets_sec),
            description=["_trial_meta"] * len(trial_onsets_sec),
        )
        trial_annotations.extras = [{"trial_id": int(lbl)} for lbl in labels]
        raw.set_annotations(raw.annotations + trial_annotations)

        # Create stim channel with trial information (i.e., symbols)
        # Specifically: 200 = symbol-0, 201 = symbol-1, 202 = symbol-2, etc.
        raw = add_stim_channel_trial(raw, onset_code, labels, offset=200)
        # Create stim channel with epoch information (i.e., 1 / 0, or on / off)
        # Specifically: 100 = "0", 101 = "1"
        raw = add_stim_channel_epoch(
            raw,
            np.concatenate([onset, onset_0]),
            np.concatenate([np.ones(onset.shape), np.zeros(onset_0.shape)]),
            offset=100,
        )

        # There is only one session, one trial of 60 subtrials
        sessions = {"0": {}}
        sessions["0"]["0"] = raw

        return sessions

    def _code2array(self, event_id):
        """Return the code of the event ID in a good format"""
        codes = OrderedDict()
        for k, v in event_id.items():
            code = k.split("_")[0]
            code = code.replace(".", "").replace("2", "")
            codes[v - 1] = np.array(list(map(int, code)))
        return codes

    def data_path(
        self,
        subject,
        paradigm_type,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
    ):
        """Return the data paths of a single subject."""
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        subject_paths = []

        url = "https://zenodo.org/records/8255618/files/4Class-CVEP.zip"
        path_zip = dl.data_dl(url, "4Class-VEP", path, force_update, verbose)
        path_folder = path_zip.rstrip("4Class-VEP.zip")

        # check if has to unzip
        if not (osp.isdir(path_folder + "4Class-VEP")):
            zip_ref = z.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)

        subject_paths.append(
            path_folder
            + "4Class-CVEP/P{:d}/P{:d}_{:s}.set".format(subject, subject, paradigm_type)
        )

        return subject_paths

    def _to_window_by_frame(
        self, data, labels, n_samples_windows, codes, offset=0, focus_rising=None
    ):
        """
        Return the window epochs, the labels and the taken index of each frame during the presentation of the
        different stimuli

        Parameters
        ----------
        data: List | np.ndarray
            The data array of the epochs of the trials of the experiment.
        labels: List | np.ndarray
            The labels of the epochs(ie. frame)
        n_samples_windows: List | np.ndarray
            The number of sample time in the window.
        codes: np.ndarray
            The codebook containing each presented code of shape (nb_bits, nb_codes), sampled at the presentation rate.
        offset: int (default: 100)
            The integer value to start the window after the onset of the corresponding frame
        focus_rising: bool (default: "stim_epoch")
            Boolean to focus on the rising or on the all the code.

        Returns
        -------
        X : np.array
            The array of the epoch starting at each frame.
        Y : np.array
            The array of the label of each epochs
        idx_taken : np.array
            The array of the array taken
        """
        length = int((2.2 - self.window_size) * self.fps)
        X = np.empty(shape=((length) * data.shape[0], self.n_channels, n_samples_windows))
        Y = np.empty(shape=((length) * data.shape[0]), dtype=int)
        idx_taken = []
        for trial_nb, trial in enumerate(data):
            lab = labels[trial_nb]
            c = codes[lab]
            labels_upsampled = np.repeat(c, self.fps // self.fps)
            labels_upsampled = np.concatenate(
                (np.zeros(int(offset * self.fps), dtype=int), np.array(labels_upsampled))
            )
            if focus_rising is not None:
                hi_indices = []
                for idx in range(1, len(labels_upsampled)):
                    if (
                        (focus_rising is not None)
                        and (labels_upsampled[idx - 1] == 0)
                        and (labels_upsampled[idx] == 1)
                    ):
                        hi_indices.append(idx)
                focused_labels = np.zeros(length)
                for idx in hi_indices:
                    focused_labels[idx : idx + 4] = 1
            else:
                focused_labels = labels_upsampled.copy()

            for idx in range(length):
                X[trial_nb * length + idx] = trial[:, idx : idx + n_samples_windows]
                Y[trial_nb * length + idx] = focused_labels[idx]
                idx_taken.append(trial_nb * length + idx)
        X = X.astype(np.float32)
        return X, Y, idx_taken

    def _onset_annotations(
        self, onset_window, label_window, onset_code, nb_seq_min, nb_seq_max
    ):
        """
        Return the onset in second of the frame where the flash is on and the onset in second of the frame where the flash is off

        Parameters
        ----------
        onset_window: List | np.ndarray
            The list of the onset of all the frame taken that appear in the stimuli
        label_window: List | np.ndarray
            The labels of the epochs(ie. frame)
        onset_code: List | np.ndarray
            The list of the onset of the first frame of each code
        nb_seq_min: int
            The first sequence (ie code) to start from
        nb_seq_max: int
            The last sequence (ie code) + 1 to finish calcul of the onset

        Returns
        -------
        onset_1 : np.array
            the onset in second of the frame where the flash is on
        onset_0 : np.array
            the onset in second of the frame where the flash is off
        """
        assert self.sfreq != 0
        new_onset_1 = []
        new_onset_0 = []
        current_code = 0
        onset_code = np.ceil(onset_code * self.fps / self.sfreq)
        nb_seq_min -= 1
        onset_shift = onset_code[current_code + nb_seq_min]
        time_trial = TRIAL_PRESENTATION_TIME - self.window_size
        for i, o in enumerate(onset_window):
            if label_window[i] == 1:
                if current_code == nb_seq_max - 1 - nb_seq_min:
                    new_onset_1.append(o + onset_shift)
                else:
                    if (
                        o + onset_shift
                        >= onset_code[current_code + nb_seq_min] + time_trial * self.fps
                    ):
                        current_code += 1
                        onset_shift = (
                            onset_code[current_code + nb_seq_min]
                            - time_trial * self.fps * current_code
                        )
                    new_onset_1.append(o + onset_shift)
            else:
                if current_code == nb_seq_max - 1 - nb_seq_min:
                    new_onset_0.append(o + onset_shift)
                else:
                    if (
                        o + onset_shift
                        >= onset_code[current_code + nb_seq_min] + time_trial * self.fps
                    ):
                        current_code += 1
                        onset_shift = (
                            onset_code[current_code + nb_seq_min]
                            - time_trial * self.fps * current_code
                        )
                    new_onset_0.append(o + onset_shift)
        new_onset_0 = np.array(list(filter(lambda i: i not in new_onset_1, new_onset_0)))
        return np.array(new_onset_1) / self.fps, np.array(new_onset_0) / self.fps


class CastillosBurstVEP100(BaseCastillos2023):
    """c-VEP and Burst-VEP dataset from Castillos et al. (2023)

    Dataset [1]_ from the study on burst-VEP [2]_.

    **Dataset description**

    Participants were comfortably seated and instructed to read and sign the informed consent. EEG data were recorded
    using a BrainProduct LiveAmp 32 active electrodes wet-EEG setup with a sample rate of 500 Hz to record the surface
    brain activity. The 32 electrodes were placed following the 10–20 international system on a BrainProduct Acticap. The
    ground electrode was placed at the FPz electrode location and all electrodes were referenced to the FCz electrode. The
    impedance of all electrodes was brought below 25kOhm prior to recording onset. Once equipped with the EEG system,
    volunteers were asked to focus on four targets that were cued sequentially in a random order for 0.5 s, followed by a
    2.2 s stimulation phase, before a 0.7 s inter-trial period. The cue sequence for each trial was pseudo-random and
    different for each block. After each block, a pause was observed and subjects had to press the space bar to continue.
    The participants were presented with fifteen blocks of four trials for each of the four conditions (burst or msequence x
    40% or 100%). The task was implemented in Python using the Psychopy toolbox. The four discs were all
    150 pixels, without borders, and were presented on the following LCD monitor: Dell P2419HC, 1920 x 1080 pixels, 265
    cd/m2, and 60 Hz refresh rate. After completing the experiment and removing the EEG equipment, the participants were
    asked to provide subjective ratings for the different stimuli conditions. These stimuli included burst c-VEP with 100%
    amplitude, burst c-VEP with 40% amplitude, m-sequences with 100% amplitude, and m-sequences with 40% amplitude. Each
    stimulus was presented three times in a pseudo-random order. Following the presentation of each stimulus, participants
    were presented with three 11-points scales and were asked to rate the visual comfort, visual tiredness, and
    intrusiveness using a mouse. In total, participants completed 12 ratings (3 repetitions x 4 types of stimuli) for
    each of the three scales.

    References
    ----------

    .. [1] Kalou Cabrera Castillos. (2023). 4-class code-VEP EEG data [Data set]. Zenodo.(dataset).
           DOI: https://doi.org/10.5281/zenodo.8255618

    .. [2] Kalou Cabrera Castillos, Simon Ladouce, Ludovic Darmet, Frédéric Dehais. Burst c-VEP Based BCI: Optimizing stimulus
           design for enhanced classification with minimal calibration data and improved user experience,NeuroImage,Volume 284,
           2023,120446,ISSN 1053-8119
           DOI: https://doi.org/10.1016/j.neuroimage.2023.120446

    Notes
    -----

    .. versionadded:: 1.1.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=32,
            channel_types={"eeg": 32},
            sensors=[
                "C3",
                "C4",
                "CP1",
                "CP2",
                "CP5",
                "CP6",
                "Cz",
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
                "Oz",
                "P10",
                "P3",
                "P4",
                "P7",
                "P8",
                "P9",
                "Pz",
                "T7",
                "T8",
            ],
            sensor_type="eeg",
            reference="FCz",
            ground="FPz",
            hardware="BrainProducts LiveAmp 32",
            software=None,
            filters={
                "notch": {
                    "freq": 50.0,
                    "bandwidth": 0.2,
                    "order": 16,
                    "type": "IIR cut-band",
                }
            },
            line_freq=50.0,
            montage="standard_1020",
            impedance_threshold_kohm=25.0,
            auxiliary_channels=None,
            cap_manufacturer="BrainProducts",
            cap_model="Acticap",
            electrode_type="active",
            electrode_material=None,
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="healthy",
            gender={"female": 4, "male": 8},
            age_mean=30.6,
            age_std=7.1,
            age_min=None,
            age_max=None,
            ages=None,
            handedness=None,
            clinical_population=None,
            bci_experience=None,
            sexes=None,
            handedness_list=None,
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="cvep",
            task_type="target selection",
            events={"0": 100, "1": 101},
            n_classes=2,
            class_labels=["0", "1"],
            trials_per_class=None,
            trial_duration=2.2,
            tasks=["visual attention", "target selection"],
            study_design="factorial within-subject",
            study_domain="BCI performance and user experience",
            feedback_type="none",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            has_training_test_split=False,
            instructions="Focus on cued targets sequentially in random order",
            cog_atlas_id=None,
            cog_po_id=None,
            stimulus_presentation={
                "software": "PsychoPy",
                "monitor": "Dell P2419HC",
                "resolution": "1920x1080",
                "refresh_rate_hz": "60",
            },
            hed_tags=None,
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.neuroimage.2023.120446",
            description="Burst c-VEP based BCI study comparing novel burst code sequences to traditional m-sequences at two amplitude depths (100% and 40%) to optimize classification performance, minimize calibration data, and improve user experience",
            investigators=[
                "Kalou Cabrera Castillos",
                "Simon Ladouce",
                "Ludovic Darmet",
                "Frédéric Dehais",
            ],
            institution="Institut Supérieur de l'Aéronautique et de l'Espace (ISAE-SUPAERO)",
            country="FR",
            repository="Zenodo",
            data_url="https://zenodo.org/record/8255618",
            license="CC BY",
            publication_year=2023,
            senior_author="Frédéric Dehais",
            contact_info=["kalou.cabrera-castillos@isae-supaero.fr"],
            associated_paper_doi="10.1016/j.neuroimage.2023.120446",
            funding=[
                "AID (Powerbrain project), France",
                "AXA Research Fund Chair for Neuroergonomics, France",
                "Chair for Neuroadaptive Technology, Artificial and Natural Intelligence Toulouse Institute (ANITI), France",
            ],
            institution_address="10 Av. Edouard Belin, Toulouse, 31400, France",
            institution_department="Human Factors and Neuroergonomics",
            ethics_approval=[
                "University of Toulouse ethics committee (CER approval number 2020-334)",
                "Declaration of Helsinki",
            ],
            acknowledgements="This work was funded by AID (Powerbrain project), France, the AXA Research Fund Chair for Neuroergonomics, France and Chair for Neuroadaptive Technology, Artificial and Natural Intelligence Toulouse Institute (ANITI), France.",
            how_to_acknowledge=None,
            keywords=[
                "Code-VEP",
                "Reactive BCI",
                "CNN",
                "Amplitude depth reduction",
                "Visual comfort",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        sessions=None,
        contributing_labs=None,
        n_contributing_labs=1,
        data_processed=False,
        file_format="EEGLAB .set",
        external_links={
            "source": "https://zenodo.org/record/8255618",
            "github": "https://github.com/neuroergoISAE/burst_codes",
        },
        tags=Tags(
            pathology=["Healthy"],
            modality=["EEG"],
            type=["reactive BCI", "c-VEP", "visual evoked potentials"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=None,
            preprocessing_steps=None,
            notch_hz=None,
            filter_type=None,
            filter_order=None,
            artifact_methods=None,
            re_reference=None,
            downsampled_to_hz=None,
            epoch_window=None,
            notes=None,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["Convolutional Neural Network (CNN)", "Pearson correlation"],
            feature_extraction=[
                "CNN spatial filtering (8x1 kernel, 16 filters)",
                "CNN temporal filtering (1x32 kernel with dilation 2, 8 filters)",
                "CNN 2D convolution (5x5 kernel, 4 filters)",
                "sliding windows (250ms, 2ms stride)",
            ],
            frequency_bands={
                "analyzed_range": [0.1, 40.0],
            },
            spatial_filters=["CNN 8x1 spatial convolution (16 filters)"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="sequential train/test split",
            cv_folds=None,
            evaluation_type=[
                "offline classification",
                "iterative calibration (1-6 blocks)",
            ],
        ),
        performance={
            "accuracy_percent": 95.6,
            "itr_bits_per_min": 67.49,
            "selection_time_s": 1.5,
            "cnn_training_time_s": 15.0,
            "burst_40_accuracy": 94.2,
            "mseq_100_accuracy": 85.0,
        },
        bci_application=BCIApplicationMetadata(
            applications=["reactive BCI"],
            environment="controlled laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="cvep",
            stimulus_frequencies_hz=None,
            frequency_resolution_hz=None,
            code_type="burst",
            code_length=None,
            n_targets=4,
            n_repetitions=None,
            isi_ms=None,
            soa_ms=None,
            imagery_tasks=None,
            cue_duration_s=0.5,
            imagery_duration_s=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials=60,
            n_trials_per_class=None,
            n_blocks=15,
            block_duration_s=None,
            trials_context="15 blocks x 4 trials per block = 60 trials per subject for burst c-VEP at 100% amplitude",
        ),
        abstract="The utilization of aperiodic flickering visual stimuli under the form of code-modulated Visual Evoked Potentials (c-VEP) represents a pivotal advancement in the field of reactive Brain–Computer Interface (rBCI). This study introduces Burst c-VEP, an innovative variant involving short bursts of aperiodic visual flashes at 2-4 flashes per second. The proposed burst c-VEP sequences exhibited higher accuracy (90.5%-95.6%) compared to m-sequence counterparts (71.4%-85.0%) with mean selection time of 1.5s. Reducing stimulus intensity to 40% amplitude depth only slightly decreased accuracy to 94.2% while substantially improving user experience. The collected dataset and CNN architecture implementation are shared through open-access repositories.",
        methodology="Twelve healthy participants completed an offline 4-class c-VEP protocol using a factorial design. EEG was recorded at 500 Hz using BrainProducts LiveAmp 32-channel system. Participants focused on cued targets with factorial manipulation of pattern type (burst vs m-sequence) and amplitude depth (100% vs 40%). Visual stimuli were presented on a 60 Hz Dell monitor. Burst codes consisted of brief flashes (~50ms) with minimum 200ms inter-burst interval, while m-sequences used Fibonacci-type LFSR with segmented 132-frame subsequences. A CNN architecture with spatial (8x1, 16 filters), temporal (1x32, 8 filters), and 2D convolution (5x5, 4 filters) layers decoded EEG using 250ms sliding windows with 2ms stride. Calibration data ranged from 1-6 blocks (8.8-52.8s). Classification used sequential train/test splits with Pearson correlation for target selection. VEP analysis examined amplitude, latency, and inter-trial coherence. Statistical analyses used 2×2 repeated measures ANOVA.",
    )

    def __init__(self, window_size=0.25, subjects=None, sessions=None, **kwargs):
        deprecated_renames = {"WindowSize": "window_size"}
        resolved = _handle_deprecated_kwargs(
            kwargs, deprecated_renames, self.__class__.__name__
        )
        window_size = resolved.get("window_size", window_size)
        super().__init__(
            events={"0": 100, "1": 101},
            sessions_per_subject=1,
            code="CastillosBurstVEP100",
            paradigm="cvep",
            paradigm_type="burst100",
            window_size=window_size,
            subjects=subjects,
            sessions=sessions,
        )


class CastillosBurstVEP40(BaseCastillos2023):
    """c-VEP and Burst-VEP dataset from Castillos et al. (2023)

    Dataset [1]_ from the study on burst-VEP [2]_.


    **Dataset description**

    Participants were comfortably seated and instructed to read and sign the informed consent. EEG data were recorded
    using a BrainProduct LiveAmp 32 active electrodes wet-EEG setup with a sample rate of 500 Hz to record the surface
    brain activity. The 32 electrodes were placed following the 10–20 international system on a BrainProduct Acticap. The
    ground electrode was placed at the FPz electrode location and all electrodes were referenced to the FCz electrode. The
    impedance of all electrodes was brought below 25kOhm prior to recording onset. Once equipped with the EEG system,
    volunteers were asked to focus on four targets that were cued sequentially in a random order for 0.5 s, followed by a
    2.2 s stimulation phase, before a 0.7 s inter-trial period. The cue sequence for each trial was pseudo-random and
    different for each block. After each block, a pause was observed and subjects had to press the space bar to continue.
    The participants were presented with fifteen blocks of four trials for each of the four conditions (burst or msequence x
    40% or 100%). The task was implemented in Python using the Psychopy toolbox. The four discs were all
    150 pixels, without borders, and were presented on the following LCD monitor: Dell P2419HC, 1920 x 1080 pixels, 265
    cd/m2, and 60 Hz refresh rate. After completing the experiment and removing the EEG equipment, the participants were
    asked to provide subjective ratings for the different stimuli conditions. These stimuli included burst c-VEP with 100%
    amplitude, burst c-VEP with 40% amplitude, m-sequences with 100% amplitude, and m-sequences with 40% amplitude. Each
    stimulus was presented three times in a pseudo-random order. Following the presentation of each stimulus, participants
    were presented with three 11-points scales and were asked to rate the visual comfort, visual tiredness, and
    intrusiveness using a mouse. In total, participants completed 12 ratings (3 repetitions x 4 types of stimuli) for
    each of the three scales.

    References
    ----------

    .. [1] Kalou Cabrera Castillos. (2023). 4-class code-VEP EEG data [Data set]. Zenodo.(dataset).
           DOI: https://doi.org/10.5281/zenodo.8255618

    .. [2] Kalou Cabrera Castillos, Simon Ladouce, Ludovic Darmet, Frédéric Dehais. Burst c-VEP Based BCI: Optimizing stimulus
           design for enhanced classification with minimal calibration data and improved user experience,NeuroImage,Volume 284,
           2023,120446,ISSN 1053-8119
           DOI: https://doi.org/10.1016/j.neuroimage.2023.120446

    Notes
    -----

    .. versionadded:: 1.1.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=32,
            channel_types={"eeg": 32},
            sensors=[
                "C3",
                "C4",
                "CP1",
                "CP2",
                "CP5",
                "CP6",
                "Cz",
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
                "Oz",
                "P10",
                "P3",
                "P4",
                "P7",
                "P8",
                "P9",
                "Pz",
                "T7",
                "T8",
            ],  # Analysis subset mentioned, full 32 from 10-20 system
            sensor_type="eeg",
            reference="FCz",
            ground="FPz",
            hardware="BrainProducts LiveAmp 32",
            software=None,
            filters={
                "line_noise": "IIR cut-band filter between 49.9 and 50.1 Hz of order 16"
            },
            line_freq=50.0,
            montage="standard_1020",
            impedance_threshold_kohm=25.0,
            auxiliary_channels=None,
            cap_manufacturer="BrainProducts",
            cap_model="Acticap",
            electrode_type="active",
            electrode_material=None,
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="healthy",
            gender={"female": 4, "male": 8},
            age_mean=30.6,
            age_std=7.1,
            age_min=None,
            age_max=None,
            ages=None,
            handedness=None,
            clinical_population=None,
            bci_experience=None,
            sexes=None,
            handedness_list=None,
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="cvep",
            task_type="reactive BCI",
            events={"0": 100, "1": 101},
            n_classes=2,
            class_labels=["0", "1"],
            trials_per_class=None,
            trial_duration=2.2,
            tasks=["attend to cued target"],
            study_design="factorial design",
            study_domain="brain-computer interface",
            feedback_type="none",
            stimulus_type="aperiodic visual flashes",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            has_training_test_split=False,
            instructions="Participants were instructed to focus on c-VEP targets cued sequentially",
            cog_atlas_id=None,
            cog_po_id=None,
            stimulus_presentation={
                "screen": "Dell P2419HC, 1920 × 1080 pixels, 265 cd/m2, 60 Hz"
            },
            hed_tags=None,
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.neuroimage.2023.120446",
            description="Burst c-VEP based BCI study optimizing stimulus design for enhanced classification with minimal calibration data and improved user experience. The study introduces an innovative variant of code-VEP called 'Burst c-VEP' involving short bursts of aperiodic visual flashes at 2-4 flashes per second.",
            investigators=[
                "Kalou Cabrera Castillos",
                "Simon Ladouce",
                "Ludovic Darmet",
                "Frédéric Dehais",
            ],
            institution="Institut Supérieur de l'Aéronautique et de l'Espace (ISAE-SUPAERO)",
            country="FR",
            repository="Zenodo",
            data_url="https://zenodo.org/record/8255618",
            license="CC BY",
            publication_year=2023,
            senior_author="Frédéric Dehais",
            contact_info=["kalou.cabrera-castillos@isae-supaero.fr"],
            associated_paper_doi="10.1016/j.neuroimage.2023.120446",
            funding=None,
            institution_address="10 Av. Edouard Belin, Toulouse, 31400, France",
            institution_department="Human Factors and Neuroergonomics",
            ethics_approval=[
                "University of Toulouse ethics committee (CER approval number 2020-334)",
                "Declaration of Helsinki",
            ],
            acknowledgements=None,
            how_to_acknowledge=None,
            keywords=[
                "Code-VEP",
                "Reactive BCI",
                "CNN",
                "Amplitude depth reduction",
                "Visual comfort",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        sessions=None,
        contributing_labs=None,
        n_contributing_labs=1,
        data_processed=False,
        file_format="EEGLAB .set",
        external_links={
            "source": "https://zenodo.org/record/8255618",
        },
        tags=Tags(
            pathology=["Healthy"],
            modality=["EEG"],
            type=["reactive BCI", "c-VEP"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=None,
            preprocessing_steps=None,
            notch_hz=None,
            filter_type=None,
            filter_order=None,
            artifact_methods=None,
            re_reference=None,
            downsampled_to_hz=None,
            epoch_window=None,
            notes=None,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CNN", "Convolutional Neural Network"],
            feature_extraction=["EEG2Code bitwise decoding"],
            frequency_bands=None,
            spatial_filters=None,
        ),
        cross_validation=CrossValidationMetadata(
            cv_method=None,
            cv_folds=None,
            evaluation_type=["offline"],
        ),
        performance={
            "accuracy_percent": 95.6,
            "burst_100_accuracy_17.6s_calibration": 90.5,
            "burst_100_accuracy_52.8s_calibration": 95.6,
            "mseq_100_accuracy_17.6s_calibration": 71.4,
            "mseq_100_accuracy_52.8s_calibration": 85.0,
            "burst_40_accuracy": 94.2,
            "mean_selection_time_s": 1.5,
        },
        bci_application=BCIApplicationMetadata(
            applications=["brain-computer interface"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="cvep",
            stimulus_frequencies_hz=[2.0, 3.0, 4.0],
            frequency_resolution_hz=None,
            code_type="burst",
            code_length=None,
            n_targets=4,
            n_repetitions=None,
            isi_ms=None,
            soa_ms=None,
            imagery_tasks=None,
            cue_duration_s=0.5,
            imagery_duration_s=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials=60,
            n_trials_per_class=None,
            n_blocks=15,
            block_duration_s=None,
            trials_context="15 blocks x 4 trials per block = 60 trials per subject for burst c-VEP at 40% amplitude",
        ),
        abstract="The utilization of aperiodic flickering visual stimuli under the form of code-modulated Visual Evoked Potentials (c-VEP) represents a pivotal advancement in the field of reactive Brain–Computer Interface (rBCI). A major advantage of the c-VEP approach is that the training of the model is independent of the number and complexity of targets, which helps reduce calibration time. Nevertheless, the existing designs of c-VEP stimuli can be further improved in terms of visual user experience but also to achieve a higher signal-to-noise ratio, while shortening the selection time and calibration process. In this study, we introduce an innovative variant of code-VEP, referred to as 'Burst c-VEP'. This original approach involves the presentation of short bursts of aperiodic visual flashes at a deliberately slow rate, typically ranging from two to four flashes per second. The rationale behind this design is to leverage the sensitivity of the primary visual cortex to transient changes in low-level stimuli features to reliably elicit distinctive series of visual evoked potentials. In comparison to other types of faster-paced code sequences, burst c-VEP exhibit favorable properties to achieve high bitwise decoding performance using convolutional neural networks (CNN), which yields potential to attain faster selection time with the need for less calibration data. Furthermore, our investigation focuses on reducing the perceptual saliency of c-VEP through the attenuation of visual stimuli contrast and intensity to significantly improve users' visual comfort. The proposed solutions were tested through an offline 4-classes c-VEP protocol involving 12 participants. Following a factorial design, participants were instructed to focus on c-VEP targets whose pattern (burst and maximum-length sequences) and amplitude (100% or 40% amplitude depth modulations) were manipulated across experimental conditions. Firstly, the full amplitude burst c-VEP sequences exhibited higher accuracy, ranging from 90.5% (with 17.6 s of calibration data) to 95.6% (with 52.8 s of calibration data), compared to its m-sequence counterpart (71.4% to 85.0%). The mean selection time for both types of codes (1.5 s) compared favorably to reports from previous studies. Secondly, our findings revealed that lowering the intensity of the stimuli only slightly decreased the accuracy of the burst code sequences to 94.2% while leading to substantial improvements in terms of user experience. Taken together, these results demonstrate the high potential of the proposed burst codes to advance reactive BCI both in terms of performance and usability. The collected dataset, along with the proposed CNN architecture implementation, are shared through open-access repositories.",
        methodology="Factorial experimental design with 12 participants. Four conditions: burst or m-sequence codes × 100% or 40% amplitude depth. Participants attended to cued targets presented as aperiodic visual flashes. Burst codes: 50ms flashes at 2-4 Hz with 200ms minimum inter-burst interval. M-sequences: pseudo-random binary sequences at ~10 Hz. EEG recorded at 500 Hz using 32-channel BrainProduct LiveAmp. Analysis on occipital/parietal electrodes. CNN-based bitwise decoding (improved EEG2Code architecture). Each participant completed 15 blocks of 4 trials per condition (60 trials per class, 240 total trials). Trial structure: 700ms ITI, 500ms cue, 2200ms stimulation. Display: Dell P2419HC 60Hz LCD. Luminance: medium grey background (124 lux), 100% condition (168 lux), 40% condition (142 lux). Preprocessing: average re-reference, 50Hz notch filter (IIR order 16), epoching 0-2.2s, baseline removal. Subjective assessments of visual comfort, tiredness, and intrusiveness collected.",
    )

    def __init__(self, window_size=0.25, subjects=None, sessions=None, **kwargs):
        deprecated_renames = {"WindowSize": "window_size"}
        resolved = _handle_deprecated_kwargs(
            kwargs, deprecated_renames, self.__class__.__name__
        )
        window_size = resolved.get("window_size", window_size)
        super().__init__(
            events={"0": 100, "1": 101},
            sessions_per_subject=1,
            code="CastillosBurstVEP40",
            paradigm="cvep",
            paradigm_type="burst40",
            window_size=window_size,
            subjects=subjects,
            sessions=sessions,
        )


class CastillosCVEP100(BaseCastillos2023):
    """c-VEP and Burst-VEP dataset from Castillos et al. (2023)

    Dataset [1]_ from the study on burst-VEP [2]_.

    **Dataset description**

    Participants were comfortably seated and instructed to read and sign the informed consent. EEG data were recorded
    using a BrainProduct LiveAmp 32 active electrodes wet-EEG setup with a sample rate of 500 Hz to record the surface
    brain activity. The 32 electrodes were placed following the 10–20 international system on a BrainProduct Acticap. The
    ground electrode was placed at the FPz electrode location and all electrodes were referenced to the FCz electrode. The
    impedance of all electrodes was brought below 25kOhm prior to recording onset. Once equipped with the EEG system,
    volunteers were asked to focus on four targets that were cued sequentially in a random order for 0.5 s, followed by a
    2.2 s stimulation phase, before a 0.7 s inter-trial period. The cue sequence for each trial was pseudo-random and
    different for each block. After each block, a pause was observed and subjects had to press the space bar to continue.
    The participants were presented with fifteen blocks of four trials for each of the four conditions (burst or msequence x
    40% or 100%). The task was implemented in Python using the Psychopy toolbox. The four discs were all
    150 pixels, without borders, and were presented on the following LCD monitor: Dell P2419HC, 1920 x 1080 pixels, 265
    cd/m2, and 60 Hz refresh rate. After completing the experiment and removing the EEG equipment, the participants were
    asked to provide subjective ratings for the different stimuli conditions. These stimuli included burst c-VEP with 100%
    amplitude, burst c-VEP with 40% amplitude, m-sequences with 100% amplitude, and m-sequences with 40% amplitude. Each
    stimulus was presented three times in a pseudo-random order. Following the presentation of each stimulus, participants
    were presented with three 11-points scales and were asked to rate the visual comfort, visual tiredness, and
    intrusiveness using a mouse. In total, participants completed 12 ratings (3 repetitions x 4 types of stimuli) for
    each of the three scales.

    References
    ----------

    .. [1] Kalou Cabrera Castillos. (2023). 4-class code-VEP EEG data [Data set]. Zenodo.(dataset).
           DOI: https://doi.org/10.5281/zenodo.8255618

    .. [2] Kalou Cabrera Castillos, Simon Ladouce, Ludovic Darmet, Frédéric Dehais. Burst c-VEP Based BCI: Optimizing stimulus
           design for enhanced classification with minimal calibration data and improved user experience,NeuroImage,Volume 284,
           2023,120446,ISSN 1053-8119
           DOI: https://doi.org/10.1016/j.neuroimage.2023.120446

    Notes
    -----

    .. versionadded:: 1.1.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=32,
            channel_types={"eeg": 32},
            sensors=[
                "C3",
                "C4",
                "CP1",
                "CP2",
                "CP5",
                "CP6",
                "Cz",
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
                "Oz",
                "P10",
                "P3",
                "P4",
                "P7",
                "P8",
                "P9",
                "Pz",
                "T7",
                "T8",
            ],
            sensor_type="EEG",
            reference="FCz",
            ground="FPz",
            hardware="BrainProducts LiveAmp",
            software=None,
            filters=None,
            line_freq=50.0,
            montage="standard_1020",
            impedance_threshold_kohm=25.0,
            cap_manufacturer="BrainProducts",
            cap_model="Acticap",
            electrode_type="active",
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="healthy",
            gender={"female": 4, "male": 8},
            age_mean=30.6,
            age_std=7.1,
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="cvep",
            task_type="visual attention",
            events={"0": 100, "1": 101},
            n_classes=2,
            class_labels=["target_1", "target_2", "target_3", "target_4"],
            trial_duration=2.2,
            study_design="factorial design (code type × amplitude depth)",
            study_domain="BCI performance and user experience",
            feedback_type="none",
            stimulus_type="visual flashing",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            has_training_test_split=False,
            instructions="focus on four targets that were cued sequentially in a random order for 0.5 s, followed by a 2.2 s stimulation phase, before a 0.7 s inter-trial period",
            stimulus_presentation={
                "display": "Dell P2419HC LCD monitor",
                "resolution": "1920×1080 pixels",
                "refresh_rate": "60 Hz",
                "brightness": "265 cd/m²",
                "stimulus_size": "150 pixels",
                "background_luminance": "124 lux (50% screen luminance)",
                "on_state_100": "168 lux (100% amplitude depth)",
                "on_state_40": "142 lux (40% amplitude depth)",
                "cue_duration": "0.5 s",
                "stimulation_duration": "2.2 s",
                "inter_trial_interval": "0.7 s",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.neuroimage.2023.120446",
            description="4-class code-VEP BCI dataset comparing burst c-VEP and m-sequence stimulation at two amplitude depths (100% and 40%) to optimize performance and user experience",
            investigators=[
                "Kalou Cabrera Castillos",
                "Simon Ladouce",
                "Ludovic Darmet",
                "Frédéric Dehais",
            ],
            institution="Institut Supérieur de l'Aéronautique et de l'Espace (ISAE-SUPAERO)",
            country="FR",
            repository="Zenodo",
            data_url="https://zenodo.org/record/8255618",
            license="CC BY",
            publication_year=2023,
            senior_author="Frédéric Dehais",
            contact_info=["kalou.cabrera-castillos@isae-supaero.fr"],
            associated_paper_doi="10.1016/j.neuroimage.2023.120446",
            funding=[
                "AID (Powerbrain project), France",
                "AXA Research Fund Chair for Neuroergonomics, France",
                "Chair for Neuroadaptive Technology, Artificial and Natural Intelligence Toulouse Institute (ANITI), France",
            ],
            institution_address="10 Av. Edouard Belin, Toulouse, 31400, France",
            institution_department="Human Factors and Neuroergonomics",
            ethics_approval=[
                "Ethics committee of the University of Toulouse (CER approval number 2020-334)",
                "Declaration of Helsinki",
            ],
            keywords=[
                "Code-VEP",
                "Reactive BCI",
                "CNN",
                "Amplitude depth reduction",
                "Visual comfort",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        data_processed=False,
        file_format="EEGLAB .set",
        external_links={
            "source": "https://zenodo.org/record/8255618",
            "github_code": "https://github.com/neuroergoISAE/burst_codes",
            "paper": "https://doi.org/10.1016/j.neuroimage.2023.120446",
        },
        tags=Tags(
            pathology=["Healthy"],
            modality=["EEG"],
            type=["reactive BCI", "visual evoked potentials"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=None,
            preprocessing_steps=None,
            notch_hz=None,
            filter_type=None,
            highpass_hz=None,
            lowpass_hz=None,
            re_reference=None,
            epoch_window=None,
            notes=None,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["Convolutional Neural Network (CNN)"],
            feature_extraction=[
                "Sliding windows (250ms, 2ms stride)",
                "Standard deviation normalization",
            ],
            spatial_filters=[
                "16 spatial filters via 1D spatial convolution (8×1 kernel)"
            ],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="sequential train/test split",
            evaluation_type=["offline classification"],
        ),
        performance={
            "accuracy_percent": 85.0,
            "itr_bits_per_min": 48.7,
            "selection_time_s": 1.5,
            "cnn_training_time_6blocks_s": 40.0,
            "calibration_data_6blocks_s": 52.8,
        },
        bci_application=BCIApplicationMetadata(
            applications=["reactive BCI"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="cvep",
            code_type="m-sequence (maximum-length sequence)",
            code_length=132,
            n_targets=4,
            n_repetitions=None,
            soa_ms=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials=60,
            n_blocks=15,
            trials_context="15 blocks × 4 trials (one per target) × 4 conditions (burst/mseq × 100%/40%)",
        ),
        abstract="The utilization of aperiodic flickering visual stimuli under the form of code-modulated Visual Evoked Potentials (c-VEP) represents a pivotal advancement in the field of reactive Brain–Computer Interface (rBCI). This study introduces an innovative variant of code-VEP, referred to as 'Burst c-VEP', involving the presentation of short bursts of aperiodic visual flashes at a deliberately slow rate, typically ranging from two to four flashes per second. The proposed solutions were tested through an offline 4-classes c-VEP protocol involving 12 participants. The full amplitude burst c-VEP sequences exhibited higher accuracy, ranging from 90.5% (with 17.6 s of calibration data) to 95.6% (with 52.8 s of calibration data), compared to its m-sequence counterpart (71.4% to 85.0%). The mean selection time for both types of codes (1.5 s) compared favorably to reports from previous studies. Lowering the intensity of the stimuli only slightly decreased the accuracy of the burst code sequences to 94.2% while leading to substantial improvements in terms of user experience.",
        methodology="Factorial experimental design with 12 healthy participants. EEG recorded with BrainProducts LiveAmp 32-channel system at 500 Hz. Four conditions tested: burst c-VEP and m-sequence c-VEP, each at 100% and 40% amplitude depth. Participants focused on cued targets (4 classes) in 15 blocks of 4 trials per condition. CNN-based decoding with 250ms sliding windows. Subjective ratings collected for visual comfort, mental tiredness, and intrusiveness. VEP analysis included amplitude, latency, and inter-trial coherence metrics.",
    )

    def __init__(self, window_size=0.25, subjects=None, sessions=None, **kwargs):
        deprecated_renames = {"WindowSize": "window_size"}
        resolved = _handle_deprecated_kwargs(
            kwargs, deprecated_renames, self.__class__.__name__
        )
        window_size = resolved.get("window_size", window_size)
        super().__init__(
            events={"0": 100, "1": 101},
            sessions_per_subject=1,
            code="CastillosCVEP100",
            paradigm="cvep",
            paradigm_type="mseq100",
            window_size=window_size,
            subjects=subjects,
            sessions=sessions,
        )


class CastillosCVEP40(BaseCastillos2023):
    """c-VEP and Burst-VEP dataset from Castillos et al. (2023)

    Dataset [1]_ from the study on burst-VEP [2]_.


    **Dataset description**

    Participants were comfortably seated and instructed to read and sign the informed consent. EEG data were recorded
    using a BrainProduct LiveAmp 32 active electrodes wet-EEG setup with a sample rate of 500 Hz to record the surface
    brain activity. The 32 electrodes were placed following the 10–20 international system on a BrainProduct Acticap. The
    ground electrode was placed at the FPz electrode location and all electrodes were referenced to the FCz electrode. The
    impedance of all electrodes was brought below 25kOhm prior to recording onset. Once equipped with the EEG system,
    volunteers were asked to focus on four targets that were cued sequentially in a random order for 0.5 s, followed by a
    2.2 s stimulation phase, before a 0.7 s inter-trial period. The cue sequence for each trial was pseudo-random and
    different for each block. After each block, a pause was observed and subjects had to press the space bar to continue.
    The participants were presented with fifteen blocks of four trials for each of the four conditions (burst or msequence x
    40% or 100%). The task was implemented in Python using the Psychopy toolbox. The four discs were all
    150 pixels, without borders, and were presented on the following LCD monitor: Dell P2419HC, 1920 x 1080 pixels, 265
    cd/m2, and 60 Hz refresh rate. After completing the experiment and removing the EEG equipment, the participants were
    asked to provide subjective ratings for the different stimuli conditions. These stimuli included burst c-VEP with 100%
    amplitude, burst c-VEP with 40% amplitude, m-sequences with 100% amplitude, and m-sequences with 40% amplitude. Each
    stimulus was presented three times in a pseudo-random order. Following the presentation of each stimulus, participants
    were presented with three 11-points scales and were asked to rate the visual comfort, visual tiredness, and
    intrusiveness using a mouse. In total, participants completed 12 ratings (3 repetitions x 4 types of stimuli) for
    each of the three scales.

    References
    ----------

    .. [1] Kalou Cabrera Castillos. (2023). 4-class code-VEP EEG data [Data set]. Zenodo.(dataset).
           DOI: https://doi.org/10.5281/zenodo.8255618

    .. [2] Kalou Cabrera Castillos, Simon Ladouce, Ludovic Darmet, Frédéric Dehais. Burst c-VEP Based BCI: Optimizing stimulus
           design for enhanced classification with minimal calibration data and improved user experience,NeuroImage,Volume 284,
           2023,120446,ISSN 1053-8119
           DOI: https://doi.org/10.1016/j.neuroimage.2023.120446

    Notes
    -----

    .. versionadded:: 1.1.0

    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=500.0,
            n_channels=32,
            channel_types={"eeg": 32},
            sensors=[
                "C3",
                "C4",
                "CP1",
                "CP2",
                "CP5",
                "CP6",
                "Cz",
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
                "Oz",
                "P10",
                "P3",
                "P4",
                "P7",
                "P8",
                "P9",
                "Pz",
                "T7",
                "T8",
            ],
            sensor_type="EEG",
            reference="FCz",
            ground="FPz",
            hardware="BrainProducts LiveAmp 32",
            software=None,
            filters={"line_noise_filter": "IIR cut-band filter 49.9-50.1 Hz, order 16"},
            line_freq=50.0,
            montage="standard_1020",
            impedance_threshold_kohm=25.0,
            auxiliary_channels=None,
            cap_manufacturer="BrainProducts",
            cap_model="Acticap",
            electrode_type="active",
            electrode_material=None,
        ),
        participants=ParticipantMetadata(
            n_subjects=12,
            health_status="healthy",
            gender={"female": 4, "male": 8},
            age_mean=30.6,
            age_std=7.1,
            age_min=None,
            age_max=None,
            ages=None,
            handedness=None,
            clinical_population=None,
            bci_experience=None,
            sexes=None,
            handedness_list=None,
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="cvep",
            task_type="reactive BCI",
            events={"0": 100, "1": 101},
            n_classes=2,
            class_labels=["0", "1"],
            trials_per_class=None,
            trial_duration=2.2,
            tasks=["visual_attention"],
            study_design="factorial design",
            study_domain="brain-computer interface",
            feedback_type="none",
            stimulus_type="visual flicker",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            has_training_test_split=False,
            instructions="focus on targets that were cued sequentially in a random order for 0.5 s, followed by a 2.2 s stimulation phase",
            cog_atlas_id=None,
            cog_po_id=None,
            stimulus_presentation={
                "cue_duration": "500 ms",
                "stimulation_duration": "2200 ms",
                "inter_trial_interval": "700 ms",
                "cue_type": "red-bordered square around target stimulus",
                "display": "Dell P2419HC, 1920×1080 pixels, 265 cd/m², 60 Hz",
            },
            hed_tags=None,
        ),
        documentation=DocumentationMetadata(
            doi="10.1016/j.neuroimage.2023.120446",
            description="Burst c-VEP Based BCI: Optimizing stimulus design for enhanced classification with minimal calibration data and improved user experience",
            investigators=[
                "Kalou Cabrera Castillos",
                "Simon Ladouce",
                "Ludovic Darmet",
                "Frédéric Dehais",
            ],
            institution="Institut Supérieur de l'Aéronautique et de l'Espace (ISAE-SUPAERO)",
            country="FR",
            repository="Zenodo",
            data_url="https://zenodo.org/record/8255618",
            license="CC BY",
            publication_year=2023,
            senior_author="Frédéric Dehais",
            contact_info=["kalou.cabrera-castillos@isae-supaero.fr"],
            associated_paper_doi="10.1016/j.neuroimage.2023.120446",
            funding=None,
            institution_address="10 Av. Edouard Belin, Toulouse, 31400, France",
            institution_department="Human Factors and Neuroergonomics",
            ethics_approval=["University of Toulouse CER approval number 2020-334"],
            acknowledgements=None,
            how_to_acknowledge=None,
            keywords=[
                "Code-VEP",
                "Reactive BCI",
                "CNN",
                "Amplitude depth reduction",
                "Visual comfort",
            ],
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        sessions=None,
        contributing_labs=None,
        n_contributing_labs=1,
        data_processed=False,
        file_format="EEGLAB .set",
        external_links={
            "source": "https://zenodo.org/record/8255618",
        },
        tags=Tags(
            pathology=["Healthy"],
            modality=["EEG"],
            type=["reactive", "code-VEP", "visual"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw",
            preprocessing_applied=None,
            preprocessing_steps=None,
            notch_hz=None,
            filter_type=None,
            filter_order=None,
            artifact_methods=None,
            re_reference=None,
            downsampled_to_hz=None,
            epoch_window=None,
            notes=None,
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CNN (Convolutional Neural Network)"],
            feature_extraction=["sliding windows", "bitwise decoding"],
            frequency_bands=None,
            spatial_filters=None,
        ),
        cross_validation=CrossValidationMetadata(
            cv_method=None,
            cv_folds=None,
            evaluation_type=["offline"],
        ),
        performance={
            "accuracy_percent": 95.6,
            "burst_100_accuracy_17.6s_calibration": 90.5,
            "burst_100_accuracy_52.8s_calibration": 95.6,
            "burst_40_accuracy": 94.2,
            "mseq_100_accuracy_17.6s_calibration": 71.4,
            "mseq_100_accuracy_52.8s_calibration": 85.0,
            "mean_selection_time": 1.5,
        },
        bci_application=BCIApplicationMetadata(
            applications=["reactive BCI"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="cvep",
            stimulus_frequencies_hz=None,
            frequency_resolution_hz=None,
            code_type="m-sequence",
            code_length=None,
            n_targets=4,
            n_repetitions=None,
            isi_ms=None,
            soa_ms=None,
            imagery_tasks=None,
            cue_duration_s=0.5,
            imagery_duration_s=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials=60,
            n_trials_per_class=None,
            n_blocks=15,
            block_duration_s=None,
            trials_context="15 blocks x 4 trials per block = 60 trials per subject for m-sequence c-VEP at 40% amplitude",
        ),
        abstract="The utilization of aperiodic flickering visual stimuli under the form of code-modulated Visual Evoked Potentials (c-VEP) represents a pivotal advancement in the field of reactive Brain–Computer Interface (rBCI). This study introduces an innovative variant of code-VEP, referred to as 'Burst c-VEP', involving the presentation of short bursts of aperiodic visual flashes at a deliberately slow rate (2-4 flashes per second). The study tested an offline 4-classes c-VEP protocol involving 12 participants with factorial design manipulating pattern (burst and m-sequences) and amplitude (100% or 40% depth modulations). Full amplitude burst c-VEP sequences exhibited higher accuracy (90.5% with 17.6s calibration to 95.6% with 52.8s calibration) compared to m-sequence (71.4% to 85.0%). Mean selection time was 1.5s. Lowering intensity to 40% decreased accuracy slightly to 94.2% while improving user experience substantially.",
        methodology="Factorial experimental design with 12 participants. Four conditions: burst vs m-sequence × 100% vs 40% amplitude depth. Participants seated comfortably, presented with 15 blocks of 4 trials for each condition. Each trial: 0.5s cue (red-bordered square), 2.2s stimulation, 0.7s inter-trial interval. Four disc targets (150 pixels) on Dell monitor (60 Hz). Background: medium grey (50% max luminance, 124 lux). 100% condition: modulation to brightest white (168 lux). 40% condition: 40% of grey-to-white range (142 lux). EEG recorded with BrainProducts LiveAmp (32 channels, 500 Hz), impedance <25kΩ. Analysis on subset: O1, O2, Oz, Pz, P3, P4, P8, P9. Preprocessing: average re-reference, IIR notch filter (49.9-50.1 Hz, order 16), epoching (0-2.2s), baseline removal. Classification: CNN architecture with sliding windows for bitwise decoding.",
    )

    def __init__(self, window_size=0.25, subjects=None, sessions=None, **kwargs):
        deprecated_renames = {"WindowSize": "window_size"}
        resolved = _handle_deprecated_kwargs(
            kwargs, deprecated_renames, self.__class__.__name__
        )
        window_size = resolved.get("window_size", window_size)
        super().__init__(
            events={"0": 100, "1": 101},
            sessions_per_subject=1,
            code="CastillosCVEP40",
            paradigm="cvep",
            paradigm_type="mseq40",
            window_size=window_size,
            subjects=subjects,
            sessions=sessions,
        )
