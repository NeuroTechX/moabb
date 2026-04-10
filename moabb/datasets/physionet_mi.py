"""Physionet Motor imagery dataset."""

import mne
import numpy as np
from mne.io import read_raw_edf

from moabb.datasets.base import BaseDataset
from moabb.datasets.download import data_dl, get_dataset_path
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)
from moabb.datasets.utils import stim_channels_with_selected_ids
from moabb.utils import _handle_deprecated_kwargs


BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0/"


class PhysionetMI(BaseDataset):
    """Physionet Motor Imagery dataset.

    Physionet MI dataset: https://physionet.org/pn4/eegmmidb/

    This data set consists of over 1500 one- and two-minute EEG recordings,
    obtained from 109 volunteers [2]_.

    Subjects performed different motor/imagery tasks while 64-channel EEG were
    recorded using the BCI2000 system (http://www.bci2000.org) [1]_.
    Each subject performed 14 experimental runs: two one-minute baseline runs
    (one with eyes open, one with eyes closed), and three two-minute runs of
    each of the four following tasks:

    1. A target appears on either the left or the right side of the screen.
       The subject opens and closes the corresponding fist until the target
       disappears. Then the subject relaxes.

    2. A target appears on either the left or the right side of the screen.
       The subject imagines opening and closing the corresponding fist until
       the target disappears. Then the subject relaxes.

    3. A target appears on either the top or the bottom of the screen.
       The subject opens and closes either both fists (if the target is on top)
       or both feet (if the target is on the bottom) until the target
       disappears. Then the subject relaxes.

    4. A target appears on either the top or the bottom of the screen.
       The subject imagines opening and closing either both fists
       (if the target is on top) or both feet (if the target is on the bottom)
       until the target disappears. Then the subject relaxes.

    .. note::
        Subject 88 was recorded at 128 Hz instead of 160 Hz like all other
        subjects. Loading subject 88 together with other subjects will cause
        errors in any paradigm due to incompatible sampling rates. To avoid
        this, exclude subject 88 when loading the full dataset, e.g.
        ``PhysionetMI(subjects=[s for s in range(1, 110) if s != 88])``.

    Parameters
    ----------

    imagined: bool (default True)
        if True, return runs corresponding to motor imagination.

    executed: bool (default False)
        if True, return runs corresponding to motor execution.

    references
    ----------

    .. [1] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N. and
           Wolpaw, J.R., 2004. BCI2000: a general-purpose brain-computer
           interface (BCI) system. IEEE Transactions on biomedical engineering,
           51(6), pp.1034-1043.

    .. [2] Goldberger, A.L., Amaral, L.A., Glass, L., Hausdorff, J.M., Ivanov,
           P.C., Mark, R.G., Mietus, J.E., Moody, G.B., Peng, C.K., Stanley,
           H.E. and PhysioBank, P., PhysioNet: components of a new research
           resource for complex physiologic signals Circulation 2000 Volume
           101 Issue 23 pp. E215–E220.
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=160.0,
            n_channels=64,
            channel_types={"eeg": 64},
            hardware="Brain Products",
            reference="mastoid",
            software="BCI2000",
            sensors=[
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
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
                "FT8",
                "T7",
                "T8",
                "T9",
                "T10",
                "TP7",
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
            line_freq=60.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_emg=True, other_physiological=["ppg"]
            ),
            sensor_type="EEG",
            montage="standard_1020",
        ),
        participants=ParticipantMetadata(
            n_subjects=109, health_status="healthy", species="human"
        ),
        experiment=ExperimentMetadata(
            events={"left_hand": 2, "right_hand": 3, "feet": 5, "hands": 4, "rest": 1},
            paradigm="imagery",
            n_classes=4,
            class_labels=["left_hand", "right_hand", "feet", "rest"],
            study_design="Multiple BCI paradigms implemented: (1) mu/beta rhythm cursor control where users control vertical cursor movement via sensorimotor rhythm amplitude, (2) SCP cursor control where users control slow cortical potentials for cursor movement, (3) P300 speller for character selection, (4) motor imagery tasks for various applications",
            feedback_type="visual",
            stimulus_type="cue-based",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="both",
            synchronicity="cued",
        ),
        documentation=DocumentationMetadata(
            doi="10.1109/TBME.2004.827072",
            investigators=[
                "Gerwin Schalk",
                "Dennis J. McFarland",
                "Thilo Hinterberger",
                "Niels Birbaumer",
                "Jonathan R. Wolpaw",
            ],
            institution="Wadsworth Center, New York State Department of Health",
            country="USA",
            publication_year=2004,
            senior_author="Jonathan R. Wolpaw",
            institution_address="Albany, NY 12201-0509 USA",
            institution_department="Laboratory of Nervous System Disorders",
            funding=[
                "National Center for Medical Rehabilitation Research, National Institute of Child Health and Human Development, National Institutes of Health (NIH) under Grant HD30146",
                "National Institute of Biomedical Imaging and Bioengineering and the National Institute of Neurological Disorders and Stroke, NIH, under Grant EB00856",
                "Deutsche Forschungsgemeinschaft (DFG)",
                "Federal Ministry of Education and Research (BMBF)",
            ],
            contact_info=["schalk@wadsworth.org"],
            keywords=[
                "Assistive devices",
                "augmentative communication",
                "brain-computer interface (BCI)",
                "ECoG",
                "electroencephalography (EEG)",
                "psychophysiology",
                "rehabilitation",
            ],
            license="ODC-By-1.0",
            repository="Physionet",
        ),
        tags=Tags(pathology=["Healthy"], modality=["Motor"], type=["Motor Imagery"]),
        preprocessing=PreprocessingMetadata(
            data_state="raw EEG stored with all event markers for offline reconstruction",
            preprocessing_applied=True,
            preprocessing_steps=[
                "calibration (linear transformation to microvolts)",
                "spatial filtering",
                "temporal filtering",
            ],
            artifact_methods=["artifact detection"],
            re_reference="common average",
        ),
        signal_processing=SignalProcessingMetadata(
            feature_extraction=[
                "CSP",
                "ERD",
                "ERS",
                "AR",
                "spectral amplitude",
                "slow cortical potentials",
                "P300 evoked potentials",
            ],
            spatial_filters=[
                "Laplacian derivation",
                "common average",
                "independent components",
                "common spatial patterns",
            ],
            frequency_bands={"mu": [8.0, 12.0], "beta": [12.0, 30.0]},
        ),
        bci_application=BCIApplicationMetadata(
            applications=[
                "speller",
                "cursor_control",
                "communication",
                "neuroprosthesis",
                "orthosis",
            ],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand", "feet", "rest"],
        ),
        performance={"itr_bits_per_min": 25.0},
        cross_validation=CrossValidationMetadata(evaluation_type=["online", "offline"]),
        sessions_per_subject=1,
        runs_per_session=6,
        data_processed=True,
        file_format="edf",
        abstract="BCI2000 is a documented general-purpose brain-computer interface (BCI) research and development platform that can incorporate alone or in combination any brain signals, signal processing methods, output devices, and operating protocols. The system is based on a modular design consisting of four modules (operator, source, signal processing, and application) that communicate through a documented network-capable protocol. BCI2000 has been used to create BCI systems for a variety of brain signals (slow cortical potentials, P300 evoked potentials, sensorimotor rhythms, cortical surface potentials, and neuronal action potentials), processing methods (spectral estimation, spatial filtering, linear classification), and applications (cursor control, word processing, wheelchair control, neuroprosthesis control). The system satisfies stringent real-time requirements and facilitates systematic research and development of BCI technology.",
        methodology="The BCI2000 system implements a four-module architecture: 1) Source module digitizes and stores brain signals without preprocessing, 2) Signal processing module performs feature extraction (calibration, spatial filtering, temporal filtering) and feature translation (linear classification, normalization), 3) User application module receives control signals and drives applications with visual/auditory/haptic feedback, 4) Operator module defines system parameters and operation timing. Signal processing uses cascaded signal operators for flexible feature extraction including autoregressive spectral estimation, FIR filtering, slow wave filtering, peak detection, and evoked response averaging. Translation algorithms use linear classifiers and normalizers with optional real-time adaptive parameter updates. All system variables (parameters, event markers, signals) are stored in documented file format with ASCII header and binary data for comprehensive offline analysis.",
    )

    def __init__(
        self,
        imagined=True,
        executed=False,
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
        **kwargs,
    ):
        deprecated_renames = {"Imagined": "imagined", "Executed": "executed"}
        resolved = _handle_deprecated_kwargs(
            kwargs, deprecated_renames, "PhysionetMotorImagery"
        )
        imagined = resolved.get("imagined", imagined)
        executed = resolved.get("executed", executed)

        super().__init__(
            subjects=list(range(1, 110)),
            sessions_per_subject=1,
            events={"left_hand": 2, "right_hand": 3, "feet": 5, "hands": 4, "rest": 1},
            code="PhysionetMotorImagery",
            # website does not specify how long the trials are, but the
            # interval between 2 trial is 4 second.
            interval=[0, 3],
            paradigm="imagery",
            doi="10.1109/TBME.2004.827072",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )
        self.events = {"left_hand": 2, "right_hand": 3, "feet": 5, "hands": 4, "rest": 1}
        self.imagined = imagined
        self.executed = executed
        self.feet_runs = []
        self.hand_runs = []

        if imagined:
            self.feet_runs += [6, 10, 14]
            self.hand_runs += [4, 8, 12]

        if executed:
            self.feet_runs += [5, 9, 13]
            self.hand_runs += [3, 7, 11]

    def _load_one_run(self, subject, run, preload=True):
        raw_fname = self._load_data(subject, runs=[run], verbose="ERROR")[0]
        raw = read_raw_edf(raw_fname, preload=preload, verbose="ERROR")
        raw.rename_channels(lambda x: x.strip("."))
        raw.rename_channels(lambda x: x.upper())
        # fmt: off
        renames = {
            "AFZ": "AFz", "PZ": "Pz", "FPZ": "Fpz", "FCZ": "FCz", "FP1": "Fp1", "CZ": "Cz",
            "OZ": "Oz", "POZ": "POz", "IZ": "Iz", "CPZ": "CPz", "FP2": "Fp2", "FZ": "Fz",
        }
        # fmt: on
        raw.rename_channels(renames)
        raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
        return raw

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        data = {}
        sign = "EEGBCI"
        get_dataset_path(sign, None)

        # hand runs
        idx = 0
        for run in self.hand_runs:
            raw = self._load_one_run(subject, run)
            stim = raw.annotations.description.astype(np.dtype("<U10"))
            stim[stim == "T0"] = "rest"
            stim[stim == "T1"] = "left_hand"
            stim[stim == "T2"] = "right_hand"
            raw.annotations.description = stim
            data[str(idx)] = stim_channels_with_selected_ids(
                raw, desired_event_id=self.events
            )
            idx += 1

        # feet runs
        for run in self.feet_runs:
            raw = self._load_one_run(subject, run)
            # modify stim channels to match new event ids. for feet runs,
            # hand = 2 modified to 4, and feet = 3, modified to 5
            stim = raw.annotations.description.astype(np.dtype("<U10"))
            stim[stim == "T0"] = "rest"
            stim[stim == "T1"] = "hands"
            stim[stim == "T2"] = "feet"
            raw.annotations.description = stim
            data[str(idx)] = stim_channels_with_selected_ids(
                raw, desired_event_id=self.events
            )
            idx += 1

        return {"0": data}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        runs = [1, 2] + self.hand_runs + self.feet_runs

        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        sign = "EEGBCI"
        get_dataset_path(sign, None)
        paths = self._load_data(subject, runs=runs, verbose=verbose)
        return paths

    def _load_data(self, subject, runs, path=None, force_update=False, verbose=None):
        # Function to load the data run by run
        if not hasattr(runs, "__iter__"):
            runs = [runs]

        # get local storage path
        sign = "EEGBCI"
        path = get_dataset_path(sign, path)

        # fetch the file(s)
        data_paths = []
        for run in runs:
            file_part = f"S{subject:03d}/S{subject:03d}R{run:02d}.edf"
            url = BASE_URL + file_part
            p = data_dl(url, sign, path, force_update, verbose)
            data_paths.append(p)
        return data_paths
