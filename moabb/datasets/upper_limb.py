import numpy as np
from mne.channels import make_standard_montage
from mne.io import read_raw_gdf

from moabb.datasets.base import BaseDataset
from moabb.datasets.metadata.schema import (
    AcquisitionMetadata,
    AuxiliaryChannelsMetadata,
    BCIApplicationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    FilterDetails,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    SignalProcessingMetadata,
    Tags,
)
from moabb.datasets.utils import stim_channels_with_selected_ids

from . import download as dl


UPPER_LIMB_URL = "https://zenodo.org/record/834976/files/"


class Ofner2017(BaseDataset):
    """Motor Imagery ataset from Ofner et al 2017.

    Upper limb Motor imagery dataset from the paper [1]_.

    **Dataset description**

    We recruited 15 healthy subjects aged between 22 and 40 years with a mean
    age of 27 years (standard deviation 5 years). Nine subjects were female,
    and all the subjects except s1 were right-handed.

    We measured each subject in two sessions on two different days, which were
    not separated by more than one week. In the first session the subjects
    performed ME, and MI in the second session. The subjects performed six
    movement types which were the same in both sessions and comprised of
    elbow flexion/extension, forearm supination/pronation and hand open/close;
    all with the right upper limb. All movements started at a
    neutral position: the hand half open, the lower arm extended to 120
    degree and in a neutral rotation, i.e. thumb on the inner side.
    Additionally to the movement classes, a rest class was recorded in which
    subjects were instructed to avoid any movement and to stay in the starting
    position. In the ME session, we instructed subjects to execute sustained
    movements. In the MI session, we asked subjects to perform kinesthetic MI
    of the movements done in the ME session (subjects performed one ME run
    immediately before the MI session to support kinesthetic MI).

    The paradigm was trial-based and cues were displayed on a computer screen
    in front of the subjects, Fig 2 shows the sequence of the paradigm.
    At second 0, a beep sounded and a cross popped up on the computer screen
    (subjects were instructed to fixate their gaze on the cross). Afterwards,
    at second 2, a cue was presented on the computer screen, indicating the
    required task (one out of six movements or rest) to the subjects. At the
    end of the trial, subjects moved back to the starting position. In every
    session, we recorded 10 runs with 42 trials per run. We presented 6
    movement classes and a rest class and recorded 60 trials per class in a
    session.

    References
    ----------
    .. [1] Ofner, P., Schwarz, A., Pereira, J. and Müller-Putz, G.R., 2017.
           Upper limb movements can be decoded from the time-domain of
           low-frequency EEG. PloS one, 12(8), p.e0182578.
           https://doi.org/10.1371/journal.pone.0182578
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=61,
            channel_types={"eeg": 61},
            hardware="g.tec",
            sensor_type="active electrodes",
            reference="right mastoid",
            software="EEGLAB",
            filters="0.3-3.0 Hz bandpass, 50 Hz notch",
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
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_emg=True,
                other_physiological=["gsr", "ppg"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=152,
            health_status="healthy",
            gender={"female": 9, "male": 6},
        ),
        experiment=ExperimentMetadata(
            paradigm="imagery",
            n_classes=4,
            class_labels=["right_hand", "right_arm", "left_hand", "feet"],
            trial_duration=17.0,
            study_design="avoid any movement and to stay in the starting position.",
            feedback_type="visual cues on computer screen",
            stimulus_type="avatar",
            synchronicity="asynchronous",
            mode="both",
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0182578",
            repository="BNCI Horizon 2020",
            data_url="https://bnci-horizon-2020.eu/database/data-sets",
            funding=["Grant ERC- ERC-"],
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor"],
        ),
        preprocessing=PreprocessingMetadata(
            filter_details=FilterDetails(
                bandpass={"low_cutoff_hz": 0.01, "high_cutoff_hz": 200.0},
                notch_hz=[50],
                filter_type="Butterworth",
                filter_order=4,
            ),
            artifact_methods=["ICA"],
            re_reference="common average reference",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["LDA", "Shrinkage LDA"],
            feature_extraction=["ERD", "Covariance/Riemannian", "ICA"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["prosthetic", "robotic_arm", "vr_ar"],
            environment="outdoor",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
        ),
        data_structure=DataStructureMetadata(
            n_trials=42,
            trials_context="per_run",
        ),
    )

    def __init__(self, imagined=True, executed=False):
        self.imagined = imagined
        self.executed = executed
        self.event_id = {
            "right_elbow_flexion": 1536,
            "right_elbow_extension": 1537,
            "right_supination": 1538,
            "right_pronation": 1539,
            "right_hand_close": 1540,
            "right_hand_open": 1541,
            "rest": 1542,
        }

        n_sessions = int(imagined) + int(executed)
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=n_sessions,
            events=self.event_id,
            code="Ofner2017",
            interval=[0, 3],  # according to paper 2-5
            paradigm="imagery",
            doi="10.1371/journal.pone.0182578",
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""

        sessions = []
        if self.imagined:
            sessions.append((1, "imagination"))

        if self.executed:
            sessions.append((0, "execution"))

        out = {}
        for ses_idx, session in sessions:
            session_name = f"{ses_idx}{session}"
            paths = self.data_path(subject, session=session)

            eog = ["eog-l", "eog-m", "eog-r"]
            montage = make_standard_montage("standard_1005")
            data = {}
            for ii, path in enumerate(paths):
                raw = read_raw_gdf(
                    path, eog=eog, misc=range(64, 96), preload=True, verbose="ERROR"
                )
                raw = raw.set_montage(montage)

                # there is nan in the data
                raw._data[np.isnan(raw._data)] = 0

                raw._data *= 1e-6

                # Modify the annotations to match the name of the command
                stim = raw.annotations.description.astype(np.dtype("<21U"))
                stim[stim == "1536"] = "right_elbow_flexion"
                stim[stim == "1537"] = "right_elbow_extension"
                stim[stim == "1538"] = "right_supination"
                stim[stim == "1539"] = "right_pronation"
                stim[stim == "1540"] = "right_hand_close"
                stim[stim == "1541"] = "right_hand_open"
                stim[stim == "1542"] = "rest"
                raw.annotations.description = stim
                data[str(ii)] = stim_channels_with_selected_ids(raw, self.event_id)

            out[session_name] = data
        return out

    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
        session=None,
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        paths = []

        if session is None:
            sessions = []
            if self.imagined:
                sessions.append("imagination")

            if self.executed:
                sessions.append("execution")
        else:
            sessions = [session]

        # FIXME check the value are in V and not uV.
        for session in sessions:
            for run in range(1, 11):
                url = (
                    f"{UPPER_LIMB_URL}motor{session}_subject{subject}" + f"_run{run}.gdf"
                )
                p = dl.data_dl(url, "UPPERLIMB", path, force_update, verbose)
                paths.append(p)

        return paths
