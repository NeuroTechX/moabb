import numpy as np
from mne.channels import make_standard_montage
from mne.io import read_raw_gdf

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
from moabb.datasets.utils import stim_channels_with_selected_ids
from moabb.utils import _handle_deprecated_kwargs

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
            channel_types={"eeg": 61, "eog": 3, "misc": 32},
            hardware="g.tec medical engineering GmbH",
            sensor_type="active",
            reference="right mastoid",
            ground="AFz",
            software=None,
            filters="0.01-200 Hz bandpass (8th order Chebyshev), 50 Hz notch",
            sensors=[
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "CCP1h",
                "CCP2h",
                "CCP3h",
                "CCP4h",
                "CCP5h",
                "CCP6h",
                "CP1",
                "CP2",
                "CP3",
                "CP4",
                "CP5",
                "CP6",
                "CPP1h",
                "CPP2h",
                "CPP3h",
                "CPP4h",
                "CPP5h",
                "CPP6h",
                "CPz",
                "Cz",
                "F1",
                "F2",
                "F3",
                "F4",
                "FC1",
                "FC2",
                "FC3",
                "FC4",
                "FC5",
                "FC6",
                "FCC1h",
                "FCC2h",
                "FCC3h",
                "FCC4h",
                "FCC5h",
                "FCC6h",
                "FCz",
                "FFC1h",
                "FFC2h",
                "FFC3h",
                "FFC4h",
                "FFC5h",
                "FFC6h",
                "FTT7h",
                "FTT8h",
                "Fz",
                "P1",
                "P2",
                "P3",
                "P4",
                "PPO1h",
                "PPO2h",
                "Pz",
                "TTP7h",
                "TTP8h",
                "armeodummy-0",
                "armeodummy-1",
                "armeodummy-10",
                "armeodummy-11",
                "armeodummy-12",
                "armeodummy-2",
                "armeodummy-3",
                "armeodummy-4",
                "armeodummy-5",
                "armeodummy-6",
                "armeodummy-7",
                "armeodummy-8",
                "armeodummy-9",
                "eog-l",
                "eog-m",
                "eog-r",
                "gesture",
                "index_far",
                "index_middle",
                "index_near",
                "litte_far",
                "litte_near",
                "middle_far",
                "middle_near",
                "middle_ring",
                "pitch",
                "ring_far",
                "ring_little",
                "ring_near",
                "roll",
                "thumb_far",
                "thumb_index",
                "thumb_near",
                "thumb_palm",
                "wrist_bend",
            ],
            line_freq=50.0,
            montage="standard_1005",
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_emg=False, other_physiological=None
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=15,
            health_status="healthy",
            gender={"female": 9, "male": 6},
            age_mean=27.0,
            age_std=5.0,
            age_min=22.0,
            age_max=40.0,
            handedness={"right": 14, "left": 1},
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={
                "right_elbow_flexion": 1536,
                "right_elbow_extension": 1537,
                "right_supination": 1538,
                "right_pronation": 1539,
                "right_hand_close": 1540,
                "right_hand_open": 1541,
                "rest": 1542,
            },
            paradigm="imagery",
            n_classes=7,
            class_labels=[
                "elbow_flexion",
                "elbow_extension",
                "forearm_supination",
                "forearm_pronation",
                "hand_open",
                "hand_close",
                "rest",
            ],
            trial_duration=None,
            study_design="Trial-based paradigm with sustained movements/motor imagery. Each trial: fixation cross at 0s, cue presentation at 2s, sustained movement/MI execution. Subjects performed both movement execution (ME) and motor imagery (MI) in separate sessions.",
            feedback_type="none",
            stimulus_type="visual cue",
            synchronicity="synchronous",
            mode="offline",
            instructions="Subjects were instructed to execute sustained movements in ME session and perform kinesthetic motor imagery in MI session. For rest class, subjects were instructed to avoid any movement and to stay in the starting position.",
            stimulus_presentation=None,
            has_training_test_split=False,
        ),
        documentation=DocumentationMetadata(
            doi="10.1371/journal.pone.0182578",
            repository="BNCI Horizon 2020",
            investigators=[
                "Patrick Ofner",
                "Andreas Schwarz",
                "Joana Pereira",
                "Gernot R. Müller-Putz",
            ],
            senior_author="Gernot R. Müller-Putz",
            contact_info=["gernot.mueller@tugraz.at"],
            institution="Graz University of Technology",
            institution_department="Institute of Neural Engineering, BCI-Lab",
            country="AT",
            publication_year=2017,
            funding=[
                "H2020-643955 MoreGrasp",
                "ERC Consolidator Grant ERC-681231 Feel Your Reach",
            ],
            ethics_approval=[
                "Medical University of Graz, approval number 28-108 ex 15/16"
            ],
            associated_paper_doi="10.1371/journal.pone.0182578",
            keywords=[
                "upper limb movements",
                "EEG",
                "motor imagery",
                "movement execution",
                "low-frequency",
                "time-domain",
                "BCI",
                "neuroprosthesis",
            ],
            acknowledgements="Data are available from the BNCI Horizon 2020 database at http://bnci-horizon-2020.eu/database/data-sets (accession number 001-2017) and from Zenodo at DOI 10.5281/zenodo.834976",
            data_url="https://bnci-horizon-2020.eu/database/data-sets",
            license="CC-BY-4.0",
        ),
        sessions_per_subject=2,
        runs_per_session=10,
        sessions=["movement_execution", "motor_imagery"],
        data_processed=False,
        file_format="gdf",
        tags=Tags(
            pathology=["Healthy"],
            modality=["Motor"],
            type=["Motor Imagery", "Motor Execution"],
        ),
        preprocessing=PreprocessingMetadata(preprocessing_applied=False),
        signal_processing=SignalProcessingMetadata(
            classifiers=["sLDA"],
            feature_extraction=[
                "time-domain signals",
                "discriminative spatial patterns (DSP)",
            ],
            spatial_filters=["sLORETA source localization"],
            frequency_bands={"analyzed_range": [0.3, 3.0]},
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="10x10-fold cross-validation",
            cv_folds=10,
            evaluation_type=["within-session"],
        ),
        performance={
            "mov_vs_mov_ME": 55.0,
            "mov_vs_rest_ME": 87.0,
            "mov_vs_mov_MI": 27.0,
            "mov_vs_rest_MI": 73.0,
        },
        bci_application=BCIApplicationMetadata(
            applications=["neuroprosthesis", "robotic_arm"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=[
                "elbow_flexion",
                "elbow_extension",
                "forearm_supination",
                "forearm_pronation",
                "hand_open",
                "hand_close",
            ],
            cue_duration_s=None,
            imagery_duration_s=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials=420,
            n_trials_per_class={
                "elbow_flexion": 60,
                "elbow_extension": 60,
                "forearm_supination": 60,
                "forearm_pronation": 60,
                "hand_open": 60,
                "hand_close": 60,
                "rest": 60,
            },
            trials_context="per_session",
        ),
        abstract="How neural correlates of movements are represented in the human brain is of ongoing interest and has been researched with invasive and non-invasive methods. In this study, we analyzed the encoding of single upper limb movements in the time-domain of low-frequency electroencephalography (EEG) signals. Fifteen healthy subjects executed and imagined six different sustained upper limb movements. We classified these six movements and a rest class and obtained significant average classification accuracies of 55% (movement vs movement) and 87% (movement vs rest) for executed movements, and 27% and 73%, respectively, for imagined movements. Furthermore, we analyzed the classifier patterns in the source space and located the brain areas conveying discriminative movement information. The classifier patterns indicate that mainly premotor areas, primary motor cortex, somatosensory cortex and posterior parietal cortex convey discriminative movement information. The decoding of single upper limb movements is specially interesting in the context of a more natural non-invasive control of e.g., a motor neuroprosthesis or a robotic arm in highly motor disabled persons.",
        methodology="Subjects performed 6 sustained upper limb movements (elbow flexion/extension, forearm supination/pronation, hand open/close) plus rest in two separate sessions (movement execution and motor imagery). EEG was recorded from 61 channels, filtered to 0.3-3 Hz, and classified using shrinkage LDA with discriminative spatial patterns. Source localization was performed using sLORETA. Classification employed both single time-point and time-window approaches with 10x10-fold cross-validation.",
    )

    def __init__(
        self,
        imagined=True,
        executed=True,
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
        **kwargs,
    ):
        deprecated_renames = {"Imagined": "imagined", "Executed": "executed"}
        resolved = _handle_deprecated_kwargs(kwargs, deprecated_renames, "Ofner2017")
        imagined = resolved.get("imagined", imagined)
        executed = resolved.get("executed", executed)

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
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
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
            # Correct channel names for subject 1 execution files where GDF
            # stores generic "eeg-0".."eeg-60" instead of 10-20 labels.
            _correct_eeg_names = [
                "F3",
                "F1",
                "Fz",
                "F2",
                "F4",
                "FFC5h",
                "FFC3h",
                "FFC1h",
                "FFC2h",
                "FFC4h",
                "FFC6h",
                "FC5",
                "FC3",
                "FC1",
                "FCz",
                "FC2",
                "FC4",
                "FC6",
                "FTT7h",
                "FCC5h",
                "FCC3h",
                "FCC1h",
                "FCC2h",
                "FCC4h",
                "FCC6h",
                "FTT8h",
                "C5",
                "C3",
                "C1",
                "Cz",
                "C2",
                "C4",
                "C6",
                "TTP7h",
                "CCP5h",
                "CCP3h",
                "CCP1h",
                "CCP2h",
                "CCP4h",
                "CCP6h",
                "TTP8h",
                "CP5",
                "CP3",
                "CP1",
                "CPz",
                "CP2",
                "CP4",
                "CP6",
                "CPP5h",
                "CPP3h",
                "CPP1h",
                "CPP2h",
                "CPP4h",
                "CPP6h",
                "P3",
                "P1",
                "Pz",
                "P2",
                "P4",
                "PPO1h",
                "PPO2h",
            ]
            data = {}
            for ii, path in enumerate(paths):
                raw = read_raw_gdf(
                    path, eog=eog, misc=range(64, 96), preload=True, verbose="ERROR"
                )
                # Fix generic channel names if present
                generic = [ch for ch in raw.ch_names if ch.startswith("eeg-")]
                if generic and len(generic) == len(_correct_eeg_names):
                    raw.rename_channels(dict(zip(generic, _correct_eeg_names)))
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
