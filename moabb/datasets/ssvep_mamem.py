"""SSVEP MAMEM1 dataset."""

import logging
import os.path as osp

import numpy as np
import pooch
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat

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

from .base import BaseDataset
from .download import (
    fs_get_file_hash,
    fs_get_file_id,
    fs_get_file_list,
    fs_get_file_name,
    get_dataset_path,
)


log = logging.getLogger(__name__)

MAMEM_URL = "https://ndownloader.figshare.com/files/"

# Specific release
# MAMEM1_URL = 'https://ndownloader.figshare.com/articles/2068677/versions/6'
# MAMEM2_URL = 'https://ndownloader.figshare.com/articles/3153409/versions/4'
# MAMEM3_URL = 'https://ndownloader.figshare.com/articles/3413851/versions/3'

# Alternate Download Location
# MAMEM1_URL = "https://archive.physionet.org/physiobank/database/mssvepdb/dataset1/"
# MAMEM2_URL = "https://archive.physionet.org/physiobank/database/mssvepdb/dataset2/"
# MAMEM3_URL = "https://archive.physionet.org/physiobank/database/mssvepdb/dataset3/"


def mamem_event(eeg, dins, labels=None):
    """Convert DIN field into events.

    Code adapted from
    https://github.com/MAMEM/eeg-processing-toolbox
    """
    thres_split = 2000
    timestamps = dins[1, :]
    samples = dins[3, :]
    numDins = dins.shape[1]

    sampleA = samples[0]
    previous = timestamps[0]
    t_start, freqs = [], []
    s, c = 0, 0
    for i in range(1, numDins):
        current = timestamps[i]
        if (current - previous) > thres_split:
            sampleB = samples[i - 1]
            freqs.append(s // c)
            if (sampleB - sampleA) > 382:
                t_start.append(sampleA)
            sampleA = samples[i]
            s = 0
            c = 0
        else:
            s = s + (current - previous)
            c = c + 1
        previous = timestamps[i]
    sampleB = samples[i - 1]
    freqs.append(s // c)
    t_start.append(sampleA)
    freqs = np.array(freqs, dtype=int) * 2
    freqs = 1000 // freqs
    t_start = np.array(t_start)

    if labels is None:
        freqs_labels = {6: 1, 7: 2, 8: 3, 9: 4, 11: 5}
        for f, t in zip(freqs, t_start):
            eeg[-1, t] = freqs_labels[f]
    else:
        for f, t in zip(labels, t_start):
            eeg[-1, t] = f
    return eeg


class BaseMAMEM(BaseDataset):
    """Base class for MAMEM datasets."""

    def __init__(
        self,
        events,
        sessions_per_subject,
        code,
        doi,
        figshare_id,
        subjects=None,
        sessions=None,
    ):
        super().__init__(
            subjects=list(range(1, 12)),
            events=events,
            interval=[1, 4],
            paradigm="ssvep",
            sessions_per_subject=sessions_per_subject,
            code=code,
            doi=doi,
            selected_subjects=subjects,
            selected_sessions=sessions,
        )
        self.figshare_id = figshare_id

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        fnames = self.data_path(subject)
        filelist = fs_get_file_list(self.figshare_id)
        fsn = fs_get_file_name(filelist)
        sessions = {}

        for fpath in fnames:
            fnamed = fsn[osp.basename(fpath)]
            if fnamed[4] == "x":
                continue
            session_name = "0"
            if self.code == "MAMEM3":
                repetition = len(fnamed) - 10
                run_name = str((ord(fnamed[4]) - 97) * 2 + repetition)
            else:
                run_name = str(ord(fnamed[4]) - 97)

            if self.code == "MAMEM3":
                m = loadmat(fpath)
                ch_names = [e[0] for e in m["info"][0, 0][9][0]]
                sfreq = 128
                montage = make_standard_montage("standard_1020")
                eeg = m["eeg"][:-1] * 1e-6
                stim = np.expand_dims(np.round(m["eeg"][-1], 0).astype(int), 0)
                eeg = np.concatenate([eeg, stim], axis=0)
            else:
                m = loadmat(fpath, squeeze_me=True)
                ch_names = [f"E{i + 1}" for i in range(0, 256)]
                ch_names.append("stim")
                sfreq = 250
                if self.code == "MAMEM2":
                    labels = m["labels"]
                else:
                    labels = None
                eeg = mamem_event(m["eeg"] * 1e-6, m["DIN_1"], labels=labels)
                montage = make_standard_montage("GSN-HydroCel-256")
            ch_types = ["eeg"] * (len(ch_names) - 1) + ["stim"]
            info = create_info(ch_names, sfreq, ch_types)
            raw = RawArray(eeg, info, verbose=False)
            raw.set_montage(montage)
            if session_name not in sessions.keys():
                sessions[session_name] = {}
            if len(sessions[session_name]) == 0:
                sessions[session_name] = {run_name: raw}
            else:
                sessions[session_name][run_name] = raw
        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        sub = f"{subject:02d}"
        sign = self.code.split("-")[0]
        key_dest = f"MNE-{sign.lower():s}-data"
        path = osp.join(get_dataset_path(sign, path), key_dest)

        filelist = fs_get_file_list(self.figshare_id)
        reg = fs_get_file_hash(filelist)
        fsn = fs_get_file_id(filelist)
        gb = pooch.create(path=path, base_url=MAMEM_URL, registry=reg)

        spath = []
        for f in fsn.keys():
            if f[2:4] == sub:
                spath.append(gb.fetch(fsn[f]))
        return spath


class MAMEM1(BaseMAMEM):
    """SSVEP MAMEM 1 dataset.

    Dataset from [1]_.

    EEG signals with 256 channels captured from 11 subjects executing a
    SSVEP-based experimental protocol. Five different frequencies
    (6.66, 7.50, 8.57, 10.00 and 12.00 Hz) have been used for the visual
    stimulation,and the EGI 300 Geodesic EEG System, using a
    stimulation, HydroCel Geodesic Sensor Net (HCGSN) and a sampling rate of
    250 Hz has been used for capturing the signals.

    Check the technical report [2]_ for more detail.
    From [1]_, subjects were exposed to non-overlapping flickering lights from five
    magenta boxes with frequencies [6.66Hz, 7.5Hz, 8.57Hz 10Hz and 12Hz].
    256 channel EEG recordings were captured.

    Each session of the experimental procedure consisted of the following:

    1. 100 seconds of rest.
    2. An adaptation period in which the subject is exposed to eight
       5 second windows of flickering from a magenta box. Each flickering
       window is of a single isolated frequency, randomly chosen from the
       above set, specified in the FREQUENCIES1.txt file under
       'adaptation'. The individual flickering windows are separated by 5
       seconds of rest.
    3. 30 seconds of rest.
    4. For each of the frequencies from the above set in ascending order,
       also specified in FREQUENCIES1.txt under 'main trials':

       1. Three 5 second windows of flickering at the chosen frequency,
           separated by 5 seconds of rest.
       2. 30 seconds of rest.

    This gives a total of 15 flickering windows, or 23 including the
    adaptation period.

    The order of chosen frequencies is the same for each session, although
    there are small-moderate variations in the actual frequencies of each
    individual window. The .freq annotations list the different frequencies at
    a higher level of precision.

    **Note**: Each 'session' in experiment 1 includes an adaptation period, unlike
    experiment 2 and 3 where each subject undergoes only one adaptation period
    before their first 'session'.

    From [3]_:

    **Eligible signals**: The EEG signal is sensitive to external factors that have
    to do with the environment or the configuration of the acquisition setup
    The research stuff was responsible for the elimination of trials that were
    considered faulty. As a result the following sessions were noted and
    excluded from further analysis:
    1. S003, during session 4 the stimulation program crashed
    2. S004, during session 2 the stimulation program crashed, and
    3. S008, during session 4 the Stim Tracker was detuned.
    Furthermore, we must also note that subject S001 participated in 3 sessions
    and subjects S003 and S004 participated in 4 sessions, compared to all
    other subjects that participated in 5 sessions (NB: in fact, there is only
    3 sessions for subjects 1, 3 and 8, and 4 sessions for subject 4 available
    to download). As a result, the utilized dataset consists of 1104 trials of
    5 seconds each.

    **Flickering frequencies**: Usually the refresh rate for an LCD Screen is 60 Hz
    creating a restriction to the number of frequencies that can be selected.
    Specifically, only the frequencies that when divided with the refresh rate
    of the screen result in an integer quotient could be selected. As a result,
    the frequendies that could be obtained were the following: 30.00. 20.00,
    15.00, 1200, 10.00, 857. 7.50 and 6.66 Hz. In addition, it is also
    important to avoid using frequencies that are multiples of another
    frequency, for example making the choice to use 10.00Hz prohibits the use
    of 20.00 and 30.00 Mhz. With the previously described limitations in mind,
    the selected frequencies for the experiment were: 12.00, 10.00, 8.57, 7.50
    and 6.66 Hz.

    **Stimuli Layout**: In an effort to keep the experimental process as simple as
    possible, we used only one flickering box instead of more common choices,
    such as 4 or 5 boxes flickering simultaneously The fact that the subject
    could focus on one stimulus without having the distraction of other
    flickering sources allowed us to minimize the noise of our signals and
    verify the appropriateness of our acquisition setup Nevertheless, having
    concluded the optimal configuration for analyzing the EEG signals, the
    experiment will be repeated with more concurrent visual stimulus.

    **Trial duration**: The duration of each trial was set to 5 seconds, as this
    time was considered adequate to allow the occipital part of the bran to
    mimic the stimulation frequency and still be small enough for making a
    selection in the context

    References
    ----------
    .. [1] Oikonomou, V. P., Liaros, G., Georgiadis, K., Chatzilari, E., Adam, K.,
           Nikolopoulos, S., & Kompatsiaris, I. (2016). Comparative evaluation of
           state-of-the-art algorithms for SSVEP-based BCIs. arXiv preprint
           arXiv:1602.00904.
    .. [2] MAMEM Steady State Visually Evoked Potential EEG Database
           `<https://archive.physionet.org/physiobank/database/mssvepdb/>`_
    .. [3] S. Nikolopoulos, 2016, DataAcquisitionDetails.pdf
           `<https://figshare.com/articles/dataset/MAMEM_EEG_SSVEP_Dataset_I_256_channels_11_subjects_5_frequencies_/2068677?file=3793738>`_
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=256,
            channel_types={"eeg": 256},
            montage="GSN-HydroCel-256",
            hardware="EGI 300 Geodesic EEG System (GES 300)",
            sensor_type="scalp electrodes",
            reference="CAR",
            software="Microsoft Visual Studio 2010 with OpenGL",
            filters="5-48 Hz bandpass, 50 Hz notch",
            sensors=[
                "E1",
                "E10",
                "E100",
                "E101",
                "E102",
                "E103",
                "E104",
                "E105",
                "E106",
                "E107",
                "E108",
                "E109",
                "E11",
                "E110",
                "E111",
                "E112",
                "E113",
                "E114",
                "E115",
                "E116",
                "E117",
                "E118",
                "E119",
                "E12",
                "E120",
                "E121",
                "E122",
                "E123",
                "E124",
                "E125",
                "E126",
                "E127",
                "E128",
                "E129",
                "E13",
                "E130",
                "E131",
                "E132",
                "E133",
                "E134",
                "E135",
                "E136",
                "E137",
                "E138",
                "E139",
                "E14",
                "E140",
                "E141",
                "E142",
                "E143",
                "E144",
                "E145",
                "E146",
                "E147",
                "E148",
                "E149",
                "E15",
                "E150",
                "E151",
                "E152",
                "E153",
                "E154",
                "E155",
                "E156",
                "E157",
                "E158",
                "E159",
                "E16",
                "E160",
                "E161",
                "E162",
                "E163",
                "E164",
                "E165",
                "E166",
                "E167",
                "E168",
                "E169",
                "E17",
                "E170",
                "E171",
                "E172",
                "E173",
                "E174",
                "E175",
                "E176",
                "E177",
                "E178",
                "E179",
                "E18",
                "E180",
                "E181",
                "E182",
                "E183",
                "E184",
                "E185",
                "E186",
                "E187",
                "E188",
                "E189",
                "E19",
                "E190",
                "E191",
                "E192",
                "E193",
                "E194",
                "E195",
                "E196",
                "E197",
                "E198",
                "E199",
                "E2",
                "E20",
                "E200",
                "E201",
                "E202",
                "E203",
                "E204",
                "E205",
                "E206",
                "E207",
                "E208",
                "E209",
                "E21",
                "E210",
                "E211",
                "E212",
                "E213",
                "E214",
                "E215",
                "E216",
                "E217",
                "E218",
                "E219",
                "E22",
                "E220",
                "E221",
                "E222",
                "E223",
                "E224",
                "E225",
                "E226",
                "E227",
                "E228",
                "E229",
                "E23",
                "E230",
                "E231",
                "E232",
                "E233",
                "E234",
                "E235",
                "E236",
                "E237",
                "E238",
                "E239",
                "E24",
                "E240",
                "E241",
                "E242",
                "E243",
                "E244",
                "E245",
                "E246",
                "E247",
                "E248",
                "E249",
                "E25",
                "E250",
                "E251",
                "E252",
                "E253",
                "E254",
                "E255",
                "E256",
                "E26",
                "E27",
                "E28",
                "E29",
                "E3",
                "E30",
                "E31",
                "E32",
                "E33",
                "E34",
                "E35",
                "E36",
                "E37",
                "E38",
                "E39",
                "E4",
                "E40",
                "E41",
                "E42",
                "E43",
                "E44",
                "E45",
                "E46",
                "E47",
                "E48",
                "E49",
                "E5",
                "E50",
                "E51",
                "E52",
                "E53",
                "E54",
                "E55",
                "E56",
                "E57",
                "E58",
                "E59",
                "E6",
                "E60",
                "E61",
                "E62",
                "E63",
                "E64",
                "E65",
                "E66",
                "E67",
                "E68",
                "E69",
                "E7",
                "E70",
                "E71",
                "E72",
                "E73",
                "E74",
                "E75",
                "E76",
                "E77",
                "E78",
                "E79",
                "E8",
                "E80",
                "E81",
                "E82",
                "E83",
                "E84",
                "E85",
                "E86",
                "E87",
                "E88",
                "E89",
                "E9",
                "E90",
                "E91",
                "E92",
                "E93",
                "E94",
                "E95",
                "E96",
                "E97",
                "E98",
                "E99",
            ],
            line_freq=50.0,
            impedance_threshold_kohm=80.0,
            cap_manufacturer="EGI",
            cap_model="HydroCel Geodesic Sensor Net (HCGSN)",
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_type=["vertical"],
                has_emg=True,
                other_physiological=["ecg", "gsr", "ppg"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=11,
            health_status="healthy",
            gender={"male": 8, "female": 3},
            age_min=24,
            age_max=39,
            ages=[24, 37, 39, 31, 27, 28, 26, 31, 29, 37, 25],
            handedness={"right": 10, "left": 1},
            handedness_list=[
                "Right",
                "Right",
                "Right",
                "Right",
                "Left",
                "Right",
                "Right",
                "Right",
                "Right",
                "Right",
                "Right",
            ],
            clinical_population="able-bodied subjects without any known neuro-muscular or mental disorders",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="ssvep",
            n_classes=5,
            class_labels=["6.66 Hz", "7.50 Hz", "8.57 Hz", "10.00 Hz", "12.00 Hz"],
            events={"6.66": 1, "7.50": 2, "8.57": 3, "10.00": 4, "12.00": 5},
            trial_duration=5.0,
            study_design="Subjects focus attention on a single violet box flickering at different frequencies (6.66, 7.50, 8.57, 10.00, 12.00 Hz) presented sequentially. Each frequency is presented for 5 seconds (trial) followed by 5 seconds rest, repeated 3 times per frequency, with 30 seconds rest between different frequencies.",
            stimulus_type="flickering box",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
            feedback_type="none",
            instructions="Subjects were instructed to focus attention on the flickering box, limit movements, and avoid swallowing or blinking during visual stimulation",
            stimulus_presentation={
                "monitor": "22 inch LCD monitor",
                "refresh_rate": "60 Hz",
                "resolution": "1680x1050 pixels",
                "stimulus_color": "violet box",
                "background_color": "black",
                "synchronization": "Stim Tracker ST-100 with light sensor",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.6084/m9.figshare.2068677.v1",
            description="Comparative evaluation of state-of-the-art algorithms for SSVEP-based BCIs",
            investigators=[
                "Vangelis P. Oikonomou",
                "Georgios Liaros",
                "Kostantinos Georgiadis",
                "Elisavet Chatzilari",
                "Katerina Adam",
                "Spiros Nikolopoulos",
                "Ioannis Kompatsiaris",
            ],
            institution="Centre for Research and Technology Hellas (CERTH)",
            country="Greece",
            repository="Figshare",
            data_url="https://dx.doi.org/10.6084/m9.figshare.2068677.v1",
            license="ODC-By-1.0",
            publication_year=2016,
            senior_author="Ioannis Kompatsiaris",
            associated_paper_doi="10.48550/arXiv.1602.00904",
            funding=["H2020-ICT-2014-644780"],
            ethics_approval=[
                "Centre for Research and Technology Hellas ethics committee, dated 3/7/2015, grant H2020-ICT-2014-644780"
            ],
            keywords=[
                "SSVEP",
                "BCI",
                "EEG",
                "brain-computer interface",
                "comparative evaluation",
                "state-of-the-art algorithms",
            ],
        ),
        sessions_per_subject=5,
        runs_per_session=3,
        sessions=["0a", "0b", "0c", "0d", "0e"],
        data_processed=False,
        file_format="MATLAB .mat",
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw EEG signals with DIN markers for synchronization",
            preprocessing_applied=True,
            preprocessing_steps=[
                "bandpass filtering (5-48 Hz)",
                "notch filtering (50 Hz line frequency)",
                "artifact removal (AMUSE or ICA)",
                "CAR re-referencing",
            ],
            highpass_hz=5.0,
            lowpass_hz=48.0,
            bandpass={"low_cutoff_hz": 5.0, "high_cutoff_hz": 48.0},
            notch_hz=50,
            filter_type="Chebyshev (IIR) or FIR",
            artifact_methods=["AMUSE", "ICA (FastICA)"],
            re_reference="CAR",
            notes="Multiple preprocessing methods evaluated in comparative study. Range 5-48 Hz chosen to capture stimulus frequencies and harmonics up to 4th order. EOG artifacts particularly noted for subject S007.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=[
                "LDA",
                "SVM",
                "Random Forest",
                "kNN",
                "Naive Bayes",
                "CCA",
                "AdaBoost",
                "Decision Trees",
            ],
            feature_extraction=[
                "Periodogram",
                "Welch Spectrum",
                "Goertzel algorithm",
                "Yule-AR Spectrum",
                "FFT",
                "PSD",
                "Discrete Wavelet Transform",
            ],
            spatial_filters=["CAR", "CSP", "Minimum Energy"],
            frequency_bands={
                "analyzed_range": [5.0, 48.0],
            },
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="leave-one-subject-out",
            evaluation_type=["cross_subject"],
        ),
        performance={
            "accuracy_percent": 74.42,
            "optimal_configuration_improvement": 7.0,
            "subject_S007_improvement_with_AMUSE": 30.0,
        },
        bci_application=BCIApplicationMetadata(
            applications=["communication", "control"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep",
            stimulus_frequencies_hz=[6.66, 7.50, 8.57, 10.00, 12.00],
            n_targets=5,
            n_repetitions=3,
        ),
        data_structure=DataStructureMetadata(
            n_trials=1104,
            trials_context="Total 1104 trials across all subjects. Each session includes 23 trials (8 adaptation + 15 main). S001: 3 sessions, S003 and S004: 4 sessions, others: 5 sessions. Some sessions excluded due to technical issues.",
            n_blocks=5,
            block_duration_s=5.0,
        ),
        abstract="Brain-computer interfaces (BCIs) have been gaining momentum in making human-computer interaction more natural, especially for people with neuro-muscular disabilities. This report focuses on SSVEP-based BCIs and performs a comparative evaluation of the most promising algorithms. A dataset of 256-channel EEG signals from 11 subjects is provided, along with a processing toolbox for reproducing results and supporting further experimentation.",
        methodology="Empirical approach where each signal processing parameter (filtering, artifact removal, feature extraction, feature selection, classification) is studied independently by keeping all other parameters fixed. Leave-one-subject-out cross-validation used to evaluate system without subject-specific training. Multiple algorithms compared for each processing stage to obtain state-of-the-art baseline.",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            events={"6.66": 1, "7.50": 2, "8.57": 3, "10.00": 4, "12.00": 5},
            sessions_per_subject=1,
            # 5 runs per sessions, except 3 for S001, S003, S008, 4 for S004
            code="MAMEM1",
            doi="10.48550/arXiv.1602.00904",
            figshare_id=2068677,
            subjects=subjects,
            sessions=sessions,
        )


class MAMEM2(BaseMAMEM):
    """SSVEP MAMEM 2 dataset.

    Dataset from [1]_.

    EEG signals with 256 channels captured from 11 subjects executing a
    SSVEP-based experimental protocol. Five different frequencies
    (6.66, 7.50, 8.57, 10.00 and 12.00 Hz) have been used for the visual
    stimulation,and the EGI 300 Geodesic EEG System, using a
    stimulation, HydroCel Geodesic Sensor Net (HCGSN) and a sampling rate of
    250 Hz has been used for capturing the signals.

    Subjects were exposed to flickering lights from five violet boxes with
    frequencies [6.66Hz, 7.5Hz, 8.57Hz, 10Hz, and 12Hz] simultaneously. Prior
    to and during each flicking window, one of the boxes is marked by a yellow
    arrow indicating the box to be focused on by the subject. 256 channel EEG
    recordings were captured.

    From [2]_, each subject underwent a single adaptation period before the first of
    their 5 sessions (unlike experiment 1 in which each session began with its own
    adaptation period). In the adaptation period, the subject is exposed to ten
    5-second flickering windows from the five boxes simultaneously, with the
    target frequencies specified in the FREQUENCIES2.txt file under
    'adaptation'. The flickering windows are separated by 5 seconds of rest,
    and the 100s adaptation period precedes the first session by 30 seconds.

    Each session consisted of the following:
    For the series of frequencies specified in the FREQUENCIES2.txt file under
    'sessions':
    A 5 second window with all boxes flickering and the subject focusing
    on the specified frequency's marked box, followed by 5 seconds of rest.
    This gives a total of 25 flickering windows for each session (not
    including the first adaptation period). Five minutes of rest before
    the next session (not including the 5th session).

    The order of chosen frequencies is the same for each session, although
    there are small-moderate variations in the actual frequencies of each
    individual window.
    **Note**: Each 'session' in experiment 1 includes an adaptation period,
    unlike experiment 2 and 3 where each subject undergoes only one adaptation
    period before their first 'session'.

    **Waveforms and Annotations**
    File names are in the form T0NNn, where NN is the subject number and n is
    a - e for the session letter or x for the adaptation period. Each session
    lasts in the order of several minutes and is sampled at 250Hz. Each session
    and adaptation period has the following files:
    A waveform file of the EEG signals (.dat) along with its header file
    (.hea). If the channel corresponds to an international 10-20 channel then
    it is labeled as such. Otherwise, it is just labeled 'EEG'. An annotation
    file (.flash) containing the locations of each individual flash. An
    annotation file (.win) containing the locations of the beginning and end
    of each 5 second flickering window. The annotations are labeled as '(' for
    start and ')' for stop, along with auxiliary strings indicating the focal
    frequency of the flashing windows.

    The FREQUENCIES2.txt file indicates the approximate marked frequencies of
    the flickering windows, equal for each session, adaptation, and subject.
    These values are equal to those contained in the .win annotations.

    **Observed  artifacts:**
    During the  stimulus  presentation  to  subject  S007  the  research stuff
    noted that the subject had a tendency to eye blink. As a result the
    interference, in matters of artifacts, on the recorded signal is expected
    to be high.

    References
    ----------
    .. [1] Oikonomou, V. P., Liaros, G., Georgiadis, K., Chatzilari, E., Adam, K.,
           Nikolopoulos, S., & Kompatsiaris, I. (2016). Comparative evaluation of
           state-of-the-art algorithms for SSVEP-based BCIs. arXiv preprint
           arXiv:1602.00904.
    .. [2] MAMEM Steady State Visually Evoked Potential EEG Database
           `<https://archive.physionet.org/physiobank/database/mssvepdb/>`_
    .. [3] S. Nikolopoulos, 2016, DataAcquisitionDetails.pdf
           `<https://figshare.com/articles/dataset/MAMEM_EEG_SSVEP_Dataset_II_256_channels_11_subjects_5_frequencies_presented_simultaneously_/3153409?file=4911931>`_
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=250.0,
            n_channels=256,
            channel_types={"eeg": 256},
            montage="10-20",
            hardware="EGI 300 Geodesic EEG System (GES 300)",
            sensor_type="scalp electrodes",
            reference="Cz",
            software="Microsoft Visual Studio 2010 with OpenGL",
            filters="5-48 Hz bandpass",
            sensors=[
                "E1",
                "E10",
                "E100",
                "E101",
                "E102",
                "E103",
                "E104",
                "E105",
                "E106",
                "E107",
                "E108",
                "E109",
                "E11",
                "E110",
                "E111",
                "E112",
                "E113",
                "E114",
                "E115",
                "E116",
                "E117",
                "E118",
                "E119",
                "E12",
                "E120",
                "E121",
                "E122",
                "E123",
                "E124",
                "E125",
                "E126",
                "E127",
                "E128",
                "E129",
                "E13",
                "E130",
                "E131",
                "E132",
                "E133",
                "E134",
                "E135",
                "E136",
                "E137",
                "E138",
                "E139",
                "E14",
                "E140",
                "E141",
                "E142",
                "E143",
                "E144",
                "E145",
                "E146",
                "E147",
                "E148",
                "E149",
                "E15",
                "E150",
                "E151",
                "E152",
                "E153",
                "E154",
                "E155",
                "E156",
                "E157",
                "E158",
                "E159",
                "E16",
                "E160",
                "E161",
                "E162",
                "E163",
                "E164",
                "E165",
                "E166",
                "E167",
                "E168",
                "E169",
                "E17",
                "E170",
                "E171",
                "E172",
                "E173",
                "E174",
                "E175",
                "E176",
                "E177",
                "E178",
                "E179",
                "E18",
                "E180",
                "E181",
                "E182",
                "E183",
                "E184",
                "E185",
                "E186",
                "E187",
                "E188",
                "E189",
                "E19",
                "E190",
                "E191",
                "E192",
                "E193",
                "E194",
                "E195",
                "E196",
                "E197",
                "E198",
                "E199",
                "E2",
                "E20",
                "E200",
                "E201",
                "E202",
                "E203",
                "E204",
                "E205",
                "E206",
                "E207",
                "E208",
                "E209",
                "E21",
                "E210",
                "E211",
                "E212",
                "E213",
                "E214",
                "E215",
                "E216",
                "E217",
                "E218",
                "E219",
                "E22",
                "E220",
                "E221",
                "E222",
                "E223",
                "E224",
                "E225",
                "E226",
                "E227",
                "E228",
                "E229",
                "E23",
                "E230",
                "E231",
                "E232",
                "E233",
                "E234",
                "E235",
                "E236",
                "E237",
                "E238",
                "E239",
                "E24",
                "E240",
                "E241",
                "E242",
                "E243",
                "E244",
                "E245",
                "E246",
                "E247",
                "E248",
                "E249",
                "E25",
                "E250",
                "E251",
                "E252",
                "E253",
                "E254",
                "E255",
                "E256",
                "E26",
                "E27",
                "E28",
                "E29",
                "E3",
                "E30",
                "E31",
                "E32",
                "E33",
                "E34",
                "E35",
                "E36",
                "E37",
                "E38",
                "E39",
                "E4",
                "E40",
                "E41",
                "E42",
                "E43",
                "E44",
                "E45",
                "E46",
                "E47",
                "E48",
                "E49",
                "E5",
                "E50",
                "E51",
                "E52",
                "E53",
                "E54",
                "E55",
                "E56",
                "E57",
                "E58",
                "E59",
                "E6",
                "E60",
                "E61",
                "E62",
                "E63",
                "E64",
                "E65",
                "E66",
                "E67",
                "E68",
                "E69",
                "E7",
                "E70",
                "E71",
                "E72",
                "E73",
                "E74",
                "E75",
                "E76",
                "E77",
                "E78",
                "E79",
                "E8",
                "E80",
                "E81",
                "E82",
                "E83",
                "E84",
                "E85",
                "E86",
                "E87",
                "E88",
                "E89",
                "E9",
                "E90",
                "E91",
                "E92",
                "E93",
                "E94",
                "E95",
                "E96",
                "E97",
                "E98",
                "E99",
            ],
            line_freq=50.0,
            impedance_threshold_kohm=80.0,
            cap_manufacturer="EGI",
            cap_model="HydroCel Geodesic Sensor Net (HCGSN)",
            electrode_type="HydroCel",
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_type=["vertical"],
                has_emg=True,
                other_physiological=["ecg", "gsr", "ppg"],
            ),
        ),
        participants=ParticipantMetadata(
            n_subjects=11,
            health_status="healthy",
            gender={"male": 8, "female": 3},
            age_min=24,
            age_max=39,
            handedness={"right": 10, "left": 1},
            clinical_population="able-bodied subjects without any known neuro-muscular or mental disorders",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"6.66": 1, "7.50": 2, "8.57": 3, "10.00": 4, "12.00": 5},
            paradigm="ssvep",
            n_classes=5,
            class_labels=["6.66 Hz", "7.50 Hz", "8.57 Hz", "10.00 Hz", "12.00 Hz"],
            trial_duration=5.0,
            study_design="Subjects focus attention on visual stimuli flickering at different frequencies (6.66, 7.50, 8.57, 10.00, 12.00 Hz) to select commands. Each stimulus presented for 5 seconds followed by 5 seconds rest.",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="offline",
            feedback_type="none",
            synchronicity="synchronous",
            stimulus_presentation={
                "device": "22 inch LCD monitor",
                "refresh_rate": "60 Hz",
                "resolution": "1680x1080",
                "stimulus_appearance": "violet box flickering at center",
                "background": "black",
                "synchronization": "Stim Tracker model ST-100 with light sensor",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.48550/arXiv.1602.00904",
            repository="GitHub",
            data_url="https://github.com/MAMEM/ssvep-eeg-processing-toolbox",
            license="ODC-By-1.0",
            investigators=[
                "Vangelis P. Oikonomou",
                "Georgios Liaros",
                "Kostantinos Georgiadis",
                "Elisavet Chatzilari",
                "Katerina Adam",
                "Spiros Nikolopoulos",
                "Ioannis Kompatsiaris",
            ],
            institution="Centre for Research and Technology Hellas (CERTH)",
            country="Greece",
            publication_year=2016,
            associated_paper_doi="arXiv:1602.00904v2",
            funding=["H2020-ICT-2014-644780"],
            ethics_approval=[
                "Approved by ethics committee of Centre for Research and Technology Hellas, date 3/7/2015, grant H2020-ICT-2014-644780"
            ],
            keywords=[
                "SSVEP",
                "BCI",
                "brain-computer interface",
                "EEG",
                "visual evoked potentials",
                "signal processing",
                "feature extraction",
                "classification",
            ],
        ),
        sessions_per_subject=5,
        runs_per_session=5,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            preprocessing_applied=True,
            preprocessing_steps=[
                "bandpass filtering (5-48 Hz)",
                "notch filtering (50 Hz)",
                "artifact removal (AMUSE, ICA)",
                "normalization to zero mean",
            ],
            highpass_hz=5.0,
            lowpass_hz=48.0,
            bandpass={"low_cutoff_hz": 5.0, "high_cutoff_hz": 48.0},
            notch_hz=50.0,
            filter_type="IIR-Elliptic",
            artifact_methods=["AMUSE", "ICA"],
            re_reference="CAR",
            notes="AMUSE removes first 15 and last 4 components. Optimal configuration uses IIR-Elliptic filter.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=[
                "LDA",
                "SVM",
                "Random Forest",
                "kNN",
                "Naive Bayes",
                "AdaBoost",
                "Decision Trees",
                "CCA",
            ],
            feature_extraction=[
                "PWelch",
                "Periodogram",
                "FFT",
                "Goertzel",
                "PYULEAR (Yule-AR)",
                "STFT",
                "DWT",
                "PSD",
                "Wavelet",
                "Spectrogram",
            ],
            spatial_filters=["CAR", "CSP", "Minimum Energy"],
            frequency_bands={
                "analyzed_range": [5.0, 48.0],
            },
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="leave-one-subject-out",
            evaluation_type=["cross_subject"],
        ),
        performance={
            "accuracy_percent": 74.42,
            "mean_accuracy_default_config": 72.47,
            "mean_accuracy_optimal_config": 74.42,
            "processing_time_msec": 68,
        },
        bci_application=BCIApplicationMetadata(
            applications=["cursor control", "command selection"],
            environment="indoor",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep",
            stimulus_frequencies_hz=[6.66, 7.50, 8.57, 10.00, 12.00],
            n_targets=5,
            n_repetitions=3,
        ),
        data_structure=DataStructureMetadata(
            n_trials=1104,
            n_trials_per_class={
                "6.66 Hz": "~220 per frequency",
                "7.50 Hz": "~220 per frequency",
                "8.57 Hz": "~220 per frequency",
                "10.00 Hz": "~220 per frequency",
                "12.00 Hz": "~220 per frequency",
            },
            trials_context="Each session includes 23 trials (8 adaptation trials excluded from analysis). 5 sessions per subject (with exceptions: S001=3 sessions, S003=4 sessions, S004=4 sessions). Total: 1104 trials of 5 seconds each.",
        ),
        abstract="Brain-computer interfaces (BCIs) have been gaining momentum in making human-computer interaction more natural, especially for people with neuro-muscular disabilities. This study focuses on SSVEP-based BCIs and performs a comparative evaluation of state-of-the-art algorithms for filtering, artifact removal, feature extraction, feature selection and classification. Dataset consists of 256-channel EEG signals from 11 subjects with 5 flickering frequencies (6.66, 7.50, 8.57, 10.00, 12.00 Hz).",
        methodology="Leave-one-subject-out cross-validation was used to evaluate a general-purpose BCI system without subject-specific training. Systematic comparison of algorithms across all signal processing stages: (1) Signal filtering: FIR vs IIR filters; (2) Artifact removal: AMUSE vs FastICA; (3) Feature extraction: PWelch, Periodogram, PYULEAR, DWT, STFT, Goertzel; (4) Feature selection: entropy-based methods and PCA/SVD; (5) Classification: SVM, LDA, KNN, Naive Bayes, Random Forest, AdaBoost. Optimal configuration achieved 74.42% mean accuracy using IIR-Elliptic filter, AMUSE artifact removal, PWelch feature extraction with nfft=512, segment length=350, overlap=0.75, and channel-138.",
        data_processed=True,
        file_format="EGI format",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            events={"6.66": 1, "7.50": 2, "8.57": 3, "10.00": 4, "12.00": 5},
            sessions_per_subject=1,
            code="MAMEM2",
            doi="10.48550/arXiv.1602.00904",
            figshare_id=3153409,
            subjects=subjects,
            sessions=sessions,
        )


class MAMEM3(BaseMAMEM):
    """SSVEP MAMEM 3 dataset.

    Dataset from [1]_.

    EEG signals with 14 channels captured from 11 subjects executing a
    SSVEP-based experimental protocol. Five different frequencies
    (6.66, 7.50, 8.57, 10.00 and 12.00 Hz) have been used for the visual
    stimulation, and the Emotiv EPOC, using 14 wireless channels has been used
    for capturing the signals.

    Subjects were exposed to flickering lights from five magenta boxes with
    frequencies [6.66Hz, 7.5Hz, 8.57Hz, 10Hz and 12Hz] simultaneously. Prior
    to and during each flicking window, one of the boxes is marked by a yellow
    arrow indicating the box to be focused on by the subject. The Emotiv EPOC
    14 channel wireless EEG headset was used to capture the subjects' signals.

    Each subject underwent a single adaptation period before the first of their
    5 sessions (unlike experiment 1 in which each session began with its own
    adaptation period). In the adaptation period, the subject is exposed to ten
    5-second flickering windows from the five boxes simultaneously, with the
    target frequencies specified in the FREQUENCIES3.txt file under
    'adaptation'. The flickering windows are separated by 5 seconds of rest,
    and the 100s adaptation period precedes the first session by 30 seconds.

    Each session consisted of the following:
    For the series of frequencies specified in the FREQUENCIES3.txt file under
    'sessions':
    A 5 second window with all boxes flickering and the subject focusing on
    the specified frequency's marked box, followed by 5 seconds of rest.
    Between the 12th and 13th flickering window, there is a 30s resting
    period. This gives a total of 25 flickering windows for each session
    (not including the first adaptation period). Five minutes of rest
    before the next session (not including the 5th session).

    The order of chosen frequencies is the same for each session, although
    there are small-moderate variations in the actual frequencies of each
    individual window.

    **Note**: Each 'session' in experiment 1 includes an adaptation period, unlike
    experiment 2 and 3 where each subject undergoes only one adaptation period
    before their first 'session' [2]_.

    **Waveforms and Annotations**
    File names are in the form U0NNn, where NN is the subject number and n is
    a - e for the session letter or x for the adaptation period. In addition,
    session file names end with either i or ii, corresponding to the first 12
    or second 13 windows of the session respectively. Each session lasts in the
    order of several minutes and is sampled at 128Hz.
    Each session half and adaptation period has the following files:
    A waveform file of the EEG signals (.dat) along with its header file
    (.hea). An annotation file (.win) containing the locations of the beginning
    and end of each 5 second flickering window. The annotations are labeled as
    '(' for start and ')' for stop, along with auxiliary strings indicating the
    focal frequency of the flashing windows.

    The FREQUENCIES3.txt file indicates the approximate marked frequencies of
    the flickering windows, equal for each session, adaptation, and subject.
    These values are equal to those contained in the .win annotations.

    **Trial  manipulation**:
    The  trial  initiation  is  defined by  an  event  code  (32779)  and  the
    end by another (32780). There are five different labels that indicate the
    box subjects were instructed to focus  on  (1, 2, 3, 4 and 5) and
    correspond to frequencies 12.00, 10.00, 8.57, 7.50 and 6.66 Hz respectively.
    5 3 2 1 4 5 2 1 4 3 is the trial sequence for the adaptation and
    4 2 3 5 1 2 5 4 2 3 1 5 4 3 2 4 1 2 5 3 4 1 3 1 3 is the sequence for each
    session.

    **Observed  artifacts**:
    During  the  stimulus  presentation to  subject  S007  the  research staff
    noted that the subject had a tendency to eye blink. As a result the
    interference, in matters of artifacts, on the recorded signal is expected
    to be high.

    References
    ----------
    .. [1] Oikonomou, V. P., Liaros, G., Georgiadis, K., Chatzilari, E., Adam, K.,
           Nikolopoulos, S., & Kompatsiaris, I. (2016). Comparative evaluation of
           state-of-the-art algorithms for SSVEP-based BCIs. arXiv preprint
           arXiv:1602.00904.
    .. [2] MAMEM Steady State Visually Evoked Potential EEG Database
           `<https://archive.physionet.org/physiobank/database/mssvepdb/>`_
    .. [3] S. Nikolopoulos, 2016, DataAcquisitionDetails.pdf
           `<https://figshare.com/articles/dataset/MAMEM_EEG_SSVEP_Dataset_III_14_channels_11_subjects_5_frequencies_presented_simultaneously_/3413851>`_
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=128.0,
            n_channels=14,
            channel_types={"eeg": 14},
            montage="10-20",
            hardware="EGI 300 Geodesic EEG System (GES 300)",
            sensor_type="scalp electrodes",
            reference="CAR",
            software="Microsoft Visual Studio 2010 with OpenGL",
            filters="5-48 Hz bandpass, 50 Hz notch",
            sensors=[
                "AF3",
                "AF4",
                "F3",
                "F4",
                "F7",
                "F8",
                "FC5",
                "FC6",
                "O1",
                "O2",
                "P7",
                "P8",
                "T7",
                "T8",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=True,
                eog_type=["vertical"],
                has_emg=True,
                other_physiological=["ecg", "gsr", "ppg"],
            ),
            cap_manufacturer="EGI",
            cap_model="HydroCel Geodesic Sensor Net (HCGSN)",
            electrode_type="wet",
            impedance_threshold_kohm=80.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=11,
            health_status="healthy",
            gender={"male": 8, "female": 3},
            age_min=24.0,
            age_max=39.0,
            ages=[24, 37, 39, 31, 27, 28, 26, 31, 29, 37, 25],
            handedness={"right": 10, "left": 1},
            bci_experience="naive",
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={
                "6.66": 33029,
                "7.50": 33028,
                "8.57": 33027,
                "10.00": 33026,
                "12.00": 33025,
            },
            paradigm="ssvep",
            n_classes=5,
            class_labels=["6.66 Hz", "7.50 Hz", "8.57 Hz", "10.00 Hz", "12.00 Hz"],
            trial_duration=5.0,
            study_design="Subjects focus attention on a violet box flickering at different frequencies (6.66, 7.50, 8.57, 10.00, 12.00 Hz) presented at the center of the monitor. Each trial lasts 5 seconds followed by 5 seconds rest.",
            stimulus_type="visual",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="offline",
            synchronicity="synchronous",
            feedback_type="none",
            has_training_test_split=False,
            instructions="Subjects were instructed to focus attention on the flickering stimulus and minimize artifacts by reducing eye blinks and movements.",
            stimulus_presentation={
                "display": "22 inch LCD monitor, 60 Hz refresh rate, 1680x1080 resolution",
                "background": "black",
                "stimulus": "violet box flickering at center of screen",
                "graphics": "Nvidia GeForce GTX 860M with vertical synchronization enabled",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.6084/m9.figshare.2068677.v1",
            description="Comparative evaluation of state-of-the-art algorithms for SSVEP-based BCIs. Dataset includes 256-channel EEG signals from 11 subjects performing SSVEP tasks with 5 different flickering frequencies.",
            investigators=[
                "Vangelis P. Oikonomou",
                "Georgios Liaros",
                "Kostantinos Georgiadis",
                "Elisavet Chatzilari",
                "Katerina Adam",
                "Spiros Nikolopoulos",
                "Ioannis Kompatsiaris",
            ],
            institution="Centre for Research and Technology Hellas (CERTH)",
            country="Greece",
            repository="Figshare",
            data_url="https://dx.doi.org/10.6084/m9.figshare.2068677.v1",
            license="ODC-By-1.0",
            publication_year=2016,
            senior_author="Ioannis Kompatsiaris",
            associated_paper_doi="arXiv:1602.00904v2",
            ethics_approval=[
                "Ethics committee of the Centre for Research and Technology Hellas, approved 3/7/2015"
            ],
            keywords=[
                "SSVEP",
                "BCI",
                "brain-computer interface",
                "EEG",
                "visual evoked potentials",
                "comparative evaluation",
                "signal processing",
            ],
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            preprocessing_applied=True,
            preprocessing_steps=[
                "bandpass filtering (5-48 Hz)",
                "notch filtering (50 Hz)",
                "artifact removal (AMUSE, ICA)",
                "Common Average Reference (CAR)",
            ],
            highpass_hz=5.0,
            lowpass_hz=48.0,
            bandpass={"low_cutoff_hz": 5.0, "high_cutoff_hz": 48.0},
            notch_hz=50.0,
            filter_type="IIR (Chebyshev, Elliptic)",
            artifact_methods=["AMUSE", "ICA", "FastICA"],
            re_reference="CAR",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=[
                "LDA",
                "SVM",
                "Random Forest",
                "kNN",
                "Naive Bayes",
                "CCA",
                "ELM",
                "Decision Trees",
            ],
            feature_extraction=[
                "Periodogram",
                "Welch",
                "Goertzel",
                "Yule-AR",
                "STFT",
                "Discrete Wavelet Transform",
                "PSD",
                "CSP",
                "ICA",
            ],
            frequency_bands={
                "analyzed_range": [5.0, 48.0],
            },
            spatial_filters=["CAR", "CSP", "Minimum Energy"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="leave-one-subject-out",
            evaluation_type=["cross_subject"],
        ),
        performance={
            "accuracy_percent": 72.47,
            "default_config_accuracy": 72.47,
            "optimal_config_accuracy": 79.47,
            "best_electrode_accuracy": 74.42,
            "execution_time_ms": 5.0,
        },
        bci_application=BCIApplicationMetadata(
            applications=["research", "comparative_study"],
            environment="laboratory",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="ssvep",
            stimulus_frequencies_hz=[6.66, 7.50, 8.57, 10.00, 12.00],
            n_targets=5,
        ),
        data_structure=DataStructureMetadata(
            n_trials=1104,
            trials_context="Total of 1104 trials (5 seconds each) across all subjects and sessions. Subject S001: 3 sessions, S003 and S004: 4 sessions each, all others: 5 sessions. Each session includes 23 trials (8 adaptation + 15 experimental).",
        ),
        sessions_per_subject=5,
        runs_per_session=10,
        data_processed=True,
        file_format="csv",
        abstract="Brain-computer interfaces (BCIs) have been gaining momentum in making human-computer interaction more natural, especially for people with neuro-muscular disabilities. This report focuses on EEG-based BCIs that rely on Steady-State-Visual-Evoked Potentials (SSVEPs) and performs a comparative evaluation of state-of-the-art algorithms for filtering, artifact removal, feature extraction, feature selection and classification. The dataset consists of 256-channel EEG signals from 11 subjects, along with a processing toolbox for reproducing results.",
        methodology="Comparative evaluation of SSVEP-based BCI algorithms using leave-one-subject-out cross-validation. The study examines filtering methods (IIR, FIR), artifact removal (AMUSE, ICA), feature extraction (Periodogram, Welch, Goertzel, Yule-AR, STFT, DWT), feature selection (Shannon entropy, PCA, ICA), and classification (LDA, SVM, kNN, Naive Bayes, Random Forest, CCA, ELM, Decision Trees). Each parameter is studied independently while keeping others fixed to identify optimal configurations.",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            events={
                "6.66": 33029,
                "7.50": 33028,
                "8.57": 33027,
                "10.00": 33026,
                "12.00": 33025,
            },
            sessions_per_subject=1,
            code="MAMEM3",
            doi="10.48550/arXiv.1602.00904",
            figshare_id=3413851,
            subjects=subjects,
            sessions=sessions,
        )
