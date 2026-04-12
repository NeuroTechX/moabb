"""BCI Competition 2020 Track 3 - Imagined Speech dataset.

Jeong et al., 2022 International BCI Competition review.
Data: https://osf.io/pq7vb/
"""

import numpy as np
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    PreprocessingMetadata,
    Tags,
)
from .utils import build_raw_from_epochs


def _speech_hed(label):
    """Build a HED tag string for an imagined-speech event (auditory cue)."""
    return (
        "(Sensory-event, Experimental-stimulus, "
        "Auditory-presentation, (Hear, Word)), "
        f"(Agent-action, (Imagine, Speak, (Word, (Label/{label}))))"
    )


_SIGN = "BCIComp2020IS"
_SFREQ = 256.0

# fmt: off
_CH_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2",
    "FC6", "T7", "C3", "Cz", "C4", "T8", "TP9", "CP5", "CP1", "CP2",
    "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz",
    "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6",
    "FT9", "FT7", "FC3", "FC4", "FT8", "FT10", "C5", "C1", "C2", "C6",
    "TP7", "CP3", "CPz", "CP4", "TP8", "P5", "P1", "P2", "P6", "PO7",
    "PO3", "POz", "PO4", "PO8",
]
# fmt: on

_CLASS_NAMES = ["Hello", "Helpme", "Stop", "Thankyou", "Yes"]

# Stable OSF download URLs for every subject file. Test split is
# excluded: the released test files have no ground-truth labels
# (competition holdout).
_OSF_URLS = {
    ("training", 1): "https://osf.io/download/us96c/",
    ("training", 2): "https://osf.io/download/yeh24/",
    ("training", 3): "https://osf.io/download/qcbv9/",
    ("training", 4): "https://osf.io/download/682y9/",
    ("training", 5): "https://osf.io/download/3txuf/",
    ("training", 6): "https://osf.io/download/q9tbf/",
    ("training", 7): "https://osf.io/download/tyzc9/",
    ("training", 8): "https://osf.io/download/mywcq/",
    ("training", 9): "https://osf.io/download/we3u7/",
    ("training", 10): "https://osf.io/download/sng2u/",
    ("training", 11): "https://osf.io/download/xc7pz/",
    ("training", 12): "https://osf.io/download/tpz6n/",
    ("training", 13): "https://osf.io/download/8md2p/",
    ("training", 14): "https://osf.io/download/m6wbu/",
    ("training", 15): "https://osf.io/download/q2ndy/",
    ("validation", 1): "https://osf.io/download/dqrfu/",
    ("validation", 2): "https://osf.io/download/f5c9n/",
    ("validation", 3): "https://osf.io/download/vyjth/",
    ("validation", 4): "https://osf.io/download/ax5v6/",
    ("validation", 5): "https://osf.io/download/qprs3/",
    ("validation", 6): "https://osf.io/download/ep7sw/",
    ("validation", 7): "https://osf.io/download/ga5r2/",
    ("validation", 8): "https://osf.io/download/jpcem/",
    ("validation", 9): "https://osf.io/download/fcj8y/",
    ("validation", 10): "https://osf.io/download/tnh2g/",
    ("validation", 11): "https://osf.io/download/w46s7/",
    ("validation", 12): "https://osf.io/download/wqm3k/",
    ("validation", 13): "https://osf.io/download/y9edh/",
    ("validation", 14): "https://osf.io/download/w9gh4/",
    ("validation", 15): "https://osf.io/download/2zh87/",
}

_SPLITS = [("training", "epo_train"), ("validation", "epo_validation")]


class BCIComp2020IS(BaseDataset):
    """BCI Competition 2020 Track 3 - Imagined Speech Classification.

    Dataset from the 2020 International BCI Competition [1]_.

    **Dataset Description**

    Fifteen subjects (aged 20-30) performed imagined speech of five
    phrases: "Hello", "Help me", "Stop", "Thank you", "Yes". EEG was
    recorded at 1000 Hz using 64 channels in a 10-20 configuration with
    a BrainAmp amplifier (BrainProducts GmbH), FCz reference, Fpz ground.
    Data is stored at the native epoch sampling rate of 256 Hz.

    Each trial begins with an auditory cue (one of the five words),
    followed by 4 repetitions of: fixation cross (0.8-1.2 s jittered)
    then 2 s imagined speech. A 3 s relaxation phase separates blocks.
    Epochs span -500 ms to 2600 ms relative to cue onset.

    Each subject has 300 training trials (60 per class) and 50
    validation trials (10 per class). Test trials (50 per subject)
    have no labels (competition holdout) and are not loaded.
    Best competition result was 82.6% accuracy.

    .. figure:: https://www.frontiersin.org/files/Articles/898300/xml-images/fnhum-16-898300-g0007.webp
       :alt: BCI Competition 2020 Track 3 trial structure — Rest,
             auditory cue, 4 imagined-speech repetitions.
       :width: 100%

       Figure 7 of [1]_ (CC-BY-4.0). Epoch window: -0.5 to +2.6 s.

    References
    ----------
    .. [1] Jeong, J.-H. et al. (2022). 2020 International brain-computer
           interface competition: A review. Frontiers in Human Neuroscience,
           16, 898300. https://doi.org/10.3389/fnhum.2022.898300
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            n_channels=64,
            channel_types={"eeg": 64},
            montage="standard_1005",
            hardware="BrainAmp (BrainProducts GmbH)",
            software="BrainVision with MATLAB 2019a",
            reference="FCz",
            ground="Fpz",
            sensors=list(_CH_NAMES),
        ),
        participants=ParticipantMetadata(
            n_subjects=15,
            health_status="healthy",
            age_min=20,
            age_max=30,
            species="human",
        ),
        experiment=ExperimentMetadata(
            events={"Hello": 1, "Helpme": 2, "Stop": 3, "Thankyou": 4, "Yes": 5},
            paradigm="imagery",
            n_classes=5,
            class_labels=_CLASS_NAMES,
            trial_duration=3.1,
            study_design=(
                "Auditory cue followed by 4 repetitions of fixation cross "
                "(0.8-1.2 s jittered) + 2 s imagined speech, with 3 s "
                "relaxation between blocks. Black screen during imagery."
            ),
            stimulus_type="auditory cue",
            stimulus_modalities=["auditory"],
            primary_modality="auditory",
            synchronicity="synchronous",
            mode="offline",
            instructions=(
                "Imagine silent pronunciation as if performing real speech. "
                "No articulator movement, no sound, no blinking."
            ),
            hed_tags={
                "Hello": _speech_hed("Hello"),
                "Helpme": _speech_hed("Helpme"),
                "Stop": _speech_hed("Stop"),
                "Thankyou": _speech_hed("Thankyou"),
                "Yes": _speech_hed("Yes"),
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.3389/fnhum.2022.898300",
            investigators=[
                "Ji-Hoon Jeong",
                "Jeong-Hyun Cho",
                "Young-Eun Lee",
                "Seo-Hyun Lee",
                "Gi-Hwan Shin",
                "Young-Seok Kweon",
                "Jose del R. Millan",
                "Klaus-Robert Mueller",
                "Seong-Whan Lee",
            ],
            institution="Korea University",
            institution_department="Department of Brain and Cognitive Engineering",
            institution_address="Seoul, South Korea",
            country="KR",
            data_url="https://osf.io/pq7vb/",
            publication_year=2022,
            license="CC-BY-4.0",
            repository="OSF",
            senior_author="Seong-Whan Lee",
            contact_info=["bcicompetition2020@gmail.com"],
            associated_paper_doi="10.3389/fnhum.2022.898300",
            keywords=[
                "brain-computer interface",
                "electroencephalogram",
                "imagined speech",
                "competition",
                "open datasets",
                "neural decoding",
            ],
            description=(
                "BCI Competition 2020 Track 3: Imagined speech classification "
                "with 5 phrases using 64-channel EEG. Best competition accuracy "
                "82.6%. IRB: KUIRB-2019-0143-01."
            ),
        ),
        sessions_per_subject=1,
        runs_per_session=2,
        tags=Tags(pathology=["Healthy"], modality=["Speech"], type=["Competition"]),
        preprocessing=PreprocessingMetadata(
            data_state="epoched", preprocessing_applied=False
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=_CLASS_NAMES,
            cue_duration_s=2.0,
            imagery_duration_s=2.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=5250,
            n_trials_per_class={
                "Hello": 1050,
                "Helpme": 1050,
                "Stop": 1050,
                "Thankyou": 1050,
                "Yes": 1050,
            },
            trials_context=("15 subjects x 350 trials (70 per class: 60 train + 10 val)"),
        ),
        data_processed=False,
        file_format="MAT",
    )

    def __init__(self, subjects=None, sessions=None):
        super().__init__(
            subjects=list(range(1, 16)),
            sessions_per_subject=1,
            events={"Hello": 1, "Helpme": 2, "Stop": 3, "Thankyou": 4, "Yes": 5},
            code="BCIComp2020IS",
            interval=[0, 3],
            paradigm="imagery",
            doi="10.3389/fnhum.2022.898300",
            selected_subjects=subjects,
            selected_sessions=sessions,
        )

    @staticmethod
    def _load_epoch_mat(fpath, epo_key):
        """Load a MATLAB v5 epoch file and return (data, labels, ch_names).

        Parameters
        ----------
        fpath : str
            Path to the .mat file.
        epo_key : str
            Key for the epoch struct (e.g. ``'epo_train'``).

        Returns
        -------
        data : ndarray, shape (n_trials, n_channels, n_times)
        labels : ndarray of int, shape (n_trials,)
            1-indexed class labels.
        ch_names : list of str
        """
        mat = loadmat(fpath, squeeze_me=False)
        epo = mat[epo_key]

        # x is (n_times, n_channels, n_trials) in the .mat;
        # transpose to (n_trials, n_channels, n_times).
        data = np.transpose(epo["x"][0, 0], (2, 1, 0))

        # y is (n_classes, n_trials) one-hot; argmax gives 0-indexed label.
        labels = np.argmax(epo["y"][0, 0], axis=0) + 1

        ch_names = [str(c[0]) for c in epo["clab"][0, 0][0]]
        return data, labels, ch_names

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        runs = {}
        for run_idx, (split, epo_key) in enumerate(_SPLITS):
            fpath = self.data_path(subject, split=split)
            data, labels, ch_names = self._load_epoch_mat(fpath, epo_key)
            runs[str(run_idx)] = build_raw_from_epochs(
                data, ch_names, _SFREQ, labels, montage_name="standard_1005"
            )
        return {"0": runs}

    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
        *,
        split=None,
    ):
        """Return the local path to a subject file.

        Downloads all (training + validation) files for ``subject``
        via :func:`moabb.datasets.download.data_dl` if they are not
        already present. Returns the path for the requested ``split``
        if one is provided, otherwise the training file path.
        """
        if subject not in self.subject_list:
            raise ValueError(f"Invalid subject number {subject}")

        paths = {}
        for split_name, _ in _SPLITS:
            url = _OSF_URLS[(split_name, subject)]
            paths[split_name] = dl.data_dl(
                url, _SIGN, path=path, force_update=force_update, verbose=verbose
            )

        return paths[split] if split else paths["training"]
