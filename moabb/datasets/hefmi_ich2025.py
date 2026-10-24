"""Hybrid EEG-fNIRS motor imagery dataset for intracerebral hemorrhage.

Shi, Chen, et al. (2025), Scientific Data.
DOI: 10.1038/s41597-025-06100-7
Data DOI: 10.6084/m9.figshare.28955456.v4
"""

import hashlib
import json
import logging
import re
import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from . import download as dl
from .base import BaseDataset
from .metadata.schema import (
    AcquisitionMetadata,
    BCIApplicationMetadata,
    CrossValidationMetadata,
    DatasetMetadata,
    DataStructureMetadata,
    DocumentationMetadata,
    ExperimentMetadata,
    ParadigmSpecificMetadata,
    ParticipantMetadata,
    SignalProcessingMetadata,
    Tags,
)
from .utils import build_raw_from_epochs


log = logging.getLogger(__name__)

# Figshare article and the version whose record gives every file's folder.
_FIGSHARE_ARTICLE = 28955456
_FIGSHARE_VERSION = 4
_RECORD_URL = (
    f"https://api.figshare.com/v2/articles/{_FIGSHARE_ARTICLE}"
    f"/versions/{_FIGSHARE_VERSION}"
)
# Cache name of that record, next to the epoch files.
_RECORD_NAME = f"figshare_article_{_FIGSHARE_ARTICLE}_v{_FIGSHARE_VERSION}.json"

# An EEG epoch file in the depositor's layout: group, folder subject N, session k.
_EPOCH_PATH = re.compile(r"epo_data/(Normals|Patients)/EEG/subject(\d+)/(\d)_epo\.mat")
_N_HEALTHY = 17  # Normals/subject1..17 are subjects 1-17; Patients/subjectN is 17 + N.

# Event codes (from the paper and MakeDatasetFromRaw.m).
_EVENTS = {"left_hand": 1, "right_hand": 2}

# 32 EEG channel names (from mne_analysis.py in the code.zip).
# fmt: off
_CH_NAMES = [
    "FC1", "AF3", "AF4", "CP1", "CP2", "CP6", "Cz", "C3",
    "C4", "T7", "T8", "FC2", "FC5", "FC6", "Pz", "CP5",
    "PO3", "PO4", "Oz", "Fp2", "Fp1", "Fz", "F3", "F4",
    "F7", "F8", "P3", "P4", "P7", "P8", "O1", "O2",
]
# fmt: on

_SFREQ = 256.0

# The 98 EEG epoch files of figshare version 4, as the record's
# ``folder_structure`` places them: (figshare file id, group, folder subject N,
# session k, size in bytes, MD5), i.e. ``epo_data/<group>/EEG/subject<N>/<k>_epo.mat``.
# Used when the record cannot be fetched; ``_epoch_files_from_record`` derives
# the same rows from the record.
# fmt: off
_EPOCH_FILES = (
    (54452621, "Normals", 1, 1, 41201769, "d108949494c5c5e51f43c1f3a7510681"),
    (54452645, "Normals", 1, 2, 41697356, "2ff2babebed82cde13538daafdf8d6e9"),
    (54452723, "Normals", 2, 1, 42452505, "8f2fe68258f4007460a6f18fa3021af4"),
    (54452729, "Normals", 2, 2, 42627242, "e4e8bf81d36208d6097820ac711f88bd"),
    (54452753, "Normals", 2, 3, 42769184, "507658b2c71609d5b36d835beb30628d"),
    (54452648, "Normals", 3, 1, 42283895, "c850f91991e4a292aaca6ed7d92b103c"),
    (54452699, "Normals", 3, 2, 42664527, "ed81aba604ced58ec8c018a5ccd0122a"),
    (54452726, "Normals", 3, 3, 42546408, "27ba61c2b133dac7e294c0a84bb6d3a7"),
    (54452678, "Normals", 4, 1, 42179123, "220cf2a42ca0bb650fcfdd3cf0eb55fa"),
    (54452702, "Normals", 4, 2, 42531888, "989dcec2fcb26d8ec5ccc70966d144ae"),
    (54452714, "Normals", 4, 3, 42354594, "6592681d457471bbe437bc9bca0b3a2a"),
    (54452681, "Normals", 5, 1, 42337367, "e7061b5b14c848e37724575d39cf7971"),
    (54452687, "Normals", 5, 2, 42398800, "60f12e9d71bf3f4e13df27732d0ef099"),
    (54452693, "Normals", 5, 3, 42270886, "9aef533a59450727be4fedd9f169d50d"),
    (54452690, "Normals", 6, 1, 42334735, "7bff8ecc04567556b4b751e6a50b3c0d"),
    (54452618, "Normals", 6, 2, 42384502, "3f7b9acb8f09590f6db7fc39c6343f3b"),
    (54452636, "Normals", 6, 3, 42339636, "a8136b83d8e10b51483a937976b5b5f6"),
    (54452606, "Normals", 7, 1, 42331473, "4199aea8f1706aca1891a7b70371c489"),
    (54452657, "Normals", 7, 2, 42428473, "f553dee643f17ea6eb95d1e4de298e72"),
    (54452189, "Normals", 7, 3, 41983865, "a51d49540dcca34325ffb080971b4503"),
    (54452150, "Normals", 8, 1, 41418331, "64ed170c8a3f5da654f8d3d116e9169a"),
    (54452165, "Normals", 8, 2, 41509197, "050673bee9c3370c16eb8576958c5ed2"),
    (54452177, "Normals", 8, 3, 41761417, "f4956a11f7e29e3561ec962d572bbe30"),
    (54452651, "Normals", 9, 1, 42216703, "feb190e7bd6b2677eb9961c5acc3c371"),
    (54452201, "Normals", 9, 2, 41529029, "01d942870108ffd9dd20373ead41ed99"),
    (54452600, "Normals", 9, 3, 42161122, "123d8fb5a8d29bd972a60563af6d7ad5"),
    (54452777, "Normals", 10, 1, 42116338, "d5aee615cdc77f8e986c042c383e63be"),
    (54452630, "Normals", 10, 2, 41749334, "16525cb6dd3f913481e2f934fa14b2f1"),
    (54452669, "Normals", 10, 3, 41843774, "afac809fad7088c524e835d402bc4891"),
    (54452780, "Normals", 11, 1, 42258093, "8130bd7b59c42e4e6e709e3d3e2eb436"),
    (54452783, "Normals", 11, 2, 42014101, "a57057d39d80e0baec2f124d699652af"),
    (54452789, "Normals", 11, 3, 42045310, "0c12c4b27be27f1b65d3d20516142d92"),
    (54452615, "Normals", 12, 1, 41948589, "1d38be3aa4fb18f707217ef11492b2c8"),
    (54452573, "Normals", 12, 2, 41882541, "627e5ba2213a75a397b29cc03c01c7e8"),
    (54452576, "Normals", 12, 3, 41831405, "459892c14e825045ee998b3efdc4777c"),
    (54452546, "Normals", 13, 1, 41611165, "017460ad15f9d335ff88dd2456779672"),
    (54452786, "Normals", 13, 2, 42018741, "5ff485cbcbeef96876eb67c198121908"),
    (54452570, "Normals", 13, 3, 41967139, "bc785f0ded43a5bcdd64c75daa1b78ec"),
    (54452756, "Normals", 14, 1, 42264564, "f046a18db10545c9f1e0577c5b57e303"),
    (54452792, "Normals", 14, 2, 42112698, "95d4b5d520fde1e6a97fdfbf4c48e716"),
    (54452762, "Normals", 14, 3, 42068361, "9b0029cf90cd4868a96a383e7c49a9b1"),
    (54452741, "Normals", 15, 1, 42896256, "51d39932dc6c5e31020507cbe85decf3"),
    (54452747, "Normals", 15, 2, 42675074, "fa335c3a02879b51bf803ba34663ffbc"),
    (54452759, "Normals", 15, 3, 42595793, "3c426abbb7cb5375f8f1d4ec513e330a"),
    (54452717, "Normals", 16, 1, 42362149, "b56c0187f3e327c584aacfd96232f4ed"),
    (54452324, "Normals", 16, 2, 41800932, "fbb4d47ddf218f6c8d6f243c1e3bcd22"),
    (54452750, "Normals", 16, 3, 42109504, "a9361ec4bb0b8be160c600057345a37a"),
    (54452282, "Normals", 17, 1, 41517760, "e23fb29692d6cfe6043db487e6fba3a4"),
    (54452294, "Normals", 17, 2, 41481531, "e89ae6f7338a56c3ea2b1bfd9374ef0f"),
    (54452288, "Normals", 17, 3, 41843779, "e718f796806347dd62470fa357a51d0a"),
    (54452255, "Patients", 1, 1, 42693848, "49f418259ee15ee5eaee8b5759f7f0cc"),
    (54452285, "Patients", 1, 2, 43283492, "9e35679ac85205131797983958099823"),
    (54452303, "Patients", 1, 3, 43018431, "5ef349257aa1f506aa12f6b77c5f2835"),
    (54452009, "Patients", 2, 1, 41492185, "8fcc46219d0119c2ddd59645df22db1d"),
    (54452024, "Patients", 2, 2, 41392136, "24093e272967279c9195226ab0a7a896"),
    (54452042, "Patients", 2, 3, 41352081, "30651edf0313400656ee548da29b8854"),
    (54452045, "Patients", 3, 1, 41773797, "b98d18e9b5ff581629dd1efc295ea465"),
    (54452006, "Patients", 3, 2, 41617968, "a1d2fb7a84c29b5480604518cba512f5"),
    (54452030, "Patients", 3, 3, 41673464, "73a3fd2ec425ab9f140ddb68cbcc1827"),
    (54452135, "Patients", 4, 1, 42254657, "62e6139ed06282a7ed34aca3c68896f3"),
    (54452132, "Patients", 4, 2, 42162411, "7362d294f7c0715ef1aeec07e2134a00"),
    (54452174, "Patients", 4, 3, 42136527, "e5f1ad587ccd98db2795624cf20cbe0e"),
    (54452117, "Patients", 5, 1, 42722570, "6afb976a3036d1045cb13449b9efc327"),
    (54452108, "Patients", 5, 2, 42645098, "1ac065ce89aa93a29206de02ec5b8646"),
    (54452138, "Patients", 5, 3, 42573545, "628ebec74e95310a7548e2ecee812a89"),
    (54452021, "Patients", 6, 1, 41269123, "6bc1c1ba726d1e75bc8417aa6eb0ebc1"),
    (54452018, "Patients", 6, 2, 41359546, "a56e8496a10d97c7ceda9ea7f63fef4c"),
    (54452078, "Patients", 7, 1, 42782326, "0265e73f2f6e8c71253b148aa123c7f0"),
    (54452081, "Patients", 7, 2, 42716054, "47ac66e2698cbe8dbe60003acfa454ac"),
    (54452075, "Patients", 8, 1, 42479668, "ed6552a83be3cf91cbc820222aed7f5c"),
    (54451994, "Patients", 9, 1, 41112262, "f5b3fd238dac45eda276442197775920"),
    (54452000, "Patients", 9, 2, 40924598, "2cc39aa6418a280752ef37e45a446601"),
    (54452312, "Patients", 10, 1, 42315258, "0fc6665b0453bc6639a6edc2e235f54a"),
    (54452252, "Patients", 10, 2, 42074741, "f4ac6a5cd3ac72f834f10376389b5d7f"),
    (54452297, "Patients", 10, 3, 42077142, "95e97978c6fa66b14e2fe989ccfd20cf"),
    (54452306, "Patients", 11, 1, 43025214, "60756ea0a6c5ac9250a02069b3503808"),
    (54452279, "Patients", 11, 2, 42772265, "f03da9f64e91613d6bdb36656a30cd99"),
    (54452198, "Patients", 12, 1, 42279945, "8acd43ca0a1578b9ae393fb550cd3d8b"),
    (54452237, "Patients", 12, 2, 42429290, "f3acbf1770b90c868ec0c1d0f43932ce"),
    (54452234, "Patients", 12, 3, 42439467, "d6c3cd42a286683548f7a351ab7e150d"),
    (54452057, "Patients", 13, 1, 41799926, "45b7a36aa6580b116871f3da6694670c"),
    (54452204, "Patients", 13, 2, 42106473, "f83d010cbca591cea6d24a5789880ca0"),
    (54452105, "Patients", 13, 3, 41931415, "ab50b538db943b186a410ccb4f96844b"),
    (54452216, "Patients", 14, 1, 42269327, "e98839fd41b9ff446556e39e2f9eaeb0"),
    (54452207, "Patients", 14, 2, 42037249, "a40e7792b86624a24dac44f27431b06f"),
    (54452213, "Patients", 14, 3, 42063789, "98111b5c3515445f640b74e805060e7c"),
    (54452231, "Patients", 15, 1, 42854159, "e7a3f4b8e0c12303f91bf1d28820a308"),
    (54452219, "Patients", 15, 2, 42709762, "b1b0a2182e43b9f0c346570127a9ab57"),
    (54452069, "Patients", 16, 1, 41969042, "f52e475dd5aa56074a9f342d91bbbe84"),
    (54452051, "Patients", 16, 2, 41672295, "226bec1b6e6de8cc81001fed1ed78ee4"),
    (54452192, "Patients", 17, 1, 42778181, "a18c1c1b367992818c102ae01de3bcc4"),
    (54452159, "Patients", 17, 2, 42126923, "f7c7efdc60db80a34a6991e8e236f487"),
    (54452147, "Patients", 18, 1, 42487602, "3d051874f2511cd3d89d5606b44582d0"),
    (54452210, "Patients", 18, 2, 42197917, "52dd06a4a8f698dcb79dce2aaf1cad1a"),
    (54452012, "Patients", 19, 1, 41094841, "b857b1f4f91e744c189b36095dc7e746"),
    (54452039, "Patients", 19, 2, 41419294, "bd1cc2a9ef5c24d652b27f930fc0bcf2"),
    (54452156, "Patients", 20, 1, 42261881, "47402e02d33f1b42e266a7f73f9f95ac"),
    (54452015, "Patients", 20, 2, 41537755, "09cfd04207cf902b936cc59b84fed568"),
)
# fmt: on


def _epoch_files_from_record(record):
    """Rows of ``_EPOCH_FILES`` derived from a figshare article record.

    A file's participant is its depositor folder, given by the record's
    ``folder_structure`` (figshare file id -> folder); its session is the
    ``<k>`` of its name. Files outside ``epo_data/*/EEG/subject<N>/`` (fNIRS,
    raw recordings, code) are skipped.
    """
    folders = record["folder_structure"]
    rows = []
    for entry in record["files"]:
        path = f"{folders.get(str(entry['id']), '')}/{entry['name']}"
        match = _EPOCH_PATH.fullmatch(path)
        if match is None:
            continue
        group, folder_subject, session = match.groups()
        rows.append(
            (
                entry["id"],
                group,
                int(folder_subject),
                int(session),
                entry["size"],
                entry["computed_md5"],
            )
        )
    return tuple(sorted(rows, key=lambda row: (row[1], row[2], row[3])))


def _is_record(obj):
    """Whether ``obj`` looks like a figshare article record with folders."""
    return isinstance(obj, dict) and "folder_structure" in obj and "files" in obj


def _subject(group, folder_subject):
    """MOABB subject of the depositor folder ``<group>/EEG/subject<N>``."""
    return folder_subject if group == "Normals" else _N_HEALTHY + folder_subject


def _md5(path, chunk=1 << 22):
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_intact(path, size, md5):
    return path.is_file() and path.stat().st_size == size and _md5(path) == md5


class HefmiIch2025(BaseDataset):
    """Hybrid EEG-fNIRS MI dataset for ICH from Shi et al 2025.

    Dataset from the article *HEFMI-ICH: a hybrid EEG-fNIRS motor
    imagery dataset for brain-computer interface in intracerebral
    hemorrhage* [1]_.

    It contains EEG data from 37 subjects (17 healthy controls +
    20 ICH patients) recorded with 32-channel EEG at 256 Hz.
    The motor imagery task involves left and right hand grasping.

    Each session has 30 trials (15 left, 15 right). Trial structure:
    2 s cue + 10 s MI + 15 s rest. Subjects have 1 to 3 sessions, 98 in
    total (2,940 trials): 3 sessions for 25 subjects, 2 for subjects 1, 23,
    24, 26, 28 and 32-37, and 1 for subject 25. The sessions of a subject are
    consecutive runs of one visit, not separate days: the raw recordings of
    every subject with several sessions start on the same day, 15-18 min
    apart.

    The data is pre-epoched in .mat format on Figshare.
    Only the EEG portion is loaded (fNIRS is excluded).

    **Subjects and sessions.** The figshare file names carry no subject
    (``1_epo.mat``, ``2_epo.mat``, ``3_epo.mat``). The depositor's folder of
    every file is given by the ``folder_structure`` of the article record
    (version 4, https://api.figshare.com/v2/articles/28955456/versions/4),
    which is also the folder layout of the article's ZIP:
    ``epo_data/{Normals,Patients}/EEG/subject<N>/<k>_epo.mat``. Subjects 1-17
    are ``Normals/EEG/subject1`` to ``subject17`` (healthy controls),
    subjects 18-37 are ``Patients/EEG/subject1`` to ``subject20`` (ICH
    patients, subject = 17 + N), and session ``str(k - 1)`` is the file
    ``<k>_epo.mat``. The loader reads this mapping from the record (fetched
    once, cached next to the data) and ships a copy of it, with the size and
    MD5 of every file, which it uses when the record cannot be fetched. Each
    file is checked against its size and MD5 before it is read, and is stored
    under the depositor's path. The files' own MAT headers agree with the
    folders: their creation times form 37 contiguous blocks, one per folder,
    with the sessions in order a few seconds apart.

    .. note::
       Earlier versions of this loader grouped the files by figshare upload
       order and started a new subject at every ``1_epo.mat``: 23 of their
       37 subjects mixed sessions of two to four people, and subjects had up
       to 6 sessions. Delete cached epochs and saved results computed with
       them; they do not describe these subjects.

    Licence: the paper states "This dataset is licensed under the Creative
    Commons Attribution (CC-BY) terms" and the figshare record is CC BY 4.0
    (the article itself is CC BY-NC-ND 4.0). Note that the figshare
    description also says the data "can only be used for research and
    learning purposes, and commercial applications are prohibited".

    Parameters
    ----------
    group : str
        Which subject group to load: ``"all"`` (default, subjects
        1-37), ``"healthy"`` (subjects 1-17), or ``"patient"``
        (subjects 18-37).

    References
    ----------
    .. [1] Shi, J., Chen, D., et al. (2025). HEFMI-ICH: a hybrid
           EEG-fNIRS motor imagery dataset for brain-computer
           interface in intracerebral hemorrhage. Scientific Data.
           https://doi.org/10.1038/s41597-025-06100-7
    """

    # NEMAR deposit nm000347 files these epochs under the subjects of the
    # earlier upload-order grouping, so it cannot serve this mapping; declare
    # it again once the deposit follows the depositor's folders.
    nemar_id = None
    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=256.0,
            channel_types={"eeg": 32},
            montage="biosemi32",
            hardware="g.HIamp (g.tec medical engineering GmbH)",
            filters={},
            sensors=list(_CH_NAMES),
            line_freq=50.0,
        ),
        participants=ParticipantMetadata(
            n_subjects=37,
            health_status="mixed (17 healthy, 20 ICH patients)",
            gender={"female": 8, "male": 29},
            age_min=20.0,
            age_max=65.0,
            species="human",
            handedness="right-handed",
            clinical_population="intracerebral hemorrhage (ICH)",
        ),
        experiment=ExperimentMetadata(
            events=dict(_EVENTS),
            paradigm="imagery",
            n_classes=2,
            class_labels=["left_hand", "right_hand"],
            trial_duration=27.0,
            study_design=(
                "2-class hand MI (left/right grasping) for ICH "
                "rehabilitation. 17 healthy + 20 ICH patients, "
                "1-3 sessions per subject (98 in total), consecutive runs "
                "of one visit."
            ),
            feedback_type="none",
            stimulus_type="directional arrow + auditory beep",
            stimulus_modalities=["visual", "auditory"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="offline",
        ),
        documentation=DocumentationMetadata(
            doi="10.1038/s41597-025-06100-7",
            investigators=[
                "Jian Shi",
                "Danyang Chen",
                "Xingwei Zhao",
                "Zhixian Zhao",
                "Shengjie Li",
                "Yeguang Xu",
                "Tao Ding",
                "Zheng Zhu",
                "Peng Zhang",
                "Qing Ye",
                "Yingxin Tang",
                "Ping Zhang",
                "Bo Tao",
                "Zhouping Tang",
            ],
            institution="Huazhong University of Science and Technology",
            country="CN",
            data_url="https://figshare.com/articles/dataset/28955456",
            publication_year=2025,
            license="CC-BY-4.0",
        ),
        sessions_per_subject=3,
        runs_per_session=1,
        tags=Tags(
            pathology=["Healthy", "Stroke"],
            modality=["Motor"],
            type=["Clinical", "Research"],
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="imagery",
            imagery_tasks=["left_hand", "right_hand"],
            cue_duration_s=2.0,
            imagery_duration_s=10.0,
        ),
        data_structure=DataStructureMetadata(
            n_trials=2940,
            trials_context=(
                "98 sessions (1-3 per subject) x 30 trials (15 per class) = 2940"
            ),
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["CSP+SVM", "FBCSP+SVM", "EEGBaseNet", "TF+SVM"],
            feature_extraction=["CSP", "FBCSP", "time-frequency features"],
            frequency_bands={"preprocessing": [0.5, 30.0]},
            spatial_filters=["CSP", "FBCSP"],
        ),
        cross_validation=CrossValidationMetadata(
            cv_method="5-fold", cv_folds=5, evaluation_type=["within_subject"]
        ),
        bci_application=BCIApplicationMetadata(
            applications=["rehabilitation"], environment="clinical", online_feedback=False
        ),
        data_processed=True,
        file_format="MAT (pre-epoched)",
    )

    def __init__(
        self, group="all", subjects=None, sessions=None, *, return_all_modalities=False
    ):
        self.group = group

        if group == "healthy":
            subj_list = list(range(1, 18))
        elif group == "patient":
            subj_list = list(range(18, 38))
        elif group == "all":
            subj_list = list(range(1, 38))
        else:
            raise ValueError(
                f"group must be 'all', 'healthy', or 'patient', got {group!r}"
            )

        super().__init__(
            subjects=subj_list,
            sessions_per_subject=3,
            events=dict(_EVENTS),
            code="HefmiIch2025",
            interval=[0, 10],
            paradigm="imagery",
            doi="10.1038/s41597-025-06100-7",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return ``{session: {"0": raw}}``; session ``str(k - 1)`` is ``<k>_epo.mat``."""
        sessions = {}
        for path in self.data_path(subject):
            session = int(Path(path).name.split("_")[0]) - 1
            sessions[str(session)] = {"0": self._load_epoch_mat(path)}
        return sessions

    def _load_epoch_mat(self, mat_path):
        """Load a pre-epoched MAT file and reconstruct continuous Raw."""
        mat = loadmat(str(mat_path), squeeze_me=True)

        # Navigate the struct: epo.X, epo.y, epo.fs
        epo = mat.get("epo", mat)
        if hasattr(epo, "dtype") and epo.dtype.names:
            X = epo["X"].item()  # (samples, channels, trials)
            y = epo["y"].item().ravel()  # (trials,) -- 0=left, 1=right
            fs = float(epo["fs"].item()) if "fs" in epo.dtype.names else _SFREQ
        else:
            X = mat.get("X", mat.get("data"))
            y = mat.get("y", mat.get("labels", np.array([]))).ravel()
            fs = float(mat.get("fs", _SFREQ))

        if X.ndim != 3:
            raise ValueError(f"Expected 3D epoch array, got shape {X.shape}")

        n_samples, n_ch, n_trials = X.shape
        ch_names = (
            list(_CH_NAMES[:n_ch])
            if n_ch <= len(_CH_NAMES)
            else [f"EEG{i + 1}" for i in range(n_ch)]
        )

        # Transpose from (samples, channels, trials) to (trials, channels, samples).
        epochs = X.transpose(2, 1, 0)

        # y: 0=left, 1=right -> MOABB: 1=left, 2=right
        event_ids = y.astype(int) + 1

        # MI onset is at 12s into each epoch (from MakeDatasetFromRaw.m).
        return build_raw_from_epochs(
            epochs,
            ch_names,
            fs,
            event_ids,
            "colin27_1005",
            buffer_samples=int(0.5 * fs),
            onset_sample=int(12 * fs),
        )

    @staticmethod
    def _figshare_record(basepath):
        """Return the figshare record of the article version, or None.

        Read from the cache, else from the active NEMAR store, else from the
        figshare API (then cached under the same name).
        """
        cache_path = basepath / _RECORD_NAME
        if dl.fill_from_store(basepath, [cache_path]):
            try:
                with open(cache_path) as f:
                    record = json.load(f)
                if not _is_record(record):
                    raise ValueError("not a figshare article record")
                return record
            except (OSError, ValueError) as e:
                log.warning("Refetching unreadable %s: %s", cache_path, e)
                cache_path.unlink(missing_ok=True)

        log.info("Fetching the HEFMI-ICH figshare record (%s)...", _RECORD_URL)
        try:
            record = dl.fs_issue_request(
                "GET", _RECORD_URL, headers={"Content-Type": "application/json"}
            )
        except Exception as e:
            log.warning("Could not fetch the figshare record: %s", e)
            return None
        if not _is_record(record):
            log.warning("The figshare API answer is not an article record.")
            return None
        try:
            basepath.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(record, f)
        except OSError:
            pass
        return record

    @classmethod
    def _epoch_files(cls, basepath):
        """The EEG epoch files, from the figshare record or the shipped copy."""
        record = cls._figshare_record(basepath)
        if record is None:
            log.info("Using the HEFMI-ICH file map shipped with MOABB.")
            return _EPOCH_FILES
        try:
            rows = _epoch_files_from_record(record)
        except (KeyError, TypeError, AttributeError) as e:
            log.warning("Unreadable figshare record (%s); using the shipped map.", e)
            return _EPOCH_FILES
        if rows != _EPOCH_FILES:
            warnings.warn(
                f"The figshare record of article {_FIGSHARE_ARTICLE} version "
                f"{_FIGSHARE_VERSION} no longer matches the file map shipped with "
                "MOABB (EEG epoch files, their folders, sizes or MD5s differ); "
                "using the shipped map, which this loader was validated against.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _EPOCH_FILES
        return rows

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the subject's epoch files, in session order.

        Each file is downloaded from figshare by its file id into the
        depositor's layout (``epo_data/<group>/EEG/subject<N>/<k>_epo.mat``
        below ``MNE-hefmiich2025-data``) and checked against its size and MD5.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        root = Path(dl.get_dataset_path(self.code, path))
        basepath = root / "MNE-hefmiich2025-data"
        basepath.mkdir(parents=True, exist_ok=True)

        rows = [
            row
            for row in self._epoch_files(basepath)
            if _subject(row[1], row[2]) == subject
        ]
        if not rows:
            raise FileNotFoundError(f"No EEG epoch files found for subject {subject}")

        paths = []
        for file_id, group, folder_subject, session, size, md5 in sorted(
            rows, key=lambda row: row[3]
        ):
            fname = f"epo_data/{group}/EEG/subject{folder_subject}/{session}_epo.mat"
            local_path = basepath / fname
            if force_update or not _is_intact(local_path, size, md5):
                local_path.unlink(missing_ok=True)
                url = f"https://ndownloader.figshare.com/files/{file_id}"
                downloaded = Path(
                    dl.data_dl(url, self.code, root, force_update, verbose, fname=fname)
                )
                if downloaded != local_path:
                    downloaded.replace(local_path)
                if not _is_intact(local_path, size, md5):
                    raise OSError(
                        f"{local_path} does not match figshare file {file_id} "
                        f"({size} bytes, MD5 {md5}); delete it and load again."
                    )
            paths.append(str(local_path))
        return paths
