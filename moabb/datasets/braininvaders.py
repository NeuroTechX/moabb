import glob
import os
import os.path as osp
import shutil
import zipfile as z
from warnings import warn

import mne
import numpy as np
import yaml
from mne.channels import make_standard_montage
from mne.utils import _open_lock
from scipy.io import loadmat

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
from moabb.datasets.utils import block_rep
from moabb.utils import _handle_deprecated_kwargs, depreciated_alias


BI2012a_URL = "https://zenodo.org/record/2649069/files/"
BI2013a_URL = "https://zenodo.org/record/2669187/files/"
BI2014a_URL = "https://zenodo.org/record/3266223/files/"
BI2014b_URL = "https://zenodo.org/record/3267302/files/"
BI2015a_URL = "https://zenodo.org/record/3266930/files/"
BI2015b_URL = "https://zenodo.org/record/3268762/files/"
VIRTUALREALITY_URL = "https://zenodo.org/record/2605205/files/"


def _bi_get_subject_data(ds, subject):  # noqa: C901
    file_path_list = ds.data_path(subject)

    sessions = {}

    for file_path in file_path_list:
        if ds.code in [
            "BrainInvaders2012",
            "BrainInvaders2014a",
            "BrainInvaders2014b",
            "BrainInvaders2015b",
        ]:
            session_name = "0"
        elif ds.code == "BrainInvaders2013a":
            session_number = file_path.split(os.sep)[-2].replace("Session", "")
            session_number = int(session_number) - 1
            session_name = str(session_number)
        elif ds.code == "BrainInvaders2015a":
            session_number = file_path.split("_")[-1][1:2]
            session_number = int(session_number) - 1
            session_name = str(session_number)
        elif ds.code == "Cattan2019-VR":
            session_map = {"VR": "0VR", "PC": "1PC"}
            session_name = file_path.split("_")[-1].split(".")[0]
            session_name = session_map[session_name]

        if session_name not in sessions.keys():
            sessions[session_name] = {}

        if ds.code == "BrainInvaders2012":
            condition_map = {"training": "0training", "online": "1online"}
            condition = file_path.split("/")[-1].split(".")[0].split(os.sep)[-1]
            run_name = condition_map[condition]
            # fmt: off
            chnames = [
                'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4',
                'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2', 'STI 014'
            ]
            # fmt: on
            chtypes = ["eeg"] * 17 + ["stim"]
            X = loadmat(file_path)[condition].T
            S = X[1:18, :] * 1e-6
            stim = (X[18, :] + X[19, :])[None, :]
            X = np.concatenate([S, stim])
            sfreq = 128
        elif ds.code == "BrainInvaders2013a":
            run_number = file_path.split(os.sep)[-1]
            run_number = run_number.split("_")[-1]
            run_number = run_number.split(".mat")[0]
            run_number = int(run_number) - 1
            run_name = str(run_number)
            # fmt: off
            chnames = [
                "Fp1", "Fp2", "F5", "AFz", "F6", "T7", "Cz", "T8", "P7",
                "P3", "Pz", "P4", "P8", "O1", "Oz", "O2", "STI 014",
            ]
            # fmt: on
            chtypes = ["eeg"] * 16 + ["stim"]
            X = loadmat(file_path)["data"].T
            sfreq = 512
        elif ds.code == "BrainInvaders2014a":
            run_name = "0"
            # fmt: off
            chnames = [
                'Fp1', 'Fp2', 'F3', 'AFz', 'F4', 'T7', 'Cz', 'T8', 'P7',
                'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'STI 014'
            ]
            # fmt: on
            chtypes = ["eeg"] * 16 + ["stim"]
            file_path = file_path_list[0]
            D = loadmat(file_path)["samples"].T
            S = D[1:17, :] * 1e-6
            stim = D[-1, :]
            X = np.concatenate([S, stim[None, :]])
            sfreq = 512
        elif ds.code == "BrainInvaders2014b":
            # fmt: off
            chnames = [
                'Fp1', 'Fp2', 'AFz', 'F7', 'F3', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
                'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6',
                'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'O1', 'Oz', 'O2', 'PO8', 'PO9',
                'PO10', 'STI 014']
            # fmt: on
            chtypes = ["eeg"] * 32 + ["stim"]
            run_name = "0"

            D = loadmat(file_path)["samples"].T
            if subject % 2 == 1:
                S = D[1:33, :] * 1e-6
            else:
                S = D[33:65, :] * 1e-6
            stim = D[-1, :]
            X = np.concatenate([S, stim[None, :]])
            sfreq = 512
        elif ds.code == "BrainInvaders2015a":
            run_name = "0"
            # fmt: off
            chnames = [
                'Fp1', 'Fp2', 'AFz', 'F7', 'F3', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
                'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3',
                'Pz', 'P4', 'P8', 'PO7', 'O1', 'Oz', 'O2', 'PO8', 'PO9', 'PO10', 'STI 014'
            ]
            # fmt: on
            chtypes = ["eeg"] * 32 + ["stim"]
            D = loadmat(file_path)["DATA"].T
            S = D[1:33, :] * 1e-6
            stim = D[-2, :] + D[-1, :]
            X = np.concatenate([S, stim[None, :]])
            sfreq = 512
        elif ds.code == "BrainInvaders2015b":
            run_number = file_path.split("_")[-1].split(".")[0][1]
            run_number = int(run_number) - 1
            run_name = str(run_number)
            # fmt: off
            chnames = [
                'Fp1', 'Fp2', 'AFz', 'F7', 'F3', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
                'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6',
                'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'O1', 'Oz', 'O2', 'PO8', 'PO9',
                'PO10', 'STI 014']
            # fmt: on
            chtypes = ["eeg"] * 32 + ["stim"]

            D = loadmat(file_path)["mat_data"].T
            if subject % 2 == 1:
                S = D[1:33, :] * 1e-6
            else:
                S = D[33:65, :] * 1e-6
            stim = D[-1, :]
            idx_target = (stim >= 60) & (stim <= 85)
            idx_nontarget = (stim >= 20) & (stim <= 45)
            stim[idx_target] = 2
            stim[idx_nontarget] = 1
            X = np.concatenate([S, stim[None, :]])
            sfreq = 512
        elif ds.code == "Cattan2019-VR":
            data = loadmat(os.path.join(file_path, os.listdir(file_path)[0]))["data"]

            chnames = [
                "Fp1",
                "Fp2",
                "Fc5",
                "Fz",
                "Fc6",
                "T7",
                "Cz",
                "T8",
                "P7",
                "P3",
                "Pz",
                "P4",
                "P8",
                "O1",
                "Oz",
                "O2",
                "stim",
            ]

            S = data[:, 1:17]
            stim = 2 * data[:, 18] + 1 * data[:, 19]
            chtypes = ["eeg"] * 16 + ["stim"]
            X = np.concatenate([S * 1e-6, stim[:, None]], axis=1).T

            sfreq = 512

        info = mne.create_info(
            ch_names=chnames,
            sfreq=sfreq,
            ch_types=chtypes,
            verbose=False,
        )

        if not ds.code == "Cattan2019-VR":
            raw = mne.io.RawArray(data=X, info=info, verbose=False)
            raw.set_montage(make_standard_montage("standard_1020"))

            if ds.code == "BrainInvaders2012":
                # get rid of the Fz channel (it is the ground)
                raw.info["bads"] = ["Fz"]
                if not ds.return_all_modalities:
                    raw.pick_types(eeg=True, stim=True)

            sessions[session_name][run_name] = raw
        else:
            idx_blockStart = np.where(data[:, 20] > 0)[0]
            idx_repetEndin = np.where(data[:, 21] > 0)[0]

            sessions[session_name] = {}
            for bi, idx_bi in enumerate(idx_blockStart):
                start = idx_bi
                end = idx_repetEndin[4::5][bi]
                Xbi = X[:, start:end]

                idx_repetEndin_local = (
                    idx_repetEndin[bi * 5 : (bi * 5 + 5)] - idx_blockStart[bi]
                )
                idx_repetEndin_local = np.concatenate([[0], idx_repetEndin_local])
                for j in range(5):
                    start = idx_repetEndin_local[j]
                    end = idx_repetEndin_local[j + 1]
                    Xbij = Xbi[:, start:end]
                    raw = mne.io.RawArray(data=Xbij, info=info, verbose=False)
                    sessions[session_name][block_rep(bi, j, 5)] = raw

    return sessions


def _bi_data_path(  # noqa: C901
    ds, subject, path=None, force_update=False, update_path=None, verbose=None
):
    if subject not in ds.subject_list:
        raise (ValueError("Invalid subject number"))

    subject_paths = []
    if ds.code == "BrainInvaders2012":
        # check if has the .zip
        url = f"{BI2012a_URL}subject_{subject:02}.zip"
        path_zip = dl.data_dl(url, "BRAININVADERS2012")
        path_folder = path_zip.strip(f"subject_{subject:02}.zip")

        # check if has to unzip
        if not (osp.isdir(path_folder + f"subject_{subject}")) and not (
            osp.isdir(path_folder + f"subject_0{subject}")
        ):
            zip_ref = z.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder)

        # filter the data regarding the experimental conditions
        if ds.training:
            subject_paths.append(
                osp.join(f"{path_folder}subject_{subject:02}", "training.mat")
            )
        if ds.online:
            subject_paths.append(
                osp.join(f"{path_folder}subject_{subject:02}", "online.mat")
            )

    elif ds.code == "BrainInvaders2013a":
        if subject in [1, 2, 3, 4, 5, 6, 7]:
            zipname_list = [
                f"subject{subject:02}_session{i:02}.zip" for i in range(1, 8 + 1)
            ]
        else:
            zipname_list = [f"subject{subject:02}.zip"]

        for i, zipname in enumerate(zipname_list):
            url = BI2013a_URL + zipname
            path_zip = dl.data_dl(url, "BRAININVADERS2013")
            path_folder = path_zip.strip(zipname)

            # check if has the directory for the subject
            directory = f"{path_folder}subject_{subject:02}"
            if not (osp.isdir(directory)):
                os.makedirs(directory)

            if not (osp.isdir(osp.join(directory, f"Session{i + 1}"))):
                zip_ref = z.ZipFile(path_zip, "r")
                zip_ref.extractall(path_folder)
                os.makedirs(osp.join(directory, f"Session{i + 1}"))
                shutil.copytree(path_zip.strip(".zip"), directory, dirs_exist_ok=True)
                shutil.rmtree(path_zip.strip(".zip"))

        # filter the data regarding the experimental conditions
        meta_file = directory + os.sep + "meta.yml"
        with _open_lock(meta_file, "r") as stream:
            meta = yaml.load(stream, Loader=yaml.FullLoader)
        conditions = []
        if ds.adaptive:
            conditions = conditions + ["adaptive"]
        if ds.non_adaptive:
            conditions = conditions + ["nonadaptive"]
        types = []
        if ds.training:
            types = types + ["training"]
        if ds.online:
            types = types + ["online"]
        filenames = []
        for run in meta["runs"]:
            run_condition = run["experimental_condition"]
            run_type = run["type"]
            if (run_condition in conditions) and (run_type in types):
                filenames = filenames + [run["filename"]]

        # list the filepaths for this subject
        for filename in filenames:
            subject_paths = subject_paths + glob.glob(
                osp.join(directory, "Session*", filename.replace(".gdf", ".mat"))
            )

    elif ds.code == "BrainInvaders2014a":
        url = f"{BI2014a_URL}subject_{subject:02}.zip"
        path_zip = dl.data_dl(url, "BRAININVADERS2014A")
        path_folder = path_zip.strip(f"subject_{subject:02}.zip")

        # check if has to unzip
        path_folder_subject = f"{path_folder}subject_{subject:02}"
        if not (osp.isdir(path_folder_subject)):
            os.mkdir(path_folder_subject)
            zip_ref = z.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder_subject)

        # filter the data regarding the experimental conditions
        subject_paths.append(osp.join(path_folder_subject, f"subject_{subject:02}.mat"))

    elif ds.code == "BrainInvaders2014b":
        group = (subject + 1) // 2
        url = f"{BI2014b_URL}group_{group:02}_mat.zip"
        path_zip = dl.data_dl(url, "BRAININVADERS2014B")
        path_folder = path_zip.strip(f"group_{group:02}_mat.zip")

        # check if has to unzip
        path_folder_subject = f"{path_folder}group_{group:02}"
        if not (osp.isdir(path_folder_subject)):
            os.mkdir(path_folder_subject)
            zip_ref = z.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder_subject)

        subject_paths = []
        # filter the data regarding the experimental conditions
        if subject % 2 == 1:
            subject_paths.append(
                osp.join(path_folder_subject, f"group_{group:02}_sujet_01.mat")
            )
        else:
            subject_paths.append(
                osp.join(path_folder_subject, f"group_{group:02}_sujet_02.mat")
            )
        # Collaborative session are not loaded
        # subject_paths.append(osp.join(path_folder_subject, f'group_{(subject+1)//2:02}.mat')

    elif ds.code == "BrainInvaders2015a":
        # TODO: possible fusion with 2014a?
        url = f"{BI2015a_URL}subject_{subject:02}_mat.zip"
        path_zip = dl.data_dl(url, "BRAININVADERS2015A")
        path_folder = path_zip.strip(f"subject_{subject:02}.zip")

        # check if has to unzip
        path_folder_subject = f"{path_folder}subject_{subject:02}"
        if not (osp.isdir(path_folder_subject)):
            os.mkdir(path_folder_subject)
            zip_ref = z.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder_subject)

        # filter the data regarding the experimental conditions
        subject_paths = []
        for session in [1, 2, 3]:
            subject_paths.append(
                osp.join(
                    path_folder_subject, f"subject_{subject:02}_session_{session:02}.mat"
                )
            )
    elif ds.code == "BrainInvaders2015b":
        # TODO: possible fusion with 2014b?
        url = f"{BI2015b_URL}group_{(subject + 1) // 2:02}_mat.zip"
        path_zip = dl.data_dl(url, "BRAININVADERS2015B")
        path_folder = path_zip.strip(f"group_{(subject + 1) // 2:02}_mat.zip")
        # check if has to unzip
        path_folder_subject = f"{path_folder}group_{(subject + 1) // 2:02}"
        if not (osp.isdir(path_folder_subject)):
            os.mkdir(path_folder_subject)
            zip_ref = z.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder_subject)

        subject_paths = []
        subject_paths = [
            osp.join(
                path_folder,
                f"group_{(subject + 1) // 2:02}",
                f"group_{(subject + 1) // 2:02}_s{i}",
            )
            for i in range(1, 5)
        ]
    elif ds.code == "Cattan2019-VR":
        subject_paths = []
        if ds.virtual_reality:
            url = "{:s}subject_{:02d}_{:s}.mat".format(VIRTUALREALITY_URL, subject, "VR")
            file_path = dl.data_path(url, "VIRTUALREALITY")
            subject_paths.append(file_path)
        if ds.personal_computer:
            url = "{:s}subject_{:02d}_{:s}.mat".format(VIRTUALREALITY_URL, subject, "PC")
            file_path = dl.data_path(url, "VIRTUALREALITY")
            subject_paths.append(file_path)

    return subject_paths


@depreciated_alias("bi2012", "1.1")
class BI2012(BaseDataset):
    """P300 dataset BI2012 from a "Brain Invaders" experiment.

    Dataset following the setup from [1]_ carried-out at University of
    Grenoble Alpes.

    This dataset contains electroencephalographic (EEG) recordings of 25 subjects testing
    the Brain Invaders, a visual P300 Brain-Computer Interface inspired by the famous vintage
    video game Space Invaders (Taito, Tokyo, Japan). The visual P300 is an event-related
    potential elicited by a visual stimulation, peaking 240-600 ms after stimulus onset. EEG
    data were recorded by 16 electrodes in an experiment that took place in the GIPSA-lab,
    Grenoble, France, in 2012). A full description of the experiment is available in [1]_.

    :Principal Investigator: B.Sc. Gijsbrecht Franciscus Petrus Van Veen

    :Technical Supervisors: Ph.D. Alexandre Barachant, Eng. Anton Andreev, Eng. Grégoire Cattan,
                            Eng. Pedro. L. C. Rodrigues

    :Scientific Supervisor: Ph.D. Marco Congedo

    :ID of the dataset: BI.EEG.2012-GIPSA

    Notes
    -----

    .. versionadded:: 0.4.6

    References
    ----------

    .. [1] Van Veen, G., Barachant, A., Andreev, A., Cattan, G., Rodrigues, P. C., &
           Congedo, M. (2019). Building Brain Invaders: EEG data of an experimental validation.
           arXiv preprint arXiv:1905.05182.
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=128.0,
            n_channels=16,
            channel_types={"eeg": 16},
            montage="standard_1020",
            hardware="NeXus-32 (MindMedia/TMSi)",
            sensor_type="EEG",
            reference="hardware common average reference",
            ground="FZ",
            software="OpenVibe",
            sensors=[
                "F7",
                "F3",
                "F4",
                "F8",
                "T7",
                "C3",
                "Cz",
                "C4",
                "T8",
                "P7",
                "P3",
                "Pz",
                "P4",
                "P8",
                "O1",
                "O2",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=False,
            ),
            electrode_type="wet",
            electrode_material="Silver/Silver Chloride",
        ),
        participants=ParticipantMetadata(
            n_subjects=25,
            health_status="healthy",
            gender=None,
            age_mean=24.4,
            age_std=2.76,
            age_min=21,
            age_max=31,
            bci_experience="half played games occasionally (around 4.5 hours a week)",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            study_design="longitudinal and transversal design with training-test mode of operation",
            feedback_type="visual (game interface)",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="both",
            has_training_test_split=True,
            instructions="limit eye blinks, head movements and face muscular contractions; silently count the number of Target flashes",
            n_classes=2,
            class_labels=["Target", "NonTarget"],
            stimulus_type="visual flashes of alien groups",
            synchronicity="synchronous",
            events={"Target": 2, "NonTarget": 1},
            stimulus_presentation={
                "repetition_structure": "12 flashes per repetition (2 Target, 10 non-Target)",
                "flash_groups": "12 groups of 6 aliens (36 total aliens)",
                "target_ratio": "1:5 (Target to non-Target)",
                "screen_distance": "75 to 115 cm",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.2649006",
            description="EEG recordings of 25 subjects testing the Brain Invaders, a visual P300 Brain-Computer Interface inspired by the famous vintage video game Space Invaders",
            investigators=[
                "G.F.P. Van Veen",
                "A. Barachant",
                "A. Andreev",
                "G. Cattan",
                "P. Rodrigues",
                "M. Congedo",
            ],
            institution="GIPSA-lab, CNRS, University Grenoble-Alpes, Grenoble INP",
            country="FR",
            repository="Zenodo",
            data_url="https://doi.org/10.5281/zenodo.2649006",
            publication_year=2019,
            senior_author="M. Congedo",
            institution_address="GIPSA-lab, 11 rue des Mathématiques, Grenoble Campus BP46, F-38402, France",
            keywords=[
                "Electroencephalography (EEG)",
                "P300",
                "Brain-Computer Interface",
                "Experiment",
            ],
            associated_paper_doi="10.5281/zenodo.2649006",
            acknowledgements="All subjects were volunteers recruited by means of flyers and of the mailing list of the University of Grenoble-Alpes. All participants provided written informed consent confirming the notification of the experimental process, the data management procedures and the right to withdraw from the experiment at any moment.",
            license="CC-BY-4.0",
        ),
        sessions_per_subject=1,
        runs_per_session=2,
        sessions=["0"],
        contributing_labs=["GIPSA-lab"],
        n_contributing_labs=1,
        data_processed=False,
        file_format="mat, csv",
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw EEG with software tagging (note: tagging introduces jitter and latency)",
            preprocessing_applied=False,
            artifact_methods=None,
            re_reference=None,
            notes="Software tagging introduces a jitter and a latency which artificially modify the ERPs onset. Strong drift over time resulting in higher jitter. Only possible to compare ERP acquired within the same experimental conditions when latency is not corrected.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["xDAWN", "Riemannian"],
            feature_extraction=["Covariance/Riemannian", "xDAWN"],
            spatial_filters=["xDAWN"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=None,
        ),
        performance={"balanced_accuracy": None},
        bci_application=BCIApplicationMetadata(
            applications=["gaming"],
            online_feedback=True,
            environment="laboratory",
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_repetitions=8,
            n_targets=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials={"Target": 128, "non-Target": 640},
            n_trials_per_class={"Target": 128, "non-Target": 640},
            trials_context="per session (Training session); variable in Online session depending on user performance",
            n_blocks=None,
            block_duration_s=None,
        ),
        abstract="We describe the experimental procedures for a dataset that we have made publicly available at https://doi.org/10.5281/zenodo.2649006 in mat and csv formats. This dataset contains electroencephalographic (EEG) recordings of 25 subjects testing the Brain Invaders (1), a visual P300 Brain-Computer Interface inspired by the famous vintage video game Space Invaders (Taito, Tokyo, Japan). The visual P300 is an event-related potential elicited by a visual stimulation, peaking 240-600 ms after stimulus onset. EEG data were recorded by 16 electrodes in an experiment that took place in the GIPSA-lab, Grenoble, France, in 2012 (2,3). Python code for manipulating the data is available at https://github.com/plcrodrigues/py.BI.EEG.2012-GIPSA. The ID of this dataset is BI.EEG.2012-GIPSA.",
        methodology="The visual P300 is an event-related potential (ERP) elicited by a visual stimulation, peaking 240-600 ms after stimulus onset. The experiment features a training-test mode of operation and both a longitudinal and transversal design. Training session: Target alien chosen randomly at each repetition, 8 Targets total, 8 repetitions each, resulting in 128 Target trials and 640 non-Target flashes. Online session: consisted of three levels with different distractor configurations, minimum 3.5 minutes per level, counter-balanced order across subjects. Interface: 36 aliens flashing in 12 groups of 6, each repetition has 12 flashes (2 Target, 10 non-Target). P300 peak latency: 240-600 ms post-stimulus.",
    )

    def __init__(
        self,
        training=True,
        online=False,
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
        **kwargs,
    ):
        deprecated_renames = {"Training": "training", "Online": "online"}
        resolved = _handle_deprecated_kwargs(kwargs, deprecated_renames, "BI2012")
        training = resolved.get("training", training)
        online = resolved.get("online", online)

        super().__init__(
            subjects=list(range(1, 26)),
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),
            code="BrainInvaders2012",
            interval=[0, 1],
            paradigm="p300",
            doi="https://doi.org/10.5281/zenodo.2649006",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

        self.training = training
        self.online = online

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        return _bi_get_subject_data(self, subject)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return _bi_data_path(self, subject, path, force_update, update_path, verbose)


@depreciated_alias("bi2013a", "1.1")
class BI2013a(BaseDataset):
    """P300 dataset BI2013a from a "Brain Invaders" experiment.

    Dataset following the setup from [1]_ carried-out at University of
    Grenoble Alpes.

    This dataset concerns an experiment carried out at GIPSA-lab
    (University of Grenoble Alpes, CNRS, Grenoble-INP) in 2013.
    The recordings concerned 24 subjects in total. Subjects 1 to 7 participated
    to eight sessions, run in different days, subject 8 to 24 participated to
    one session. Each session consisted in two runs, one in a Non-Adaptive
    (classical) and one in an Adaptive (calibration-less) mode of operation.
    The order of the runs was randomized for each session. In both runs there
    was a Training (calibration) phase and an Online phase, always passed in
    this order. In the non-Adaptive run the data from the Training phase was
    used for classifying the trials on the Online phase using the training-test
    version of the MDM algorithm [2]_. In the Adaptive run, the data from the
    training phase was not used at all, instead the classifier was initialized
    with generic class geometric means and continuously adapted to the incoming
    data using the Riemannian method explained in [2]_. Subjects were completely
    blind to the mode of operation and the two runs appeared to them identical.

    In the Brain Invaders P300 paradigm, a repetition is composed of 12
    flashes, of which 2 include the Target symbol (Target flashes) and 10 do
    not (non-Target flash). Please see [3]_ for a description of the paradigm.
    For this experiment, in the Training phases the number of flashes is fixed
    (80 Target flashes and 400 non-Target flashes). In the Online phases the
    number of Target and non-Target still are in a ratio 1/5, however their
    number is variable because the Brain Invaders works with a fixed number of
    game levels, however the number of repetitions needed to destroy the target
    (hence to proceed to the next level) depends on the user’s performance
    [2]_. In any case, since the classes are unbalanced, an appropriate score
    must be used for quantifying the performance of classification methods
    (e.g., balanced accuracy, AUC methods, etc).

    Data were acquired with a Nexus (TMSi, The Netherlands) EEG amplifier:

    * Sampling Frequency: 512 samples per second
    * Digital Filter: no
    * Electrodes:  16 wet Silver/Silver Chloride electrodes positioned at
      FP1, FP2, F5, AFz, F6, T7, Cz, T8, P7, P3, Pz, P4, P8, O1, Oz, O2
      according to the 10/20 international system.
    * Reference: left ear-lobe.
    * Ground: N/A.

    :Principal Investigators: Erwan Vaineau, Dr. Alexandre Barachant
    :Scientific Supervisor:  Dr. Marco Congedo
    :Technical Supervisor: Anton Andreev

    References
    ----------

    .. [1] Vaineau, E., Barachant, A., Andreev, A., Rodrigues, P. C.,
           Cattan, G. & Congedo, M. (2019). Brain invaders adaptive
           versus non-adaptive P300 brain-computer interface dataset.
           arXiv preprint arXiv:1904.09111.

    .. [2] Barachant A, Congedo M (2014) A Plug & Play P300 BCI using
           Information Geometry.
           arXiv:1409.0107.

    .. [3] Congedo M, Goyat M, Tarrin N, Ionescu G, Rivet B,Varnet L, Rivet B,
           Phlypo R, Jrad N, Acquadro M, Jutten C (2011) “Brain Invaders”: a
           prototype of an open-source P300-based video game working with the
           OpenViBE platform. Proc. IBCI Conf., Graz, Austria, 280-283.
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=16,
            channel_types={"eeg": 16},
            montage="standard_1020",
            hardware="g.USBamp (g.tec, Schiedlberg, Austria)",
            sensor_type="wet Silver/Silver Chloride electrodes",
            reference="left earlobe",
            ground="FZ",
            software="OpenVibe",
            sensors=[
                "Fp1",
                "Fp2",
                "F5",
                "AFz",
                "F6",
                "T7",
                "Cz",
                "T8",
                "P7",
                "P3",
                "Pz",
                "P4",
                "P8",
                "O1",
                "Oz",
                "O2",
            ],
            line_freq=50.0,
            filters="no digital filter applied",
            cap_manufacturer="g.tec",
            cap_model="g.GAMMAcap",
            electrode_type="wet",
            electrode_material="Silver/Silver Chloride",
            auxiliary_channels=None,
        ),
        participants=ParticipantMetadata(
            n_subjects=24,
            health_status="healthy",
            gender={"male": 12, "female": 12},
            age_mean=25.96,
            age_std=4.46,
            age_min=20.0,
            age_max=30.0,
            species="human",
            bci_experience="volunteers recruited via flyers and university mailing list",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            task_type="visual P300 BCI",
            events={"Target": 33285, "NonTarget": 33286},
            n_classes=2,
            class_labels=["Target", "NonTarget"],
            trial_duration=None,
            study_design="compare P300-based BCI with and without adaptive calibration using Riemannian geometry; randomised order of runs (adaptive vs non-adaptive)",
            feedback_type="visual (Brain Invaders video game interface)",
            stimulus_type="visual flashes",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="both",
            has_training_test_split=True,
            instructions="destroy targets in Brain Invaders BCI video game",
            stimulus_presentation={
                "distance_from_screen": "75 to 115 cm",
                "screen": "ViewSonic 22 inch",
                "flash_groups": "36 symbols distributed in 12 groups",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.1494163",
            description="EEG recordings of 24 subjects doing a visual P300 Brain-Computer Interface experiment comparing adaptive vs non-adaptive calibration using Riemannian geometry",
            investigators=[
                "E. Vaineau",
                "A. Barachant",
                "A. Andreev",
                "P. Rodrigues",
                "G. Cattan",
                "M. Congedo",
            ],
            institution="GIPSA-lab, CNRS, University Grenoble-Alpes, Grenoble INP",
            country="FR",
            repository="Zenodo",
            data_url="https://doi.org/10.5281/zenodo.1494163",
            publication_year=2019,
            senior_author="M. Congedo",
            associated_paper_doi="10.5281/zenodo.2649006",
            institution_address="GIPSA-lab, 11 rue des Mathématiques, Grenoble Campus BP46, F-38402, France",
            ethics_approval=[
                "Approved by the Ethical Committee of the University of Grenoble Alpes (Comité d'Ethique pour la Recherche Non-Interventionnelle)"
            ],
            keywords=[
                "Electroencephalography (EEG)",
                "P300",
                "Brain-Computer Interface",
                "Experiment",
                "Adaptive",
                "Calibration",
            ],
            license="CC-BY-1.0",
        ),
        sessions_per_subject=8,
        runs_per_session=2,
        contributing_labs=["GIPSA-lab"],
        n_contributing_labs=1,
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw EEG with software tagging via USB (note: tagging introduces jitter and latency)",
            preprocessing_applied=False,
            artifact_methods=None,
            re_reference=None,
            notes="Tags sent by application to amplifier through USB port and recorded as supplementary channel; tagging process identical in all experimental conditions",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=[
                "xDAWN",
                "Riemannian",
                "RMDM (Riemannian Minimum Distance to Mean)",
            ],
            feature_extraction=[
                "Covariance/Riemannian",
                "xDAWN",
                "common spatiotemporal pattern",
            ],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["cross_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["gaming"],
            environment="small room (4 square meters) with one-way glass window for experimenter observation",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=None,
            n_repetitions=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials={
                "Training_Target": 80,
                "Training_non-Target": 400,
                "Online": "variable (depends on user performance)",
            },
            trials_context="per_phase",
        ),
        performance={
            "Balanced_Accuracy": "used due to unbalanced classes (1:5 ratio Target to non-Target)",
        },
        data_processed=False,
        file_format="mat, csv, gdf",
        abstract="This dataset contains electroencephalographic (EEG) recordings of 24 subjects doing a visual P300 Brain-Computer Interface experiment on PC. The visual P300 is an event-related potential elicited by visual stimulation, peaking 240-600 ms after stimulus onset. The experiment was designed to compare the use of a P300-based brain-computer interface with and without adaptive calibration using Riemannian geometry. EEG data were recorded using 16 electrodes during an experiment at GIPSA-lab, Grenoble, France, in 2013.",
        methodology="Subjects participated in sessions with two runs (Non-Adaptive and Adaptive, randomised order). Each run had Training (calibration) and Online phases. In Non-Adaptive mode, Training data calibrated the MDM classifier for Online phase. In Adaptive mode, classifier initialized with generic class geometric means from previous experiment and continuously adapted using Riemannian method. Brain Invaders interface: 36 symbols in 12 groups, one repetition = 12 flashes (2 Target, 10 non-Target). Training phase: 80 Target and 400 non-Target flashes (fixed). Online phase: variable repetitions based on performance to destroy targets. Subjects blind to mode of operation.",
    )

    def __init__(
        self,
        non_adaptive=True,
        adaptive=False,
        training=True,
        online=False,
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
        **kwargs,
    ):
        deprecated_renames = {
            "NonAdaptive": "non_adaptive",
            "Adaptive": "adaptive",
            "Training": "training",
            "Online": "online",
        }
        resolved = _handle_deprecated_kwargs(kwargs, deprecated_renames, "BI2013a")
        non_adaptive = resolved.get("non_adaptive", non_adaptive)
        adaptive = resolved.get("adaptive", adaptive)
        training = resolved.get("training", training)
        online = resolved.get("online", online)

        super().__init__(
            subjects=list(range(1, 25)),
            sessions_per_subject=8,
            events=dict(Target=33285, NonTarget=33286),
            code="BrainInvaders2013a",
            interval=[0, 1],
            paradigm="p300",
            doi="https://doi.org/10.5281/zenodo.2669187",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

        self.adaptive = adaptive
        self.non_adaptive = non_adaptive
        self.training = training
        self.online = online

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        return _bi_get_subject_data(self, subject)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return _bi_data_path(self, subject, path, force_update, update_path, verbose)


@depreciated_alias("bi2014a", "1.1")
class BI2014a(BaseDataset):
    """P300 dataset BI2014a from a "Brain Invaders" experiment.

    This dataset contains electroencephalographic (EEG) recordings of 71 subjects
    playing to a visual P300 Brain-Computer Interface (BCI) videogame named Brain Invaders.
    The interface uses the oddball paradigm on a grid of 36 symbols (1 Target, 35 Non-Target)
    that are flashed pseudo-randomly to elicit the P300 response. EEG data were recorded
    using 16 active dry electrodes with up to three game sessions. The experiment took place
    at GIPSA-lab, Grenoble, France, in 2014. A full description of the experiment is available
    at [1]_. The ID of this dataset is BI2014a.

    :Investigators: Eng. Louis Korczowski, B. Sc. Ekaterina Ostaschenko
    :Technical Support: Eng. Anton Andreev, Eng. Grégoire Cattan, Eng. Pedro. L. C. Rodrigues,
                        M. Sc. Violette Gautheret
    :Scientific Supervisor: Ph.D. Marco Congedo

    Notes
    -----

    .. versionadded:: 0.4.6

    References
    ----------

    .. [1] Korczowski, L., Ostaschenko, E., Andreev, A., Cattan, G., Rodrigues, P. L. C.,
           Gautheret, V., & Congedo, M. (2019). Brain Invaders calibration-less P300-based
           BCI using dry EEG electrodes Dataset (BI2014a).
           https://hal.archives-ouvertes.fr/hal-02171575
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=16,
            channel_types={"eeg": 16},
            montage="standard_1010",
            hardware="g.USBamp (g.tec, Schiedlberg, Austria)",
            sensor_type="dry electrodes",
            reference="right earlobe",
            ground="FZ",
            software="OpenVibe",
            sensors=[
                "Fp1",
                "Fp2",
                "F5",
                "AFz",
                "F6",
                "T7",
                "Cz",
                "T8",
                "P7",
                "P3",
                "Pz",
                "P4",
                "P8",
                "O1",
                "Oz",
                "O2",
            ],
            line_freq=50.0,
            auxiliary_channels=AuxiliaryChannelsMetadata(
                has_eog=False,
            ),
            electrode_type="dry 8-pins gold-alloy electrodes",
            electrode_material="gold-alloy",
            cap_manufacturer="g.tec",
            cap_model="g.Sahara",
            filters="no digital filter applied",
        ),
        participants=ParticipantMetadata(
            n_subjects=64,
            gender={"male": 49, "female": 22},
            age_mean=23.55,
            age_std=3.13,
            health_status="healthy",
            bci_experience="57 were naïve BCI participants",
            handedness="not specified",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            task_type="oddball",
            n_classes=2,
            class_labels=["Target", "Non-Target"],
            trial_duration=None,
            study_design="calibration-less P300-based BCI system with dry electrodes; screening session for potential candidates for a broader multi-user BCI study",
            feedback_type="visual",
            stimulus_type="visual flashes",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="online",
            instructions="Destroy the target symbol by focusing attention on it. Players had up to eight attempts to destroy the target symbol per level.",
            stimulus_presentation={
                "n_symbols": "36",
                "n_groups": "12",
                "symbols_per_group": "6",
                "target_flashes_per_repetition": "2",
                "non_target_flashes_per_repetition": "10",
                "animation": "symbols slowly and regularly moving according to predefined path",
            },
            events={"Target": 2, "NonTarget": 1},
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.3266223",
            description="Dataset contains electroencephalographic (EEG) recordings of 71 subjects playing to a visual P300 Brain-Computer Interface (BCI) videogame named Brain Invaders. The interface uses the oddball paradigm on a grid of 36 symbols (1 Target, 35 Non-Target) that are flashed pseudo-randomly to elicit the P300 response.",
            investigators=[
                "Louis Korczowski",
                "Ekaterina Ostaschenko",
                "Anton Andreev",
                "Grégoire Cattan",
                "Pedro Luiz Coelho Rodrigues",
                "Violette Gautheret",
                "Marco Congedo",
            ],
            institution="GIPSA-lab, CNRS, University Grenoble-Alpes, Grenoble INP",
            country="FR",
            repository="Zenodo",
            data_url="https://doi.org/10.5281/zenodo.3266223",
            publication_year=2019,
            senior_author="Marco Congedo",
            associated_paper_doi="hal-02171575",
            institution_address="GIPSA-lab, 11 rue des Mathématiques, Grenoble Campus BP46, F-38402, France",
            ethics_approval=[
                "Approved by the Ethical Committee of the University of Grenoble Alpes (Comité d'Ethique pour la Recherche Non-Interventionnelle)"
            ],
            keywords=[
                "Electroencephalography (EEG)",
                "P300",
                "Brain-Computer Interface",
                "Experiment",
                "Collaboration",
                "Multi-User",
                "Hyperscanning",
            ],
            acknowledgements="At the end of the experiment one ticket of cinema was offered to each subject, for a value of 7.5 euros per subject.",
            license="CC-BY-4.0",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        sessions=None,
        data_processed=False,
        file_format="mat and csv",
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw EEG with hardware tagging (USB digital-to-analog converter for synchronization)",
            preprocessing_applied=False,
            artifact_methods=[],
            re_reference=None,
            notes="No digital filter applied during recording. USB digital-to-analog converter used to reduce jitter and synchronize experimental tags with EEG signals.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=[
                "Riemannian Minimum Distance to Mean (RMDM)",
                "xDAWN",
                "Riemannian",
            ],
            feature_extraction=["Covariance/Riemannian", "xDAWN"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["cross_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["gaming"],
            environment="laboratory",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=1,
            n_repetitions=12,
        ),
        data_structure=DataStructureMetadata(
            n_trials="variable; up to 8 attempts per level, 9 levels per session",
            n_blocks=9,
            block_duration_s=None,
            trials_context="9 levels per session, up to 8 attempts per level to destroy target",
        ),
        performance={
            "note": "Real-time adaptive RMDM classifier used for assessing participants' command with calibration-free procedure",
        },
        abstract="We describe the experimental procedures for the bi2014a dataset that contains electroencephalographic (EEG) recordings of 71 subjects playing to a visual P300 Brain-Computer Interface (BCI) videogame named Brain Invaders. The interface uses the oddball paradigm on a grid of 36 symbols (1 Target, 35 Non-Target) that are flashed pseudo-randomly to elicit the P300 response. EEG data were recorded using 16 active dry electrodes with up to three game sessions. The experiment took place at GIPSA-lab, Grenoble, France, in 2014.",
        methodology="The experiment was designed to study the viability of a calibration-less P300-based BCI system with dry electrodes. Visual P300 is an event-related potential (ERP) elicited by an expected but unpredictable target visual stimulation (oddball paradigm), with peaking amplitude 240-600 ms after stimulus onset. Two event-related stimuli: Target (P300 expected) and Non-Target (no P300). The experiment used Brain Invaders, a P300-based BCI open-source software. A repetition is composed of 12 flashes (one for each group), of which two include the Target symbol (Target flashes) and 10 do not (non-Target flashes). The ratio of Target versus non-Target epochs in the whole datasets is one-to-five. During the experiment, the output of a real-time adaptive Riemannian Minimum Distance to Mean (RMDM) classifier was used for assessing the participants' command. Game session was compounded by nine levels, consisting in a unique and predefined configuration of the 36 symbols of the interface. Players had up to eight attempts to destroy the target symbol. If the player missed all eight attempts, the level was started once again from the beginning. Average duration of five minutes for the nine levels. Experimenter could end the experiment if no control over the BCI system was gained after 10 minutes.",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 65)),
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),
            code="BrainInvaders2014a",
            interval=[0, 1],
            paradigm="p300",
            doi="https://doi.org/10.5281/zenodo.3266222",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        return _bi_get_subject_data(self, subject)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return _bi_data_path(self, subject, path, force_update, update_path, verbose)


@depreciated_alias("bi2014b", "1.1")
class BI2014b(BaseDataset):
    """P300 dataset BI2014b from a "Brain Invaders" experiment.

    This dataset contains electroencephalographic (EEG) recordings of 38 subjects playing in
    pair (19 pairs) to the multi-user version of a visual P300-based Brain-Computer Interface (BCI)
    named Brain Invaders. The interface uses the oddball paradigm on a grid of 36 symbols (1 Target,
    35 Non-Target) that are flashed pseudo-randomly to elicit a P300 response, an evoked-potential
    appearing about 300ms after stimulation onset. EEG data were recorded using 32 active wet
    electrodes per subjects (total: 64 electrodes) during three randomized conditions
    (Solo1, Solo2, Collaboration). The experiment took place at GIPSA-lab, Grenoble, France, in 2014.
    A full description of the experiment is available at [1]_. The ID of this dataset is BI2014b.

    :Investigators: Eng. Louis Korczowski, B. Sc. Ekaterina Ostaschenko
    :Technical Support: Eng. Anton Andreev, Eng. Grégoire Cattan, Eng. Pedro. L. C. Rodrigues,
                        M. Sc. Violette Gautheret
    :Scientific Supervisor: Ph.D. Marco Congedo

    Notes
    -----

    .. versionadded:: 0.4.6

    References
    ----------

    .. [1] Korczowski, L., Ostaschenko, E., Andreev, A., Cattan, G., Rodrigues, P. L. C.,
           Gautheret, V., & Congedo, M. (2019). Brain Invaders Solo versus Collaboration:
           Multi-User P300-Based Brain-Computer Interface Dataset (BI2014b).
           https://hal.archives-ouvertes.fr/hal-02173958
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="standard_1010",
            hardware="g.USBamp (g.tec, Schiedlberg, Austria)",
            sensor_type="wet electrodes",
            reference="right earlobe",
            ground="Fz",
            software="OpenVibe",
            sensors=[
                "Fp1",
                "Fp2",
                "AFz",
                "F7",
                "F3",
                "F4",
                "F8",
                "FC5",
                "FC1",
                "FC2",
                "FC6",
                "T7",
                "C3",
                "Cz",
                "C4",
                "T8",
                "CP5",
                "CP1",
                "CP2",
                "CP6",
                "P7",
                "P3",
                "Pz",
                "P4",
                "P8",
                "PO7",
                "O1",
                "Oz",
                "O2",
                "PO8",
                "PO9",
                "PO10",
            ],
            line_freq=50.0,
            cap_manufacturer="g.tec",
            cap_model="g.GAMMAcap",
            electrode_type="wet",
            electrode_material="Ag/AgCl",
        ),
        participants=ParticipantMetadata(
            n_subjects=38,
            health_status="healthy",
            gender={"male": 24, "female": 14},
            age_mean=24.10,
            age_std=3.09,
            bci_experience="not naïve users - selected on the basis of their individual score during a preliminary session of Brain Invaders",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            task_type="oddball",
            events={"Target": 2, "NonTarget": 1},
            n_classes=2,
            class_labels=["Target", "NonTarget"],
            trial_duration=None,
            study_design="multi-user/hyperscanning experiment with three randomized conditions (Solo1, Solo2, Collaboration). Subjects played in pairs. Solo conditions used a control design where non-playing participant focused on unanimated cross to prevent stimulus observation while EEG was recorded (to correct for fake inter-brain synchrony).",
            study_domain="inter-brain synchrony in collaborative BCI",
            feedback_type="visual",
            stimulus_type="visual flashes",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            synchronicity="synchronous",
            mode="online",
            has_training_test_split=False,
            instructions="destroy the target alien symbol as fast as possible. Up to eight attempts per level. If all attempts missed, level restarted.",
            stimulus_presentation={
                "repetition_structure": "12 flashes per repetition of pseudo-random groups of 6 symbols, such that each symbol flashes exactly twice per repetition",
                "target_ratio": "1:5 (Target vs Non-Target)",
                "flash_groups": "6 rows and 6 columns (pseudo-random groups, not physical arrangement)",
                "animation": "aliens slowly and regularly moved according to predefined path with constant inter-distance",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.3267301",
            description="EEG recordings of 38 subjects playing in pairs to the multi-user version of Brain Invaders P300-based BCI. Contains three conditions: Solo1, Solo2, and Collaboration.",
            investigators=[
                "Louis Korczowski",
                "Ekaterina Ostaschenko",
                "Anton Andreev",
                "Grégoire Cattan",
                "Pedro Luiz Coelho Rodrigues",
                "Violette Gautheret",
                "Marco Congedo",
            ],
            institution="GIPSA-lab, CNRS, University Grenoble-Alpes, Grenoble INP",
            country="FR",
            repository="Zenodo",
            data_url="https://doi.org/10.5281/zenodo.3267301",
            publication_year=2019,
            senior_author="Marco Congedo",
            associated_paper_doi="hal-02173958",
            institution_address="GIPSA-lab, 11 rue des Mathématiques, Grenoble Campus BP46, F-38402, France",
            ethics_approval=[
                "Ethical Committee of the University of Grenoble Alpes (Comité d'Ethique pour la Recherche Non-Interventionnelle)"
            ],
            acknowledgements="At the end of the experiment two tickets of cinema were offered to each subject, for a total value of 15 euros per subject.",
            keywords=[
                "Electroencephalography (EEG)",
                "P300",
                "Brain-Computer Interface (BCI)",
                "Experiment",
                "Collaboration",
                "Multi-User",
                "Hyperscanning",
            ],
            license="CC-BY-4.0",
        ),
        sessions_per_subject=1,
        runs_per_session=1,
        sessions=None,
        data_processed=False,
        file_format="mat and csv",
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw EEG with no digital filter applied, synchronized experimental tags using USB analog-to-digital converter to reduce jitter",
            preprocessing_applied=False,
            notes="Experimental tags produced by Brain Invaders 2 were synchronized with EEG signals using USB analog-to-digital converter connected to g.USBamp trigger channel. This tagging procedure allows consistent tagging latency and jitter.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["RMDM (Riemannian Minimum Distance to Mean)", "Riemannian"],
            feature_extraction=["Covariance/Riemannian"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["cross_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["gaming"],
            environment="small room with 24' screen, subjects sitting side by side at ~125cm distance, experimenter in adjacent room with one-way glass window",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=1,
            n_repetitions=None,
            soa_ms=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials="variable per session (9 levels, up to 8 attempts per level)",
            n_blocks=9,
            block_duration_s="variable, average ~33 seconds per level (5 minutes total for 9 levels)",
            trials_context="9 levels per game session, each with unique predefined spatial configuration of 36 aliens. Up to 8 attempts to destroy target per level.",
        ),
        performance={
            "classifier": "real-time adaptive RMDM classifier (calibration-free procedure)",
        },
        abstract="We describe the experimental procedures for a dataset containing electroencephalographic (EEG) recordings of 38 subjects playing in pairs to the multi-user version of a visual P300-based Brain-Computer Interface (BCI) named Brain Invaders. The interface uses the oddball paradigm on a grid of 36 symbols (1 Target, 35 Non-Target) that are flashed pseudo-randomly to elicit a P300 response. EEG data were recorded using 32 active wet electrodes per subject (total: 64 electrodes) during three randomised conditions (Solo1, Solo2, Collaboration). The experiment took place at GIPSA-lab, Grenoble, France, in 2014.",
        methodology="Multi-user hyperscanning P300 BCI experiment designed to study inter-brain synchrony. Participants played Brain Invaders 2 in three conditions: Solo1 (player1 plays, player2 watches cross), Solo2 (roles reversed), and Collaboration (4 game sessions with both players). Each game session consisted of 9 levels with predefined alien configurations. A repetition used 12 flashes of pseudo-random groups of 6 symbols, ensuring each symbol flashed twice per repetition (1:5 Target:Non-Target ratio). Real-time adaptive RMDM classifier provided online feedback. Control condition (non-playing participant) allowed correction for fake inter-brain synchrony.",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 39)),
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),
            code="BrainInvaders2014b",
            interval=[0, 1],
            paradigm="p300",
            doi="https://doi.org/10.5281/zenodo.3267301",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        return _bi_get_subject_data(self, subject)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return _bi_data_path(self, subject, path, force_update, update_path, verbose)


@depreciated_alias("bi2015a", "1.1")
class BI2015a(BaseDataset):
    """P300 dataset BI2015a from a "Brain Invaders" experiment.

    This dataset contains electroencephalographic (EEG) recordings
    of 43 subjects playing to a visual P300 Brain-Computer Interface (BCI)
    videogame named Brain Invaders. The interface uses the oddball paradigm
    on a grid of 36 symbols (1 Target, 35 Non-Target) that are flashed
    pseudo-randomly to elicit the P300 response. EEG data were recorded using
    32 active wet electrodes with three conditions: flash duration 50ms, 80ms
    or 110ms. The experiment took place at GIPSA-lab, Grenoble, France, in 2015.
    A full description of the experiment is available at [1]_. The ID of this
    dataset is BI2015a.

    :Investigators: Eng. Louis Korczowski, B. Sc. Martine Cederhout
    :Technical Support: Eng. Anton Andreev, Eng. Grégoire Cattan, Eng. Pedro. L. C. Rodrigues,
                        M. Sc. Violette Gautheret
    :Scientific Supervisor: Ph.D. Marco Congedo

    Notes
    -----

    .. versionadded:: 0.4.6

    References
    ----------

    .. [1] Korczowski, L., Cederhout, M., Andreev, A., Cattan, G., Rodrigues, P. L. C.,
           Gautheret, V., & Congedo, M. (2019). Brain Invaders calibration-less P300-based
           BCI with modulation of flash duration Dataset (BI2015a)
           https://hal.archives-ouvertes.fr/hal-02172347
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="10-10",
            hardware="g.USBamp (g.tec, Schiedlberg, Austria)",
            sensor_type="wet electrodes",
            reference="right earlobe",
            ground="Fz",
            software="OpenVibe",
            sensors=[
                "Fp1",
                "Fp2",
                "AFz",
                "F7",
                "F3",
                "F4",
                "F8",
                "FC5",
                "FC1",
                "FC2",
                "FC6",
                "T7",
                "C3",
                "Cz",
                "C4",
                "T8",
                "CP5",
                "CP1",
                "CP2",
                "CP6",
                "P7",
                "P3",
                "Pz",
                "P4",
                "P8",
                "PO7",
                "O1",
                "Oz",
                "O2",
                "PO8",
                "PO9",
                "PO10",
            ],
            line_freq=50.0,
            cap_manufacturer="g.tec",
            cap_model="g.GAMMAcap",
            electrode_type="wet",
            electrode_material="Silver/Silver Chloride",
            filters="no digital filter applied",
        ),
        participants=ParticipantMetadata(
            n_subjects=43,
            gender={"male": 31, "female": 12},
            age_mean=23.70,
            age_std=3.19,
            health_status="healthy",
            bci_experience="mostly students and young researchers",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            events={"Target": 2, "NonTarget": 1},
            paradigm="p300",
            study_design="calibration-less P300-based BCI with modulation of flash duration; three game sessions (9 levels each) with different flash durations (110ms, 80ms, 50ms); resting state and eyes closed recorded before and after sessions; subjects instructed to limit eye blinks, head movements and face muscular contractions",
            feedback_type="visual (game interface with real-time adaptive Riemannian RMDM classifier)",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="online",
            stimulus_type="oddball paradigm on grid of 36 symbols (1 Target, 35 Non-Target) flashed pseudo-randomly",
            n_classes=2,
            class_labels=["Target", "Non-Target"],
            synchronicity="synchronous",
            task_type="target detection",
            instructions="destroy target symbol within 8 attempts; aliens move slowly and regularly according to predefined path to maintain attention",
            has_training_test_split=False,
            stimulus_presentation={
                "SoftwareName": "OpenViBE",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.3266930",
            repository="Zenodo",
            data_url="https://doi.org/10.5281/zenodo.3266930",
            associated_paper_doi="hal-02172347",
            investigators=[
                "Louis Korczowski",
                "Martine Cederhout",
                "Anton Andreev",
                "Grégoire Cattan",
                "Pedro Luiz Coelho Rodrigues",
                "Violette Gautheret",
                "Marco Congedo",
            ],
            senior_author="Marco Congedo",
            institution="GIPSA-lab, CNRS, University Grenoble-Alpes, Grenoble INP",
            institution_address="GIPSA-lab, 11 rue des Mathématiques, Grenoble Campus BP46, F-38402, France",
            country="FR",
            publication_year=2019,
            keywords=[
                "Electroencephalography (EEG)",
                "P300",
                "Brain-Computer Interface",
                "Experiment",
            ],
            ethics_approval=[
                "Ethical Committee of the University of Grenoble Alpes (Comité d'Ethique pour la Recherche Non-Interventionnelle)"
            ],
            how_to_acknowledge="Korczowski, L., Cederhout, M., Andreev, A., Cattan, G., Rodrigues, P.L.C., Gautheret, V., Congedo, M. (2019). Brain Invaders calibration-less P300-based BCI with modulation of flash duration Dataset (bi2015a). Technical Report, GIPSA-lab.",
            license="CC-BY-4.0",
        ),
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw EEG with synchronized USB tagging (reduced jitter using USB digital-to-analog converter)",
            preprocessing_applied=False,
            notes="no digital filter applied during acquisition; tags synchronized with EEG signals to reduce jitter; consistent tagging latency across Brain Invaders databases",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["Riemannian Minimum Distance to Mean (RMDM)", "adaptive"],
            feature_extraction=["Covariance/Riemannian"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["cross_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["gaming"],
            environment="small room (4 square meters) with 24 inch screen",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=1,
            n_repetitions=12,
            soa_ms=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials="variable per subject (up to 8 attempts per level, 9 levels per session, 3 sessions)",
            n_blocks=3,
            trials_context="9 levels per session with variable duration (average ~5 minutes per session, max 10 minutes)",
        ),
        sessions_per_subject=3,
        runs_per_session=1,
        data_processed=False,
        file_format="mat and csv",
        abstract="This dataset contains electroencephalographic (EEG) recordings of 50 subjects playing to a visual P300 Brain-Computer Interface (BCI) videogame named Brain Invaders. The interface uses the oddball paradigm on a grid of 36 symbols (1 Target, 35 Non-Target) that are flashed pseudo-randomly to elicit the P300 response. EEG data were recorded using 32 active wet electrodes with three conditions: flash duration 50ms, 80ms or 110ms. The experiment took place at GIPSA-lab, Grenoble, France, in 2015.",
        methodology="The experiment was designed to study the influence of the flash duration on a calibration-less P300-based BCI system with wet electrodes and as a screening session for potential candidates for a broader multi-user BCI study. The visual P300 is an event-related potential (ERP) elicited by an expected but unpredictable target visual stimulation (oddball paradigm), with peaking amplitude 240-600 ms after stimulus onset. During the experiment, the output of a real-time adaptive Riemannian Minimum Distance to Mean (RMDM) classifier was used for assessing the participants' command. This scheme allows a calibration-free classifier. Before and after the three game sessions, around one minute of resting state and eyes closed conditions were recorded. The interface of Brain Invaders is composed of 36 aliens. In the Brain Invaders P300 paradigm, a repetition is composed of 12 flashes of pseudo-random groups of six symbols chosen in such a way that after each repetition each symbol has flashed exactly two times. A game session was compounded by nine levels, consisting in a unique and predefined configuration of the 36 symbols of the interface. Aliens slowly and regularly moved according to a predefined path keeping constant the inter-distance between adjacent aliens to maintain high player's attention during the whole experiment.",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 44)),
            sessions_per_subject=3,
            events=dict(Target=2, NonTarget=1),
            code="BrainInvaders2015a",
            interval=[0, 1],
            paradigm="p300",
            doi="https://doi.org/10.5281/zenodo.3266929",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        return _bi_get_subject_data(self, subject)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return _bi_data_path(self, subject, path, force_update, update_path, verbose)


@depreciated_alias("bi2015b", "1.1")
class BI2015b(BaseDataset):
    """P300 dataset BI2015b from a "Brain Invaders" experiment.

    This dataset contains electroencephalographic (EEG) recordings
    of 44 subjects playing in pair to the multi-user version of a visual
    P300 Brain-Computer Interface (BCI) named Brain Invaders. The interface
    uses the oddball paradigm on a grid of 36 symbols (1 or 2 Target,
    35 or 34 Non-Target) that are flashed pseudo-randomly to elicit the
    P300 response. EEG data were recorded using 32 active wet electrodes
    per subjects (total: 64 electrodes) during four randomised conditions
    (Cooperation 1-Target, Cooperation 2-Targets, Competition 1-Target,
    Competition 2-Targets). The experiment took place at GIPSA-lab, Grenoble,
    France, in 2015. A full description of the experiment is available at
    A full description of the experiment is available at [1]_. The ID of this
    dataset is BI2015a.

    :Investigators: Eng. Louis Korczowski, B. Sc. Martine Cederhout
    :Technical Support: Eng. Anton Andreev, Eng. Grégoire Cattan, Eng. Pedro. L. C. Rodrigues,
                        M. Sc. Violette Gautheret
    :Scientific Supervisor: Ph.D. Marco Congedo

    Notes
    -----

    .. versionadded:: 0.4.6

    References
    ----------

    .. [1] Korczowski, L., Cederhout, M., Andreev, A., Cattan, G., Rodrigues, P. L. C.,
           Gautheret, V., & Congedo, M. (2019). Brain Invaders Cooperative versus Competitive:
           Multi-User P300-based Brain-Computer Interface Dataset (BI2015b)
           https://hal.archives-ouvertes.fr/hal-02172347
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=32,
            channel_types={"eeg": 32},
            montage="10-10",
            hardware="g.USBamp (g.tec, Schiedlberg, Austria)",
            sensor_type="wet Silver/Silver Chloride electrodes",
            reference="right earlobe",
            ground="Fz",
            software="OpenVibe",
            sensors=[
                "AFz",
                "C3",
                "C4",
                "CP1",
                "CP2",
                "CP5",
                "CP6",
                "Cz",
                "F3",
                "F4",
                "F7",
                "F8",
                "FC1",
                "FC2",
                "FC5",
                "FC6",
                "Fp1",
                "Fp2",
                "O1",
                "O2",
                "Oz",
                "P3",
                "P4",
                "P7",
                "P8",
                "PO10",
                "PO7",
                "PO8",
                "PO9",
                "Pz",
                "T7",
                "T8",
            ],
            line_freq=50.0,
            cap_manufacturer="g.tec",
            cap_model="g.GAMMAcap",
            electrode_type="wet",
            electrode_material="Silver/Silver Chloride",
            filters="no digital filter applied",
        ),
        participants=ParticipantMetadata(
            n_subjects=50,
            health_status="Healthy",
            gender={"male": 36, "female": 14},
            age_mean=23.70,
            age_std=3.19,
            bci_experience="mostly students and young researchers",
            species="human",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            events={"Target": 1, "Non-Target": 2},
            n_classes=2,
            class_labels=["Target", "Non-Target"],
            study_design="Three game sessions with different flash durations (110ms, 80ms, 50ms), with resting state and eyes closed conditions recorded before and after. Subjects were instructed to limit eye blinks, head movements and face muscular contractions.",
            feedback_type="visual (game interface with reward screen)",
            stimulus_type="visual flash",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="online",
            instructions="Players had up to eight attempts to destroy the target symbol per level. Target symbol identification using oddball paradigm with 36 aliens flashing in pseudo-random groups of six symbols.",
            has_training_test_split=False,
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.3266930",
            description="EEG recordings of 50 subjects playing to a visual P300 Brain-Computer Interface (BCI) videogame named Brain Invaders. The interface uses the oddball paradigm on a grid of 36 symbols (1 Target, 35 Non-Target) that are flashed pseudo-randomly to elicit the P300 response. Three conditions: flash duration 50ms, 80ms or 110ms.",
            investigators=[
                "Louis Korczowski",
                "Martine Cederhout",
                "Anton Andreev",
                "Grégoire Cattan",
                "Pedro Luiz Coelho Rodrigues",
                "Violette Gautheret",
                "Marco Congedo",
            ],
            institution="GIPSA-lab, CNRS, University Grenoble-Alpes, Grenoble INP",
            country="France",
            repository="Zenodo",
            data_url="https://doi.org/10.5281/zenodo.3266930",
            publication_year=2019,
            senior_author="Marco Congedo",
            associated_paper_doi="hal-02172347",
            institution_address="GIPSA-lab, 11 rue des Mathématiques, Grenoble Campus BP46, F-38402, France",
            ethics_approval=[
                "Ethical Committee of the University of Grenoble Alpes (Comité d'Ethique pour la Recherche Non-Interventionnelle)"
            ],
            keywords=[
                "Electroencephalography (EEG)",
                "P300",
                "Brain-Computer Interface",
                "Experiment",
            ],
            license="CC-BY-4.0",
        ),
        sessions_per_subject=3,
        runs_per_session=4,
        contributing_labs=["GIPSA-lab"],
        n_contributing_labs=1,
        data_processed=False,
        file_format="mat and csv",
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw EEG with synchronized hardware tagging via USB digital-to-analog converter (reduced jitter compared to software tagging)",
            preprocessing_applied=False,
            notes="Data were stored with no digital filter applied. USB digital-to-analog converter connected to the g.USBamp trigger channel was used to synchronize experimental tags produced by Brain Invaders with EEG signals to reduce jitter.",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=[
                "Riemannian Minimum Distance to Mean (RMDM)",
                "xDAWN",
                "Riemannian MDM",
            ],
            feature_extraction=["Covariance/Riemannian", "xDAWN"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["cross_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["gaming"],
            environment="small room with a surface of four meters square, containing a 24' screen",
            online_feedback=True,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=1,
            n_repetitions=12,
            soa_ms=None,
        ),
        data_structure=DataStructureMetadata(
            n_trials="variable per subject (up to 8 attempts per level, 9 levels per session, 3 sessions)",
            n_blocks=9,
            trials_context="per session (9 levels per session, 3 sessions with different flash durations)",
        ),
        performance={
            "note": "Real-time adaptive classifier used during experiment, performance variable per subject",
        },
        abstract="We describe the experimental procedures for an experiment dataset that we have made publicly available at https://doi.org/10.5281/zenodo.3266930 in mat and csv formats. This dataset contains electroencephalographic (EEG) recordings of 50 subjects playing to a visual P300 Brain-Computer Interface (BCI) videogame named Brain Invaders. The interface uses the oddball paradigm on a grid of 36 symbols (1 Target, 35 Non-Target) that are flashed pseudo-randomly to elicit the P300 response. EEG data were recorded using 32 active wet electrodes with three conditions: flash duration 50ms, 80ms or 110ms. The experiment took place at GIPSA-lab, Grenoble, France, in 2015.",
        methodology="The experiment consisted of three game sessions of Brain Invaders of 9 levels each with different flash duration (110ms, 80ms, 50ms). Before and after the three game sessions, around one minute of resting state and eyes closed conditions were recorded. The interface is composed of 36 aliens. A repetition is composed of 12 flashes of pseudo-random groups of six symbols chosen in such a way that after each repetition each symbol has flashed exactly two times. The ratio of Target versus non-Target is one-to-five. During the experiment, the output of a real-time adaptive Riemannian Minimum Distance to Mean (RMDM) classifier was used for assessing the participants' command. This scheme allows a calibration-free classifier.",
    )

    def __init__(self, subjects=None, sessions=None, *, return_all_modalities=False):
        super().__init__(
            subjects=list(range(1, 45)),
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),
            code="BrainInvaders2015b",
            interval=[0, 1],
            paradigm="p300",
            doi="https://doi.org/10.5281/zenodo.3267307",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        return _bi_get_subject_data(self, subject)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return _bi_data_path(self, subject, path, force_update, update_path, verbose)


@depreciated_alias("VirtualReality", "1.1")
class Cattan2019_VR(BaseDataset):
    """Dataset of an EEG-based BCI experiment in Virtual Reality using P300.

    We describe the experimental procedures for a dataset that we have made publicly
    available at https://doi.org/10.5281/zenodo.2605204 in mat (Mathworks, Natick, USA)
    and csv formats [1]_. This dataset contains electroencephalographic recordings on 21
    subjects doing a visual P300 experiment on non-VR (PC display) and VR (virtual
    reality). The visual P300 is an event-related potential elicited by a visual
    stimulation, peaking 240-600 ms after stimulus onset. The experiment was designed
    in order to compare the use of a P300-based brain-computer interface on a PC and
    with a virtual reality headset, concerning the physiological, subjective and
    performance aspects. The brain-computer interface is based on electroencephalography
    (EEG). EEG data were recorded thanks to 16 electrodes. The virtual reality headset
    consisted of a passive head-mounted display, that is, a head-mounted display which
    does not include any electronics at the exception of a smartphone. A full description
    of the experiment is available at https://hal.archives-ouvertes.fr/hal-02078533.

    See the example `plot_vr_pc_p300_different_epoch_size` to compare the performance
    between PC and VR.

    Parameters
    ----------
    virtual_reality: bool (default False)
        if True, return runs corresponding to P300 experiment on virtual reality.
    screen_display: bool (default True)
        if True, return runs corresponding to P300 experiment on personal computer.

    Notes
    -----
    .. versionadded:: 0.5.0

    References
    ----------
    .. [1] G. Cattan, A. Andreev, P. L. C. Rodrigues, and M. Congedo (2019).
            Dataset of an EEG-based BCI experiment in Virtual Reality and
            on a Personal Computer. Research Report, GIPSA-lab; IHMTEK.
            https://doi.org/10.5281/zenodo.2605204

    .. versionadded:: 0.5.0
    """

    METADATA = DatasetMetadata(
        acquisition=AcquisitionMetadata(
            sampling_rate=512.0,
            n_channels=16,
            channel_types={"eeg": 16},
            montage="10-10",
            hardware="g.USBamp (g.tec, Schiedlberg, Austria)",
            sensor_type="wet electrodes",
            reference="right earlobe",
            ground="AFZ",
            software="OpenVibe",
            sensors=[
                "Fp1",
                "Fp2",
                "Fc5",
                "Fz",
                "Fc6",
                "T7",
                "Cz",
                "T8",
                "P7",
                "P3",
                "Pz",
                "P4",
                "P8",
                "O1",
                "Oz",
                "O2",
            ],
            line_freq=50.0,
            cap_manufacturer="EasyCap",
            cap_model="EC20",
            filters="no digital filter applied",
        ),
        participants=ParticipantMetadata(
            n_subjects=21,
            health_status="healthy",
            gender={"male": 14, "female": 7},
            age_mean=26.38,
            age_std=5.78,
            age_min=19.0,
            age_max=44.0,
            bci_experience="varied gaming experience: some played video games occasionally, some played First Person Shooters; varied VR experience from none to repetitive",
            species="homo sapiens",
        ),
        experiment=ExperimentMetadata(
            paradigm="p300",
            events={
                "Target": 2,
                "NonTarget": 1,
            },
            n_classes=2,
            class_labels=["target", "non_target"],
            trial_duration=None,
            study_design="randomized session order (PC vs VR); limit eye blinks, head movements and face muscular contractions",
            feedback_type="visual",
            stimulus_type="flashing white crosses in 6x6 matrix",
            stimulus_modalities=["visual"],
            primary_modality="visual",
            mode="offline",
            has_training_test_split=False,
            instructions="focus on a red-squared target symbol while groups of six symbols flash",
            stimulus_presentation={
                "description": "6x6 matrix of white crosses; groups of 6 symbols flash; each symbol flashes exactly 2 times per repetition",
                "platform": "Unity engine exported to PC and VR",
            },
        ),
        documentation=DocumentationMetadata(
            doi="10.5281/zenodo.2605204",
            description="EEG recordings of 21 subjects doing a visual P300 experiment on PC and VR to compare BCI performance and user experience",
            investigators=[
                "Grégoire Cattan",
                "Anton Andreev",
                "Pedro Luiz Coelho Rodrigues",
                "Marco Congedo",
            ],
            institution="GIPSA-lab",
            country="FR",
            repository="Zenodo",
            data_url="https://doi.org/10.5281/zenodo.2605204",
            publication_year=2019,
            senior_author="Marco Congedo",
            associated_paper_doi="hal-02078533v3",
            funding=["IHMTEK Company (Interaction Homme-Machine Technologie)"],
            institution_address="GIPSA-lab, 11 rue des Mathématiques, Grenoble Campus BP46, F-38402, France",
            institution_department="GIPSA-lab, CNRS, University Grenoble-Alpes, Grenoble INP",
            ethics_approval=[
                "Ethical Committee of the University of Grenoble Alpes (Comité d'Ethique pour la Recherche Non-Interventionnelle)"
            ],
            acknowledgements="promoted by the IHMTEK Company",
            keywords=[
                "Electroencephalography (EEG)",
                "P300",
                "Brain-Computer Interface (BCI)",
                "Virtual Reality (VR)",
                "experiment",
            ],
            license="CC-BY-4.0",
        ),
        sessions_per_subject=2,
        runs_per_session=60,
        sessions=["PC", "VR"],
        contributing_labs=["GIPSA-lab"],
        n_contributing_labs=1,
        data_processed=False,
        file_format="mat, csv",
        tags=Tags(
            pathology=["Healthy"],
            modality=["Visual"],
            type=["Perception"],
        ),
        preprocessing=PreprocessingMetadata(
            data_state="raw EEG with software tagging via USB (note: tagging introduces jitter and latency - mean 38ms in PC, 117ms in VR)",
            preprocessing_applied=False,
            artifact_methods=None,
            re_reference=None,
            notes="mean tagging latency: ~38 ms in PC, ~117 ms in VR due to different hardware/software setup; these latencies should be used to correct ERPs",
        ),
        signal_processing=SignalProcessingMetadata(
            classifiers=["xDAWN", "Riemannian"],
            feature_extraction=["Covariance/Riemannian", "xDAWN"],
        ),
        cross_validation=CrossValidationMetadata(
            evaluation_type=["cross_session"],
        ),
        bci_application=BCIApplicationMetadata(
            applications=["speller"],
            environment="PC and Virtual Reality (VRElegiant HMD with Huawei Ascend Mate 7 smartphone)",
            online_feedback=False,
        ),
        paradigm_specific=ParadigmSpecificMetadata(
            detected_paradigm="p300",
            n_targets=1,
            n_repetitions=12,
        ),
        data_structure=DataStructureMetadata(
            n_trials={"target": 120, "non_target": 600},
            n_trials_per_class={"target": 120, "non_target": 600},
            n_blocks=12,
            trials_context="per session: 12 blocks × 5 repetitions × 12 flashes per repetition (2 target, 10 non-target)",
        ),
        abstract="Dataset contains electroencephalographic recordings on 21 subjects doing a visual P300 experiment on PC and VR. The visual P300 is an event-related potential elicited by a visual stimulation, peaking 240–600 ms after stimulus onset. The experiment compares P300-based BCI on PC vs VR headset (passive HMD with smartphone) concerning physiological, subjective and performance aspects. EEG recorded with 16 electrodes. Experiment conducted at GIPSA-lab in 2018.",
        methodology="Two randomized sessions (PC and VR). Each session: 12 blocks of 5 repetitions. Each repetition: 12 flashes of groups of 6 symbols, ensuring each symbol flashes exactly 2 times. Target flashes twice per repetition (2 target flashes), non-target flashes 10 times. Random feedback given after each repetition (70% expected accuracy). P300 interface: 6x6 matrix of white flashing crosses with red-squared target. VR used passive HMD (VRElegiant) with Huawei Mate 7 smartphone. IMU deactivated to prevent drift. Unity engine used for identical visual stimulation across PC and VR.",
    )

    def __init__(
        self,
        virtual_reality=True,
        screen_display=True,
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
        **kwargs,
    ):
        deprecated_renames = {
            "VirtualReality": "virtual_reality",
            "ScreenDisplay": "screen_display",
        }
        resolved = _handle_deprecated_kwargs(kwargs, deprecated_renames, "Cattan2019-VR")
        virtual_reality = resolved.get("virtual_reality", virtual_reality)
        screen_display = resolved.get("screen_display", screen_display)

        self.n_repetitions = 5
        super().__init__(
            subjects=list(range(1, 21 + 1)),
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),
            code="Cattan2019-VR",  # before: "VR-P300"
            interval=[0, 1.0],
            paradigm="p300",
            doi="https://doi.org/10.5281/zenodo.2605204",
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )

        self.virtual_reality = virtual_reality
        self.personal_computer = screen_display
        if not self.virtual_reality and not self.personal_computer:
            warn(
                "[Cattan2019-VR dataset] virtual_reality and screen display are False. No data will be downloaded, unless you change these parameters after initialization."
            )

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        return _bi_get_subject_data(self, subject)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return _bi_data_path(self, subject, path, force_update, update_path, verbose)

    def _block_rep(self, block, repetition):
        return block_rep(block, repetition, self.n_repetitions)
