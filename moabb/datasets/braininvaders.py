import glob
import os
import os.path as osp
import shutil
import zipfile as z
from distutils.dir_util import copy_tree

import mne
import numpy as np
import pandas as pd
import yaml
from mne.channels import make_standard_montage
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


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
            "Brain Invaders 2012",
            "Brain Invaders 2014a",
            "Brain Invaders 2014b",
            "Brain Invaders 2015b",
        ]:
            session_name = "session_1"
        elif ds.code == "Brain Invaders 2013a":
            session_number = file_path.split(os.sep)[-2].replace("Session", "")
            session_name = "session_" + session_number
        elif ds.code == "Brain Invaders 2015a":
            session_name = f'session_{file_path.split("_")[-1][1:2]}'
        elif ds.code == "P300-VR":
            session_name = file_path.split(".")[0].split("_")[-1]

        if session_name not in sessions.keys():
            sessions[session_name] = {}

        if ds.code == "Brain Invaders 2012":
            condition = file_path.split("/")[-1].split(".")[0].split(os.sep)[-1]
            run_name = "run_" + condition
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
        elif ds.code == "Brain Invaders 2013a":
            run_number = file_path.split(os.sep)[-1]
            run_number = run_number.split("_")[-1]
            run_number = run_number.split(".mat")[0]
            run_name = "run_" + run_number
            # fmt: off
            chnames = [
                "Fp1", "Fp2", "F5", "AFz", "F6", "T7", "Cz", "T8", "P7",
                "P3", "Pz", "P4", "P8", "O1", "Oz", "O2", "STI 014",
            ]
            # fmt: on
            chtypes = ["eeg"] * 16 + ["stim"]
            X = loadmat(file_path)["data"].T
            sfreq = 512
        elif ds.code == "Brain Invaders 2014a":
            run_name = "run_1"
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
        elif ds.code == "Brain Invaders 2014b":
            # fmt: off
            chnames = [
                'Fp1', 'Fp2', 'AFz', 'F7', 'F3', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
                'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6',
                'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'O1', 'Oz', 'O2', 'PO8', 'PO9',
                'PO10', 'STI 014']
            # fmt: on
            chtypes = ["eeg"] * 32 + ["stim"]
            run_name = "run_1"

            D = loadmat(file_path)["samples"].T
            if subject % 2 == 1:
                S = D[1:33, :] * 1e-6
            else:
                S = D[33:65, :] * 1e-6
            stim = D[-1, :]
            X = np.concatenate([S, stim[None, :]])
            sfreq = 512
        elif ds.code == "Brain Invaders 2015a":
            run_name = "run_1"
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
        elif ds.code == "Brain Invaders 2015b":
            run_name = "run_" + file_path.split("_")[-1].split(".")[0][1]
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
        elif ds.code == "Virtual Reality dataset":
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
            X = np.concatenate([S, stim[:, None]], axis=1).T

            sfreq = 512

        info = mne.create_info(
            ch_names=chnames,
            sfreq=sfreq,
            ch_types=chtypes,
            verbose=False,
        )

        if not ds.code == "Virtual Reality dataset":
            raw = mne.io.RawArray(data=X, info=info, verbose=False)
            raw.set_montage(make_standard_montage("standard_1020"))

            if ds.code == "Brain Invaders 2012":
                # get rid of the Fz channel (it is the ground)
                raw.info["bads"] = ["Fz"]
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
                    sessions[session_name][
                        "block_" + str(bi + 1) + "-repetition_" + str(j + 1)
                    ] = raw

    return sessions


def _bi_data_path(  # noqa: C901
    ds, subject, path=None, force_update=False, update_path=None, verbose=None
):
    if subject not in ds.subject_list:
        raise (ValueError("Invalid subject number"))

    subject_paths = []
    if ds.code == "Brain Invaders 2012":
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

    elif ds.code == "Brain Invaders 2013a":
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
                copy_tree(path_zip.strip(".zip"), directory)
                shutil.rmtree(path_zip.strip(".zip"))

        # filter the data regarding the experimental conditions
        meta_file = directory + os.sep + "meta.yml"
        with open(meta_file, "r") as stream:
            meta = yaml.load(stream, Loader=yaml.FullLoader)
        conditions = []
        if ds.adaptive:
            conditions = conditions + ["adaptive"]
        if ds.nonadaptive:
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

    elif ds.code == "Brain Invaders 2014a":
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

    elif ds.code == "Brain Invaders 2014b":
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

    elif ds.code == "Brain Invaders 2015a":
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
    elif ds.code == "Brain Invaders 2015b":
        # TODO: possible fusion with 2014b?
        url = f"{BI2015b_URL}group_{(subject+1)//2:02}_mat.zip"
        path_zip = dl.data_dl(url, "BRAININVADERS2015B")
        path_folder = path_zip.strip(f"group_{(subject+1)//2:02}_mat.zip")
        # check if has to unzip
        path_folder_subject = f"{path_folder}group_{(subject+1)//2:02}"
        if not (osp.isdir(path_folder_subject)):
            os.mkdir(path_folder_subject)
            zip_ref = z.ZipFile(path_zip, "r")
            zip_ref.extractall(path_folder_subject)

        subject_paths = []
        subject_paths = [
            osp.join(
                path_folder,
                f"group_{(subject+1)//2:02}",
                f"group_{(subject+1)//2:02}_s{i}",
            )
            for i in range(1, 5)
        ]
    elif ds.code == "Virtual Reality dataset":
        subject_paths = []
        url = "{:s}subject_{:02d}_{:s}.mat".format(
            VIRTUALREALITY_URL,
            subject,
            "VR" if ds.virtual_reality else ds.personal_computer,
        )
        file_path = dl.data_path(url, "VIRTUALREALITY")
        subject_paths.append(file_path)

    return subject_paths


class bi2012(BaseDataset):
    """P300 dataset bi2012 from a "Brain Invaders" experiment

    .. admonition:: Dataset summary
        ================ ======= ======= ================ =============== =============== ===========
         Name             #Subj   #Chan   #Trials/class    Trials length   Sampling Rate   #Sessions
        ================ ======= ======= ================ =============== =============== ===========
         bi2013a           25      16     6140 NT / 128 T       1s              512Hz          2
        ================ ======= ======= ================ =============== =============== ===========

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

    def __init__(self, Training=True, Online=False):
        super().__init__(
            subjects=list(range(1, 26)),
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),
            code="Brain Invaders 2012",
            interval=[0, 1],
            paradigm="p300",
            doi="https://doi.org/10.5281/zenodo.2649006",
        )

        self.training = Training
        self.online = Online

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        return _bi_get_subject_data(self, subject)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return _bi_data_path(self, subject, path, force_update, update_path, verbose)


class bi2013a(BaseDataset):
    """P300 dataset bi2013a from a "Brain Invaders" experiment

    .. admonition:: Dataset summary


        =======  =======  =======  =================  ===============  ===============  =================
        Name       #Subj    #Chan  #Trials / class    Trials length    Sampling rate    #Sessions
        =======  =======  =======  =================  ===============  ===============  =================
        bi2013a       24       16  3200 NT / 640 T    1s               512Hz            (1-7)8 s|(8-24)1s
        =======  =======  =======  =================  ===============  ===============  =================

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

    def __init__(self, NonAdaptive=True, Adaptive=False, Training=True, Online=False):
        super().__init__(
            subjects=list(range(1, 25)),
            sessions_per_subject=1,
            events=dict(Target=33285, NonTarget=33286),
            code="Brain Invaders 2013a",
            interval=[0, 1],
            paradigm="p300",
            doi="https://doi.org/10.5281/zenodo.2669187",
        )

        self.adaptive = Adaptive
        self.nonadaptive = NonAdaptive
        self.training = Training
        self.online = Online

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        return _bi_get_subject_data(self, subject)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return _bi_data_path(self, subject, path, force_update, update_path, verbose)


class bi2014a(BaseDataset):
    """P300 dataset bi2014a from a "Brain Invaders" experiment

    .. admonition:: Dataset summary
        ================ ======= ======= ================ =============== =============== ===========
         Name             #Subj   #Chan   #Trials/class    Trials length   Sampling Rate   #Sessions
        ================ ======= ======= ================ =============== =============== ===========
         bi2014a           71      16        5 NT x 1 T         1s              512Hz       up to 3
        ================ ======= ======= ================ =============== =============== ===========

    This dataset contains electroencephalographic (EEG) recordings of 71 subjects
    playing to a visual P300 Brain-Computer Interface (BCI) videogame named Brain Invaders.
    The interface uses the oddball paradigm on a grid of 36 symbols (1 Target, 35 Non-Target)
    that are flashed pseudo-randomly to elicit the P300 response. EEG data were recorded
    using 16 active dry electrodes with up to three game sessions. The experiment took place
    at GIPSA-lab, Grenoble, France, in 2014. A full description of the experiment is available
    at [1]_. The ID of this dataset is bi2014a.

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
           BCI using dry EEG electrodes Dataset (bi2014a).
           https://hal.archives-ouvertes.fr/hal-02171575
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 65)),
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),
            code="Brain Invaders 2014a",
            interval=[0, 1],
            paradigm="p300",
            doi="https://doi.org/10.5281/zenodo.3266222",
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        return _bi_get_subject_data(self, subject)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return _bi_data_path(self, subject, path, force_update, update_path, verbose)


class bi2014b(BaseDataset):
    """P300 dataset bi2014b from a "Brain Invaders" experiment

    .. admonition:: Dataset summary
        ================ ======= ======= ================ =============== =============== ===========
         Name             #Subj   #Chan   #Trials/class    Trials length   Sampling Rate   #Sessions
        ================ ======= ======= ================ =============== =============== ===========
         bi2014b           38      32        5 NT x 1 T         1s              512Hz           3
        ================ ======= ======= ================ =============== =============== ===========

    This dataset contains electroencephalographic (EEG) recordings of 38 subjects playing in
    pair (19 pairs) to the multi-user version of a visual P300-based Brain-Computer Interface (BCI)
    named Brain Invaders. The interface uses the oddball paradigm on a grid of 36 symbols (1 Target,
    35 Non-Target) that are flashed pseudo-randomly to elicit a P300 response, an evoked-potential
    appearing about 300ms after stimulation onset. EEG data were recorded using 32 active wet
    electrodes per subjects (total: 64 electrodes) during three randomized conditions
    (Solo1, Solo2, Collaboration). The experiment took place at GIPSA-lab, Grenoble, France, in 2014.
    A full description of the experiment is available at [1]_. The ID of this dataset is bi2014b.

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
           Multi-User P300-Based Brain-Computer Interface Dataset (bi2014b).
           https://hal.archives-ouvertes.fr/hal-02173958
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 38)),
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),
            code="Brain Invaders 2014b",
            interval=[0, 1],
            paradigm="p300",
            doi="https://doi.org/10.5281/zenodo.3267301",
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        return _bi_get_subject_data(self, subject)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return _bi_data_path(self, subject, path, force_update, update_path, verbose)


class bi2015a(BaseDataset):
    """P300 dataset bi2015a from a "Brain Invaders" experiment

    .. admonition:: Dataset summary
        ================ ======= ======= ================ =============== =============== ===========
         Name             #Subj   #Chan   #Trials/class    Trials length   Sampling Rate   #Sessions
        ================ ======= ======= ================ =============== =============== ===========
         bi2015a           50      32        5 NT x 1 T         1s              512Hz           3
        ================ ======= ======= ================ =============== =============== ===========

    This dataset contains electroencephalographic (EEG) recordings
    of 43 subjects playing to a visual P300 Brain-Computer Interface (BCI)
    videogame named Brain Invaders. The interface uses the oddball paradigm
    on a grid of 36 symbols (1 Target, 35 Non-Target) that are flashed
    pseudo-randomly to elicit the P300 response. EEG data were recorded using
    32 active wet electrodes with three conditions: flash duration 50ms, 80ms
    or 110ms. The experiment took place at GIPSA-lab, Grenoble, France, in 2015.
    A full description of the experiment is available at [1]_. The ID of this
    dataset is bi2015a.

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
           BCI with modulation of flash duration Dataset (bi2015a)
           https://hal.archives-ouvertes.fr/hal-02172347
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 44)),
            sessions_per_subject=3,
            events=dict(Target=2, NonTarget=1),
            code="Brain Invaders 2015a",
            interval=[0, 1],
            paradigm="p300",
            doi="https://doi.org/10.5281/zenodo.3266929",
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        return _bi_get_subject_data(self, subject)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return _bi_data_path(self, subject, path, force_update, update_path, verbose)


class bi2015b(BaseDataset):
    """P300 dataset bi2015b from a "Brain Invaders" experiment

       .. admonition:: Dataset summary
        ================ ======= ======= ================ =============== =============== ===========
         Name             #Subj   #Chan   #Trials/class    Trials length   Sampling Rate   #Sessions
        ================ ======= ======= ================ =============== =============== ===========
         bi2015b           44      32        5 NT x 1 T         1s              512Hz           2
        ================ ======= ======= ================ =============== =============== ===========

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
    dataset is bi2015a.

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
           Multi-User P300-based Brain-Computer Interface Dataset (bi2015b)
           https://hal.archives-ouvertes.fr/hal-02172347
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 45)),
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),
            code="Brain Invaders 2015b",
            interval=[0, 1],
            paradigm="p300",
            doi="https://doi.org/10.5281/zenodo.3267307",
        )

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        return _bi_get_subject_data(self, subject)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return _bi_data_path(self, subject, path, force_update, update_path, verbose)


class VirtualReality(BaseDataset):
    """Dataset of an EEG-based BCI experiment in Virtual Reality using P300

    .. admonition:: Dataset summary
        ================ ======= ======= ================ =============== =============== ===========
         Name             #Subj   #Chan   #Trials/class    Trials length   Sampling Rate   #Sessions
        ================ ======= ======= ================ =============== =============== ===========
         VirtualReality   24      16      600 NT / 120 T   1s              512Hz           2
        ================ ======= ======= ================ =============== =============== ===========


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

    def __init__(self, virtual_reality=False, screen_display=True):
        super().__init__(
            subjects=list(range(1, 20 + 1)),
            sessions_per_subject=1,
            events=dict(Target=2, NonTarget=1),
            code="P300-VR",
            interval=[0, 1.0],
            paradigm="p300",
            doi="https://doi.org/10.5281/zenodo.2605204",
        )

        self.virtual_reality = virtual_reality
        self.personal_computer = screen_display

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
        return _bi_get_subject_data(self, subject)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        return _bi_data_path(self, subject, path, force_update, update_path, verbose)

    def get_block_repetition(self, paradigm, subjects, block_list, repetition_list):
        """Select data for all provided subjects, blocks and repetitions.
        Each subject has 5 blocks of 12 repetitions.

        The returned data is a dictionary with the folowing structure::

            data = {'subject_id' :
                        {'session_id':
                            {'run_id': raw}
                        }
                    }

        See also
        --------
        BaseDataset.get_data

        Parameters
        ----------
        subjects: List of int
            List of subject number
        block_list: List of int
            List of block number (from 1 to 5)
        repetition_list: List of int
            List of repetition number inside a block (from 1 to 12)

        Returns
        -------
        data: Dict
            dict containing the raw data
        """
        X, labels, meta = paradigm.get_data(self, subjects)
        X_select = []
        labels_select = []
        meta_select = []
        for block in block_list:
            for repetition in repetition_list:
                X_select.append(
                    X[
                        meta["run"]
                        == "block_" + str(block) + "-repetition_" + str(repetition)
                    ]
                )
                labels_select.append(
                    labels[
                        meta["run"]
                        == "block_" + str(block) + "-repetition_" + str(repetition)
                    ]
                )
                meta_select.append(
                    meta[
                        meta["run"]
                        == "block_" + str(block) + "-repetition_" + str(repetition)
                    ]
                )
        X_select = np.concatenate(X_select)
        labels_select = np.concatenate(labels_select)
        meta_select = np.concatenate(meta_select)
        df = pd.DataFrame(meta_select, columns=meta.columns)
        meta_select = df

        return X_select, labels_select, meta_select
