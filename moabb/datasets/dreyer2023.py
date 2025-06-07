"""
A large EEG right-left hand motor imagery dataset.
It is organized into three A, B, C datasets.
URL PATH: https://zenodo.org/record/7554429
"""

import warnings
import zipfile
from pathlib import Path

import pandas as pd
from mne_bids import BIDSPath, get_entity_vals, read_raw_bids
from tqdm import tqdm

from moabb.datasets import download as dl

from .base import BaseDataset


_manifest_link = "https://osf.io/download/67c9abecc1b99765d8bb36b0/"
_metainfo_link = "https://osf.io/download/67c9e8234f014fc76e0411ba/"

_osf_tag = "8tdk5"
_api_base_url = f"https://files.de-1.osf.io/v1/resources/{_osf_tag}/providers/osfstorage/"


class _Dreyer2023Base(BaseDataset):
    """
    Parent class of Dreyer2023A, Dreyer2023B and Dreyer2023C.
    Should not be instantiated.
    """

    def __init__(self, subjects, sub_id=""):

        self.sub_id = sub_id

        if sub_id is None:
            self.sub_id = ""

        self.subject_list = subjects

        super().__init__(
            self.subject_list,
            sessions_per_subject=1,
            events=dict(left_hand=1, right_hand=2),
            code="Dreyer2023" + self.sub_id,
            interval=[3, 8],
            paradigm="imagery",
            doi="10.5281/zenodo.7554429",
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.

        Returns
        -------
        dict
            A dictionary containing the raw data for the subject.
        """
        # Get the file path for the subject's data
        files_path = self.data_path(subject)
        runs = {}
        for run_id, file in enumerate(files_path):

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Read the subject's raw data and set the montage
                raw = read_raw_bids(bids_path=file, verbose=False)
                raw = raw.load_data()
                # We are losting several annotations because there is no fuck
                # place explaining what it is the events ids :)
                raw.annotations.rename({"769": "left_hand", "770": "right_hand"})
                runs.update({f"{run_id}{file.task}": raw})

        sessions = {"0": runs}

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """
        Return the data BIDS paths of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
            If it doesn’t exist, the “~/mne_data” directory is used. If the
            dataset is not found under the given path, the data
            will be automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python config
            to the given path.
            If None, the user is prompted.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose()).

        Returns
        -------
        list
            A list containing the BIDSPath object for the subject's data file.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        # Download and extract the dataset
        dataset_path = self.download_by_subject(subject=subject, path=path)

        tasks = get_entity_vals(dataset_path, "task")

        bids_path_list = []
        for task in tasks:
            if "baseline" in task or "rest" in task:
                continue

            if subject == 59 and ("R5online" in task) or ("R6online" in task):
                continue

            # Create a BIDSPath object for all the tasks
            bids_path = BIDSPath(
                subject=f"{subject:02d}",
                suffix="eeg",
                task=task,
                root=dataset_path,
                check=True,
            )
            bids_path_list.append(bids_path)

        return bids_path_list

    def download_by_subject(self, subject, path=None):
        """
        Download and extract the dataset.

        Parameters
        ----------
        subject : int
            The subject number to download the dataset for.

        path : str | None
            The path to the directory where the dataset should be downloaded.
            If None, the default directory is used.


        Returns
        -------
        path : str
            The dataset path.
        """
        path = Path(dl.get_dataset_path(self.code, path)) / (f"MNE-{self.code}-data")

        # checking it there is manifest file in the dataset folder.
        dl.download_if_missing(path / "dreyer2023_manifest.tsv", _manifest_link)

        manifest = pd.read_csv(path / "dreyer2023_manifest.tsv", sep="\t")

        subject_index = manifest["filename"] == f"sub-{subject:02d}.zip"

        dataset_index = ~manifest["filename"].str.contains("sub")

        manifest_subject = manifest[subject_index | dataset_index]

        manifest_subject = manifest_subject.copy()

        for _, row in tqdm(manifest_subject.iterrows()):
            download_url = _api_base_url + row["url"].replace(
                "https://osf.io/download/", ""
            ).replace("/", "")
            dl.download_if_missing(
                path / row["filename"], download_url, warn_missing=False
            )

        for _, row in manifest_subject.iterrows():
            if row["filename"].endswith(".zip"):
                if not (path / row["filename"].replace(".zip", "")).exists():
                    with zipfile.ZipFile(path / row["filename"], "r") as zip_ref:
                        zip_ref.extractall(path)

        return path

    def get_subject_info(self, path=None):
        """
        Return the demographic information of the subjects.

        Returns
        -------
        DataFrame
            A DataFrame containing the demographic information of the subjects.
        """
        path = Path(dl.get_dataset_path(self.code, path)) / (f"MNE-{self.code}-data")

        # checking it there is manifest file in the dataset folder.
        dl.download_if_missing(path / "performance.csv", _metainfo_link)

        metainfo = pd.read_csv(path / "performance.csv", sep=";")

        if self.sub_id == "":
            return metainfo
        else:
            return metainfo[metainfo["SUBDATASET"] == self.sub_id].reset_index(drop=True)


class Dreyer2023A(_Dreyer2023Base):
    """Class for Dreyer2023A dataset management. MI dataset.

    **Dataset description**

    "A large EEG database with users' profile information for motor imagery
    Brain-Computer Interface research" [1]_ [2]_

    :Data collectors: Appriou Aurélien; Caselli Damien; Benaroch Camille;
                      Yamamoto Sayu Maria; Roc Aline; Lotte Fabien;
                      Dreyer Pauline; Pillette Léa
    :Data manager: Dreyer Pauline
    :Project leader: Lotte Fabien
    :Project members: Rimbert Sébastien; Monseigne Thibaut

    Dataset Dreyer2023A contains EEG, EOG and EMG signals recorded on 60 healthy subjects
    performing Left-Right Motor Imagery experiments
    (29 women, age 19-59, M = 29, SD = 9.32) [1]_.
    Experiments were conducted by six experimenters. In addition, for each recording
    the following pieces of information are provided:
    subject's demographic, personality and cognitive profiles, the OpenViBE experimental
    instructions and codes, and experimenter's gender.

    The experiment is designed for the investigation of the impact of the participant's
    and experimenter's gender on MI BCI performance [1]_.

    A recording contains open and closed eyes baseline recordings and 6 runs of the MI
    experiments. First 2 runs (acquisition runs) were used to train system and
    the following 4 runs (training runs) to train the participant. Each run contained
    40 trials [1]_.

    Each trial was recorded as follows [1]_:
        - t=0.00s  cross displayed on screen
        - t=2.00s  acoustic signal announced appearance of a red arrow
        - t=3.00s  a red arrow appears (subject starts to perform task)
        - t=4.25s  the red arrow disappears
        - t=4.25s  the feedback on performance is given in form of a blue bar
          with update frequency of 16 Hz
        - t=8.00s  cross turns off (subject stops to perform task)

    EEG signals [1]_:
        - recorded with 27 electrodes, namely:
          Fz, FCz, Cz, CPz, Pz, C1, C3, C5, C2, C4, C6, F4, FC2, FC4, FC6, CP2,
          CP4, CP6, P4, F3, FC1, FC3, FC5, CP1, CP3, CP5, P3 (10-20 system),
          referenced to the left earlobe.

    EOG signals [1]_:
        - recorded with 3 electrodes, namely: EOG1, EOG2, EOG3
          placed below, above and on the side of one eye.

    EMG signals [1]_:
        - recorded with 2 electrodes, namely: EMGg, EMGd
          placed 2.5cm below the skinfold on each wrist.

    Demographic and biosocial information includes:
        - gender, birth year, laterality
        - vision, vision assistance
        - familiarity to cognitive science or neurology, level of education
        - physical activity, meditation
        - attentional, neurological, psychiatrics symptoms

    Personality and the cognitive profile [1]_:
        - evaluated via 5th edition of the 16 Personality Factors (16PF5) test
        - and mental rotation test
        - index of learning style

    Pre and post experiment questionnaires [1]_:
        - evaluation of pre and post mood, mindfulness and motivational states

    The online OpenViBE BCI classification performance [1]_:
        - only performance measure used to give the feedback to the participants

    * Subject 59 contains only 4 runs

    References
    ----------

    .. [1] Pillette, L., Roc, A., N’kaoua, B., & Lotte, F. (2021).
        Experimenters' influence on mental-imagery based brain-computer interface user training.
        International Journal of Human-Computer Studies, 149, 102603.
    .. [2] Benaroch, C., Yamamoto, M. S., Roc, A., Dreyer, P., Jeunet, C., & Lotte, F. (2022).
        When should MI-BCI feature optimization include prior knowledge, and which one?.
        Brain-Computer Interfaces, 9(2), 115-128.
    """

    def __init__(self):
        super().__init__(subjects=list(range(1, 61)), sub_id="A")


class Dreyer2023B(_Dreyer2023Base):
    """Class for Dreyer2023B dataset management. MI dataset.

    **Dataset description**

    "A large EEG database with users' profile information for motor imagery
    Brain-Computer Interface research" [1]_ [2]_

    :Data collectors: Appriou Aurélien; Caselli Damien; Benaroch Camille;
                      Yamamoto Sayu Maria; Roc Aline; Lotte Fabien;
                      Dreyer Pauline; Pillette Léa
    :Data manager: Dreyer Pauline
    :Project leader: Lotte Fabien
    :Project members: Rimbert Sébastien; Monseigne Thibaut

    Dataset Dreyer2023B contains EEG, EOG and EMG signals recorded on 21 healthy subjects
    performing Left-Right Motor Imagery experiments
    (8 women, age 19-37, M = 29, SD = 9.318) [2]_.
    Experiments were conducted by female experimenters. In addition, for each recording
    the following pieces of information are provided:
    subject's demographic, personality and cognitive profiles, the OpenViBE experimental
    instructions and codes, and experimenter's gender.

    The experiment is designed for the investigation of the relation between MI-BCI online
    performance and Most Discriminant Frequency Band (MDFB) [2]_.

    A recording contains open and closed eyes baseline recordings and 6 runs of the MI
    experiments. First 2 runs (acquisition runs) were used to train system and
    the following 4 runs (training runs) to train the participant. Each run contained
    40 trials [1]_.

    Each trial was recorded as follows [1]_:
        - t=0.00s  cross displayed on screen
        - t=2.00s  acoustic signal announced appearance of a red arrow
        - t=3.00s  a red arrow appears (subject starts to perform task)
        - t=4.25s  the red arrow disappears
        - t=4.25s  the feedback on performance is given in form of a blue bar
          with update frequency of 16 Hz
        - t=8.00s  cross turns off (subject stops to perform task)

    EEG signals [1]_:
        - recorded with 27 electrodes, namely:
          Fz, FCz, Cz, CPz, Pz, C1, C3, C5, C2, C4, C6, F4, FC2, FC4, FC6, CP2,
          CP4, CP6, P4, F3, FC1, FC3, FC5, CP1, CP3, CP5, P3 (10-20 system),
          referenced to the left earlobe.

    EOG signals [1]_:
        - recorded with 3 electrodes, namely: EOG1, EOG2, EOG3
          placed below, above and on the side of one eye.

    EMG signals [1]_:
        - recorded with 2 electrodes, namely: EMGg, EMGd
          placed 2.5cm below the skinfold on each wrist.

    Demographic and biosocial information includes:
        - gender, birth year, laterality
        - vision, vision assistance
        - familiarity to cognitive science or neurology, level of education
        - physical activity, meditation
        - attentional, neurological, psychiatrics symptoms

    Personality and the cognitive profile [1]_:
        - evaluated via 5th edition of the 16 Personality Factors (16PF5) test
        - and mental rotation test
        - index of learning style

    Pre and post experiment questionnaires [1]_:
        - evaluation of pre and post mood, mindfulness and motivational states

    The online OpenViBE BCI classification performance [1]_:
        - only performance measure used to give the feedback to the participants


    References
    ----------

    .. [1] Pillette, L., Roc, A., N’kaoua, B., & Lotte, F. (2021).
        Experimenters' influence on mental-imagery based brain-computer interface user training.
        International Journal of Human-Computer Studies, 149, 102603.
    .. [2] Benaroch, C., Yamamoto, M. S., Roc, A., Dreyer, P., Jeunet, C., & Lotte, F. (2022).
        When should MI-BCI feature optimization include prior knowledge, and which one?.
        Brain-Computer Interfaces, 9(2), 115-128.
    """

    def __init__(self):
        super().__init__(subjects=list(range(61, 82)), sub_id="B")


class Dreyer2023C(_Dreyer2023Base):
    """Class for Dreyer2023C dataset management. MI dataset.

    **Dataset description**

    "A large EEG database with users' profile information for motor imagery
    Brain-Computer Interface research" [1]_ [2]_.

    :Data collectors: Appriou Aurélien; Caselli Damien; Benaroch Camille;
                      Yamamoto Sayu Maria; Roc Aline; Lotte Fabien;
                      Dreyer Pauline; Pillette Léa
    :Data manager: Dreyer Pauline
    :Project leader: Lotte Fabien
    :Project member: Rimbert Sébastien; Monseigne Thibaut

    Dataset Dreyer2023C contains EEG, EOG and EMG signals recorded on 6 healthy subjects
    performing Left-Right Motor Imagery experiments (4 women) who participated in datasets
    A or B.
    In addition, for each recording the following pieces of information are provided:
    subject's demographic, personality and cognitive profiles, the OpenViBE experimental
    instructions and codes, and experimenter's gender.

    A recording contains open and closed eyes baseline recordings and 6 runs of the MI
    experiments. First 2 runs (acquisition runs) were used to train system and
    the following 4 runs (training runs) to train the participant. Each run contained
    40 trials [1]_.

    Each trial was recorded as follows [1]_:
        - t=0.00s  cross displayed on screen
        - t=2.00s  acoustic signal announced appearance of a red arrow
        - t=3.00s  a red arrow appears (subject starts to perform task)
        - t=4.25s  the red arrow disappears
        - t=4.25s  the feedback on performance is given in form of a blue bar
          with update frequency of 16 Hz
        - t=8.00s  cross turns off (subject stops to perform task)

    EEG signals [1]_:
        - recorded with 27 electrodes, namely:
          Fz, FCz, Cz, CPz, Pz, C1, C3, C5, C2, C4, C6, F4, FC2, FC4, FC6, CP2,
          CP4, CP6, P4, F3, FC1, FC3, FC5, CP1, CP3, CP5, P3 (10-20 system),
          referenced to the left earlobe.

    EOG signals [1]_:
        - recorded with 3 electrodes, namely: EOG1, EOG2, EOG3
          placed below, above and on the side of one eye.

    EMG signals [1]_:
        - recorded with 2 electrodes, namely: EMGg, EMGd
          placed 2.5cm below the skinfold on each wrist.

    Demographic and biosocial information includes:
        - gender, birth year, laterality
        - vision, vision assistance
        - familiarity to cognitive science or neurology, level of education
        - physical activity, meditation
        - attentional, neurological, psychiatrics symptoms

    Personality and the cognitive profile [1]_:
        - evaluated via 5th edition of the 16 Personality Factors (16PF5) test
        - and mental rotation test
        - index of learning style

    Pre and post-experiment questionnaires [1]_:
        - evaluation of pre and post mood, mindfulness and motivational states

    The online OpenViBE BCI classification performance [1]_:
        - only performance measure used to give the feedback to the participants


    References
    ----------

    .. [1] Pillette, L., Roc, A., N’kaoua, B., & Lotte, F. (2021).
        Experimenters' influence on mental-imagery based brain-computer interface user training.
        International Journal of Human-Computer Studies, 149, 102603.
    .. [2] Benaroch, C., Yamamoto, M. S., Roc, A., Dreyer, P., Jeunet, C., & Lotte, F. (2022).
        When should MI-BCI feature optimization include prior knowledge, and which one?.
        Brain-Computer Interfaces, 9(2), 115-128.
    """

    def __init__(self):
        super().__init__(subjects=list(range(82, 88)), sub_id="C")


class Dreyer2023(_Dreyer2023Base):
    """Class for Dreyer2023 dataset management. MI dataset.

    **Dataset description**

    "A large EEG database with users' profile information for motor imagery
    Brain-Computer Interface research" [1]_ [2]_

    :Data Collectors: Appriou Aurélien; Caselli Damien; Benaroch Camille;
                      Yamamoto Sayu Maria; Roc Aline; Lotte Fabien;
                      Dreyer Pauline; Pillette Léa
    :Data Manager: Dreyer Pauline
    :Project leader: Lotte Fabien
    :Project members: Rimbert Sébastien; Monseigne Thibaut

    Dataset Dreyer2023 contains concatenated datasets Dreyer2023A, Dreyer2023B and Dreyer2023C.

    Experiments were conducted by six experimenters. In addition, for each recording
    the following pieces of information are provided:
    subject's demographic, personality and cognitive profiles, the OpenViBE experimental
    instructions and codes, and experimenter's gender.

    The experiment is designed for the investigation of the impact of the participant's
    and experimenter's gender on MI BCI performance [1]_.

    A recording contains open and closed eyes baseline recordings and 6 runs of the MI
    experiments. First 2 runs (acquisition runs) were used to train system and
    the following 4 runs (training runs) to train the participant. Each run contained
    40 trials [1]_.

    Each trial was recorded as follows [1]_:
        - t=0.00s  cross displayed on screen
        - t=2.00s  acoustic signal announced appearance of a red arrow
        - t=3.00s  a red arrow appears (subject starts to perform task)
        - t=4.25s  the red arrow disappears
        - t=4.25s  the feedback on performance is given in the form of a blue bar
          with update frequency of 14 Hz
        - t=8.00s  cross turns off (subject stops to perform task)

    EEG signals [1]_:
        - recorded with 27 electrodes, namely:
          Fz, FCz, Cz, CPz, Pz, C1, C3, C5, C2, C4, C6, F4, FC2, FC4, FC6, CP2,
          CP4, CP6, P4, F3, FC1, FC3, FC5, CP1, CP3, CP5, P3 (10-20 system),
          referenced to the left earlobe.

    EOG signals [1]_:
        - recorded with 3 electrodes, namely: EOG1, EOG2, EOG3
          placed below, above and on the side of one eye.

    EMG signals [1]_:
        - recorded with 2 electrodes, namely: EMGg, EMGd
          placed 2.5cm below the skinfold on each wrist.

    Demographic and biosocial information includes:
        - gender, birth year, laterality
        - vision, vision assistance
        - familiarity to cognitive science or neurology, level of education
        - physical activity, meditation
        - attentional, neurological, psychiatrics symptoms

    Personality and the cognitive profile [1]_:
        - evaluated via 5th edition of the 16 Personality Factors (16PF5) test
        - and mental rotation test
        - index of learning style

    Pre and post experiment questionnaires [1]_:
        - evaluation of pre and post mood, mindfulness and motivational states

    The online OpenViBE BCI classification performance [1]_:
        - only performance measure used to give the feedback to the participants


    References
    ----------

    .. [1] Pillette, L., Roc, A., N’kaoua, B., & Lotte, F. (2021).
        Experimenters' influence on mental-imagery based brain-computer interface user training.
        International Journal of Human-Computer Studies, 149, 102603.
    .. [2] Benaroch, C., Yamamoto, M. S., Roc, A., Dreyer, P., Jeunet, C., & Lotte, F. (2022).
        When should MI-BCI feature optimization include prior knowledge, and which one?.
        Brain-Computer Interfaces, 9(2), 115-128.
    """

    def __init__(self):
        super().__init__(subjects=list(range(1, 88)))
