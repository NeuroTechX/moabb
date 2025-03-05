"""
A large EEG right-left hand motor imagery dataset.
It is organized into three A, B, C datasets.
URL PATH: https://zenodo.org/record/7554429
"""

import os
import zipfile
from os.path import exists, join

import pandas as pd
from mne.io import read_raw_gdf
from pooch import retrieve

from .base import BaseDataset
from .download import get_dataset_path


# fmt: off
RECORD_INFO = {
    "Demo_Bio": ["SUJ_gender", "Birth_year", "Vision", "Vision_assistance",
                 "Symptoms_TXT", "Level of study", "Level_knowledge neuro",
                 "Meditation practice", "Laterality answered", "Manual activity",
                 "Manual activity TXT"],
    "OpenVibe_Perf": ["Perf_RUN_3", "Perf_RUN_4", "Perf_RUN_5", "Perf_RUN_6"],
    "Mental_Rotation": ["score", "time_1", "time_2"],
    "PRE_session": ["PRE_Mood", "PRE_Mindfulness", "PRE_Motivation",
                    "PRE_Hours_sleep_last_night", "PRE_Usual_sleep",
                    "PRE_Level_of_alertness", "PRE_Stimulant_doses_12h",
                    "PRE_Stimulant_doses_2h", "PRE_Stim_normal", "PRE_Tabacco",
                    "PRE_Tabacco_normal", "PRE_Alcohol", "PRE_Last_meal",
                    "PRE_Last_pills", "PRE_Pills_TXT", "PRE_Nervousness",
                    "PRE_Awakening", "PRE_Concentration"],
    "POST_session": ["POST_Mood", "POST_Mindfulness", "POST_Motivation",
                     "POST_Cognitive load", "POST_Agentivity",
                     "POST_Expectations_filled"],
    "Index_of_Learnig_Style": ["active", "reflexive", "sensory", "intuitive", "visual",
                               "verbal", "sequential", "global"],
    "16Pf5": ["A", "B", "C_", "E", "F", "G", "H", "I", "L", "M", "N", "O", "Q1", "Q2",
              "Q3", "Q4", "IM", "EX", "AX", "TM", "IN", "SC", "Interrogation"],
    "Experimenter_Gender": ['EXP_gender']
}
# fmt: on

DREYER2023_URL = "https://zenodo.org/record/7554429/files/BCI Database.zip"


def dreyer2023_subject_path(basepath, db_id, subject):
    """Returns subject path. If it does not exist, it downloads data first."""
    """
        Arguments:
            basepath [str]: path to the datasets
            db_id [str]: database ID (options A, B, C)
            subject [int]: subject number
        Returns:
            str: path to the subject's data
    """
    subj_path = join(basepath, "BCI Database", "Signals", "DATA {0}", "{0}{1}").format(
        db_id, subject
    )
    if not exists(subj_path):
        if not exists(join(basepath, "data.zip")):
            retrieve(
                DREYER2023_URL, None, fname="data.zip", path=basepath, progressbar=True
            )
            with zipfile.ZipFile(os.path.join(basepath, "data.zip"), "r") as f:
                f.extractall(basepath)
            os.remove(join(basepath, "data.zip"))
    return subj_path


class _Dreyer2023Base(BaseDataset):
    """Class for Dreyer2023 dataset management. MI dataset."""

    """
        Parent class of Dreyer2023A, Dreyer2023B and Dreyer2023C.
        Should not be instantiated.
    """

    def __init__(self, subjects, db_id="A"):
        assert db_id in [
            "A",
            "B",
            "C",
        ], "Invalid dataset selection! Existing Dreyer2023 datasets: A, B, and C."
        self.db_id = db_id
        self.db_idx_off = dict(A=0, B=60, C=81)

        super().__init__(
            subjects,
            sessions_per_subject=1,
            events=dict(left_hand=1, right_hand=2),
            code="Dreyer2023" + self.db_id,
            interval=[3, 8],
            paradigm="imagery",
            doi="10.5281/zenodo.7554429",
        )

    def get_subject_info(self, path=None, subjects=None, infos=None):
        """Loads subject info."""
        """
            Arguments:
                path: path to the dataset
                subjects: list of subjects
                infos: list of recording infos to load
            Returns:
                DataFrame: selected recording info for given subjects
        """
        if isinstance(subjects, type(None)):
            subjects = self.subject_list
        if len([s for s in subjects if s not in self.subject_list]):
            raise ValueError("Invalid subject selection")

        if isinstance(infos, type(None)):
            infos = list(RECORD_INFO.keys())

        path = get_dataset_path("DREYER", path)
        basepath = join(path, "MNE-dreyer-2023")

        perform_path = join(basepath, "BCI Database", "Perfomances.xlsx")

        df = pd.read_excel(perform_path)

        if self.db_id == "A":
            df.columns = df.iloc[1]
            df = df.iloc[list(range(2, 62)), :]
        if self.db_id == "B":
            df.columns = df.iloc[65]
            df = df.iloc[list(range(66, 87)), :]
        if self.db_id == "C":
            df.columns = df.iloc[90]
            df = df.iloc[list(range(91, 97)), :]
        df.reset_index(drop=True, inplace=True)
        df.columns.name = None

        subjects = [
            (
                self.db_id + str(s + self.db_idx_off[self.db_id])
                if not str(s).startswith(self.db_id)
                else str(s)
            )
            for s in subjects
        ]

        assert not any(
            [s for s in subjects if s not in df["SUJ_ID"].tolist()]
        ), "Invalid subject selection."
        df = df.loc[df["SUJ_ID"].isin(subjects)]

        info_select = ["SUJ_ID"]
        for i in infos:
            if i in RECORD_INFO.keys():
                for j in RECORD_INFO[i]:
                    if j in df.columns:
                        info_select.append(j)
            elif i in df.columns:
                info_select.append(i)
            else:
                raise ValueError("Invalid info selection.")

        return df[info_select].reset_index(drop=True)

    def _get_single_subject_data(self, subject):
        subj_dir = self.data_path(subject)

        subj_id = self.db_id + str(subject + self.db_idx_off[self.db_id])
        # fmt: off
        ch_names = ["Fz", "FCz", "Cz", "CPz", "Pz", "C1", "C3", "C5", "C2", "C4", "C6",
                    "EOG1", "EOG2", "EOG3", "EMGg", "EMGd", "F4", "FC2", "FC4", "FC6",
                    "CP2", "CP4", "CP6", "P4", "F3", "FC1", "FC3", "FC5", "CP1", "CP3",
                    "CP5", "P3"]
        # fmt: on
        ch_types = ["eeg"] * 11 + ["eog"] * 3 + ["emg"] * 2 + ["eeg"] * 16
        ch_map = dict(zip(ch_names, ch_types))

        # Closed and open eyes baselines
        baselines = {}
        baselines["ce"] = read_raw_gdf(
            join(subj_dir, subj_id + "_{0}_baseline.gdf").format("CE"),
            eog=["EOG1", "EOG2", "EOG3"],
            misc=["EMGg", "EMGd"],
            verbose="WARNING",
        )
        baselines["ce"].set_channel_types(ch_map)
        baselines["oe"] = read_raw_gdf(
            join(subj_dir, subj_id + "_{0}_baseline.gdf").format("OE"),
            eog=["EOG1", "EOG2", "EOG3"],
            misc=["EMGg", "EMGd"],
            verbose="WARNING",
        )
        baselines["oe"].set_channel_types(ch_map)
        # Recordings
        recordings = {}
        # i - index, n - name, t - type
        for r_i, (r_n, r_t) in enumerate(
            zip(
                ["R1", "R2", "R3", "R4", "R5", "R6"],
                ["acquisition"] * 2 + ["onlineT"] * 4,
            )
        ):
            # One subject of dataset A has 4 recordings
            if r_i > 3 and self.db_id == "A" and subject == 59:
                continue

            recordings["%d" % r_i] = read_raw_gdf(
                join(subj_dir, subj_id + "_{0}_{1}.gdf".format(r_n, r_t)),
                preload=True,
                eog=["EOG1", "EOG2", "EOG3"],
                misc=["EMGg", "EMGd"],
                verbose="WARNING",
            )
            recordings["%d" % r_i].set_channel_types(ch_map)

            recordings["%d" % r_i].annotations.rename(
                {"769": "left_hand", "770": "right_hand"}
            )

        return {"0": recordings}

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        path = get_dataset_path("DREYER", path)
        basepath = join(path, "MNE-dreyer-2023")
        if not os.path.isdir(basepath):
            os.makedirs(basepath)
        return dreyer2023_subject_path(
            basepath, self.db_id, subject + self.db_idx_off[self.db_id]
        )


class Dreyer2023A(_Dreyer2023Base):
    """Class for Dreyer2023A dataset management. MI dataset.

    **Dataset description**

    "A large EEG database with users' profile information for motor imagery
    Brain-Computer Interface research" [1]_ [2]_

    Data collectors : Appriou Aurélien; Caselli Damien; Benaroch Camille;
                      Yamamoto Sayu Maria; Roc Aline; Lotte Fabien;
                      Dreyer Pauline; Pillette Léa
    Data manager    : Dreyer Pauline
    Project leader  : Lotte Fabien
    Project members : Rimbert Sébastien; Monseigne Thibaut

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

    # TO DO:
        * Article [1]_ states there is 29/60 women, in the excel file it is 30/60
        * Sampling frequency? 256 Hz in [1]_, 512 in loaded info and at URL

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
        super().__init__(subjects=list(range(1, 61)), db_id="A")


class Dreyer2023B(_Dreyer2023Base):
    """Class for Dreyer2023B dataset management. MI dataset.

    **Dataset description**

    "A large EEG database with users' profile information for motor imagery
    Brain-Computer Interface research" [1]_ [2]_

    Data collectors : Appriou Aurélien; Caselli Damien; Benaroch Camille;
                      Yamamoto Sayu Maria; Roc Aline; Lotte Fabien;
                      Dreyer Pauline; Pillette Léa
    Data manager    : Dreyer Pauline
    Project leader  : Lotte Fabien
    Project members : Rimbert Sébastien; Monseigne Thibaut

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

    # TO DO:
        * Sampling frequency? 256 Hz in [1]_, 512 in loaded info and at URL
        Mapping based on MDFB as in [2]_
        database_B = ['B' + str(i) for i in range(61, 82)]
        database_A = ['A' + str(i) for i in [43, 44, 6, 10, 52, 23, 48, 24, 40, 43, 2, 1,
                                             13, 22, 25, 29, 3, 11, 30, 19, 21]]
        cross_database_mapping = dict(zip(database_B, database_A))

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
        super().__init__(subjects=list(range(1, 22)), db_id="B")


class Dreyer2023C(_Dreyer2023Base):
    """Class for Dreyer2023C dataset management. MI dataset.

    **Dataset description**

    "A large EEG database with users' profile information for motor imagery
    Brain-Computer Interface research" [1]_ [2]_

    Data collectors : Appriou Aurélien; Caselli Damien; Benaroch Camille;
                      Yamamoto Sayu Maria; Roc Aline; Lotte Fabien;
                      Dreyer Pauline; Pillette Léa
    Data manager    : Dreyer Pauline
    Project leader  : Lotte Fabien
    Project members : Rimbert Sébastien; Monseigne Thibaut

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

    Pre and post experiment questionnaires [1]_:
        - evaluation of pre and post mood, mindfulness and motivational states

    The online OpenViBE BCI classification performance [1]_:
        - only performance measure used to give the feedback to the participants

    # TO DO:
        * Sampling frequency? 256 Hz in [1]_, 512 in loaded info and at URL

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
        super().__init__(subjects=list(range(1, 7)), db_id="C")
