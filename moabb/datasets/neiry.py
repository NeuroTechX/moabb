import zipfile
from pathlib import Path

import h5py
import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray

from . import download as dl
from .base import BaseDataset


class DemonsP300(BaseDataset):
    """Visual P300 dataset recorded in Virtual Reality (VR) game Raccoons versus Demons.

    **Dataset Description**

    We publish dataset of visual P300 BCI performed in Virtual Reality (VR) game Raccoons versus
    Demons (RvD). Data contains reach labels incorporating information about stimulus chosen enabling us
    to estimate model’s confidence at each stimulus prediction stage.
    `target` channel contains standard P300 target/non-target labels,
    while `mult_target` channel contains multiclass labels (numbers of activated stimuli).

    **Participants**

    60 healthy participants (23 males) naive to BCI with mean age 28 years from 19 to 45 y.o. took part in the study.
    All subject signed informed consent and passed primary prerequisites on their health and condition.

    **Stimulation and EEG recording**

    The EEG was recorded with NVX-52 encephalograph (MCS, Zelenograd, Russia) at 500 Hz. We used 8 sponge
    electrodes (Cz, P3, P4, PO3, POz, PO4, O1, O2). Stimuli were presented with HTC Vive Pro VR headset with
    TTL hardware sync

    **Experimental procedure**

    Participants were asked to play the P300 BCI game in virtual reality.
    BCI was embedded into a game plot with the player posing as a forest warden.
    The player was supposed to feed animals and protect them from demons.
    Game mechanics consisted in demons jumping (visually activating),
    so player have to concentrate on one demon (chosen freely). That produced
    P300 response in time of the deamon jump. That was the way to trigger fireball
    torwards a deamon predicted by classifier from EEG data.

    More info can be found in [1]_ [2]_ [3]_.

    References
    ----------

    .. [1] Goncharenko V., Grigoryan R., and Samokhina A. (May 12, 2020),
           Raccoons vs Demons: multiclass labeled P300 dataset,
           https://arxiv.org/abs/2005.02251
    .. [2] Goncharenko V., Grigoryan R., and Samokhina A.,
           Approaches to multiclass classifcation of P300 potential datasets,
           Intelligent Data Processing: Theory and Applications:Book of abstract of
           the 13th International Conference, Moscow, 2020. — Moscow: Russian
           Academy of Sciences, 2020. — 472 p.ISBN 978-5-907366-16-9
           http://www.machinelearning.ru/wiki/images/3/31/Idp20.pdf
    .. [3] Goncharenko V., Grigoryan R., and Samokhina A.,
           P300 potentials dataset and approaches to its processing,
           Труды 63-й Всероссийской научной конференции МФТИ. 23–29 ноября 2020
           года. Прикладные математика и информатика. —  Москва : МФТИ, 2020. – 334 с.
           ISBN 978-5-7417-0757-9
           https://mipt.ru/science/5top100/education/courseproposal/%D0%A4%D0%9F%D0%9C%D0%98%20%D1%84%D0%B8%D0%BD%D0%B0%D0%BB-compressed2.pdf
    """

    ch_names = ["Cz", "P3", "Pz", "P4", "PO3", "PO4", "O1", "O2"]
    sampling_rate = 500.0
    url = "https://gin.g-node.org/v-goncharenko/neiry-demons/raw/master/nery_demons_dataset.zip"

    _ms_in_sec = 1000
    _hdf_path = "p300dataset"
    _ds_folder_name = "demons"

    _act_dtype = np.dtype(
        [
            ("id", np.int),
            ("target", np.int),
            ("is_train", np.bool),
            ("prediction", np.int),
            ("sessions", np.object),  # list of `_session_dtype`
        ]
    )
    _session_dtype = np.dtype(
        [
            ("eeg", np.object),
            ("starts", np.object),
            ("stimuli", np.object),
        ]
    )

    def __init__(self):
        super().__init__(
            subjects=list(range(60)),
            sessions_per_subject=1,
            events={"Target": 1, "NonTarget": 2},
            code="Demons P300",
            interval=[0, 1],
            paradigm="p300",
        )
        self.path = None
        self.subjects_filenames = None

    @staticmethod
    def _strip(session) -> tuple:
        """Strips nans (from right side of all channels) added during hdf5 packaging

        Returns:
            tuple ready to be converted to `_session_dtype`
        """
        eeg, *rest = session
        ind = -next(i for i, value in enumerate(eeg[0, ::-1]) if not np.isnan(value))
        if ind == 0:
            ind = None
        return tuple((eeg[:, :ind], *rest))

    @classmethod
    def read_hdf(cls, filename) -> np.ndarray:
        """Reads data from HDF file

        Returns:
            array of `_act_dtype`
        """
        with h5py.File(filename, "r") as hfile:
            group = hfile[cls._hdf_path]
            record = np.empty(len(group), cls._act_dtype)
            for i, act in enumerate(group.values()):
                record[i]["sessions"] = np.array(
                    [cls._strip(item) for item in act], cls._session_dtype
                )
                for name, value in act.attrs.items():
                    record[i][name] = value
        return record

    def _get_single_subject_data(self, subject: int):
        record = self.read_hdf(self.data_path(subject))

        info = create_info(
            self.ch_names + ["mult_target", "target"],
            self.sampling_rate,
            ["eeg"] * len(self.ch_names) + ["misc", "stim"],
        )
        montage = make_standard_montage("standard_1020")

        runs_raw = {}
        for i, act in enumerate(record):
            # target and stims are increased by 1
            # because the channel is filled with zeros by default
            target = act["target"] + 1
            run_data = []
            for eeg, starts, stims in act["sessions"]:
                starts = starts * self.sampling_rate / self._ms_in_sec
                starts = starts.round().astype(np.int)
                stims = stims + 1
                stims_channel = np.zeros(eeg.shape[1])
                target_channel = np.zeros(eeg.shape[1])

                for start, stimul in zip(starts, stims):
                    stims_channel[start] = stimul
                    target_channel[start] = 1 if stimul == target else 2

                round_data = np.vstack(
                    (eeg, stims_channel[None, :], target_channel[None, :])
                )
                run_data.append(round_data)

            raw = RawArray(np.hstack(run_data), info)
            raw.set_montage(montage)
            runs_raw[f"run_{i}"] = raw
        return {"session_0": runs_raw}

    def data_path(
        self, subject: int, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        zip_path = Path(dl.data_dl(self.url, self._ds_folder_name))
        self.path = zip_path.parent / self._ds_folder_name / zip_path.stem

        if not self.path.exists():
            with zipfile.ZipFile(zip_path) as zip_file:
                zip_file.extractall(self.path.parent)

        self.subjects_filenames = sorted(self.path.glob("*.hdf5"))

        return self.subjects_filenames[subject].as_posix()
