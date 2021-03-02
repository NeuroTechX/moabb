import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray

from . import download as dl
from .base import BaseDataset


class DemonsP300(BaseDataset):
    '''The dataset of visual P300 BCI performed in Virtual Reality (VR) game Raccoons versus Demons.
    **Abstract**
    We publish dataset of visual P300 BCI performed in Virtual Reality (VR) game Raccoons versus
    Demons (RvD). Data contains reach labels incorporating information about stimulus chosen enabling us
    to estimate model’s confidence at each stimulus prediction stage.
    Data and experiments code are available at [3]_.
    raccoons-vs-demons.
    **Participants**
    61 healthy participants (23 males) naive to BCI with mean age 28 years from 19 to 45 y.o. took part in the study.
    All subject signed informed consent and passed primary prerequisites on their health and condition.
    **Stimulation and EEG recording**
    The EEG was recorded with NVX-52 encephalograph (MCS, Zelenograd, Russia) at 500 Hz. We used 8 sponge
    electrodes (Cz, P3, P4, PO3, POz, PO4, O1, O2). Stimuli were presented with HTC Vive Pro VR headset with
    TTL hardware sync
    **Experimental procedure**
    Participants were asked to play the P300 BCI game in virtual reality. BCI was embedded into a game plot with the
    player posing as a forest warden. The player was supposed to feed animals and protect them from demons.
    More info is here [1]_ [2]_.
    References
    ----------
    .. [1] Goncharenko V., Grigoryan R., and Samokhina A. (May 12, 2020),
        Raccoons vs Demons: multiclass labeled P300 dataset
        https://arxiv.org/abs/2005.02251
    .. [2] Impulse Neiry website: https://impulse-neiry.com/
    .. [3] Impulse Neiry repository https://gitlab.com/impulse-neiry_public/
    '''

    ch_names = ['Cz', 'P3', 'Pz', 'P4', 'PO3', 'PO4', 'O1', 'O2']
    sampling_rate = 500.0
    url = f'FIXME'

    hdf_path = 'p300dataset'
    ds_folder_name = 'demons'

    act_dtype = np.dtype([
        ('id', np.int),
        ('target', np.int),
        ('is_train', np.bool),
        ('prediction', np.int),
        ('sessions', np.object),  # list of `session_dtype`
    ])

    session_dtype = np.dtype([
        ('eeg', np.object),
        ('starts', np.object),
        ('stimuli', np.object),
    ])


    def __init__(self):
        super().__init__(
            subjects=range(60),
            sessions_per_subject=17,
            events={'Target': 1, 'NonTarget': 2},
            code='Demons P300',
            interval=[0, 1],
            paradigm='p300',
        )
        self.path = None
        self.subjects_filenames = None

    @staticmethod
    def _strip(session) -> tuple:
        '''Strips nans (from right side of all channels) added during hdf5 packaging
        Returns:
            tuple ready to be converted to `session_dtype`
        '''
        eeg, *rest = session
        ind = -next(i for i, value in enumerate(eeg[0, ::-1]) if not np.isnan(value))
        if ind == 0:
            ind = None
        return tuple((eeg[:, :ind], *rest))

    @classmethod
    def read_hdf(cls, filename) -> np.ndarray:
        '''Reads data from HDF file
        Returns:
            array of act_dtype
        '''
        with h5py.File(filename, 'r') as hfile:
            group = hfile[cls.hdf_path]
            record = np.empty(len(group), cls.act_dtype)
            for i, act in enumerate(group.values()):
                record[i]['sessions'] = np.array(
                    [cls._strip(item) for item in act], cls.session_dtype
                )
                for name, value in act.attrs.items():
                    record[i][name] = value
        return record

    def convert_num_to_ms(self, starts):
        '''Convertation for RawArray
        For example lets convert record=600 to a position in a channel.
        Frequency of the EEG is 500Hz, so there are 500 records per second.
        Start records are in ms, so 600ms = 600/1000 = 0.6 sec
        To get position of current record
            multiply 0.6 sec with frequency, 0.6sec x 500Hz = 300
        '''
        return starts * self.sampling_rate / 1000

    def _get_single_subject_data(self, subject: int):
        record = self.read_hdf(self.data_path(subject)[0])

        info = create_info(
            self.ch_names + ['stim', 'target'],
            self.sampling_rate,
            ['eeg'] * len(self.ch_names) + ['stim', 'stim'],
        )
        montage = make_standard_montage('standard_1020')

        data = {}
        for i, act in enumerate(record):
            # target and stims are increased by 1
            # because the channel is filled with zeros by default
            target = act['target'] + 1
            runs = {}
            for j, run in enumerate(act['sessions']):
                eeg_data = run[0]
                starts = run[1]
                stims = run[2] + 1

                eeg_len = len(eeg_data[0])
                stims_channel = np.zeros(eeg_len)
                target_channel = np.zeros(eeg_len)

                starts = self.convert_num_to_ms(starts)

                for start, stimul in zip(starts.astype(int), stims):
                    stims_channel[start] = stimul
                    target_channel[start] = 1 if stimul == target else 2

                eeg_data = np.vstack(
                    (eeg_data, stims_channel[None, :], target_channel[None, :])
                )
                raw = RawArray(eeg_data, info)
                raw.set_montage(montage)
                runs[f'run_{j}'] = raw

            data[f'session_{i}'] = runs

        return data

    def data_path(
        self, subject: int, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError('Invalid subject number'))

        if self.subjects_filenames is None:
            zip_path = dl.data_path(self.url, self.ds_folder_name)
            zip_path = Path(zip_path)
            self.path = zip_path.with_name(self.ds_folder_name).resolve()

            if not self.path.exists():
                with zipfile.ZipFile(zip_path) as zip_file:
                    zip_file.extractall(self.path)

            self.subjects_filenames = sorted(self.path.glob('*.hdf5'))

        return [self.subjects_filenames[subject].as_posix()]
