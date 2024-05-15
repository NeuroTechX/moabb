import logging
import os

import mne
import numpy as np
import pooch
from scipy.io import loadmat

import moabb.datasets.download as dl

from .base import BaseDataset


log = logging.getLogger(__name__)
BASE_URL = "https://ndownloader.figshare.com/files/"


class Stieger2021(BaseDataset):

    def __init__(self, interval=[0, 3], sessions=None):
        super().__init__(
            subjects=list(range(1, 63)),
            sessions_per_subject=11,
            events=dict(right_hand=1, left_hand=2, both_hand=3, rest=4),
            code="Stieger2021",
            interval=interval,
            paradigm="imagery",
            doi="10.1038/s41597-021-00883-1",
        )

        self.sessions = sessions
        self.figshare_id = 13123148  # id on figshare

        assert interval[0] >= 0.00  # the interval has to start after the cue onset

    def preprocess(self, raw):
        # interpolate channels marked as bad
        if len(raw.info["bads"]) > 0:
            raw.interpolate_bads()
        return super().preprocess(raw)

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        key_dest = f"MNE-{self.code:s}-data"
        path = os.path.join(dl.get_dataset_path(self.code, path), key_dest)

        filelist = dl.fs_get_file_list(self.figshare_id)
        reg = dl.fs_get_file_hash(filelist)
        fsn = dl.fs_get_file_id(filelist)

        spath = []
        for f in fsn.keys():
            if ".mat" not in f:
                continue
            sbj = int(f.split("_")[0][1:])
            ses = int(f.split("_")[-1].split(".")[0])
            if sbj == subject and ses in self.sessions:
                fpath = os.path.join(path, f)
                if not os.path.exists(fpath):
                    pooch.retrieve(
                        Stieger2021.BASE_URL + fsn[f],
                        reg[fsn[f]],
                        f,
                        path,
                        downloader=pooch.HTTPDownloader(progressbar=True),
                    )
                spath.append(fpath)
        return spath

    def _get_single_subject_data(self, subject):
        fname = self.data_path(subject)

        subject_data = {}

        for fn in fname:

            session = int(os.path.basename(fn).split("_")[2].split(".")[0])

            if self.sessions is not None:
                if session not in set(self.sessions):
                    continue

            container = loadmat(
                fn,
                squeeze_me=True,
                struct_as_record=False,
                verify_compressed_data_integrity=False,
            )["BCI"]

            srate = container.SRATE

            eeg_ch_names = container.chaninfo.label.tolist()
            # adjust naming convention
            eeg_ch_names = [
                ch.replace("Z", "z").replace("FP", "Fp") for ch in eeg_ch_names
            ]
            # extract all standard EEG channels
            montage = mne.channels.make_standard_montage("standard_1005")
            channel_mask = np.isin(eeg_ch_names, montage.ch_names)
            ch_names = [ch for ch, found in zip(eeg_ch_names, channel_mask) if found] + [
                "Stim"
            ]
            ch_types = ["eeg"] * channel_mask.sum() + ["stim"]

            X_flat = []
            stim_flat = []
            for i in range(container.data.shape[0]):
                x = container.data[i][channel_mask, :]
                y = container.TrialData[i].targetnumber
                stim = np.zeros_like(container.time[i])
                if (
                    container.TrialData[i].artifact == 0
                    and (container.TrialData[i].triallength + 2) > self.interval[1]
                ):
                    assert (
                        container.time[i][2 * srate] == 0
                    )  # this should be the cue time-point
                    stim[2 * srate] = y
                X_flat.append(x)
                stim_flat.append(stim[None, :])

            X_flat = np.concatenate(X_flat, axis=1)
            stim_flat = np.concatenate(stim_flat, axis=1)

            p_keep = np.flatnonzero(stim_flat).shape[0] / container.data.shape[0]

            message = f"Record {subject}/{session} (subject/session): rejecting {(1 - p_keep) * 100:.0f}% of the trials."
            if p_keep < 0.5:
                log.warning(message)
            else:
                log.info(message)

            eeg_data = np.concatenate([X_flat * 1e-6, stim_flat], axis=0)

            info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=srate)
            raw = mne.io.RawArray(data=eeg_data, info=info, verbose=False)
            raw.set_montage(montage)
            if isinstance(container.chaninfo.noisechan, int):
                badchanidxs = [container.chaninfo.noisechan]
            elif isinstance(container.chaninfo.noisechan, np.ndarray):
                badchanidxs = container.chaninfo.noisechan
            else:
                badchanidxs = []

            for idx in badchanidxs:
                used_channels = ch_names if self.channels is None else self.channels
                if eeg_ch_names[idx - 1] in used_channels:
                    raw.info["bads"].append(eeg_ch_names[idx - 1])

            if len(raw.info["bads"]) > 0:
                log.info(
                    f'Record {subject}/{session} (subject/session): bad channels that will be interpolated: {raw.info["bads"]}'
                )

            # subject_data[session] = {"run_0": self._common_prep(raw)}\
            subject_data[session] = {"run_0": self.preprocess(raw)}
        return subject_data
