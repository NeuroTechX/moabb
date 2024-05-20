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
    """Motor Imagery dataset from Stieger et al. 2021.

    .. admonition:: Dataset summary


        ============= ======= ======= ========== ================= ============ =============== ===========
        Name          #Subj   #Chan   #Classes   #Trials / class   Trials len   Sampling rate   #Sessions
        ============= ======= ======= ========== ================= ============ =============== ===========
        Stieger2021   62      64      4          450               3s           1000Hz          10
        ============= ======= ======= ========== ================= ============ =============== ===========

    The main goals of our original study were to characterize how individuals
    learn to control SMR-BCIs and to test whether this learning can be improved
    through behavioral interventions such as mindfulness training30. Participants
    were initially assessed for baseline BCI proficiency and then randomly
    assigned to an 8-week mindfulness intervention (Mindfulness-based stress
    reduction), or waitlist control condition where participants waited for
    the same duration as the MBSR class before starting BCI training, but
    were offered a comparable MBSR course after completing all experimental
    requirements46. Following the 8-weeks, participants returned to the lab
    for 6–10 sessions of BCI training.

    All experiments were approved by the institutional review boards of the
    University of Minnesota and Carnegie Mellon University. Informed consents
    were obtained from all subjects. In total, 144 participants were enrolled
    in the study and 76 participants completed all experimental requirements.
    Seventy-two participants were assigned to each intervention by block
    randomization, with 42 participants completing all sessions in the
    experimental group (MBSR before BCI training; MBSR subjects) and 34
    completing experimentation in the control group. Four subjects were
    excluded from the analysis due to non-compliance with the task demands and
    one was excluded due to experimenter error. We were primarily interested
    in how individuals learn to control BCIs, therefore analysis focused on
    those that did not demonstrate ceiling performance in the baseline BCI
    assessment (accuracy above 90% in 1D control). The dataset descriptor
    presented here describes data collected from 62 participants:
    33 MBSR participants (Age=42+/−15, (F)emale=26) and 29 controls
    (Age=36+/−13, F=23). In the United States, women are twice as likely to
    practice meditation compared to men47,48. Therefore, the gender
    imbalance in our study may result from a greater likelihood of
    women to respond to flyers offering a meditation class in exchange
    for participating in our study.

    For all BCI sessions, participants were seated comfortably in a chair and
    faced a computer monitor that was placed approximately 65cm in front of them.
    After the EEG capping procedure (see data acquisition), the BCI tasks began.
    Before each task, participants received the appropriate instructions. During
    the BCI tasks, users attempted to steer a virtual cursor from the center of
    the screen out to one of four targets. Participants initially received the
    following instructions: “Imagine your left (right) hand opening and closing
    to move the cursor left (right). Imagine both hands opening and closing to
    move the cursor up. Finally, to move the cursor down, voluntarily rest; in
    other words, clear your mind.” In separate blocks of trials, participants
    directed the cursor toward a target that required left/right (LR) movement
    only, up/down (UD) only, and combined 2D movement (2D)30. Each experimental
    block (LR, UD, 2D) consisted of 3 runs, where each run was composed of 25
    trials. After the first three blocks, participants were given a short break
    (5–10 minutes) that required rating comics by preference. The break task was
    chosen to standardize subject experience over the break interval. Following
    the break, participants competed the same 3 blocks as before. In total,
    each session consisted of 2 blocks of each task (6 runs total of LR, UD,
    and 2D control), which culminated in 450 trials performed each day.

    References
    ----------

    .. [1] Stieger, J. R., Engel, S. A., & He, B. (2021).
           Continuous sensorimotor rhythm based brain computer interface
           learning in a large population. Scientific Data, 8(1), 98.
           https://doi.org/10.1038/s41597-021-00883-1
    """

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
                    f"Record {subject}/{session} (subject/session): "
                    f'bad channels that need to be interpolated: {raw.info["bads"]}'
                )

            subject_data[session] = {"run_0": raw}
        return subject_data
