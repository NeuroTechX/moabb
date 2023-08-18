import h5py
import mne
import numpy as np
from mne import create_info
from mne.io import RawArray
from scipy.io import loadmat

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


Thielen2021_URL = "https://public.data.donders.ru.nl/dcc/DSC_2018.00122_448_v3"
ELECTRODE_MAPPING = {
    "AF3": "Fz",
    "F3": "T7",
    "FC5": "O1",
    "P7": "POz",
    "P8": "Oz",
    "FC6": "Iz",
    "F4": "O2",
    "AF4": "T8",
}
SESSIONS = (
    "20181128",
    "20181206",
    "20181217",
    "20181217",
    "20181217",
    "20181218",
    "20181218",
    "20181219",
    "20181219",
    "20181220",
    "20181220",
    "20181220",
    "20190107",
    "20190107",
    "20190110",
    "20190110",
    "20190110",
    "20190117",
    "20190117",
    "20190118",
    "20190118",
    "20190118",
    "20190220",
    "20190222",
    "20190225",
    "20190301",
    "20190307",
    "20190308",
    "20190311",
    "20190311",
)
NR_BLOCKS = 5  # Each session consisted of 5 blocks (i.e., runs)
NR_CYCLES_PER_TRIAL = 15  # Each trial contained 15 cycles of a 2.1 second code
PRESENTATION_RATE = 60  # Codes were presented at a 60 Hz monitor refresh rate


class Thielen2021(BaseDataset):
    """c-VEP dataset from Thielen et al. (2021)

    Dataset [1]_ from the study on zero-training c-VEP [2]_.

    .. admonition:: Dataset summary

        =============  =======  =======  ==================  ===============  ===============  ===========
        Name             #Subj    #Chan     #Trials / class  Trials length    Sampling rate      #Sessions
        =============  =======  =======  ==================  ===============  ===============  ===========
        Thielen2021         30        8  18900 NT / 18900 T  0.3s             512Hz                      1
        =============  =======  =======  ==================  ===============  ===============  ===========

    **Dataset description**

    EEG recordings were acquired at a sampling rate of 512 Hz, employing 8 Ag/AgCl electrodes. The Biosemi ActiveTwo EEG
    amplifier was utilized during the experiment. The electrode array consisted of Fz, T7, O1, POz, Oz, Iz, O2, and T8,
    connected as EXG channels.

    During the experimental sessions, participants engaged in passive operation (i.e., without feedback) of a 4 x 5
    visual speller Brain-Computer Interface (BCI), comprising 20 distinct classes. Each cell of the symbol grid
    underwent luminance modulation at full contrast, accomplished through pseudo-random noise-codes derived from a
    collection of modulated Gold codes. These codes are binary, have a balanced distribution of ones and zeros, and
    adhering to a limited run-length pattern (maximum run-length of 2 bits). The codes were presented at a presentation
    rate of 60 Hz. As one cycle of these modulated Gold codes contains 126 bits, the duration of one cycle is 2.1
    seconds.

    For each of the five blocks, a trial started with a cueing phase, during which the target symbol was highlighted in
    a green hue for a duration of 1 second. Following this, participants maintained their gaze fixated on the target
    symbol while all symbols flashed in accordance with their respective pseudo-random noise-codes for a duration of
    31.5 seconds (i.e., 15 code cycles). Each block encompassed 20 trials, presented in a randomized sequence, thereby
    ensuring that each symbol was attended to once within the span of a block.

    Note, here, we only load the offline data of this study, and ignore the online phase.

    References
    ----------

    .. [1] Thielen, J. (Jordy), Pieter Marsman, Jason Farquhar, Desain, P.W.M. (Peter) (2023): From full calibration to
           zero training for a code-modulated visual evoked potentials brain computer interface. Version 3. Radboud
           University. (dataset). DOI: https://doi.org/10.34973/9txv-z787

    .. [2] Thielen, J., Marsman, P., Farquhar, J., & Desain, P. (2021). From full calibration to zero training for a
           code-modulated visual evoked potentials for brain–computer interface. Journal of Neural Engineering, 18(5),
           056007. DOI: https://doi.org/10.1088/1741-2552/abecef

    Notes
    -----

    .. versionadded:: 0.6.0

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 30 + 1)),
            sessions_per_subject=1,
            events={"1.0": 101, "0.0": 100},
            code="Thielen2021",
            interval=(0, 0.3),
            paradigm="cvep",
            doi="10.34973/9txv-z787",
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""
        file_path_list = self.data_path(subject)

        # Codes
        codes = np.tile(loadmat(file_path_list[-1])["codes"], (NR_CYCLES_PER_TRIAL, 1))

        # There is only one session, each of 5 blocks (i.e., runs)
        sessions = {"session_1": {}}
        for i_b in range(NR_BLOCKS):
            # EEG
            raw = mne.io.read_raw_gdf(
                file_path_list[2 * i_b],
                stim_channel="status",
                preload=True,
                verbose=False,
            )
            mne.rename_channels(raw.info, ELECTRODE_MAPPING)

            # Labels at trial level (i.e., symbols)
            labels = (
                np.array(h5py.File(file_path_list[2 * i_b + 1], "r")["v"])
                .astype("uint8")
                .flatten()
                - 1
            )

            # Find onsets of trials
            # N.B. Every 2.1 seconds an event was generated (15 times per trial, plus one 16th "leaking epoch")
            # N.B. This "leaking epoch" is not always present, so taking epoch[::16, :] won't work
            events = mne.find_events(raw, verbose=False)
            cond = np.logical_or(
                np.diff(events[:, 0]) < 1.8 * raw.info["sfreq"],
                np.diff(events[:, 0]) > 2.4 * raw.info["sfreq"],
            )
            idx = np.concatenate(([0], 1 + np.where(cond)[0]))
            onsets = events[idx, 0]

            # Create stim channel with trial information (i.e., symbols)
            # N.B. 200 = symbol-0, 201 = symbol-1, 202 = symbol-2, etc.
            stim_chan = np.zeros((1, len(raw)))
            for onset, label in zip(onsets, labels):
                stim_chan[0, onset] = 200 + label
            info = create_info(
                ch_names=["stim_trial"],
                ch_types=["stim"],
                sfreq=raw.info["sfreq"],
                verbose=False,
            )
            raw = raw.add_channels([RawArray(data=stim_chan, info=info, verbose=False)])

            # Create stim channel with epoch information (i.e., 1 / 0, or on / off)
            # N.B. 100 = "0", 101 = "1"
            stim_chan = np.zeros((1, len(raw)))
            for onset, label in zip(onsets, labels):
                idx = (
                    onset
                    + np.arange(codes.shape[0]) / PRESENTATION_RATE * raw.info["sfreq"]
                ).astype("int")
                stim_chan[0, idx] = 100 + codes[:, label]
            info = create_info(
                ch_names=["stim_epoch"],
                ch_types=["stim"],
                sfreq=raw.info["sfreq"],
                verbose=False,
            )
            raw = raw.add_channels([RawArray(data=stim_chan, info=info, verbose=False)])

            # Add data as a new run
            sessions["session_1"][f"run_{1 + i_b:02d}"] = raw

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the data paths of a single subject."""
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        sub = f"sub-{subject:02d}"
        ses = SESSIONS[subject - 1]
        subject_paths = []
        for i_b in range(NR_BLOCKS):
            blk = f"block_{1 + i_b:d}"

            # EEG
            url = f"{Thielen2021_URL:s}/sourcedata/offline/{sub}/{blk}/{sub}_{ses}_{blk}_main_eeg.gdf"
            subject_paths.append(dl.data_dl(url, self.code, path, force_update, verbose))

            # Labels at trial level (i.e., symbols)
            url = f"{Thielen2021_URL:s}/sourcedata/offline/{sub}/{blk}/trainlabels.mat"
            subject_paths.append(dl.data_dl(url, self.code, path, force_update, verbose))

        # Codes
        url = f"{Thielen2021_URL:s}/resources/mgold_61_6521_flip_balanced_20.mat"
        subject_paths.append(dl.data_dl(url, self.code, path, force_update, verbose))

        return subject_paths
