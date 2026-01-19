import tempfile
import traceback
import zipfile
from datetime import timezone
from glob import glob

import mne
import numpy as np
from dateutil import parser

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset
from moabb.datasets.bson_loader import load_bson
from moabb.datasets.utils import add_stim_channel_epoch, add_stim_channel_trial


MARTINEZCAGIGAL2023_PARY_URL = "https://uvadoc.uva.es/handle/10324/70945"
HANDLE_URI = "https://uvadoc.uva.es/bitstream/handle/10324/70945"

SUBJECTS = (
    "hgab",
    "btfc",
    "lvpg",
    "qtce",
    "pcsa",
    "pkel",
    "vfab",
    "bvgy",
    "eimw",
    "sbbh",
    "iuvo",
    "hmfm",
    "pqgs",
    "opqm",
    "fjax",
    "zdvm",
)

ALL_CONDITIONS = ("2", "3", "5", "7", "11")

# M-sequence lengths for each base (p-ary)
MSEQUENCE_LENGTHS = {
    "2": 63,  # GF(2^6)
    "3": 80,  # GF(3^4)
    "5": 124,  # GF(5^3)
    "7": 48,  # GF(7^2)
    "11": 120,  # GF(11^2)
}

# Event descriptions for p-ary datasets
# Note: offset of 100 is added by add_stim_channel_epoch
EVENTS = {
    "0.0": 100,
    "1.0": 101,
    "2.0": 102,
    "3.0": 103,
    "4.0": 104,
    "5.0": 105,
    "6.0": 106,
    "7.0": 107,
    "8.0": 108,
    "9.0": 109,
    "10.0": 110,
}


class MartinezCagigal2023Pary(BaseDataset):
    """P-ary m-sequence-based c-VEP dataset from Martínez-Cagigal et al. (2023)

    **Dataset Description**

    This dataset was originally recorded for study [1]_, which evaluated
    different non-binary encoding strategies. Specifically, five different
    conditions were tested in a 16-command speller. Each condition used a
    different p-ary m-sequence to encode the commands via circular shifting.
    One command was encoded using the original m-sequence, while the remaining
    commands were encoded using shifted versions of that sequence [2]_.

    A p-ary m-sequence means it contains *p* different events, which were
    encoded using different shades of gray. For example, in the binary case
    (p=2), events 0 and 1 were encoded using white and black flashes,
    respectively. For p=3, black, white, and mid-gray flashes were used [1]_.

    The evaluated conditions were:

    - Base 2: GF(2^6) m-sequence of 63 bits
    - Base 3: GF(3^4) m-sequence of 80 bits
    - Base 5: GF(5^3) m-sequence of 124 bits
    - Base 7: GF(7^2) m-sequence of 48 bits
    - Base 11: GF(11^2) m-sequence of 120 bits

    The dataset includes recordings from 16 healthy subjects performing
    a copy-spelling task under each condition. The evaluation was conducted in
    a single session, during which each participant completed:

    1. A calibration phase consisting of 30 trials using the original
       m-sequence (divided into six recordings of five trials each), and
    2. An online copy-spelling task of 32 trials (divided into two recordings
       of 16 trials each).

    Each trial consisted of 10 cycles (i.e., repetitions of the same code).
    Additionally, participants completed questionnaires to assess satisfaction
    and perceived eyestrain for each m-sequence condition. Questionnaire
    results are available in [3]_.

    The encoding was displayed at a 120 Hz refresh rate. EEG signals were
    recorded using a g.USBamp amplifier (g.Tec, Guger Technologies, Austria)
    with 16 active electrodes and a sampling rate of 256 Hz. Electrodes were
    placed at: F3, Fz, F4, C3, Cz, C4, CPz, P3, Pz, P4, PO7, PO8, Oz, I1, and I2;
    grounded at AFz and referenced to the earlobe.

    .. note::
       Recordings of user "zdvm" for bases 2, 3, 5, and 7 had a sampling rate
       of 600 Hz. The rest of recordings have all a sampling rate of 256 Hz.

    The experimental paradigm was executed using the MEDUSA© software [4]_.

    Parameters
    ----------
    conditions : tuple of str, optional
        Which conditions to load. Default is all conditions:
        ("2", "3", "5", "7", "11").
        Each condition corresponds to a different p-ary m-sequence base.

    References
    ----------
    .. [1] Martínez-Cagigal, V., Santamaría-Vázquez, E., Pérez-Velasco, S.,
       Marcos-Martínez, D., Moreno-Calderón, S., & Hornero, R. (2023).
       Non-binary m-sequences for more comfortable brain–computer interfaces
       based on c-VEPs. *Expert Systems with Applications, 232*, 120815.
       https://doi.org/10.1016/j.eswa.2023.120815

    .. [2] Martínez-Cagigal, V., Thielen, J., Santamaría-Vázquez, E.,
       Pérez-Velasco, S., Desain, P., & Hornero, R. (2021).
       Brain–computer interfaces based on code-modulated visual evoked
       potentials (c-VEP): A literature review. *Journal of Neural Engineering,
       18*(6), 061002. https://doi.org/10.1088/1741-2552/ac38cf

    .. [3] Martínez-Cagigal, V. (2025). Dataset: Non-binary m-sequences for
       more comfortable brain–computer interfaces based on c-VEPs.
       https://doi.org/10.35376/10324/70945

    .. [4] Santamaría-Vázquez, E., Martínez-Cagigal, V., Marcos-Martínez, D.,
       Rodríguez-González, V., Pérez-Velasco, S., Moreno-Calderón, S., &
       Hornero, R. (2023). MEDUSA©: A novel Python-based software ecosystem to
       accelerate brain–computer interface and cognitive neuroscience research.
       *Computer Methods and Programs in Biomedicine, 230*, 107357.
       https://doi.org/10.1016/j.cmpb.2023.107357

    Notes
    -----
    Although the dataset was recorded in a single session, each condition is
    stored as a separate session to match the MOABB structure. Within each
    session, eight runs are available (six for training, two for testing).

    .. versionadded:: 1.2.0
    """

    def __init__(self, conditions=ALL_CONDITIONS):
        # Validate conditions
        for cond in conditions:
            if cond not in ALL_CONDITIONS:
                raise ValueError(
                    f"Invalid condition '{cond}'. "
                    f"Valid conditions are: {ALL_CONDITIONS}"
                )
        self.conditions = conditions

        super().__init__(
            subjects=list(range(1, len(SUBJECTS) + 1)),
            sessions_per_subject=len(conditions),
            events=EVENTS,
            code="MartinezCagigal2023Parycvep",
            interval=(0, 1),  # Don't use this, it depends on the condition
            paradigm="cvep",
            doi="https://doi.org/10.71569/025s-eq10",
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""
        file_path_list = self.data_path(subject)
        user = SUBJECTS[subject - 1]

        # Get the EEG files
        zf = zipfile.ZipFile(file_path_list[0])
        sessions = {}

        with tempfile.TemporaryDirectory() as tempdir:
            zf.extractall(tempdir)

            for i, base in enumerate(self.conditions):
                session_name = f"{i}base{base}"  # e.g., "0base2", "1base3"
                sessions[session_name] = {}

                # Training signals
                train_paths = glob(f"{tempdir}/{user}/*{base}_train*")
                for j, train_path in enumerate(train_paths):
                    try:
                        print(f"> Loading {user}, base {base}, train {j + 1}")
                        sessions[session_name][f"{j + 1}train"] = (
                            self._convert_to_mne_format(train_path)
                        )
                    except Exception:
                        print(
                            f"[EXCEPTION] Cannot convert signal {train_path}."
                            f" More information: {traceback.format_exc()}"
                        )
                n = len(train_paths)

                # Load the true labels for testing
                test_labels = glob(f"{tempdir}/{user}/*{base}_labels*")
                assert len(test_labels) == 1
                with open(test_labels[0], "r", encoding="utf-8") as f:
                    true_labels = [line.strip() for line in f.readlines()]

                # Testing signals
                test_paths = glob(f"{tempdir}/{user}/*{base}_test*")
                assert len(test_paths) == len(true_labels)
                for j, test_path in enumerate(test_paths):
                    try:
                        print(f"> Loading {user}, base {base}, test {j+n+1}")
                        sessions[session_name][f"{j + n + 1}test"] = (
                            self._convert_to_mne_format(test_path, true_labels[j])
                        )
                    except Exception:
                        print(
                            f"[EXCEPTION] Cannot convert signal {test_path}."
                            f" More information: {traceback.format_exc()}"
                        )

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return the data paths of a single subject."""
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        # Get subject (anonymized name)
        sub = SUBJECTS[subject - 1]

        # Get subject data
        url = f"{HANDLE_URI}/{sub}.zip"
        subject_paths = list()
        subject_paths.append(dl.data_dl(url, self.code, path, force_update, verbose))

        return subject_paths

    @staticmethod
    def _get_camel_case_labels(ch_labels):
        """Converts a given list of channel labels to the common camel-case
        format (e.g., 'FPZ' -> 'FPz').
        """
        assert isinstance(ch_labels, list)
        camel_case_labels = []
        for ch in ch_labels:
            new_ch = ch.replace("Z", "z").replace("H", "h").replace("FP", "Fp")
            camel_case_labels.append(new_ch)
        return camel_case_labels

    @staticmethod
    def _load_bson_recording(path):
        """Load a BSON recording file and return the data dictionary."""
        return load_bson(path)

    @staticmethod
    def _trim_unfinished_trial(cvep_data):
        """Trim incomplete trials from the CVEP data."""
        cycle_idx = np.array(cvep_data["cycle_idx"])
        if np.max(cycle_idx) != cycle_idx[-1]:
            last_idx = np.where(cycle_idx == np.max(cycle_idx))[0][-1] + 1
            cvep_data["cycle_idx"] = cvep_data["cycle_idx"][:last_idx]
            cvep_data["level_idx"] = cvep_data["level_idx"][:last_idx]
            cvep_data["matrix_idx"] = cvep_data["matrix_idx"][:last_idx]
            cvep_data["onsets"] = cvep_data["onsets"][:last_idx]
            cvep_data["trial_idx"] = cvep_data["trial_idx"][:last_idx]
            cvep_data["unit_idx"] = cvep_data["unit_idx"][:last_idx]
        return cvep_data

    def _convert_to_mne_format(self, path, true_labels=None):
        """Convert a BSON recording to MNE Raw format."""
        # Load BSON file
        rec = self._load_bson_recording(path)

        # Get CVEP speller data
        cvep_data = rec["cvepspellerdata"]
        cvep_data = self._trim_unfinished_trial(cvep_data)

        # Get EEG data
        eeg = rec["eeg"]
        signal = np.array(eeg["signal"])
        times = np.array(eeg["times"])
        sampling_freq = eeg["fs"]

        # Create the info
        ch_names = self._get_camel_case_labels(eeg["channel_set"]["l_cha"])
        ch_types = ["eeg"] * len(ch_names)
        meas_date = parser.parse(rec["date"])
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sampling_freq)
        info["subject_info"] = {"his_id": str(rec["subject_id"])}
        info["description"] = str(rec["recording_id"])
        info.set_meas_date(meas_date.replace(tzinfo=timezone.utc))
        info.set_montage("standard_1005", match_case=False, on_missing="warn")

        # Set data (signal shape is samples x channels, need to transpose)
        raw_data = mne.io.RawArray(signal.T, info, verbose=False)

        # Get timing information
        fps = cvep_data["fps_resolution"]
        sample_onsets = np.array(cvep_data["onsets"]) - times[0]

        # Get bit-wise sequences for each cycle
        seqs_by_cycle = list()
        commands_info = cvep_data["commands_info"]

        if cvep_data["mode"] == "train":
            command_idx = cvep_data["command_idx"]
            matrix_idx = cvep_data["matrix_idx"]
            for idx in range(len(command_idx)):
                m_ = int(matrix_idx[idx])
                c_ = str(int(command_idx[idx]))
                seqs_by_cycle.append(commands_info[m_][c_]["sequence"])
        else:
            # For test mode, need to look up sequences by label
            assert true_labels is not None
            seqs_by_trial = list()
            for label in true_labels:
                for item in commands_info[0].values():
                    if item["label"] == label:
                        seqs_by_trial.append(item["sequence"])
                        break
            trial_idx = cvep_data["trial_idx"]
            for t_idx in trial_idx:
                seqs_by_cycle.append(seqs_by_trial[int(t_idx)])

        # Calculate trial onsets in samples
        trial_onsets_samples = (sample_onsets * sampling_freq).astype(int)

        # Get unique trial indices and their labels
        trial_idx = np.array(cvep_data["trial_idx"])
        unique_trials = np.unique(trial_idx)
        trial_labels = unique_trials.astype(int)

        # Find the first onset for each trial
        first_trial_onsets = []
        for t in unique_trials:
            mask = trial_idx == t
            first_onset_idx = np.where(mask)[0][0]
            first_trial_onsets.append(trial_onsets_samples[first_onset_idx])
        first_trial_onsets = np.array(first_trial_onsets)

        # Add trial-level stimulus channel (offset=200)
        raw_data = add_stim_channel_trial(
            raw_data, first_trial_onsets, trial_labels, offset=200
        )

        # Build a codebook from the sequences (shape: n_bits x n_codes)
        # For p-ary sequences, codes contain values 0 to p-1
        max_seq_len = max(len(s) for s in seqs_by_cycle)
        unique_seqs = {}
        for i, seq in enumerate(seqs_by_cycle):
            trial_id = int(trial_idx[i])
            if trial_id not in unique_seqs:
                unique_seqs[trial_id] = seq

        # Create codebook matrix
        n_codes = len(unique_seqs)
        codes = np.zeros((max_seq_len, n_codes), dtype=int)
        for trial_id, seq in unique_seqs.items():
            codes[: len(seq), trial_id] = seq

        # Add epoch-level stimulus channel with bit-wise codes (offset=100)
        raw_data = add_stim_channel_epoch(
            raw_data, first_trial_onsets, trial_labels, codes, fps, offset=100
        )

        return raw_data


if __name__ == "__main__":
    dataset = MartinezCagigal2023Pary()
    sessions = dataset.get_data(subjects=[1])
    print(sessions[1].keys())
