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


MARTINEZCAGIGAL2023_CHECKER_URL = "https://uvadoc.uva.es/handle/10324/70973"
HANDLE_URI = "https://uvadoc.uva.es/bitstream/handle/10324/70973"

SUBJECTS = (
    "SF01",
    "SF02",
    "SF03",
    "SF04",
    "SF05",
    "SF06",
    "SF07",
    "SF08",
    "SF09",
    "SF10",
    "SF11",
    "SF12",
    "SF13",
    "SF14",
    "SF15",
    "SF16",
)

ALL_CONDITIONS = ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")

# Event descriptions for checkerboard dataset
# Note: offset of 100 is added by add_stim_channel_epoch
EVENTS = {"0.0": 100, "1.0": 101}


class MartinezCagigal2023Checker(BaseDataset):
    """Checkerboard m-sequence-based c-VEP dataset from
    Martínez-Cagigal et al. (2025) and Fernández-Rodríguez et al. (2023).

    **Dataset Description**

    This dataset, accessible at [1]_, was originally recorded for study [2]_,
    which evaluated 8 different stimuli in a c-VEP circular shifting paradigm
    using binary m-sequences. The conditions were tested in a 9-command
    speller. The stimulus was composed of a black-background checkerboard
    (BB-CB) pattern, i.e. event 1 was encoded with a checkerboard pattern and
    event 0 with a white flash. The stimuli were encoded using circularly
    shifting versions of a 63-bit binary m-sequence. The different conditions
    evaluated different spatial frequency variations of the BB-CB pattern
    (i.e., the number of squares inside the checkerboard pattern).

    The evaluated conditions were:

    - c1: C001 (0 c/º, 1x1 squares).
    - c2: C002 (0.15 c/º, 2x2 squares).
    - c3: C004 (0.3 c/º, 4x4 squares).
    - c4: C008 (0.6 c/º, 8x8 squares).
    - c5: C016 (1.2 c/º, 16x16 squares).
    - c6: C032 (2.4 c/º, 32x32 squares).
    - c7: C064 (4.79 c/º, 64x64 squares).
    - c8: C128 (9.58 c/º, 128x128 squares).

    The dataset includes recordings from 16 healthy subjects performing
    a copy-spelling task under each condition. The evaluation was conducted in
    a single session, during which each participant completed:

    1. A calibration phase consisting of 30 trials using the original
       m-sequence (divided into two recordings of 15 trials each), and
    2. An online copy-spelling task of 18 trials (in one run).

    Each trial consisted of 8 cycles (i.e., repetitions of the same code).
    Additionally, participants completed questionnaires to assess satisfaction
    and perceived eyestrain for each m-sequence condition. Questionnaire
    results are available in [1]_.

    The encoding was displayed at a 120 Hz refresh rate. EEG signals were
    recorded using a g.USBamp amplifier (g.Tec, Guger Technologies, Austria)
    with 16 active electrodes and a sampling rate of 256 Hz. Electrodes were
    placed at: Oz, F3, Fz, F4, I1, I2, C3, Cz, C4, CPz, P3, Pz, P4, PO7, POz,
    PO8, grounded at AFz and referenced to the earlobe.

    The experimental paradigm was executed using the MEDUSA© software [3]_.

    Parameters
    ----------
    conditions : tuple of str, optional
        Which conditions to load. Default is all conditions:
        ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8").
        Each condition corresponds to a different spatial frequency of the
        checkerboard pattern.

    References
    ----------
    .. [1] Martínez Cagigal, V. (2025). Dataset: Influence of spatial frequency
       in visual stimuli for cVEP-based BCIs: evaluation of performance and
       user experience. https://doi.org/10.71569/7c67-v596

    .. [2] Fernández-Rodríguez, Á., Martínez-Cagigal, V., Santamaría-Vázquez,
       E., Ron-Angevin, R., & Hornero, R. (2023). Influence of spatial frequency
       in visual stimuli for cVEP-based BCIs: evaluation of performance and user
       experience. Frontiers in Human Neuroscience, 17, 1288438.
       https://doi.org/10.3389/fnhum.2023.1288438

    .. [3] Santamaría-Vázquez, E., Martínez-Cagigal, V., Marcos-Martínez, D.,
       Rodríguez-González, V., Pérez-Velasco, S., Moreno-Calderón, S., &
       Hornero, R. (2023). MEDUSA©: A novel Python-based software ecosystem to
       accelerate brain–computer interface and cognitive neuroscience research.
       Computer Methods and Programs in Biomedicine, 230, 107357.
       https://doi.org/10.1016/j.cmpb.2023.107357

    Notes
    -----
    Although the dataset was recorded in a single session, each condition is
    stored as a separate session to match the MOABB structure. Within each
    session, three runs are available (two for training, one for testing).

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
            code="MartinezCagigal2023Checkercvep",
            interval=(0, 1),  # Don't use this, it depends on the condition
            paradigm="cvep",
            doi="https://doi.org/10.71569/7c67-v596",
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

            for i, cond in enumerate(self.conditions):
                session_name = f"{i}{cond}"  # e.g., "0c1", "1c2"
                sessions[session_name] = {}

                # Training signals
                train_paths = glob(f"{tempdir}/{user}/{cond}/*_calib*")
                for j, train_path in enumerate(train_paths):
                    try:
                        print(f"> Loading {user}, cond {cond}, train {j + 1}")
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
                test_labels = glob(f"{tempdir}/{user}/{cond}/*_labels*")
                assert len(test_labels) == 1
                with open(test_labels[0], "r", encoding="utf-8") as f:
                    true_labels = [line.strip() for line in f.readlines()]

                # Testing signals
                test_paths = glob(f"{tempdir}/{user}/{cond}/*_online*")
                assert len(test_paths) == len(true_labels)
                for j, test_path in enumerate(test_paths):
                    try:
                        print(f"> Loading {user}, cond {cond}, test {j+n+1}")
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
        # For binary sequences, codes contain values 0 and 1
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
    dataset = MartinezCagigal2023Checker()
    sessions = dataset.get_data(subjects=[1])
    print(sessions[1].keys())
