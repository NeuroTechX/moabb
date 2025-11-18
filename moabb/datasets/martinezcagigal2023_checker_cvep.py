import tempfile
import traceback
import zipfile
from datetime import timezone
from glob import glob

import mne
import numpy as np
from dateutil import parser
from medusa import components

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


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

CONDITIONS = ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")


class MartinezCagigal2023C(BaseDataset):
    """Checkerboard m-sequence-based c-VEP dataset from Martínez-Cagigal et
    al. (2025) and Fernández-Rodríguez et al. (2023).

    **Dataset Description**
    This dataset, accesible at [1], was originally recorded for study [2],
    which evaluated 8 different stimuli in a c-VEP circular shifting paradigm
    using binary m-sequences. The conditions were tested in a 9-command
    speller. The stimulus was composed of a black-background checkerboard (
    BB-CB) pattern, i.e. event 1 was encoded with a checkerboard patterna nd
    event 0 with a black flash. The stimuli were encoded using circularly
    shifting versions of a 63-bit binary m-sequence. The different conditions
    evaluated different spatial frequency variations of the BB-CB pattern (
    i.e., the number of squares inside the checkerboard pattern). The evaluated
    conditions were:
    - 1c: C001 (0 c/º, 1x1 squares).
    - 2c: C002 (0.15 c/º, 2x2 squares).
    - 3c: C004 (0.3 c/º, 4x4 squares).
    - 4c: C008 (0.6 c/º, 8x8 squares).
    - 5c: C016 (1.2 c/º, 16x16 squares).
    - 6c: C032 (2.4 c/º, 32x32 squares).
    - 7c: C064 (4.79 c/º, 64x64 squares).
    - 8c: C128 (9.58 c/º, 128x128 squares).

    The dataset includes recordings from 16 healthy subjects performing
    a copy-spelling task under each condition. The evaluation was conducted in
    a single session, during which each participant completed:
    (1) a calibration phase consisting of 30 trials using the original
        m-sequence (divided into two recordings of 15 trials each), and
    (2) an online copy-spelling task of 18 trials (in one run).

    Each trial consisted of 8 cycles (i.e., repetitions of the same code).
    Additionally, participants completed questionnaires to assess satisfaction
    and perceived eyestrain for each m-sequence condition. Questionnaire
    results are available in [1].

    The encoding was displayed at a 120 Hz refresh rate. EEG signals were
    recorded using a g.USBamp amplifier (g.Tec, Guger Technologies, Austria)
    with 16 active electrodes and a sampling rate of 256 Hz. Electrodes were
    placed at: Oz, F3, Fz, F4, I1, I2, C3, Cz, C4, CPz, P3, Pz, P4, PO7, POz,
    PO8, grounded at AFz and referenced to the earlobe.

    The experimental paradigm was executed using the MEDUSA© software [3],
    with the publicly available application "c-VEP Speller":
    https://www.medusabci.com/market/cvep_speller/

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
    session, three MNE arrays are available (two for training, one for testing).

    Due to limitations in the MNE format, both cycle onsets and event onsets
    are stored as annotations, with labels included in their descriptions:
        - "cycle_onset": marks the start of a stimulation cycle
        - "0": marks the onset of a 0 (white) event
        - "1": marks the onset of a 1 (black) event

    The MNE format does not support encoding further paradigm-specific
    information. For full access to the dataset's metadata and structure,
    users are encouraged to load the original recordings using the MEDUSA Kernel.

    Example:
    >> pip install medusa-kernel
    >> from medusa import components
    >> from medusa.bci import cvep_spellers
    >> rec = components.Recording.load(path)

    The "rec" object will contain all available information.

    .. versionadded:: 1.2.0
    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, len(SUBJECTS) + 1)),
            sessions_per_subject=len(CONDITIONS),
            events={},
            code="MartinezCagigal2023Checkercvep",
            interval=(0, 1),  # Don't use this, it depends on the condition
            paradigm="cvep",
            doi="https://doi.org/10.71569/7c67-v596",
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""
        file_path_list = self.data_path(subject)

        # Each subject evaluated 8 different conditions, where
        # checkerboard-like stimuli spatial frequency varied
        sessions = dict()
        for cond in CONDITIONS:
            sessions[cond[::-1]] = dict()
        user = SUBJECTS[subject - 1]

        # Get the EEG files
        zf = zipfile.ZipFile(file_path_list[0])
        with tempfile.TemporaryDirectory() as tempdir:
            zf.extractall(tempdir)
            # For each base (condition)
            for cond in CONDITIONS:
                cname = cond[::-1]
                # Training signals
                train_paths = glob(f"{tempdir}/{user}/{cond}/*_calib*")
                for i, train_path in enumerate(train_paths):
                    try:
                        print(f"> Loading {user}, cond {cname}, train {i + 1}")
                        sessions[cname][f"{i + 1}train"] = self.__convert_to_mne_format(
                            train_path
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
                for i, test_path in enumerate(test_paths):
                    try:
                        print(f"> Loading {user}, cond {cname}, test {i+n+1}")
                        sessions[cname][f"{i + n + 1}test"] = (
                            self.__convert_to_mne_format(test_path, true_labels[i])
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
    def __get_camel_case_labels(ch_labels):
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
    def __trim_unfinished_trial(rec):
        if np.max(rec.cvepspellerdata.cycle_idx) != rec.cvepspellerdata.cycle_idx[-1]:
            last_idx = (
                np.where(
                    rec.cvepspellerdata.cycle_idx == np.max(rec.cvepspellerdata.cycle_idx)
                )[0][-1]
                + 1
            )
            rec.cvepspellerdata.cycle_idx = rec.cvepspellerdata.cycle_idx[:last_idx]
            rec.cvepspellerdata.level_idx = rec.cvepspellerdata.level_idx[:last_idx]
            rec.cvepspellerdata.matrix_idx = rec.cvepspellerdata.matrix_idx[:last_idx]
            rec.cvepspellerdata.onsets = rec.cvepspellerdata.onsets[:last_idx]
            rec.cvepspellerdata.trial_idx = rec.cvepspellerdata.trial_idx[:last_idx]
            rec.cvepspellerdata.unit_idx = rec.cvepspellerdata.unit_idx[:last_idx]
        return rec

    def __convert_to_mne_format(self, path, true_labels=None):
        # Load in MEDUSA format
        rec = components.Recording.load(path)

        # Trim unfinished trials
        rec = self.__trim_unfinished_trial(rec)
        signal = rec.get_biosignals_with_class_name("EEG")["eeg"]

        # Create the info
        ch_names = self.__get_camel_case_labels(signal.channel_set.l_cha)
        ch_types = ["eeg"] * len(ch_names)
        sampling_freq = signal.fs
        meas_date = parser.parse(rec.date)
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sampling_freq)
        info["subject_info"] = {"his_id": str(rec.subject_id)}
        info["description"] = str(rec.recording_id)
        info.set_meas_date(meas_date.replace(tzinfo=timezone.utc))
        info.set_montage("standard_1005", match_case=False, on_missing="warn")

        # Set data
        raw_data = mne.io.RawArray(np.array(signal.signal).T, info, verbose=False)
        exp_annotations = {}

        # Cycle onsets
        sample_onsets = rec.cvepspellerdata.onsets - signal.times[0]
        exp_annotations["onset"] = sample_onsets.tolist()
        exp_annotations["description"] = ["cycle_onset"] * len(sample_onsets)

        # Get bit-wise sequences
        fps = rec.cvepspellerdata.fps_resolution
        seqs_by_cycle = list()
        if rec.cvepspellerdata.mode == "train":
            for idx in range(len(rec.cvepspellerdata.command_idx)):
                m_ = int(rec.cvepspellerdata.matrix_idx[idx])
                c_ = str(int(rec.cvepspellerdata.command_idx[idx]))
                seqs_by_cycle.append(
                    rec.cvepspellerdata.commands_info[m_][c_]["sequence"]
                )
        else:
            assert true_labels is not None
            seqs_by_trial = list()
            for label in true_labels:
                for item in rec.cvepspellerdata.commands_info[0].values():
                    if item["label"] == label:
                        seqs_by_trial.append(item["sequence"])
                        break
            for t_idx in rec.cvepspellerdata.trial_idx:
                seqs_by_cycle.append(seqs_by_trial[int(t_idx)])

        # Set the duration for cycle onsets
        exp_annotations["duration"] = [len(s) / fps for s in seqs_by_cycle]

        # Annotate bit-wise
        for o_idx in range(len(sample_onsets)):
            bw_onsets = [
                sample_onsets[o_idx] + i / fps for i in range(len(seqs_by_cycle[o_idx]))
            ]
            bw_duration = [1 / fps] * len(bw_onsets)
            bw_desc = seqs_by_cycle[o_idx]
            exp_annotations["onset"] += bw_onsets
            exp_annotations["duration"] += bw_duration
            exp_annotations["description"] += bw_desc

        # Set annotations
        annotations = mne.Annotations(
            onset=exp_annotations["onset"],
            duration=exp_annotations["duration"],
            description=exp_annotations["description"],
        )
        raw_data.set_annotations(annotations)

        return raw_data


if __name__ == "__main__":
    dataset = MartinezCagigal2023C()
    sessions = dataset.get_data(subjects=[1])
