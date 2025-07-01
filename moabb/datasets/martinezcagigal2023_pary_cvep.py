import mne
import numpy as np
import zipfile, tempfile
import traceback
from glob import glob
from medusa import components
from medusa.bci import cvep_spellers
from dateutil import parser
from datetime import timezone

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


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
    "zdvm"
)

CONDITIONS = (
    "2",
    "3",
    "5",
    "7",
    "11"
)

class MartinezCagigal2023P(BaseDataset):
    """P-ary m-sequence-based c-VEP dataset from Martínez-Cagigal et al. (2023)

    **Dataset Description**
    This dataset was originally recorded for study [1], which evaluated
    different non-binary encoding strategies. Specifically, five different
    conditions were tested in a 16-command speller. Each condition used a
    different p-ary m-sequence to encode the commands via circular shifting.
    One command was encoded using the original m-sequence, while the remaining
    commands were encoded using shifted versions of that sequence [2].

    A p-ary m-sequence means it contains *p* different events, which were
    encoded using different shades of gray. For example, in the binary case
    (p=2), events 0 and 1 were encoded using white and black flashes,
    respectively. For p=3, black, white, and mid-gray flashes were used [1].

    The evaluated conditions were:
    - Base 2: GF(2^6) m-sequence of 63 bits
    - Base 3: GF(3^4) m-sequence of 80 bits
    - Base 5: GF(5^3) m-sequence of 124 bits
    - Base 7: GF(7^2) m-sequence of 48 bits
    - Base 11: GF(11^2) m-sequence of 120 bits

    The dataset includes recordings from 16 healthy subjects performing
    a copy-spelling task under each condition. The evaluation was conducted in
    a single session, during which each participant completed:
    (1) a calibration phase consisting of 30 trials using the original
        m-sequence (divided into six recordings of five trials each), and
    (2) an online copy-spelling task of 32 trials (divided into two recordings
        of 16 trials each).

    Each trial consisted of 10 cycles (i.e., repetitions of the same code).
    Additionally, participants completed questionnaires to assess satisfaction
    and perceived eyestrain for each m-sequence condition. Questionnaire
    results are available in [3].

    The encoding was displayed at a 120 Hz refresh rate. EEG signals were
    recorded using a g.USBamp amplifier (g.Tec, Guger Technologies, Austria)
    with 16 active electrodes and a sampling rate of 256 Hz. Electrodes were
    placed at: F3, Fz, F4, C3, Cz, C4, CPz, P3, Pz, P4, PO7, PO8, Oz, I1, and I2;
    grounded at AFz and referenced to the earlobe. NOTE: Recordings of user
    “zdvm” for bases 2, 3, 5, and 7 had a sampling rate of 600 Hz.
    The rest of recordings have all a sampling rate of 256 Hz.

    The experimental paradigm was executed using the MEDUSA© software [4],
    with the publicly available application "P-ary c-VEP Speller":
    https://www.medusabci.com/market/pary_cvep/

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
    session, eight MNE arrays are available (six for training, two for testing).

    Due to limitations in the MNE format, both cycle onsets and event onsets
    are stored as annotations, with labels included in their descriptions.
    For example, for a binary m-sequence, the possible annotations are:
        - "cycle_onset": marks the start of a stimulation cycle
        - "0": marks the onset of a 0 (white) event
        - "1": marks the onset of a 1 (black) event
    For other p-ary sequences (e.g., p=7), additional event labels (0–6) are included.

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
            code="MartinezCagigal2023Parycvep",
            interval=(0, 1),    # Don't use this, it depends on the condition
            paradigm="cvep",
            doi="https://doi.org/10.71569/025s-eq10"
        )

    def _get_single_subject_data(self, subject):
        """Return the data of a single subject."""
        file_path_list = self.data_path(subject)

        # Each subject evaluated 5 different conditions, composed of p-ary
        # m-sequences of bases 2, 3, 5, 7, and 11
        sessions = dict()
        for cond in CONDITIONS:
            sessions[cond] = dict()
        user = SUBJECTS[subject - 1]

        # Get the EEG files
        zf = zipfile.ZipFile(file_path_list[0])
        with tempfile.TemporaryDirectory() as tempdir:
            zf.extractall(tempdir)
            # For each base (condition)
            for base in CONDITIONS:
                # Training signals
                train_paths = glob(f"{tempdir}/{user}/*{base}_train*")
                for i, train_path in enumerate(train_paths):
                    try:
                        print(f"> Loading {user}, base {base}, train {i + 1}")
                        sessions[str(base)][f"{i + 1}train"] = \
                            self.__convert_to_mne_format(train_path)
                    except Exception as e:
                        print(f"[EXCEPTION] Cannot convert signal {train_path}."
                              f" More information: {traceback.format_exc()}")
                n = len(train_paths)

                # Load the true labels for testing
                test_labels = glob(f"{tempdir}/{user}/*{base}_labels*")
                assert len(test_labels) == 1
                with open(test_labels[0], 'r', encoding='utf-8') as f:
                    true_labels = [line.strip() for line in f.readlines()]

                # Testing signals
                test_paths = glob(f"{tempdir}/{user}/*{base}_test*")
                assert len(test_paths) == len(true_labels)
                for i, test_path in enumerate(test_paths):
                    try:
                        print(f"> Loading {user}, base {base}, test {i+n+1}")
                        sessions[str(base)][f"{i + n + 1}test"] = \
                            self.__convert_to_mne_format(
                                test_path, true_labels[i])
                    except Exception as e:
                        print(f"[EXCEPTION] Cannot convert signal {test_path}."
                              f" More information: {traceback.format_exc()}")

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
        subject_paths.append(
            dl.data_dl(url, self.code, path, force_update, verbose)
        )

        return subject_paths

    @staticmethod
    def __get_camel_case_labels(ch_labels):
        """ Converts a given list of channel labels to the common camel-case
        format (e.g., 'FPZ' -> 'FPz').
        """
        assert isinstance(ch_labels, list)
        camel_case_labels = []
        for ch in ch_labels:
            new_ch = ch.replace('Z', 'z').replace('H', 'h').replace('FP', 'Fp')
            camel_case_labels.append(new_ch)
        return camel_case_labels

    @staticmethod
    def __trim_unfinished_trial(rec):
        if np.max(rec.cvepspellerdata.cycle_idx) != \
                rec.cvepspellerdata.cycle_idx[-1]:
            last_idx = np.where(rec.cvepspellerdata.cycle_idx == np.max(
                rec.cvepspellerdata.cycle_idx))[0][-1] + 1
            rec.cvepspellerdata.cycle_idx = rec.cvepspellerdata.cycle_idx[:last_idx]
            rec.cvepspellerdata.level_idx = rec.cvepspellerdata.level_idx[:last_idx]
            rec.cvepspellerdata.matrix_idx = rec.cvepspellerdata.matrix_idx[
                                             :last_idx]
            rec.cvepspellerdata.onsets = rec.cvepspellerdata.onsets[:last_idx]
            rec.cvepspellerdata.trial_idx = rec.cvepspellerdata.trial_idx[:last_idx]
            rec.cvepspellerdata.unit_idx = rec.cvepspellerdata.unit_idx[:last_idx]
        return rec

    def __convert_to_mne_format(self, path, true_labels=None):
        # Load in MEDUSA format
        rec = components.Recording.load(path)

        # Trim unfinished trials
        rec = self.__trim_unfinished_trial(rec)
        signal = rec.get_biosignals_with_class_name('EEG')['eeg']

        # Create the info
        ch_names = self.__get_camel_case_labels(signal.channel_set.l_cha)
        ch_types = ['eeg'] * len(ch_names)
        sampling_freq = signal.fs
        meas_date = parser.parse(rec.date)
        info = mne.create_info(
            ch_names=ch_names,
            ch_types=ch_types,
            sfreq=sampling_freq
        )
        info["subject_info"] = {"his_id": str(rec.subject_id)}
        info["description"] = str(rec.recording_id)
        info.set_meas_date(meas_date.replace(tzinfo=timezone.utc))
        info.set_montage('standard_1005', match_case=False, on_missing='warn')

        # Set data
        raw_data = mne.io.RawArray(np.array(signal.signal).T, info,
                                   verbose=False)
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
                    rec.cvepspellerdata.commands_info[m_][c_]['sequence']
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
            bw_onsets = [sample_onsets[o_idx] + i / fps for i in range(
                len(seqs_by_cycle[o_idx]))]
            bw_duration = [1 / fps] * len(bw_onsets)
            bw_desc = seqs_by_cycle[o_idx]
            exp_annotations["onset"] += bw_onsets
            exp_annotations["duration"] += bw_duration
            exp_annotations["description"] += bw_desc

        # Set annotations
        annotations = mne.Annotations(
            onset=exp_annotations["onset"],
            duration=exp_annotations["duration"],
            description=exp_annotations["description"]
        )
        raw_data.set_annotations(annotations)

        return raw_data

if __name__ == "__main__":
    dataset = MartinezCagigal2023P()
    sessions = dataset.get_data(subjects=[1])
