import re
import zipfile
from abc import ABC
from pathlib import Path

import mne
import numpy as np

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


VSPELL_BASE_URL = "https://zenodo.org/record/"
VISUAL_SPELLER_LLP_URL = VSPELL_BASE_URL + "5831826/files/"
VISUAL_SPELLER_MIX_URL = VSPELL_BASE_URL + "5831879/files/"
OPTICAL_MARKER_CODE = 500


class _BaseVisualMatrixSpellerDataset(BaseDataset, ABC):
    def __init__(
        self, src_url, n_subjects, raw_slice_offset, use_blocks_as_sessions=True, **kwargs
    ):
        self.n_channels = 31  # all channels except 5 times x_* CH and EOGvu
        if kwargs["interval"] is None:
            # "Epochs were windowed to [−200, 700] ms relative to the stimulus onset [...]."
            kwargs["interval"] = [-0.2, 0.7]

        super().__init__(
            events=dict(Target=10002, NonTarget=10001),
            paradigm="p300",
            subjects=(np.arange(n_subjects) + 1).tolist(),
            **kwargs,
        )

        self.raw_slice_offset = 2_000 if raw_slice_offset is None else raw_slice_offset
        self._src_url = src_url
        self.use_blocks_as_sessions = use_blocks_as_sessions

    @staticmethod
    def _filename_trial_info_extraction(vhdr_file_path):
        vhdr_file_path = Path(vhdr_file_path)
        vhdr_file_name = vhdr_file_path.name
        run_file_pattern = "^matrixSpeller_Block([0-9]+)_Run([0-9]+)\\.vhdr$"
        vhdr_file_patter_match = re.match(run_file_pattern, vhdr_file_name)

        if not vhdr_file_patter_match:
            # TODO: raise a wild exception?
            print(vhdr_file_path)

        session_name = vhdr_file_path.parent.name
        block_idx = vhdr_file_patter_match.group(1)
        run_idx = vhdr_file_patter_match.group(2)
        return session_name, block_idx, run_idx

    def _get_single_subject_data(self, subject):
        subject_data_vhdr_files = self.data_path(subject)
        sessions = dict()

        for _file_idx, subject_data_vhdr_file in enumerate(subject_data_vhdr_files):
            (
                session_name,
                block_idx,
                run_idx,
            ) = Huebner2017._filename_trial_info_extraction(subject_data_vhdr_file)

            raw_bvr_list = _read_raw_llp_study_data(
                vhdr_fname=subject_data_vhdr_file,
                raw_slice_offset=self.raw_slice_offset,
                verbose=None,
            )

            if self.use_blocks_as_sessions:
                session_name = f"{session_name}_block_{block_idx}"
            else:
                session_name = f"{session_name}"
            if session_name not in sessions.keys():
                sessions[session_name] = dict()
            sessions[session_name][run_idx] = raw_bvr_list[0]

        return sessions

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        url = f"{self._src_url}subject{subject:02d}.zip"
        zipfile_path = Path(dl.data_dl(url, "llp"))
        zipfile_extracted_path = zipfile_path.parent

        subject_dir_path = zipfile_extracted_path / f"subject{subject:02d}"

        if not subject_dir_path.is_dir():
            _BaseVisualMatrixSpellerDataset._extract_data(
                zipfile_extracted_path, zipfile_path
            )

        subject_paths = zipfile_extracted_path / f"subject{subject:02d}"
        subject_paths = subject_paths.glob("matrixSpeller_Block*_Run*.vhdr")
        subject_paths = [str(p) for p in subject_paths]
        return sorted(subject_paths)

    @staticmethod
    def _extract_data(data_dir_extracted_path, data_archive_path):
        zip_ref = zipfile.ZipFile(data_archive_path, "r")
        zip_ref.extractall(data_dir_extracted_path)


class Huebner2017(_BaseVisualMatrixSpellerDataset):
    """
    Learning from label proportions for a visual matrix speller (ERP) dataset from Hübner et al 2017 [1]_.

    .. admonition:: Dataset summary


        ===========  =======  =======  =================  ===============  ===============  ===========
        Name           #Subj    #Chan  #Trials / class    Trials length    Sampling rate      #Sessions
        ===========  =======  =======  =================  ===============  ===============  ===========
        Huebner2017       13       31                     0.9s             1000Hz                     1
        ===========  =======  =======  =================  ===============  ===============  ===========

    **Dataset description**

    The subjects were asked to spell the sentence: “Franzy jagt im komplett verwahrlosten Taxi quer durch Freiburg”.
    The sentence was chosen because it contains each letter used in German at least once. Each subject spelled this
    sentence three times. The stimulus onset asynchrony (SOA) was 250 ms (corresponding to 15 frames on the LCD screen
    utilized) while the stimulus duration was 100 ms (corresponding to 6 frames on the LCD screen utilized). For each
    character, 68 highlighting events occurred and a total of 63 characters were spelled three times. This resulted in
    a total of 68 ⋅ 63 ⋅ 3 = 12852 EEG epochs per subject. Spelling one character took around 25 s including 4 s for
    cueing the current symbol, 17 s for highlighting and 4 s to provide feedback to the user. Assuming a perfect
    decoding, these timing constraints would allow for a maximum spelling speed of 2.4 characters per minute. Fig 1
    shows the complete experimental structure and how LLP is used to reconstruct average target and non-target ERP
    responses.

    Subjects were placed in a chair at 80 cm distance from a 24-inch flat screen. EEG signals from 31 passive Ag/AgCl
    electrodes (EasyCap) were recorded, which were placed approximately equidistantly according to the extended
    10–20 system, and whose impedances were kept below 20 kΩ. All channels were referenced against the nose and the
    ground was at FCz. The signals were registered by multichannel EEG amplifiers (BrainAmp DC, Brain Products) at a
    sampling rate of 1 kHz. To control for vertical ocular movements and eye blinks, we recorded with an EOG electrode
    placed below the right eye and referenced against the EEG channel Fp2 above the eye. In addition, pulse and
    breathing activity were recorded.

    Parameters
    ----------
    interval: array_like
        range/interval in milliseconds in which the brain response/activity relative to an event/stimulus onset lies in.
        Default is set to [-.2, .7].
    raw_slice_offset: int, None
        defines the crop offset in milliseconds before the first and after the last event (target or non-targeet) onset.
        Default None which crops with an offset 2,000 ms.

    References
    ----------
    .. [1] Hübner, D., Verhoeven, T., Schmid, K., Müller, K. R., Tangermann, M., & Kindermans, P. J. (2017)
           Learning from label proportions in brain-computer interfaces: Online unsupervised learning with guarantees.
           PLOS ONE 12(4): e0175856.
           https://doi.org/10.1371/journal.pone.0175856

    .. versionadded:: 0.4.5
    """

    def __init__(self, interval=None, raw_slice_offset=None, use_blocks_as_sessions=True):
        llp_speller_paper_doi = "10.1371/journal.pone.0175856"
        super().__init__(
            src_url=VISUAL_SPELLER_LLP_URL,
            raw_slice_offset=raw_slice_offset,
            n_subjects=13,
            sessions_per_subject=1,  # if varying, take minimum
            code="Visual Speller LLP",
            interval=interval,
            doi=llp_speller_paper_doi,
            use_blocks_as_sessions=use_blocks_as_sessions,
        )


class Huebner2018(_BaseVisualMatrixSpellerDataset):
    """
    Mixture of LLP and EM for a visual matrix speller (ERP) dataset from Hübner et al 2018 [1]_.

    .. admonition:: Dataset summary


        ===========  =======  =======  =================  ===============  ===============  ===========
        Name           #Subj    #Chan  #Trials / class    Trials length    Sampling rate      #Sessions
        ===========  =======  =======  =================  ===============  ===============  ===========
        Huebner2018       12       31                     0.9s             1000Hz                     1
        ===========  =======  =======  =================  ===============  ===============  ===========

    **Dataset description**

    Within a single session, a subject was asked to spell the beginning of a sentence in each of three blocks.The text
    consists of the 35 symbols “Franzy jagt im Taxi quer durch das ”. Each block, one of the three decoding
    algorithms (EM, LLP, MIX) was used in order to guess the attended symbol. The order of the blocks was
    pseudo-randomized over subjects, such that each possible order of the three decoding algorithms was used twice.
    This randomization should reduce systematic biases by order effects or temporal effects, e.g., due to fatigue or
    task-learning.

    A trial describes the process of spelling one character. Each of the 35 trials per block contained 68 highlighting
    events. The stimulus onset asynchrony (SOA) was 250 ms and the stimulus duration was 100 ms leading to an
    interstimulus interval (ISI) of 150 ms.

    Parameters
    ----------
    interval: array_like
        range/interval in milliseconds in which the brain response/activity relative to an event/stimulus onset lies in.
        Default is set to [-.2, .7].
    raw_slice_offset: int, None
        defines the crop offset in milliseconds before the first and after the last event (target or non-targeet) onset.
        Default None which crops with an offset 2,000 ms.

    References
    ----------
    .. [1] Huebner, D., Verhoeven, T., Mueller, K. R., Kindermans, P. J., & Tangermann, M. (2018).
           Unsupervised learning for brain-computer interfaces based on event-related potentials: Review and online comparison [research frontier].
           IEEE Computational Intelligence Magazine, 13(2), 66-77.
           https://doi.org/10.1109/MCI.2018.2807039

    .. versionadded:: 0.4.5
    """

    def __init__(self, interval=None, raw_slice_offset=None, use_blocks_as_sessions=True):
        mix_speller_paper_doi = "10.1109/MCI.2018.2807039"
        super().__init__(
            src_url=VISUAL_SPELLER_MIX_URL,
            raw_slice_offset=raw_slice_offset,
            n_subjects=12,
            sessions_per_subject=1,  # if varying, take minimum
            code="Visual Speller MIX",
            interval=interval,
            doi=mix_speller_paper_doi,
            use_blocks_as_sessions=use_blocks_as_sessions,
        )


def _read_raw_llp_study_data(vhdr_fname, raw_slice_offset, verbose=None):
    """
    Read LLP BVR recordings file. Ignore the different sequence lengths. Just tag event as target or non-target if it
    contains a target or does not contain a target.

    Parameters
    ----------
    vhdr_fname: str
        Path to the EEG header file.
    verbose : bool, int, None
        specify the loglevel.

    Returns
    -------
    raw_object: mne.io.Raw
        the loaded BVR raw object.
    """
    non_scalp_channels = ["EOGvu", "x_EMGl", "x_GSR", "x_Respi", "x_Pulse", "x_Optic"]
    raw_bvr = mne.io.read_raw_brainvision(
        vhdr_fname=vhdr_fname,  # eog='EOGvu',
        misc=non_scalp_channels,
        preload=True,
        verbose=verbose,
    )  # type: mne.io.Raw
    raw_bvr.set_montage("standard_1020")

    events = _parse_events(raw_bvr)

    onset_arr_list, marker_arr_list = _extract_target_non_target_description(events)

    def annotate_and_crop_raw(onset_arr, marker_arr):
        raw = raw_bvr

        raw_annotated = raw.set_annotations(
            _create_annotations_from(marker_arr, onset_arr, raw)
        )

        tmin = max((onset_arr[0] - raw_slice_offset) / 1e3, 0)
        tmax = min((onset_arr[-1] + raw_slice_offset) / 1e3, raw.times[-1])
        return raw_annotated.crop(tmin=tmin, tmax=tmax, include_tmax=True)

    return list(map(annotate_and_crop_raw, onset_arr_list, marker_arr_list))


def _create_annotations_from(marker_arr, onset_arr, raw_bvr):
    default_bvr_marker_duration = raw_bvr.annotations[0]["duration"]

    onset = onset_arr / 1e3  # convert onset in seconds to ms
    durations = np.repeat(default_bvr_marker_duration, len(marker_arr))
    description = list(map(lambda m: f"Stimulus/S {m:3}", marker_arr))
    orig_time = raw_bvr.annotations[0]["orig_time"]
    return mne.Annotations(
        onset=onset, duration=durations, description=description, orig_time=orig_time
    )


def _parse_events(raw_bvr):
    stimulus_pattern = re.compile("(Stimulus/S|Optic/O) *([0-9]+)")

    def parse_marker(desc):
        match = stimulus_pattern.match(desc)
        if match is None:
            return None
        if match.group(1) == "Optic/O":
            return OPTICAL_MARKER_CODE

        return int(match.group(2))

    events, _ = mne.events_from_annotations(
        raw=raw_bvr, event_id=parse_marker, verbose=None
    )
    return events


def _find_single_trial_start_end_idx(events):
    trial_start_end_markers = [21, 22, 10]
    return np.where(np.isin(events[:, 2], trial_start_end_markers))[0]


def _extract_target_non_target_description(events):
    single_trial_start_end_idx = _find_single_trial_start_end_idx(events)

    n_events = single_trial_start_end_idx.size - 1

    onset_arr = np.empty((n_events,), dtype=np.int64)
    marker_arr = np.empty((n_events,), dtype=np.int64)

    broken_events_idx = list()
    for epoch_idx in range(n_events):
        epoch_start_idx = single_trial_start_end_idx[epoch_idx]
        epoch_end_idx = single_trial_start_end_idx[epoch_idx + 1]

        epoch_events = events[epoch_start_idx:epoch_end_idx]

        onset_ms = _find_epoch_onset(epoch_events)
        if onset_ms == -1:
            broken_events_idx.append(epoch_idx)
            continue

        onset_arr[epoch_idx] = onset_ms
        marker_arr[epoch_idx] = int(
            _single_trial_contains_target(epoch_events)
        )  # 1/true if single trial has target

    return [np.delete(onset_arr, broken_events_idx)], [
        np.delete(marker_arr, broken_events_idx)
    ]


def _find_epoch_onset(epoch_events):
    optical_idx = epoch_events[:, 2] == OPTICAL_MARKER_CODE
    stimulus_onset_time = epoch_events[optical_idx, 0]

    def second_optical_is_feedback():
        if stimulus_onset_time.size != 2:
            return False

        stimulus_prior_second_optical_marker = epoch_events[
            np.where(optical_idx)[0][1] - 1, 2
        ]
        return stimulus_prior_second_optical_marker in [50, 51, 11]

    if stimulus_onset_time.size == 1 or second_optical_is_feedback():
        return stimulus_onset_time[0]

    # broken epoch: no true onset found..
    return -1


def _single_trial_contains_target(trial_events):
    trial_markers = trial_events[:, 2]
    return np.any((trial_markers > 100) & (trial_markers <= 142))
