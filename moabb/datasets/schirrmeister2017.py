import logging
import re

import h5py
import mne

# import requests
import numpy as np

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


log = logging.getLogger(__name__)

GIN_URL = (
    "https://web.gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data"  # noqa
)


class Schirrmeister2017(BaseDataset):
    """High-gamma dataset discribed in Schirrmeister et al. 2017

    Dataset from [1]_

    Our “High-Gamma Dataset” is a 128-electrode dataset (of which we later only use
    44 sensors covering the motor cortex, (see Section 2.7.1), obtained from 14
    healthy subjects (6 female, 2 left-handed, age 27.2 ± 3.6 (mean ± std)) with
    roughly 1000 (963.1 ± 150.9, mean ± std) four-second trials of executed
    movements divided into 13 runs per subject.  The four classes of movements were
    movements of either the left hand, the right hand, both feet, and rest (no
    movement, but same type of visual cue as for the other classes).  The training
    set consists of the approx.  880 trials of all runs except the last two runs,
    the test set of the approx.  160 trials of the last 2 runs.  This dataset was
    acquired in an EEG lab optimized for non-invasive detection of high- frequency
    movement-related EEG components (Ball et al., 2008; Darvas et al., 2010).

    Depending on the direction of a gray arrow that was shown on black back-
    ground, the subjects had to repetitively clench their toes (downward arrow),
    perform sequential finger-tapping of their left (leftward arrow) or right
    (rightward arrow) hand, or relax (upward arrow).  The movements were selected
    to require little proximal muscular activity while still being complex enough
    to keep subjects in- volved.  Within the 4-s trials, the subjects performed the
    repetitive movements at their own pace, which had to be maintained as long as
    the arrow was showing.  Per run, 80 arrows were displayed for 4 s each, with 3
    to 4 s of continuous random inter-trial interval.  The order of presentation
    was pseudo-randomized, with all four arrows being shown every four trials.
    Ideally 13 runs were performed to collect 260 trials of each movement and rest.
    The stimuli were presented and the data recorded with BCI2000 (Schalk et al.,
    2004).  The experiment was approved by the ethical committee of the University
    of Freiburg.

    References
    ----------

    .. [1] Schirrmeister, Robin Tibor, et al. "Deep learning with convolutional
           neural networks for EEG decoding and visualization." Human brain mapping 38.11
           (2017): 5391-5420.

    """

    def __init__(self):
        super().__init__(
            subjects=list(range(1, 15)),
            sessions_per_subject=1,
            events=dict(right_hand=1, left_hand=2, rest=3, feet=4),
            code="Schirrmeister2017",
            interval=[0, 4],
            paradigm="imagery",
            doi="10.1002/hbm.23730",
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))

        def _url(prefix):
            return "/".join([GIN_URL, prefix, "{:d}.mat".format(subject)])

        return [
            dl.data_dl(_url(t), "SCHIRRMEISTER2017", path, force_update, verbose)
            for t in ["train", "test"]
        ]

    def _get_single_subject_data(self, subject):
        train, test = [BBCIDataset(path) for path in self.data_path(subject)]
        sessions = {}
        sessions["session_1"] = {"train": train.load(), "test": test.load()}
        return sessions


class BBCIDataset(object):
    """
    Loader class for files created by saving BBCI files in matlab (make
    sure to save with '-v7.3' in matlab, see
    https://de.mathworks.com/help/matlab/import_export/mat-file-versions.html#buk6i87
    )
    Parameters
    ----------
    filename: str
    load_sensor_names: list of str, optional
        Also speeds up loading if you only load some sensors.
        None means load all sensors.

    Copyright Robin Schirrmeister, 2017
    Altered by Vinay Jayaram, 2018
    """

    def __init__(self, filename, load_sensor_names=None):
        self.__dict__.update(locals())
        del self.self

    def load(self):
        cnt = self._load_continuous_signal()
        cnt = self._add_markers(cnt)
        return cnt

    def _load_continuous_signal(self):
        wanted_chan_inds, wanted_sensor_names = self._determine_sensors()
        fs = self._determine_samplingrate()
        with h5py.File(self.filename, "r") as h5file:
            samples = int(h5file["nfo"]["T"][0, 0])
            cnt_signal_shape = (samples, len(wanted_chan_inds))
            continuous_signal = np.ones(cnt_signal_shape, dtype=np.float32) * np.nan
            for chan_ind_arr, chan_ind_set in enumerate(wanted_chan_inds):
                # + 1 because matlab/this hdf5-naming logic
                # has 1-based indexing
                # i.e ch1,ch2,....
                chan_set_name = "ch" + str(chan_ind_set + 1)
                # first 0 to unpack into vector, before it is 1xN matrix
                chan_signal = h5file[chan_set_name][
                    :
                ].squeeze()  # already load into memory
                continuous_signal[:, chan_ind_arr] = chan_signal
            assert not np.any(np.isnan(continuous_signal)), "No NaNs expected in signal"

        # Assume we cant know channel type here automatically
        ch_types = ["eeg"] * len(wanted_chan_inds)
        info = mne.create_info(ch_names=wanted_sensor_names, sfreq=fs, ch_types=ch_types)
        # Scale to volts from microvolts, (VJ 19.6.18)
        continuous_signal = continuous_signal * 1e-6
        cnt = mne.io.RawArray(continuous_signal.T, info)
        return cnt

    def _determine_sensors(self):
        all_sensor_names = self.get_all_sensors(self.filename, pattern=None)
        if self.load_sensor_names is None:

            # if no sensor names given, take all EEG-chans
            eeg_sensor_names = all_sensor_names
            eeg_sensor_names = filter(lambda s: not s.startswith("BIP"), eeg_sensor_names)
            eeg_sensor_names = filter(lambda s: not s.startswith("E"), eeg_sensor_names)
            eeg_sensor_names = filter(
                lambda s: not s.startswith("Microphone"), eeg_sensor_names
            )
            eeg_sensor_names = filter(
                lambda s: not s.startswith("Breath"), eeg_sensor_names
            )
            eeg_sensor_names = filter(lambda s: not s.startswith("GSR"), eeg_sensor_names)
            eeg_sensor_names = list(eeg_sensor_names)
            assert len(eeg_sensor_names) in set(
                [128, 64, 32, 16]
            ), "check this code if you have different sensors..."  # noqa
            self.load_sensor_names = eeg_sensor_names
        chan_inds = self._determine_chan_inds(all_sensor_names, self.load_sensor_names)
        return chan_inds, self.load_sensor_names

    def _determine_samplingrate(self):
        with h5py.File(self.filename, "r") as h5file:
            fs = h5file["nfo"]["fs"][0, 0]
            assert isinstance(fs, int) or fs.is_integer()
            fs = int(fs)
        return fs

    @staticmethod
    def _determine_chan_inds(all_sensor_names, sensor_names):
        assert sensor_names is not None
        chan_inds = [all_sensor_names.index(s) for s in sensor_names]
        assert len(chan_inds) == len(sensor_names), "All" "sensors" "should be there."
        # TODO: is it possible for this to fail? the list
        # comp fails first right?
        assert len(set(chan_inds)) == len(chan_inds), "No" "duplicated sensors" "wanted."
        return chan_inds

    @staticmethod
    def get_all_sensors(filename, pattern=None):
        """
        Get all sensors that exist in the given file.

        Parameters
        ----------
        filename: str
        pattern: str, optional
            Only return those sensor names that match the given pattern.
        Returns
        -------
        sensor_names: list of str
            Sensor names that match the pattern or all
            sensor names in the file.
        """
        with h5py.File(filename, "r") as h5file:
            clab_set = h5file["nfo"]["clab"][:].squeeze()
            all_sensor_names = [
                "".join(chr(c.squeeze()) for c in h5file[obj_ref]) for obj_ref in clab_set
            ]
            if pattern is not None:
                all_sensor_names = filter(
                    lambda sname: re.search(pattern, sname), all_sensor_names
                )
        return all_sensor_names

    def _add_markers(self, cnt):
        with h5py.File(self.filename, "r") as h5file:
            event_times_in_ms = h5file["mrk"]["time"][:].squeeze()
            event_classes = h5file["mrk"]["event"]["desc"][:].squeeze().astype(np.int64)

            # Check whether class names known and correct order
            # class_name_set = h5file['nfo']['className'][:].squeeze()
            # all_class_names = [''.join(chr(c) for c in h5file[obj_ref])
            #                    for obj_ref in class_name_set]

        event_times_in_samples = event_times_in_ms * cnt.info["sfreq"] / 1000.0
        event_times_in_samples = np.uint32(np.round(event_times_in_samples))

        # Check if there are markers at the same time
        previous_i_sample = -1
        for i_event, (i_sample, _) in enumerate(
            zip(event_times_in_samples, event_classes)
        ):
            if i_sample == previous_i_sample:
                info = "{:d}: ({:.0f} and {:.0f}).\n".format(
                    i_sample, event_classes[i_event - 1], event_classes[i_event]
                )
                log.warning(
                    f"Same sample has at least two markers.\n{info}Marker codes will be summed."
                )
            previous_i_sample = i_sample

        # Now create stim chan
        stim_chan = np.zeros_like(cnt.get_data()[0])
        for i_sample, id_class in zip(event_times_in_samples, event_classes):
            stim_chan[i_sample] += id_class
        info = mne.create_info(
            ch_names=["STI 014"], sfreq=cnt.info["sfreq"], ch_types=["stim"]
        )
        stim_cnt = mne.io.RawArray(stim_chan[None], info, verbose="WARNING")
        cnt = cnt.add_channels([stim_cnt])
        event_arr = [
            event_times_in_samples,
            [0] * len(event_times_in_samples),
            event_classes,
        ]
        cnt.info["events"] = np.array(event_arr).T
        return cnt
