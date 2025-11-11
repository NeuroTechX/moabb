"""Utils for easy database selection."""

from __future__ import annotations

import abc
import inspect
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import mne_bids
import numpy as np
from mne import create_info
from mne.io import RawArray

import moabb.datasets as db
from moabb.analysis.plotting import (
    _get_dataset_parameters,
    dataset_bubble_plot,
    get_dataset_area,
)
from moabb.datasets.base import BaseDataset
from moabb.utils import aliases_list


logger = logging.getLogger(__name__)

dataset_list = []
dataset_dict = {}


def _init_dataset():

    for ds in inspect.getmembers(db, inspect.isclass):
        if issubclass(ds[1], BaseDataset):
            dataset_list.append(ds[1])

    dataset_class = {
        dataset.__name__: dataset
        for dataset in dataset_list
        if dataset.__name__ not in list(zip(*aliases_list))[0]
    }

    if not dataset_dict:
        dataset_dict.update(dataset_class)


def dataset_search(  # noqa: C901
    paradigm=None,
    multi_session=False,
    events=None,
    has_all_events=False,
    interval=None,
    min_subjects=1,
    channels=(),
):
    """Returns a list of datasets that match a given criteria.

    Parameters
    ----------
    paradigm: str | None
        'imagery', 'p300', 'ssvep', 'cvep', None

    multi_session: bool
        if True only returns datasets with more than one session per subject.
        If False return all

    events: list of strings
        events to select

    has_all_events: bool
        skip datasets that don't have all events in events

    interval:
        Length of motor imagery interval, in seconds. Only used in imagery
        paradigm

    min_subjects: int,
        minimum subjects in dataset

    channels: list of str
        list or set of channels
    """
    if len(dataset_list) == 0:
        _init_dataset()

    if not dataset_dict:
        _init_dataset()

    deprecated_names, _, _ = zip(*aliases_list)

    channels = set(channels)
    out_data = []
    if events is not None and has_all_events:
        n_classes = len(events)
    else:
        n_classes = None
    assert paradigm in ["imagery", "p300", "ssvep", "cvep", None]

    for type_d in dataset_list:
        if type_d.__name__ in deprecated_names:
            continue

        d = type_d()
        skip_dataset = False
        if multi_session and d.n_sessions < 2:
            continue

        if len(d.subject_list) < min_subjects:
            continue

        if paradigm is not None and paradigm != d.paradigm:
            continue

        if interval is not None and d.interval[1] - d.interval[0] < interval:
            continue

        keep_event_dict = {}
        if events is None:
            keep_event_dict = d.event_id.copy()
        else:
            n_events = 0
            for e in events:
                if n_classes is not None:
                    if n_events == n_classes:
                        break
                if e in d.event_id.keys():
                    keep_event_dict[e] = d.event_id[e]
                    n_events += 1
                else:
                    if has_all_events:
                        skip_dataset = True
        if keep_event_dict and not skip_dataset:
            if len(channels) > 0:
                s1 = d.get_data([1])[1]
                sess1 = s1[list(s1.keys())[0]]
                raw = sess1[list(sess1.keys())[0]]
                raw.pick_types(eeg=True)
                if channels <= set(raw.info["ch_names"]):
                    out_data.append(d)
            else:
                out_data.append(d)
    return out_data


def find_intersecting_channels(datasets, verbose=False):
    """Given a list of dataset instances return a list of channels shared by
    all datasets. Skip datasets which have 0 overlap with the others.

    returns: set of common channels, list of datasets with valid channels
    """
    allchans = set()
    dset_chans = []
    keep_datasets = []
    for d in datasets:
        logger.info("Searching dataset: {:s}".format(type(d).__name__))
        s1 = d.get_data([1])[1]
        sess1 = s1[list(s1.keys())[0]]
        raw = sess1[list(sess1.keys())[0]]
        raw.pick_types(eeg=True)
        processed = []
        for ch in raw.info["ch_names"]:
            ch = ch.upper()
            if ch.find("EEG") == -1:
                # TODO: less hacky way of finding poorly labeled datasets
                processed.append(ch)
        allchans.update(processed)
        if len(processed) > 0:
            if verbose:
                logger.info("Found EEG channels: {}".format(processed))
            dset_chans.append(processed)
            keep_datasets.append(d)
        else:
            logger.warning(
                "Dataset {:s} has no recognizable EEG channels".format(type(d).__name__)
            )  # noqa
    allchans.intersection_update(*dset_chans)
    allchans = [s.replace("Z", "z") for s in allchans]
    return allchans, keep_datasets


def _download_all(update_path=True, verbose=None):
    """Download all data.

    This function is mainly used to generate the data cache.
    """

    # iterate over dataset
    for ds in dataset_list:
        # call download
        ds().download(update_path=True, verbose=verbose, accept=True)


def block_rep(block: int, rep: int, n_rep: int):
    idx = block * n_rep + rep
    return f"{idx}block{block}rep{rep}"


def blocks_reps(blocks: list, reps: list, n_rep: int):
    return [block_rep(b, r, n_rep) for b in blocks for r in reps]


def add_stim_channel_trial(raw, onsets, labels, offset=200, ch_name="stim_trial"):
    """
    Add a stimulus channel with trial onsets and their labels.

    Parameters
    ----------
    raw: mne.Raw
        The raw object to add the stimulus channel to.
    onsets: List | np.ndarray
        The onsets of the trials in sample numbers.
    labels: List | np.ndarray
        The labels of the trials.
    offset: int (default: 200)
        The integer value to start markers with. For instance, if 200, then label 0 will be marker 200, label 1
        will be marker 201, etc.
    ch_name: str (default: "stim_trial")
        The name of the added stimulus channel.

    Returns
    -------
    mne.Raw
        The raw object with the added stimulus channel.

    Notes
    -----
    .. versionadded:: 1.1.0
    """
    stim_chan = np.zeros((1, len(raw)))
    for onset, label in zip(onsets, labels):
        stim_chan[0, onset] = offset + label
    info = create_info(
        ch_names=[ch_name],
        ch_types=["stim"],
        sfreq=raw.info["sfreq"],
        verbose=False,
    )
    raw = raw.add_channels([RawArray(data=stim_chan, info=info, verbose=False)])
    return raw


def add_stim_channel_epoch(
    raw,
    onsets,
    labels,
    codes=None,
    presentation_rate=None,
    offset=100,
    ch_name="stim_epoch",
):
    """
    Add a stimulus channel with epoch onsets and their labels, which are the values of the presented code for each
    of the trials.

    Parameters
    ----------
    raw: mne.Raw
        The raw object to add the stimulus channel to.
    onsets: List | np.ndarray
        The onsets of the trials in sample numbers.
    labels: List | np.ndarray
        The labels of the trials.
    codes: np.ndarray (default: None)
        The codebook containing each presented code of shape (nr_bits, nr_codes), sampled at the presentation rate.
        If None, the labels information is used directly.
    presentation_rate: int (default: None):
        The presentation rate (e.g., frame rate) at which the codes were presented in Hz.
        If None, the raw object's sampling frequency is used.
    offset: int (default: 100)
        The integer value to start markers with. For instance, if 100, then label 0 will be marker 100, label 1
        will be marker 101, etc.
    ch_name: str (default: "stim_epoch")
        The name of the added stimulus channel.

    Returns
    -------
    mne.Raw
        The raw object with the added stimulus channel.

    Notes
    -----
    .. versionadded:: 1.1.0
    """
    if presentation_rate is None:
        presentation_rate = raw.info["sfreq"]
    stim_chan = np.zeros((1, len(raw)))
    for onset, label in zip(onsets, labels):
        if codes is None:
            stim_chan[0, int(onset * presentation_rate)] = offset + label
        else:
            idx = np.round(
                onset + np.arange(codes.shape[0]) / presentation_rate * raw.info["sfreq"]
            ).astype("int")
            stim_chan[0, idx] = offset + codes[:, label]

    info = create_info(
        ch_names=[ch_name],
        ch_types=["stim"],
        sfreq=raw.info["sfreq"],
        verbose=False,
    )
    raw = raw.add_channels([RawArray(data=stim_chan, info=info, verbose=False)])
    return raw


def stim_channels_with_selected_ids(
    raw: mne.io.BaseRaw, desired_event_id: dict, stim_channel_name="STIM"
):
    """
    Add a stimulus channel with filtering and renaming based on events_ids.

    Parameters
    ----------
    raw: mne.Raw
        The raw object to add the stimulus channel to.
    desired_event_id: dict
        Dictionary with events
    """

    # Get events using the consistent event_id mapping
    events, _ = mne.events_from_annotations(raw, event_id=desired_event_id)

    # Filter the events array to include only desired events
    desired_event_ids = list(desired_event_id.values())
    filtered_events = events[np.isin(events[:, 2], desired_event_ids)]

    # Create annotations from filtered events using the inverted mapping
    event_desc = {v: k for k, v in desired_event_id.items()}
    annot_from_events = mne.annotations_from_events(
        events=filtered_events,
        event_desc=event_desc,
        sfreq=raw.info["sfreq"],
        orig_time=raw.info["meas_date"],
    )
    raw.set_annotations(annot_from_events)

    # Create the stim channel data array
    stim_channs = np.zeros((1, raw.n_times))
    for event in filtered_events:
        sample_index = event[0]
        event_code = event[2]  # Consistent event IDs
        stim_channs[0, sample_index] = event_code

    # Create the stim channel and add it to raw

    stim_info = mne.create_info(
        [stim_channel_name], sfreq=raw.info["sfreq"], ch_types=["stim"]
    )
    stim_raw = mne.io.RawArray(stim_channs, stim_info, verbose=False)
    raw_with_stim = raw.copy().add_channels([stim_raw], force_update_info=True)

    return raw_with_stim


def bids_metainfo(bids_path: Path) -> dict:
    """Create metadata for the BIDS dataset.

    To allow lazy loading of the metadata, we store the metadata in a JSON file
    in the root of the BIDS dataset.

    Parameters
    ----------
    bids_path : Path
        The path to the BIDS dataset.
    """
    json_data = {}

    paths = mne_bids.find_matching_paths(
        root=bids_path,
        datatypes="eeg",
    )

    for path in paths:
        uid = path.fpath.name
        json_data[uid] = path.entities
        json_data[uid]["fpath"] = str(path.fpath)

    return json_data


class _BubbleChart:
    def __init__(self, area, bubble_spacing=0.0):
        """
        Setup for bubble collapse.

        From https://matplotlib.org/stable/gallery/misc/packed_bubbles.html

        Parameters
        ----------
        area : array-like
            Area of the bubbles.
        bubble_spacing : float, default: 0
            Minimal spacing between bubbles after collapsing.

        Notes
        -----
        If "area" is sorted, the results might look weird.
        """
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[: len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[: len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3])

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0], bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return np.argmin(distance, keepdims=True)

    def collapse(self, n_iterations=50):
        """
        Move bubbles to the center of mass.

        Parameters
        ----------
        n_iterations : int, default: 50
            Number of moves to perform.
        """
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = self.bubbles[i, :2] + orth * self.step_dist
                        new_point2 = self.bubbles[i, :2] - orth * self.step_dist
                        dist1 = self.center_distance(self.com, np.array([new_point1]))
                        dist2 = self.center_distance(self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def get_centers(self):
        return self.bubbles[:, :2]


class _BaseDatasetPlotter:
    def __init__(self, datasets, meta_gap, kwargs, n_col=None):
        self.datasets = datasets = (
            datasets
            if datasets is not None
            else sorted(
                [dataset() for dataset in dataset_list if "Fake" not in dataset.__name__],
                key=lambda x: x.__class__.__name__,
            )
        )
        areas_list = []
        for d in datasets:
            if isinstance(d, dict):
                n_subjects = d["n_subjects"]
                n_sessions = d["n_sessions"]
                n_trials = d["n_trials"]
                trial_len = d["trial_len"]
            else:
                _, _, n_subjects, n_sessions, n_trials, trial_len = (
                    _get_dataset_parameters(d)
                )
            areas_list.append(
                get_dataset_area(
                    n_subjects=n_subjects,
                    n_sessions=n_sessions,
                    n_trials=n_trials,
                    trial_len=trial_len,
                )
            )
        self.areas = np.array(areas_list)
        self.radii = np.sqrt(self.areas / np.pi)
        self.meta_gap = meta_gap
        self.kwargs = kwargs
        self.n_col = n_col

    @abc.abstractmethod
    def _get_centers(self) -> np.ndarray:
        pass

    def plot(self):

        centers = self._get_centers()

        rm = self.radii + self.meta_gap
        xlim = ((centers[:, 0] - rm).min(), (centers[:, 0] + rm).max() + self.meta_gap)
        ylim = ((centers[:, 1] - rm).min(), (centers[:, 1] + rm).max())
        lx, ly = xlim[1] - self.meta_gap, ylim[0] + self.meta_gap

        factor = 0.05
        # No need to expose the factor parameter as users can already
        # tune the scale and fontsize arguments, which will have the same effect.
        fig, ax = plt.subplots(
            subplot_kw={"aspect": "equal"},
            figsize=(factor * (xlim[1] - xlim[0]), factor * (ylim[1] - ylim[0])),
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        for i, dataset in enumerate(self.datasets):
            dataset_kwargs = (
                dataset if isinstance(dataset, dict) else {"dataset": dataset}
            )
            dataset_bubble_plot(
                **dataset_kwargs,
                ax=ax,
                center=centers[i],
                legend=i == len(self.datasets) - 1,
                legend_position=(lx, ly),
                scale_ax=False,
                **self.kwargs,
            )
        return fig


class _ClusterDatasetPlotter(_BaseDatasetPlotter):
    def _get_centers(self):
        bubble_chart = _BubbleChart(self.areas, bubble_spacing=self.meta_gap)
        bubble_chart.collapse(n_iterations=100)
        return bubble_chart.get_centers()


class _GridDatasetPlotter(_BaseDatasetPlotter):
    def _get_centers(self):
        assert isinstance(self.n_col, int)
        height = self.radii.max() * 2
        i = np.arange(len(self.datasets))
        x = i % self.n_col
        y = -(i // self.n_col)
        return np.stack([x, y], axis=1) * height


def plot_datasets_grid(
    datasets: list[BaseDataset | dict] | None = None,
    n_col: int = 10,
    margin: float = 10.0,
    **kwargs,
):
    """Plots all the MOABB datasets in one figure, distributed on a grid.

    This uses the :func:`~moabb.analysis.plotting.dataset_bubble_plot` function to
    plot the datasets.
    The datasets are sorted in alphabetical order and
    plotted in a grid with n_col columns.

    Parameters
    ----------
    datasets: list[BaseDataset | dict] | None
        List of datasets to plot. If None, all datasets are plotted.
        If an element of the list is a dictionary, it is assumed to
        have the following keys:

            dataset_name: str
                Name of the dataset.
            paradigm: str
                Paradigm of the dataset (e.g., 'imagery', 'p300').
            n_subjects: int
                Number of subjects in the dataset.
            n_sessions: int
                Number of sessions in the dataset.
            n_trials: int
                Number of trials in the dataset.
            trial_len: float
                Length of each trial in seconds.

    n_col: int
        Number of columns in the figure.
    margin: float
        Margin around the plots.
    kwargs: dict
        Additional arguments to pass to the dataset_bubble_plot function.

    Returns
    -------
    fig: Figure
        Pyplot handle
    """
    plotter = _GridDatasetPlotter(
        datasets=datasets,
        meta_gap=margin,
        n_col=n_col,
        kwargs=kwargs,
    )
    return plotter.plot()


def plot_datasets_cluster(
    datasets: list[BaseDataset | dict] | None = None,
    meta_gap: float = 10.0,
    **kwargs,
):
    """Plots all the MOABB datasets in one figure, grouped in one cluster.

    This uses the :func:`~moabb.analysis.plotting.dataset_bubble_plot` function to
    plot the datasets.
    The datasets are sorted in alphabetical order and
    plotted in a grid with n_col columns.

    Parameters
    ----------
    datasets: list[BaseDataset | dict] | None
        List of datasets to plot. If None, all datasets are plotted.
        If an element of the list is a dictionary, it is assumed to
        have the following keys:

            dataset_name: str
                Name of the dataset.
            paradigm: str
                Paradigm of the dataset (e.g., 'imagery', 'p300').
            n_subjects: int
                Number of subjects in the dataset.
            n_sessions: int
                Number of sessions in the dataset.
            n_trials: int
                Number of trials in the dataset.
            trial_len: float
                Length of each trial in seconds.

    meta_gap: float
        Gap between the different datasets in the cluster.
    kwargs: dict
        Additional arguments to pass to the dataset_bubble_plot function.

    Returns
    -------
    fig: Figure
        Pyplot handle
    """
    plotter = _ClusterDatasetPlotter(
        datasets=datasets,
        meta_gap=meta_gap,
        kwargs=kwargs,
    )
    return plotter.plot()
