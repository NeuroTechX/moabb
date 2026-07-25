import os
import re
import tempfile
from pathlib import Path

import numpy as np
from mne import Annotations, annotations_from_events, create_info, get_config
from mne.channels import make_standard_montage
from mne.io import RawArray

from moabb.datasets.base import BaseDataset
from moabb.datasets.utils import block_rep
from moabb.utils import _handle_deprecated_kwargs


class FakeDataset(BaseDataset):
    """Fake Dataset for test purpose.

    By default, the dataset has 2 sessions, 10 subjects, and 3 classes.

    .. versionchanged:: 0.4.3
        Added ``annotations`` parameter.

    Parameters
    ----------
    event_list: list or tuple of str
        List of event to generate, default: ("fake1", "fake2", "fake3")
    n_sessions: int, default 2
        Number of session to generate
    n_runs: int, default 2
        Number of runs to generate
    n_subjects: int, default 10
        Number of subject to generate
    paradigm : str
        Defines what sort of dataset this is. Allowed values are
        'p300', 'imagery', and 'ssvep'.
    channels: list or tuple of str
        List of channels to generate, default ("C3", "Cz", "C4")
    duration: float or list of float
        Duration of each run in seconds. If float, same duration for all
        runs. If list, duration for each run.
    n_events: int or list of int
        Number of events per run. If int, same number of events
        for all runs. If list, number of events for each run.
    stim: bool
        If True, pass events through stim channel.
    annotations: bool
        If True, pass events through Annotations.
    """

    def __init__(
        self,
        event_list=("fake1", "fake2", "fake3"),
        n_sessions=2,
        n_runs=2,
        n_subjects=10,
        code="FakeDataset",
        paradigm="imagery",
        channels=("C3", "Cz", "C4"),
        seed=None,
        sfreq=128,
        duration=120,
        n_events=60,
        stim=True,
        annotations=False,
        subjects=None,
        sessions=None,
        *,
        return_all_modalities=False,
        **kwargs,
    ):
        deprecated_renames = {
            "EventList": "event_list",
            "NSessions": "n_sessions",
            "NRuns": "n_runs",
            "NSubjects": "n_subjects",
            "Code": "code",
            "Paradigm": "paradigm",
            "Channels": "channels",
            "Seed": "seed",
            "Sfreq": "sfreq",
            "Duration": "duration",
            "NEvents": "n_events",
            "Stim": "stim",
            "Annotations": "annotations",
            "Subjects": "subjects",
            "Sessions": "sessions",
        }
        resolved = _handle_deprecated_kwargs(kwargs, deprecated_renames, "FakeDataset")
        event_list = resolved.get("event_list", event_list)
        n_sessions = resolved.get("n_sessions", n_sessions)
        n_runs = resolved.get("n_runs", n_runs)
        n_subjects = resolved.get("n_subjects", n_subjects)
        code = resolved.get("code", code)
        paradigm = resolved.get("paradigm", paradigm)
        channels = resolved.get("channels", channels)
        seed = resolved.get("seed", seed)
        sfreq = resolved.get("sfreq", sfreq)
        duration = resolved.get("duration", duration)
        n_events = resolved.get("n_events", n_events)
        stim = resolved.get("stim", stim)
        annotations = resolved.get("annotations", annotations)
        subjects = resolved.get("subjects", subjects)
        sessions = resolved.get("sessions", sessions)

        self.n_sessions = n_sessions
        self.n_runs = n_runs
        self.n_events = n_events if isinstance(n_events, list) else [n_events] * n_runs
        self.duration = duration if isinstance(duration, list) else [duration] * n_runs
        assert len(self.n_events) == n_runs
        assert len(self.duration) == n_runs
        self.sfreq = sfreq
        event_id = {ev: ii + 1 for ii, ev in enumerate(event_list)}
        self.channels = channels
        self.stim = stim
        self.annotations = annotations
        self.seed = seed
        code = (
            f"{code}-{paradigm.lower()}-{n_subjects}-{n_sessions}--"
            f"{'-'.join([str(n) for n in self.n_events])}--"
            f"{'-'.join([str(int(n)) for n in self.duration])}--"
            f"{'-'.join([re.sub('[^A-Za-z0-9]', '', e).lower() for e in event_list])}--"
            f"{'-'.join([c.lower() for c in channels])}"
        )
        super().__init__(
            subjects=list(range(1, n_subjects + 1)),
            sessions_per_subject=n_sessions,
            events=event_id,
            code=code,
            interval=[0, 3],
            paradigm=paradigm,
            selected_subjects=subjects,
            selected_sessions=sessions,
            return_all_modalities=return_all_modalities,
        )
        key = "MNE_DATASETS_{:s}_PATH".format(self.code.upper())
        temp_dir = get_config(key)
        if temp_dir is None or not Path(temp_dir).is_dir():
            temp_dir = tempfile.mkdtemp()
            # Use os.environ instead of mne.set_config to avoid
            # "Setting non-standard config type" warnings.
            # MNE's get_config() checks environment variables first.
            os.environ[key] = temp_dir

    def _get_single_subject_data(self, subject):
        if self.seed is not None:
            np.random.seed(self.seed + subject)
        data = {}
        for session in range(self.n_sessions):
            data[f"{session}"] = {
                f"{ii}": self._generate_raw(n, d)
                for ii, (n, d) in enumerate(zip(self.n_events, self.duration))
            }
        return data

    def _generate_events(self, n_events, duration):
        start = max(0, int(self.interval[0] * self.sfreq)) + 1
        stop = (
            min(
                int((duration - self.interval[1]) * self.sfreq),
                int(duration * self.sfreq),
            )
            - 1
        )
        onset = np.linspace(start, stop, n_events)
        events = np.zeros((n_events, 3), dtype="int32")
        events[:, 0] = onset
        for ii, ev in enumerate(self.event_id):
            events[ii :: len(self.event_id), 2] = self.event_id[ev]
        return events

    def _generate_raw(self, n_events, duration):
        montage = make_standard_montage("standard_1005")
        sfreq = self.sfreq
        eeg_data = 2e-5 * np.random.randn(int(duration * sfreq), len(self.channels))
        events = self._generate_events(n_events, duration)
        ch_types = ["eeg"] * len(self.channels)
        ch_names = list(self.channels)

        if self.stim:
            y = np.zeros(eeg_data.shape[0])
            y[events[:, 0]] = events[:, 2]
            ch_types += ["stim"]
            ch_names += ["stim"]
            eeg_data = np.c_[eeg_data, y]

        info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        raw = RawArray(data=eeg_data.T, info=info, verbose=False)
        raw.set_montage(montage)

        if self.annotations:
            event_desc = {v: k for k, v in self.event_id.items()}
            if len(events) != 0:
                annotations = annotations_from_events(
                    events, sfreq=sfreq, event_desc=event_desc
                )
                annotations.set_durations(self.interval[1] - self.interval[0])
            else:
                annotations = Annotations([], [], [])
            raw.set_annotations(annotations)

        return raw

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        pass


class FakeVirtualRealityDataset(FakeDataset):
    """Fake Cattan2019_VR dataset for test purpose.

    .. versionadded:: 0.5.0
    """

    def __init__(self, seed=None, subjects=None, sessions=None, **kwargs):
        deprecated_renames = {
            "Seed": "seed",
            "Subjects": "subjects",
            "Sessions": "sessions",
        }
        resolved = _handle_deprecated_kwargs(
            kwargs, deprecated_renames, "FakeVirtualRealityDataset"
        )
        seed = resolved.get("seed", seed)
        subjects = resolved.get("subjects", subjects)
        sessions = resolved.get("sessions", sessions)

        self.n_blocks = 5
        self.n_repetitions = 12
        self.n_events_rep = [60] * self.n_repetitions
        self.duration_rep = [120] * self.n_repetitions
        super().__init__(
            n_sessions=1,
            n_runs=self.n_blocks * self.n_repetitions,
            n_subjects=21,
            code="FakeVirtualRealityDataset",
            event_list={"Target": 2, "NonTarget": 1},
            paradigm="p300",
            seed=seed,
            duration=self.duration_rep * self.n_blocks,
            n_events=self.n_events_rep * self.n_blocks,
            stim=True,
            annotations=False,
            subjects=subjects,
            sessions=sessions,
        )

    def _get_single_subject_data(self, subject):
        if self.seed is not None:
            np.random.seed(self.seed + subject)
        data = {}
        for session in range(self.n_sessions):
            data[f"{session}"] = {}
            for block in range(self.n_blocks):
                for repetition, (n, d) in enumerate(
                    zip(self.n_events_rep, self.duration_rep)
                ):
                    data[f"{session}"][
                        block_rep(block, repetition, self.n_repetitions)
                    ] = self._generate_raw(n, d)
        return data

    def _block_rep(self, block, repetition):
        return block_rep(block, repetition, self.n_repetitions)
