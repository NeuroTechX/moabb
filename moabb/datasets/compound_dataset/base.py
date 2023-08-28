"""Build a custom dataset using subjects from other datasets."""

from ..base import BaseDataset


class CompoundDataset(BaseDataset):
    """CompoundDataset class.
    With this dataset, you can merge different dataset
    by selecting among subjects in all datasets
    to build a custom dataset.


    Parameters
    ----------
    subjects_list: List[Union[tuple, CompoundDataset]]
        A list of subject or CompoundDataset (exclusive).
        Example, with a list of selected subject:
        [
            (bi2013(), 1, "0", "0")   # dataset, subject 1, session 0, run 0
            (bi2014(), 1, "0", None)  # dataset, subject 1, session 0, all runs
        ]
        Example of building a dataset compounded of CompoundDatasets:
        [
            CompoundDataset(subjects_list1),
            CompoundDataset(subjects_list2)
        ]

    sessions_per_subject: int
        Number of sessions per subject (if varying, take minimum)

    events: dict of strings
        String codes for events matched with labels in the stim channel.
        See `BaseDataset`.

    code: string
        Unique identifier for dataset, used in all plots

    interval: list with 2 entries
        See `BaseDataset`.

    paradigm: ['p300','imagery', 'ssvep', 'rstate']
        Defines what sort of dataset this is
    """

    def __init__(
        self, subjects_list: list, events: dict, code: str, interval: list, paradigm: str
    ):
        self._set_subjects_list(subjects_list)
        super().__init__(
            subjects=list(range(1, self.count + 1)),
            sessions_per_subject=self._get_sessions_per_subject(),
            events=events,
            code=code,
            interval=interval,
            paradigm=paradigm,
        )

    @property
    def count(self):
        return len(self.subjects_list)

    def _get_sessions_per_subject(self):
        n_sessions = -1
        for value in self.subjects_list:
            sessions = value[2]
            size = len(sessions) if isinstance(sessions, list) else 1
            if sessions is None:
                dataset = value[0]
                size = dataset.n_sessions
            if n_sessions == -1:
                n_sessions = size
            else:
                n_sessions = min(n_sessions, size)
        return n_sessions

    def _set_subjects_list(self, subjects_list: list):
        if isinstance(subjects_list[0], tuple):
            self.subjects_list = subjects_list
        else:
            self.subjects_list = []
            for compoundDataset in subjects_list:
                self.subjects_list.extend(compoundDataset.subjects_list)

    def _get_single_subject_data(self, shopped_subject):
        """Return data for a single subject."""
        dataset, subject, sessions, runs = self.subjects_list[shopped_subject - 1]
        subject_data = dataset._get_single_subject_data(subject)
        if sessions is None:
            return subject_data
        elif isinstance(sessions, list):
            sessions_data = {f"{session}": subject_data[session] for session in sessions}
        else:
            sessions_data = {f"{sessions}": subject_data[sessions]}

        if runs is None:
            return sessions_data
        elif isinstance(runs, list):
            for session in sessions_data.keys():
                sessions_data[session] = {
                    f"{run}": sessions_data[session][run] for run in runs
                }
            return sessions_data
        else:
            for session in sessions_data.keys():
                sessions_data[session] = {f"{runs}": sessions_data[session][runs]}
            return sessions_data

    def data_path(
        self,
        shopped_subject,
        path=None,
        force_update=False,
        update_path=None,
        verbose=None,
    ):
        dataset, subject, _, _ = self.subjects_list[shopped_subject - 1]
        path = dataset.data_path(subject)
        return path
