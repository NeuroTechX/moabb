"""Build a custom dataset using subjects from other datasets."""

from sklearn.pipeline import Pipeline

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

    code: string
        Unique identifier for dataset, used in all plots

    interval: list with 2 entries
        See `BaseDataset`.

    """

    def __init__(self, subjects_list: list, code: str, interval: list):
        self._set_subjects_list(subjects_list)
        dataset, _, _, _ = self.subjects_list[0]
        paradigm = self._get_paradigm()
        super().__init__(
            subjects=list(range(1, self.count + 1)),
            sessions_per_subject=self._get_sessions_per_subject(),
            events=dataset.event_id,
            code=code,
            interval=interval,
            paradigm=paradigm,
        )

    @property
    def datasets(self):
        all_datasets = [entry[0] for entry in self.subjects_list]
        found_flags = set()
        filtered_dataset = []
        for dataset in all_datasets:
            if dataset.code not in found_flags:
                filtered_dataset.append(dataset)
                found_flags.add(dataset.code)
        return filtered_dataset

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

    def _get_paradigm(self):
        dataset, _, _, _ = self.subjects_list[0]
        paradigm = dataset.paradigm
        # Check all of the datasets have the same paradigm
        for i in range(1, len(self.subjects_list)):
            entry = self.subjects_list[i]
            dataset = entry[0]
            assert dataset.paradigm == paradigm
        return paradigm

    def _with_data_origin(self, data: dict, shopped_subject):
        data_origin = self.subjects_list[shopped_subject - 1]

        class dict_with_hidden_key(dict):
            def __getitem__(self, item):
                # ensures data_origin is never accessed when iterating with dict.keys()
                # that would make iterating over runs and sessions failing.
                if item == "data_origin":
                    return data_origin
                else:
                    return super().__getitem__(item)

        return dict_with_hidden_key(data)

    def _get_single_subject_data_using_cache(
        self, shopped_subject, cache_config, process_pipeline
    ):
        # change this compound dataset target event_id to match the one of the hidden dataset
        # as event_id can varies between datasets
        dataset, _, _, _ = self.subjects_list[shopped_subject - 1]
        self.event_id = dataset.event_id

        # regenerate the process_pipeline by overriding all `event_id`
        steps = []
        for step in process_pipeline.steps:
            label, op = step
            if hasattr(op, "event_id"):
                op.event_id = self.event_id
            steps.append((label, op))
        process_pipeline = Pipeline(steps)

        # don't forget to continue on preprocessing by calling super
        data = super()._get_single_subject_data_using_cache(
            shopped_subject, cache_config, process_pipeline
        )
        return self._with_data_origin(data, shopped_subject)

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
