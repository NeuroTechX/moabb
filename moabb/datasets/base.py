"""
Base class for a dataset
"""
import abc


class BaseDataset(metaclass=abc.ABCMeta):
    """Base dataset"""

    def __init__(self, subjects, sessions_per_subject, events, code, interval, paradigm):
        self.subject_list = subjects
        self.n_sessions = sessions_per_subject
        self.event_id = events
        self.selected_events = None
        self.code = code
        self.interval = interval
        self.paradigm = paradigm

    def get_data(self, subjects=None, stack_sessions=False):
        """return data for a (list of) subject(s)
        If sessions are not stacked, return each session as a separate dataset"""
        data = []
        if type(subjects) is int:
            subjects = [subjects]
        elif subjects is None:
            subjects = self.subject_list
        for subject in subjects:
            if subject not in self.subject_list:
                raise ValueError('Invalid subject {:s} given'.format(subject))
            data.extend(self._get_single_subject_data(subject, stack_sessions))
        return data

    def download(self, path=None, force_update=False,
                 update_path=None, verbose=None):
        """Download all data from the dataset.

        This function is only usefull to download all the dataset at once.


        Parameters
        ----------
        path : None | str
            Location of where to look for the data storing location.
            If None, the environment variable or config parameter
            ``MNE_DATASETS_(dataset)_PATH`` is used. If it doesn't exist, the
            "~/mne_data" directory is used. If the dataset
            is not found under the given path, the data
            will be automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python
            config to the given path. If None, the user is prompted.
        verbose : bool, str, int, or None
            If not None, override default verbose level
            (see :func:`mne.verbose`).
        """
        for subject in self.subject_list:
            self.data_path(subject=subject, path=path,
                           force_update=force_update,
                           update_path=update_path, verbose=verbose)

    @abc.abstractmethod
    def _get_single_subject_data(self, subject, stack_sessions):
        """get data from a single subject."""
        pass

    @abc.abstractmethod
    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None):
        """Get path to local copy of a subject data.

        Parameters
        ----------
        subject : int
            Number of subject to use
        path : None | str
            Location of where to look for the data storing location.
            If None, the environment variable or config parameter
            ``MNE_DATASETS_(dataset)_PATH`` is used. If it doesn't exist, the
            "~/mne_data" directory is used. If the dataset
            is not found under the given path, the data
            will be automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python
            config to the given path. If None, the user is prompted.
        verbose : bool, str, int, or None
            If not None, override default verbose level
            (see :func:`mne.verbose`).

        Returns
        -------
        path : list of str
            Local path to the given data file. This path is contained inside a
            list of length one, for compatibility.
        """  # noqa: E501
        pass
