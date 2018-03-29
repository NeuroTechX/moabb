"""
Base class for a dataset
"""
import abc


class BaseDataset(metaclass=abc.ABCMeta):
    """Base dataset"""

    def __init__(self, subjects, sessions_per_subject, events, code, interval,
                 paradigm):

        if not isinstance(subjects, list):
            raise(ValueError("subjects must be a list"))

        self.subject_list = subjects
        self.n_sessions = sessions_per_subject
        self.event_id = events
        self.code = code
        self.interval = interval
        self.paradigm = paradigm

    def get_data(self, subjects=None):
        """
        Return the data correspoonding to a list of subjects.

        The returned data is a dictionary with the folowing structure

        data = {'subject_id' :
                    {'session_id':
                        {'run_id': raw}
                    }
                }

        subjects are on top, then we have sessions, then runs.
        A sessions is a recording done in a single day, without removing the
        EEG cap. A session is constitued of at least one run. A run is a single
        contigous recording. Some dataset break session in multiple runs.

        parameters
        ----------
        subjects: List of int
            List of subject number

        returns
        -------
        data: Dict
            dict containing the raw data
        """
        data = []

        if subjects is None:
            subjects = self.subject_list

        if not isinstance(subjects, list):
            raise(ValueError('subjects must be a list'))

        data = dict()
        for subject in subjects:
            if subject not in self.subject_list:
                raise ValueError('Invalid subject {:s} given'.format(subject))
            data[subject] = self._get_single_subject_data(subject)

        return data

    @abc.abstractmethod
    def _get_single_subject_data(self, subject):
        """
        Return the data of a single subject

        The returned data is a dictionary with the folowing structure

        data = {'session_id':
                    {'run_id': raw}
                }

        parameters
        ----------
        subject: int
            subject number

        returns
        -------
        data: Dict
            dict containing the raw data
        """
        pass
