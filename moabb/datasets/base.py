"""
Base class for a dataset
"""


class BaseDataset():
    """Base dataset"""

    def __init__(self):
        pass

    def get_subject_list(self):
        """return the list of subjects"""
        return self.subject_list

    def get_data(self, subjects):
        """return data for a (list of) subject(s)"""
        pass

    def get_events_id(self):
        """return event ids"""
        return self.events_id

    def get_name(self):
        """return name of the dataset"""
        return self.name
