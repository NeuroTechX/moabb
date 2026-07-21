"""Wang2026 BCI dataset."""

from moabb.datasets import download as dl
from moabb.datasets.base import BaseDataset


# URL for the data
WANG2026_URL = "https://ndownloader.figshare.com/files/32293995"

class Wang2026(BaseDataset):
    """
    Add description.
    """


    def __init__(self, subjects=None, sessions=None, **kwargs):
        super().__init__(
            subjects=subjects,
            sessions=sessions,
            code="Wang2026",
            interval=(0, None),
            **kwargs,
        )
    
    def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
        """Return the data paths of a single subject.

        Parameters
        ----------
        subject : int
            The subject number to fetch data for.
        path : None | str
            Location of where to look for the data storing location. If None,
            the environment variable or config parameter MNE_(dataset) is used.
            If it doesn’t exist, the “~/mne_data” directory is used. If the
            dataset is not found under the given path, the data
            will be automatically downloaded to the specified folder.
        force_update : bool
            Force update of the dataset even if a local copy exists.
        update_path : bool | None
            If True, set the MNE_DATASETS_(dataset)_PATH in mne-python config
            to the given path.
            If None, the user is prompted.
        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose()).

        Returns
        -------
        list
            A list containing the path to the subject's data file.
        """
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")