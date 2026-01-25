"""BNCI dataset base helpers."""

from moabb.datasets.base import BaseDataset

from .utils import BNCI_URL


class BNCIBaseDataset(BaseDataset):
    """Base dataset with shared BNCI loading behavior."""

    def __init__(self, *, load_fn, base_url=BNCI_URL, **kwargs):
        self._load_fn = load_fn
        self._base_url = base_url
        super().__init__(**kwargs)

    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
        return self._load_fn(
            subject=subject,
            path=None,
            force_update=False,
            update_path=None,
            base_url=self._base_url,
            only_filenames=False,
            verbose=False,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        """Return paths to data files for a single subject."""
        return self._load_fn(
            subject=subject,
            path=path,
            force_update=force_update,
            update_path=update_path,
            base_url=self._base_url,
            only_filenames=True,
            verbose=verbose,
        )
