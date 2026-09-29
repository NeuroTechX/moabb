"""Provider routing for loaders that already consume an OpenNeuro BIDS release.

Unlike converted deposits, these public mirrors store the loader's input in
raw BIDS, not sourcedata. Keep native parsing and subject/run selection intact.
"""

import json
import warnings
from pathlib import Path

import numpy as np

from moabb.datasets.download import NemarDownloadError
from moabb.utils import get_download_provider


def drop_native_stim(raw):
    """Drop recorded trigger channels; the loader rebuilds STIM from its events."""
    kinds = zip(raw.ch_names, raw.get_channel_types())
    return raw.drop_channels([ch for ch, kind in kinds if kind == "stim"])


def relabel_annotations(raw, mapping):
    """Rename annotation descriptions; unmapped ones never match ``event_id``."""
    raw.annotations.description = np.array(
        [mapping.get(desc, desc) for desc in raw.annotations.description], dtype=str
    )


def write_dataset_description(bids_root, name, bids_version, doi, authors):
    """Write the minimal ``dataset_description.json`` mne-bids needs, if missing."""
    dd_path = Path(bids_root) / "dataset_description.json"
    if not dd_path.exists():
        description = {
            "Name": name,
            "BIDSVersion": bids_version,
            "License": "CC0",
            "Authors": list(authors),
            "DatasetDOI": doi,
        }
        dd_path.write_text(json.dumps(description, indent=2))


class OpenNeuroMirrorMixin:
    """Fetch a verified NEMAR raw mirror, with explicit upstream fallback."""

    nemar_bids_filters = {"scope": "raw", "datatype": "eeg"}

    def _prefetch_nemar_sourcedata(self, subjects, verbose=None):
        # data_path/_download_subject fetch the actual raw BIDS input instead.
        return None

    def download(
        self,
        subject_list=None,
        path=None,
        force_update=False,
        update_path=None,
        accept=False,
        verbose=None,
    ):
        for subject in self.subject_list if subject_list is None else subject_list:
            self.data_path(subject, path, force_update, update_path, verbose)

    def _mirror_root(self, subject, path, force_update, update_path, verbose):
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")
        provider = get_download_provider()
        if provider == "upstream":
            return None
        try:
            return self._download_nemar(subject, path, force_update, update_path, verbose)
        except NemarDownloadError as exc:
            if provider == "nemar":
                raise
            warnings.warn(
                f"Could not fetch {self.nemar_id}; using its OpenNeuro source: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
