"""Dataset-specific raw-mirror routing for the second OpenNeuro batch.

The shared ``OpenNeuroMirrorMixin`` contract (provider policy, SDK raw
selection with a mocked HTTP transport, no sourcedata prefetch) is owned by
``test_openneuro_mirror.py`` in PR #1186. ``test_provider_policy`` below is a
single-case copy kept only so this branch covers its own copy of
``_openneuro_mirror.py`` until #1186 lands; delete it when rebasing on it.
"""

from unittest.mock import Mock

import pytest

from moabb.datasets import Daly2020, Damm2026, Peterson2022
from moabb.datasets.download import NemarDownloadError


CASES = [
    (Daly2020, "on002720", 1, "01"),
    (Damm2026, "on008446", 1, "01"),
    (Peterson2022, "on003810", 2, "02"),
]


@pytest.mark.parametrize("cls,nemar_id,subject,label", CASES)
def test_mirror_flags(cls, nemar_id, subject, label, monkeypatch, tmp_path):
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    transport = Mock()
    monkeypatch.setattr("moabb.datasets.download.nemar.download", transport)
    ds = cls()
    assert ds.nemar_id == nemar_id
    ds.download([subject], tmp_path, True, False, verbose="ERROR")
    transport.assert_called_once_with(
        dataset=nemar_id,
        target_dir=tmp_path / f"MNE-{ds.code.lower()}-data" / nemar_id,
        subject=label,
        trust_existing=False,
        scope="raw",
        datatype="eeg",
    )
    # Raw mirror loaders must never prefetch converted sourcedata.
    monkeypatch.setattr(ds, "sourcedata_path", Mock(side_effect=AssertionError))
    ds._prefetch_nemar_sourcedata([subject])


def test_provider_policy(monkeypatch, tmp_path):
    cls, _, subject, _ = CASES[0]
    ds = cls()
    transport = Mock(side_effect=NemarDownloadError("offline"))
    monkeypatch.setattr(ds, "_download_nemar", transport)
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "nemar")
    with pytest.raises(NemarDownloadError):
        ds._mirror_root(subject, tmp_path, False, False, None)
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "auto")
    with pytest.warns(RuntimeWarning, match="OpenNeuro"):
        assert ds._mirror_root(subject, tmp_path, False, False, None) is None
    transport.reset_mock()
    monkeypatch.setenv("MOABB_DOWNLOAD_PROVIDER", "upstream")
    assert ds._mirror_root(subject, tmp_path, False, False, None) is None
    transport.assert_not_called()
