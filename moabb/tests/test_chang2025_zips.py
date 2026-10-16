"""Offline regression: Chang2025 downloads the ZIP that holds each subject.

The figshare ZIPs are split by original subject ID (S201-S230, without S207
and S218), but the loader picked them by MOABB subject number. On a fresh
cache MOABB subjects 10, 15, 19, 20, 24 and 25 got the neighbouring ZIP, so
their folder was missing and they were skipped with a warning. No downloads:
the ZIPs are synthetic, with the released member layout.
"""

import zipfile

import pytest

from moabb.datasets import Chang2025, chang2025


# Top-level folders of each released ZIP (figshare article 28831730).
_RELEASED = {
    57432889: ["S201", "S202", "S203", "S204", "S205"],
    57433693: ["S206", "S208", "S209", "S210"],
    57435691: ["S211", "S212", "S213", "S214", "S215"],
    57435694: ["S216", "S217", "S219", "S220"],
    57560707: ["S221", "S222", "S223", "S224", "S225"],
    57560710: ["S226", "S227", "S228", "S229", "S230"],
}


@pytest.fixture
def fake_figshare(monkeypatch, tmp_path):
    requested = []

    def data_dl(url, sign, path=None, force_update=False, verbose=None):
        file_id = int(url.rsplit("/", 1)[-1])
        requested.append(file_id)
        zpath = tmp_path / f"{file_id}.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            for sid in _RELEASED[file_id]:
                zf.writestr(f"{sid}/{sid}_MI1.set", b"")
        return str(zpath)

    monkeypatch.setattr(chang2025.dl, "data_dl", data_dl)
    return requested


def test_zip_table_matches_the_released_zips():
    assert dict(chang2025._ZIP_FILES.values()) == _RELEASED


@pytest.mark.parametrize("subject", range(1, 29))
def test_every_subject_gets_its_own_folder(fake_figshare, tmp_path, subject):
    ds = Chang2025()
    base = ds.data_path(subject, path=str(tmp_path / "mne"))
    orig_id = chang2025._ORIG_IDS[subject - 1]
    assert (tmp_path / "mne" / "MNE-chang2025-data" / orig_id).is_dir(), base
    assert len(fake_figshare) == 1
    assert orig_id in _RELEASED[fake_figshare[0]]
