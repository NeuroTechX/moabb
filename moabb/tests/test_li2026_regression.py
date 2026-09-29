"""Regression tests for the Li2026 Curry fallback reader."""

import numpy as np

from moabb.datasets.li2026 import Li2026


def test_legacy_curry_fallback_reads_float32_sidecars(tmp_path):
    cdt = tmp_path / "recording.cdt"
    np.array([[1, 2, 3], [4, 5, 6]], dtype="<f4").tofile(cdt)
    cdt.with_suffix(".cdt.dpa").write_text(
        """NumSamples = 2\nNumChannels = 3\nSampleFreqHz = 1000\n
LABELS START_LIST
Cz
C3
C4
LABELS END_LIST
LABELS_OTHERS START_LIST
LABELS_OTHERS END_LIST
""",
        encoding="utf-8",
    )
    cdt.with_suffix(".cdt.ceo").write_text(
        """NUMBER_LIST START_LIST
0 0 1 -1
1 0 2 -1
NUMBER_LIST END_LIST
""",
        encoding="utf-8",
    )

    raw = Li2026._read_legacy_curry(cdt)

    assert raw.ch_names == ["Cz", "C3", "C4"]
    np.testing.assert_allclose(raw.get_data()[:, 0], [1e-6, 2e-6, 3e-6])
    assert raw.annotations.description.tolist() == ["1", "2"]
    np.testing.assert_allclose(raw.annotations.onset, [0.0, 0.001])


def test_archive_transport_and_missing_task(tmp_path, monkeypatch):
    import zipfile

    import pytest

    from moabb.datasets import li2026

    archive = tmp_path / "MI_A_Dataset.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for task in li2026._TASKS:
            for subject in ("01", "50", "100", "150", "200"):
                stream.writestr(
                    f"MI_A_Dataset/MI_A_Dataset/Raw_data/{task}/Sub_{subject}.cdt", b""
                )
    calls = []

    def transport(url, sign, path, force_update, verbose):
        calls.append((path, force_update, verbose))
        return str(archive)

    monkeypatch.setattr(li2026.dl, "data_dl", transport)
    paths = Li2026().data_path(5, path=tmp_path, force_update=True, verbose="ERROR")
    assert len(paths) == 5
    assert all(path.endswith("Sub_200.cdt") for path in paths)
    assert calls == [(tmp_path, True, "ERROR")]
    from pathlib import Path

    Path(paths[-1]).unlink()
    with pytest.raises(FileNotFoundError, match="Expected at least"):
        Li2026().data_path(5)
