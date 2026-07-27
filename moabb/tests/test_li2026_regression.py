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
