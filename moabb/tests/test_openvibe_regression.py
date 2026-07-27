"""Regression tests for OpenViBE CSV header variants."""

import bz2

import numpy as np
import pandas as pd

from moabb.datasets.openvibe import (
    CODE_LEFT,
    CODE_RIGHT,
    OpenViBE,
    _CHANNELS,
    _CSV_CHANNELS,
)


def test_openvibe_accepts_nz_reference_header_alias(tmp_path, monkeypatch):
    """Subjects 5--14 use ``Nz`` instead of the legacy ``Ref_Nose`` header."""
    columns = ["Nz" if channel == "Ref_Nose" else channel for channel in _CSV_CHANNELS]
    frame = pd.DataFrame(
        {column: np.arange(3, dtype=float) for column in columns}
        | {"Event Id": [np.nan, str(CODE_LEFT), str(CODE_RIGHT)]}
    )
    path = tmp_path / "05-signal.csv.bz2"
    with bz2.open(path, "wt") as fout:
        frame.to_csv(fout, index=False)

    dataset = OpenViBE()
    monkeypatch.setattr(dataset, "data_path", lambda subject: [str(path)])
    raw = dataset._get_single_subject_data(5)["0"]["0"]

    assert raw.ch_names == _CHANNELS
    np.testing.assert_allclose(raw.get_data()[2], np.arange(3) * 1e-6)
    assert list(raw.annotations.description) == ["left_hand", "right_hand"]
