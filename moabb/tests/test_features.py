"""Tests for feature transformers."""

import numpy as np

from moabb.pipelines.features import LogVariance


def test_log_variance_floors_only_zero_variance():
    for dtype in (np.float32, np.float64):
        X = np.array(
            [[[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]]], dtype=dtype
        )

        transformed = LogVariance().transform(X)

        variance = np.var(X, axis=-1)
        np.testing.assert_allclose(
            transformed[0, 0],
            np.log(np.finfo(variance.dtype).tiny),
            rtol=0,
            atol=0,
        )
        np.testing.assert_array_equal(
            transformed[variance > 0], np.log(variance[variance > 0])
        )
        assert np.isfinite(transformed).all()
