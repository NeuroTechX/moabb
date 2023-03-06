import random

import numpy as np
import torch
from braindecode.datasets import create_from_X_y
from sklearn.base import BaseEstimator, TransformerMixin


class Transformer(BaseEstimator, TransformerMixin):
    """
    Class to Load the data from MOABB in a format compatible with braindecode
    """

    def __init__(self, kw_args=None):
        self.kw_args = kw_args

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):
        dataset = create_from_X_y(
            X.get_data(),
            y=self.y,
            window_size_samples=X.get_data().shape[2],
            window_stride_samples=X.get_data().shape[2],
            drop_last_window=False,
            sfreq=X.info["sfreq"],
        )

        return dataset

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True
