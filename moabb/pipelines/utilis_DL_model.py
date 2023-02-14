import os
from typing import Any, Dict, Iterable

import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.constraints import max_norm
from keras.layers import (
    GRU,
    LSTM,
    Activation,
    Add,
    AveragePooling2D,
    AvgPool2D,
    Concatenate,
    Conv1D,
    Conv2D,
    Conv3D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Flatten,
    Input,
    Lambda,
    LayerNormalization,
    MaxPooling1D,
    MaxPooling2D,
    Reshape,
    SeparableConv2D,
)
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
from scikeras.wrappers import KerasClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow import keras  # Super important for Tensorflow 2.11


def EEGNet(self, input_layer, F1=8, kernLength=64, D=2, dropout=0.5, activation="elu"):
    F2 = F1 * D

    block1 = Conv2D(
        F1,
        kernel_size=(1, kernLength),
        padding="same",
        input_shape=(self.X_shape_[1], self.X_shape_[2], 1),
        use_bias=False,
    )(input_layer)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D(
        kernel_size=(self.X_shape_[1], 1),
        use_bias=False,
        depth_multiplier=2,
        depthwise_constraint=max_norm(1.0),
    )(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation(activation)(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropout)(block1)

    block2 = SeparableConv2D(F2, kernel_size=(1, 16), use_bias=False, padding="same")(
        block1
    )
    block2 = BatchNormalization()(block2)
    block2 = Activation(activation)(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = Dropout(dropout)(block2)

    return block2


def TCN_block(
    input_layer, input_dimension, depth, kernel_size, filters, dropout, activation
):
    """TCN_block from Bai et al 2018
    Temporal Convolutional Network (TCN)

    Notes
    -----
    THe original code available at https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    This implementation has a slight modification from the original code
    and it is taken from the code by Ingolfsson et al at https://github.com/iis-eth-zurich/eeg-tcnet
    See details at https://arxiv.org/abs/2006.00622
    References
    ----------
    .. Bai, S., Kolter, J. Z., & Koltun, V. (2018).
       An empirical evaluation of generic convolutional and recurrent networks
       for sequence modeling.
       arXiv preprint arXiv:1803.01271.
    """

    block = Conv1D(
        filters,
        kernel_size=kernel_size,
        dilation_rate=1,
        activation="linear",
        padding="causal",
        kernel_initializer="he_uniform",
    )(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    block = Conv1D(
        filters,
        kernel_size=kernel_size,
        dilation_rate=1,
        activation="linear",
        padding="causal",
        kernel_initializer="he_uniform",
    )(block)
    block = BatchNormalization()(block)
    block = Activation(activation)(block)
    block = Dropout(dropout)(block)
    if input_dimension != filters:
        conv = Conv1D(filters, kernel_size=1, padding="same")(input_layer)
        added = Add()([block, conv])
    else:
        added = Add()([block, input_layer])
    out = Activation(activation)(added)

    for i in range(depth - 1):
        block = Conv1D(
            filters,
            kernel_size=kernel_size,
            dilation_rate=2 ** (i + 1),
            activation="linear",
            padding="causal",
            kernel_initializer="he_uniform",
        )(out)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        block = Conv1D(
            filters,
            kernel_size=kernel_size,
            dilation_rate=2 ** (i + 1),
            activation="linear",
            padding="causal",
            kernel_initializer="he_uniform",
        )(block)
        block = BatchNormalization()(block)
        block = Activation(activation)(block)
        block = Dropout(dropout)(block)
        added = Add()([block, out])
        out = Activation(activation)(added)

    return out
