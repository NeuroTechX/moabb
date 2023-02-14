"""
Utils for Deep learning that work on Moabb.
Implementation using the tensorflow, keras and scikeras framework.
"""

# Authors: Igor Carrara <igor.carrara@inria.fr>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#          Sylvain Chevallier <sylvain.chevallier@universite-paris-saclay.fr>

# License: BSD (3-clause)

from keras.constraints import max_norm
from keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    Conv1D,
    Conv2D,
    DepthwiseConv2D,
    Dropout,
    SeparableConv2D,
)
from keras.layers.normalization.batch_normalization import BatchNormalization


def EEGNet(
    data, input_layer, filters_1=8, kernel_size=64, depth=2, dropout=0.5, activation="elu"
):
    """
    EEGNet implementation.

    """
    filters_2 = filters_1 * depth

    block1 = Conv2D(
        filters=filters_1,
        kernel_size=(1, kernel_size),
        padding="same",
        input_shape=(data.X_shape_[1], data.X_shape_[2], 1),
        use_bias=False,
    )(input_layer)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D(
        kernel_size=(data.X_shape_[1], 1),
        use_bias=False,
        depth_multiplier=2,
        depthwise_constraint=max_norm(1.0),
    )(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation(activation)(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropout)(block1)

    block2 = SeparableConv2D(
        filters_2, kernel_size=(1, 16), use_bias=False, padding="same"
    )(block1)
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
