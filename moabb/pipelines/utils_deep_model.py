"""
Utils for Deep learning integrated on MOABB.
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
    EEGNet block implementation as described in [1]_.

    This implementation is taken from code by The Integrated Systems Laboratory of ETH Zurich
    at https://github.com/iis-eth-zurich/eeg-tcnet

    We use the original parameter implemented on the paper.

    Note that this implementation has not been verified by the original
    authors.

    References
    ----------
    .. [1] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet:
           a compact convolutional neural network for EEG-based brain–computer interfaces. Journal of neural
           engineering, 15(5), 056013.
           https://doi.org/10.1088/1741-2552/aace8c
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
    """Temporal Convolutional Network (TCN), TCN_block from [1]_.

    This implementation is taken from code by The Integrated Systems Laboratory of ETH Zurich
    at https://github.com/iis-eth-zurich/eeg-tcnet

    References
    ----------
    .. [1] Ingolfsson, T. M., Hersche, M., Wang, X., Kobayashi, N., Cavigelli, L., & Benini, L. (2020, October).
           EEG-TCNet: An accurate temporal convolutional network for embedded motor-imagery brain–machine interfaces.
           In 2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC) (pp. 2958-2965). IEEE.
           https://doi.org/10.48550/arXiv.2006.00622
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


def EEGNet_TC(self, input_layer, F1=8, kernLength=64, D=2, dropout=0.1, activation="elu"):
    F2 = F1 * D

    block1 = Conv2D(
        F1,
        kernel_size=(kernLength, 1),
        padding="same",
        use_bias=False,
        data_format="channels_last",
    )(input_layer)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = DepthwiseConv2D(
        kernel_size=(1, self.X_shape_[1]),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.0),
        data_format="channels_last",
    )(block1)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation(activation)(block1)
    block1 = AveragePooling2D((8, 1), data_format="channels_last")(block1)
    block1 = Dropout(dropout)(block1)

    block2 = SeparableConv2D(
        F2,
        kernel_size=(16, 1),
        use_bias=False,
        padding="same",
        data_format="channels_last",
    )(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation(activation)(block2)
    block2 = AveragePooling2D((8, 1), data_format="channels_last")(block2)
    block2 = Dropout(dropout)(block2)

    return block2
