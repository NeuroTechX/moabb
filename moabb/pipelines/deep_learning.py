"""
Deep learning integrated in MOABB
Implementation using the tensorflow, keras and scikeras framework.
"""

# Authors: Igor Carrara <igor.carrara@inria.fr>
#          Bruno Aristimunha <b.aristimunha@gmail.com>
#          Sylvain Chevallier <sylvain.chevallier@universite-paris-saclay.fr>

# License: BSD (3-clause)

from typing import Any, Dict

import tensorflow as tf
from keras import backend as K
from keras.constraints import max_norm
from keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    AvgPool2D,
    Concatenate,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Flatten,
    Input,
    Lambda,
    LayerNormalization,
    MaxPooling2D,
    Permute,
)
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.models import Model, Sequential
from scikeras.wrappers import KerasClassifier

from moabb.pipelines.utils_deep_model import EEGNet, EEGNet_TC, TCN_block


# =====================================================================================
# ShallowConvNet
# =====================================================================================
def square(x):
    """
    Function to square the input tensor element-wise.
    Element-wise square.
    """
    return K.square(x)


def log(x):
    """
    Function to take the log of the input tensor element-wise.
    We use a clip to avoid taking the log of 0.
    min_value=1e-7, max_value=10000
    Parameters
    ----------
    x: tensor

    Returns
    -------
    tensor
    """
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


class KerasShallowConvNet(KerasClassifier):
    """Keras implementation of the Shallow Convolutional Network as described in [1]_.

    This implementation is taken from code by the Army Research Laboratory (ARL)
    at https://github.com/vlawhern/arl-eegmodels

    We use the original parameter implemented on the paper.

    Note that this implementation has not been verified by the original
    authors.

    References
    ----------
    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger,
           K., Tangermann, M., ... & Ball, T. (2017). Deep learning with convolutional neural networks
           for EEG decoding and visualization. Human brain mapping, 38(11), 5391-5420.
           https://doi.org/10.1002/hbm.23730

    Notes
    -----
    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        loss,
        optimizer="Adam",
        epochs=1000,
        batch_size=64,
        verbose=0,
        random_state=None,
        validation_split=0.2,
        history_plot=False,
        path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.loss = loss
        if optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.validation_split = validation_split
        self.history_plot = history_plot
        self.path = path

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        input_main = Input(shape=(self.X_shape_[1], self.X_shape_[2], 1))
        block1 = Conv2D(
            40,
            (1, 25),
            input_shape=(self.X_shape_[1], self.X_shape_[2], 1),
            kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
        )(input_main)
        block1 = Conv2D(
            40,
            (self.X_shape_[1], 1),
            use_bias=False,
            kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
        )(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
        block1 = Activation(square)(block1)
        block1 = AveragePooling2D(pool_size=(1, 75), strides=(1, 15))(block1)
        block1 = Activation(log)(block1)
        block1 = Dropout(0.5)(block1)
        flatten = Flatten()(block1)
        dense = Dense(self.n_classes_, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation("softmax")(dense)

        model = Model(inputs=input_main, outputs=softmax)

        model.compile(loss=compile_kwargs["loss"], optimizer=compile_kwargs["optimizer"])

        return model


# =================================================================================
# DeepConvNet
# =================================================================================
class KerasDeepConvNet(KerasClassifier):
    """Keras implementation of the Deep Convolutional Network as described in [1]_.

    This implementation is taken from code by the Army Research Laboratory (ARL)
    at https://github.com/vlawhern/arl-eegmodels

    We use the original parameter implemented on the paper.

    Note that this implementation has not been verified by the original
    authors.

    References
    ----------
    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger,
           K., Tangermann, M., ... & Ball, T. (2017). Deep learning with convolutional neural networks
           for EEG decoding and visualization. Human brain mapping, 38(11), 5391-5420.
           https://doi.org/10.1002/hbm.23730

    Notes
    -----
    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        loss,
        optimizer="Adam",
        epochs=1000,
        batch_size=64,
        verbose=0,
        random_state=None,
        validation_split=0.2,
        history_plot=False,
        path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.loss = loss
        if optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0009)

        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.validation_split = validation_split
        self.history_plot = history_plot
        self.path = path

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        input_main = Input(shape=(self.X_shape_[1], self.X_shape_[2], 1))
        block1 = Conv2D(
            25,
            (1, 10),
            input_shape=(self.X_shape_[1], self.X_shape_[2], 1),
            kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
        )(input_main)
        block1 = Conv2D(
            25,
            (self.X_shape_[1], 1),
            kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
        )(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
        block1 = Activation("elu")(block1)
        block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
        block1 = Dropout(0.5)(block1)

        block2 = Conv2D(50, (1, 10), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(
            block1
        )
        block2 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
        block2 = Activation("elu")(block2)
        block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
        block2 = Dropout(0.5)(block2)

        block3 = Conv2D(100, (1, 10), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(
            block2
        )
        block3 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
        block3 = Activation("elu")(block3)
        block3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block3)
        block3 = Dropout(0.5)(block3)

        block4 = Conv2D(200, (1, 10), kernel_constraint=max_norm(2.0, axis=(0, 1, 2)))(
            block3
        )
        block4 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
        block4 = Activation("elu")(block4)
        block4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block4)
        block4 = Dropout(0.5)(block4)

        flatten = Flatten()(block4)

        dense = Dense(self.n_classes_, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation("softmax")(dense)

        model = Model(inputs=input_main, outputs=softmax)

        model.compile(loss=compile_kwargs["loss"], optimizer=compile_kwargs["optimizer"])

        return model


# ===========================================================================
# EEGNet_8_2
# ===========================================================================
class KerasEEGNet_8_2(KerasClassifier):
    """Keras implementation of the EEGNet as described in [1]_.

    This implementation is taken from code by the Army Research Laboratory (ARL)
    at https://github.com/vlawhern/arl-eegmodels

    We use the original parameter implemented on the paper.

    Note that this implementation has not been verified by the original
    authors.

    References
    ----------
    .. [1] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet:
           a compact convolutional neural network for EEG-based brain–computer interfaces. Journal of neural
           engineering, 15(5), 056013.
           https://doi.org/10.1088/1741-2552/aace8c

    Notes
    -----
    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        loss,
        optimizer="Adam",
        epochs=1000,
        batch_size=64,
        verbose=0,
        random_state=None,
        validation_split=0.2,
        history_plot=False,
        path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.loss = loss
        if optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0009)
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.validation_split = validation_split
        self.history_plot = history_plot
        self.path = path

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        # Parameter of the Article
        F1 = 8
        kernLength = 64
        D = 2
        dropout = 0.5

        # Architecture
        # Input
        input_main = Input(shape=(self.X_shape_[1], self.X_shape_[2], 1))
        # EEGNet Block
        eegnet = EEGNet(
            self,
            input_layer=input_main,
            filters_1=F1,
            kernel_size=kernLength,
            depth=D,
            dropout=dropout,
        )
        flatten = Flatten()(eegnet)
        # Classification Block
        dense = Dense(self.n_classes_, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation("softmax")(dense)
        # Creation of the Model
        model = Model(inputs=input_main, outputs=softmax)

        # Compile Model
        model.compile(loss=compile_kwargs["loss"], optimizer=compile_kwargs["optimizer"])

        return model


# =======================================================================
# EEGTCNet
# =======================================================================
class KerasEEGTCNet(KerasClassifier):
    """Keras implementation of the EEGTCNet as described in [1]_.

    This implementation is taken from code by
    at https://github.com/iis-eth-zurich/eeg-tcnet

    We use the original parameter implemented on the paper.

    Note that this implementation has not been verified by the original
    authors.

    References
    ----------
    .. [1] Ingolfsson, T. M., Hersche, M., Wang, X., Kobayashi, N., Cavigelli, L., & Benini, L. (2020, October).
           EEG-TCNet: An accurate temporal convolutional network for embedded motor-imagery brain–machine interfaces.
           In 2020 IEEE International Conference on Systems, Man, and Cybernetics (SMC) (pp. 2958-2965). IEEE.
           https://doi.org/10.1109/SMC42975.2020.9283028

    Notes
    -----
    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        loss,
        optimizer="Adam",
        epochs=1000,
        batch_size=64,
        verbose=0,
        random_state=None,
        validation_split=0.2,
        history_plot=False,
        path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.loss = loss
        if optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0009)

        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.validation_split = validation_split
        self.history_plot = history_plot
        self.path = path

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        # Parameter of the Article
        F1 = 8
        kernLength = 64
        D = 2
        dropout = 0.5
        F2 = F1 * D

        # Architecture
        # Input
        input_1 = Input(shape=(self.X_shape_[1], self.X_shape_[2], 1))
        input_2 = Permute((2, 1, 3))(input_1)
        # EEGNet Block
        eegnet = EEGNet_TC(
            self, input_layer=input_2, F1=F1, kernLength=kernLength, D=D, dropout=dropout
        )
        block2 = Lambda(lambda x: x[:, :, -1, :])(eegnet)
        # TCN Block
        outs = TCN_block(
            input_layer=block2,
            input_dimension=F2,
            depth=2,
            kernel_size=4,
            filters=12,
            dropout=dropout,
            activation="elu",
        )
        out = Lambda(lambda x: x[:, -1, :])(outs)
        # Classification Block
        dense = Dense(self.n_classes_, kernel_constraint=max_norm(0.5))(out)
        softmax = Activation("softmax")(dense)
        # Creation of the Model
        model = Model(inputs=input_1, outputs=softmax)

        # Compile Model
        model.compile(loss=compile_kwargs["loss"], optimizer=compile_kwargs["optimizer"])

        return model


# =====================================================================
# EEGNeX
# =====================================================================
class KerasEEGNeX(KerasClassifier):
    """Keras implementation of the EEGNex as described in [1]_.

    This implementation is taken from code by
    at https://github.com/chenxiachan/EEGNeX

    We use the original parameter implemented on the paper.

    Note that this implementation has not been verified by the original
    authors.

    References
    ----------
    .. [1] Chen, X., Teng, X., Chen, H., Pan, Y., & Geyer, P. (2022). Toward reliable signals decoding for
           electroencephalogram: A benchmark study to EEGNeX. arXiv preprint arXiv:2207.12369.
           https://doi.org/10.48550/arXiv.2207.12369

    Notes
    -----
    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        loss,
        optimizer="Adam",
        epochs=1000,
        batch_size=64,
        verbose=0,
        random_state=None,
        validation_split=0.2,
        history_plot=False,
        path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.loss = loss
        if optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0009)
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.validation_split = validation_split
        self.history_plot = history_plot
        self.path = path

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        # Architecture
        # Input
        model = Sequential()
        model.add(Input(shape=(self.X_shape_[1], self.X_shape_[2], 1)))
        # EEGNeX
        model.add(
            Conv2D(
                filters=8,
                kernel_size=(1, 32),
                use_bias=False,
                padding="same",
                data_format="channels_last",
            )
        )
        model.add(LayerNormalization())
        model.add(Activation(activation="elu"))
        model.add(
            Conv2D(
                filters=32,
                kernel_size=(1, 32),
                use_bias=False,
                padding="same",
                data_format="channels_last",
            )
        )
        model.add(LayerNormalization())
        model.add(Activation(activation="elu"))

        model.add(
            DepthwiseConv2D(
                kernel_size=(self.X_shape_[1], 1),
                depth_multiplier=2,
                use_bias=False,
                depthwise_constraint=max_norm(1.0),
                data_format="channels_last",
            )
        )
        model.add(LayerNormalization())
        model.add(Activation(activation="elu"))
        model.add(
            AvgPool2D(pool_size=(1, 4), padding="same", data_format="channels_last")
        )
        model.add(Dropout(0.5))

        model.add(
            Conv2D(
                filters=32,
                kernel_size=(1, 16),
                use_bias=False,
                padding="same",
                dilation_rate=(1, 2),
                data_format="channels_last",
            )
        )
        model.add(LayerNormalization())
        model.add(Activation(activation="elu"))

        model.add(
            Conv2D(
                filters=8,
                kernel_size=(1, 16),
                use_bias=False,
                padding="same",
                dilation_rate=(1, 4),
                data_format="channels_last",
            )
        )

        model.add(LayerNormalization())
        model.add(Activation(activation="elu"))
        model.add(Dropout(0.5))

        model.add(Flatten())
        # Classification Block
        model.add(Dense(self.n_classes_, kernel_constraint=max_norm(0.5)))
        model.add(Activation(activation="softmax"))

        # Compile Model
        model.compile(loss=compile_kwargs["loss"], optimizer=compile_kwargs["optimizer"])

        return model


# =================================================================
# EEGITNet
# =================================================================

n_ff = [2, 4, 8]
n_sf = [1, 1, 1]


class KerasEEGITNet(KerasClassifier):
    """Keras implementation of the EEGITNet as described in [1]_.

    This implementation is taken from code by
    at https://github.com/AbbasSalami/EEG-ITNet

    We use the original parameter implemented on the paper.

    Note that this implementation has not been verified by the original
    authors.

    References
    ----------
    .. [1] Salami, A., Andreu-Perez, J., & Gillmeister, H. (2022). EEG-ITNet: An explainable inception temporal
           convolutional network for motor imagery classification. IEEE Access, 10, 36672-36685.
           https://doi.org/10.1109/ACCESS.2022.3161489

    Notes
    -----
    .. versionadded:: 0.5.0
    """

    def __init__(
        self,
        loss,
        optimizer="Adam",
        epochs=1000,
        batch_size=64,
        verbose=0,
        random_state=None,
        validation_split=0.2,
        history_plot=False,
        path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.loss = loss
        if optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0009)
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.validation_split = validation_split
        self.history_plot = history_plot
        self.path = path

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        input_main = Input(shape=(self.X_shape_[1], self.X_shape_[2], 1))

        block1 = Conv2D(
            n_ff[0],
            (1, 16),
            use_bias=False,
            activation="linear",
            padding="same",
            name="Spectral_filter_1",
        )(input_main)
        block1 = BatchNormalization()(block1)
        block1 = DepthwiseConv2D(
            (self.X_shape_[1], 1),
            use_bias=False,
            padding="valid",
            depth_multiplier=n_sf[0],
            activation="linear",
            depthwise_constraint=tf.keras.constraints.MaxNorm(max_value=1),
            name="Spatial_filter_1",
        )(block1)
        block1 = BatchNormalization()(block1)
        block1 = Activation("elu")(block1)

        # ================================

        block2 = Conv2D(
            n_ff[1],
            (1, 32),
            use_bias=False,
            activation="linear",
            padding="same",
            name="Spectral_filter_2",
        )(input_main)
        block2 = BatchNormalization()(block2)
        block2 = DepthwiseConv2D(
            (self.X_shape_[1], 1),
            use_bias=False,
            padding="valid",
            depth_multiplier=n_sf[1],
            activation="linear",
            depthwise_constraint=tf.keras.constraints.MaxNorm(max_value=1),
            name="Spatial_filter_2",
        )(block2)
        block2 = BatchNormalization()(block2)
        block2 = Activation("elu")(block2)

        # ================================

        block3 = Conv2D(
            n_ff[2],
            (1, 64),
            use_bias=False,
            activation="linear",
            padding="same",
            name="Spectral_filter_3",
        )(input_main)
        block3 = BatchNormalization()(block3)
        block3 = DepthwiseConv2D(
            (self.X_shape_[1], 1),
            use_bias=False,
            padding="valid",
            depth_multiplier=n_sf[2],
            activation="linear",
            depthwise_constraint=tf.keras.constraints.MaxNorm(max_value=1),
            name="Spatial_filter_3",
        )(block3)
        block3 = BatchNormalization()(block3)
        block3 = Activation("elu")(block3)

        # ================================

        block = Concatenate(axis=-1)([block1, block2, block3])

        # ================================

        block = AveragePooling2D((1, 4), padding="same")(block)
        block_in = Dropout(0.4)(block)

        # ================================

        paddings = tf.constant([[0, 0], [0, 0], [3, 0], [0, 0]])
        block = tf.pad(block_in, paddings, "CONSTANT")
        block = DepthwiseConv2D(
            (1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1)
        )(block)
        block = BatchNormalization()(block)
        block = Activation("elu")(block)
        block = Dropout(0.4)(block)
        block = tf.pad(block, paddings, "CONSTANT")
        block = DepthwiseConv2D(
            (1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 1)
        )(block)
        block = BatchNormalization()(block)
        block = Activation("elu")(block)
        block = Dropout(0.4)(block)
        block_out = Add()([block_in, block])

        paddings = tf.constant([[0, 0], [0, 0], [6, 0], [0, 0]])
        block = tf.pad(block_out, paddings, "CONSTANT")
        block = DepthwiseConv2D(
            (1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2)
        )(block)
        block = BatchNormalization()(block)
        block = Activation("elu")(block)
        block = Dropout(0.4)(block)
        block = tf.pad(block, paddings, "CONSTANT")
        block = DepthwiseConv2D(
            (1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 2)
        )(block)
        block = BatchNormalization()(block)
        block = Activation("elu")(block)
        block = Dropout(0.4)(block)
        block_out = Add()([block_out, block])

        paddings = tf.constant([[0, 0], [0, 0], [12, 0], [0, 0]])
        block = tf.pad(block_out, paddings, "CONSTANT")
        block = DepthwiseConv2D(
            (1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4)
        )(block)
        block = BatchNormalization()(block)
        block = Activation("elu")(block)
        block = Dropout(0.4)(block)
        block = tf.pad(block, paddings, "CONSTANT")
        block = DepthwiseConv2D(
            (1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 4)
        )(block)
        block = BatchNormalization()(block)
        block = Activation("elu")(block)
        block = Dropout(0.4)(block)
        block_out = Add()([block_out, block])

        paddings = tf.constant([[0, 0], [0, 0], [24, 0], [0, 0]])
        block = tf.pad(block_out, paddings, "CONSTANT")
        block = DepthwiseConv2D(
            (1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8)
        )(block)
        block = BatchNormalization()(block)
        block = Activation("elu")(block)
        block = Dropout(0.4)(block)
        block = tf.pad(block, paddings, "CONSTANT")
        block = DepthwiseConv2D(
            (1, 4), padding="valid", depth_multiplier=1, dilation_rate=(1, 8)
        )(block)
        block = BatchNormalization()(block)
        block = Activation("elu")(block)
        block = Dropout(0.4)(block)
        block_out = Add()([block_out, block])

        # ================================

        block = block_out

        # ================================

        block = Conv2D(28, (1, 1))(block)
        block = BatchNormalization()(block)
        block = Activation("elu")(block)
        block = AveragePooling2D((4, 1), padding="same")(block)
        block = Dropout(0.4)(block)
        embedded = Flatten()(block)

        dense = Dense(self.n_classes_, kernel_constraint=max_norm(0.5))(embedded)
        softmax = Activation("softmax")(dense)

        model = Model(inputs=input_main, outputs=softmax)

        model.compile(loss=compile_kwargs["loss"], optimizer=compile_kwargs["optimizer"])

        return model
