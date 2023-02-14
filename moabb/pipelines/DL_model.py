from typing import Any, Dict

import tensorflow as tf
from keras.constraints import max_norm
from keras.layers import (
    Activation,
    AvgPool2D,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    Flatten,
    Input,
    SeparableConv2D,
    AveragePooling2D,
    MaxPooling2D
)
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.models import Sequential
from keras.models import Model
from scikeras.wrappers import KerasClassifier
from keras import backend as K


# ====================================================================================================================
# ShallowConvNet
# ====================================================================================================================
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))

class Keras_ShallowConvNet(KerasClassifier):
    """ Keras implementation of the Shallow Convolutional Network as described
        in Schirrmeister et. al. (2017), Human Brain Mapping.

        This implementation is taken from code by the Army Research Laboratory (ARL)
        at https://github.com/vlawhern/arl-eegmodels

        We use the original parameter implemented on the paper.

        Note that this implementation has not been verified by the original
        authors. We do note that this implementation reproduces the results in the
        original paper with minor deviations.
        """
    def __init__(self,
                 loss,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.0009),
                 epochs=1000,
                 batch_size=64,
                 verbose=0,
                 random_state=42,
                 validation_split=0.2,
                 history_plot=False,
                 path = None,
                 **kwargs,):
        super().__init__(**kwargs)

        self.loss = loss
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
        block1 = Conv2D(40, (1, 25),
                        input_shape=(self.X_shape_[1], self.X_shape_[2], 1),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
        block1 = Conv2D(40, (self.X_shape_[1], 1), use_bias=False,
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
        block1 = Activation(square)(block1)
        block1 = AveragePooling2D(pool_size=(1, 75), strides=(1, 15))(block1)
        block1 = Activation(log)(block1)
        block1 = Dropout(0.5)(block1)
        flatten = Flatten()(block1)
        dense = Dense(self.n_classes_, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation('softmax')(dense)

        model = Model(inputs=input_main, outputs=softmax)

        model.compile(loss=compile_kwargs["loss"], optimizer=compile_kwargs["optimizer"])

        return model


# ====================================================================================================================
# DeepConvNet
# ====================================================================================================================
class Keras_DeepConvNet(KerasClassifier):
    """ Keras implementation of the Shallow Convolutional Network as described
        in Schirrmeister et. al. (2017), Human Brain Mapping.

        This implementation is taken from code by the Army Research Laboratory (ARL)
        at https://github.com/vlawhern/arl-eegmodels

        We use the original parameter implemented on the paper.

        Note that this implementation has not been verified by the original
        authors. We do note that this implementation reproduces the results in the
        original paper with minor deviations.
        """

    def __init__(self,
                 loss,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.0009),
                 epochs=1000,
                 batch_size=64,
                 verbose=0,
                 random_state=42,
                 validation_split=0.2,
                 history_plot=False,
                 path=None,
                 **kwargs, ):
        super().__init__(**kwargs)

        self.loss = loss
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
        block1 = Conv2D(25, (1, 10),
                        input_shape=(self.X_shape_[1], self.X_shape_[2], 1),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
        block1 = Conv2D(25, (self.X_shape_[1], 1),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
        block1 = Activation('elu')(block1)
        block1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block1)
        block1 = Dropout(0.5)(block1)

        block2 = Conv2D(50, (1, 10),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block2 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
        block2 = Activation('elu')(block2)
        block2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block2)
        block2 = Dropout(0.5)(block2)

        block3 = Conv2D(100, (1, 10),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
        block3 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
        block3 = Activation('elu')(block3)
        block3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block3)
        block3 = Dropout(0.5)(block3)

        block4 = Conv2D(200, (1, 10),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
        block4 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
        block4 = Activation('elu')(block4)
        block4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3))(block4)
        block4 = Dropout(0.5)(block4)

        flatten = Flatten()(block4)

        dense = Dense(self.n_classes_, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation('softmax')(dense)

        model = Model(inputs=input_main, outputs=softmax)

        model.compile(loss=compile_kwargs["loss"], optimizer=compile_kwargs["optimizer"])

        return model


# ====================================================================================================================
# EEGNet_8_2
# ====================================================================================================================
class Keras_EEGNet_8_2(KerasClassifier):
    def __init__(
        self,
        loss,
        optimizer=None,
        epochs=200,
        batch_size=128,
        verbose=0,
        random_state=42,
        validation_split=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.loss = loss
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self.validation_split = validation_split

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        model = Sequential()
        model.add(Input(shape=(self.X_shape_[1], self.X_shape_[2], 1)))
        #
        model.add(
            Conv2D(
                filters=8,
                kernel_size=(1, 64),
                use_bias=False,
                padding="same",
                data_format="channels_last",
            )
        )
        model.add(BatchNormalization())

        model.add(
            DepthwiseConv2D(
                kernel_size=(self.X_shape_[1], 1),
                depth_multiplier=2,
                use_bias=False,
                depthwise_constraint=max_norm(1.0),
                data_format="channels_last",
            )
        )
        model.add(BatchNormalization())
        model.add(Activation(activation="elu"))
        model.add(
            AvgPool2D(pool_size=(1, 4), padding="same", data_format="channels_last")
        )
        model.add(Dropout(0.5))

        model.add(
            SeparableConv2D(
                filters=16,
                kernel_size=(1, 16),
                use_bias=False,
                padding="same",
                data_format="channels_last",
            )
        )
        model.add(BatchNormalization())
        model.add(Activation(activation="elu"))
        model.add(
            AvgPool2D(pool_size=(1, 8), padding="same", data_format="channels_last")
        )
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(self.n_classes_, kernel_constraint=max_norm(0.25)))
        model.add(Activation("softmax"))
        model.compile(loss=compile_kwargs["loss"], optimizer=compile_kwargs["optimizer"])
        # model.summary()

        return model
