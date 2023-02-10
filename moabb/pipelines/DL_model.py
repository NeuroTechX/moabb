from typing import Any, Dict, Iterable

import tensorflow as tf
from tensorflow import keras
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
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier


# ====================================================================================================================
# EEGNet_8_2
# ====================================================================================================================
funct_parser = {
    "tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)": tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20
    ),
    "tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)": tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5
    ),
}


class Keras_EEGNet_8_2(KerasClassifier):
    def __init__(
        self,
        loss,
        optimizer=tf.keras.optimizers.Adam(),
        epochs=200,
        batch_size=128,
        verbose=0,
        random_state=42,
        validation_split=0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.loss = loss
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
