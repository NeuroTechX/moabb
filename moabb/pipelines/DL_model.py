import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow import keras  # Super important for Tensorflow 2.11
from scikeras.wrappers import KerasClassifier
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Dense, Flatten, Conv1D, Dropout, Input, MaxPooling2D, Activation, \
    Conv3D, Reshape
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.layers import GRU, LSTM, MaxPooling1D, Input, DepthwiseConv2D, AvgPool2D, SeparableConv2D, \
    LayerNormalization, Lambda, DepthwiseConv2D, Concatenate, Add
from keras.regularizers import l2
from keras.constraints import max_norm
from typing import Dict, Iterable, Any


# ====================================================================================================================
# EEGNet_8_2
# ====================================================================================================================
def EEGNet_8_2_(meta: Dict[str, Any]):
    model = Sequential()
    model.add(Input(shape=(meta["X_shape_"][1], meta["X_shape_"][2], 1)))
    #
    model.add(Conv2D(filters=8, kernel_size=(1, 64), use_bias=False, padding='same', data_format="channels_last"))
    model.add(BatchNormalization())

    model.add(DepthwiseConv2D(kernel_size=(meta["X_shape_"][1], 1), depth_multiplier=2, use_bias=False,
                              depthwise_constraint=max_norm(1.), data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(AvgPool2D(pool_size=(1, 4), padding='same', data_format="channels_last"))
    model.add(Dropout(0.5))

    model.add(
        SeparableConv2D(filters=16, kernel_size=(1, 16), use_bias=False, padding='same', data_format="channels_last"))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(AvgPool2D(pool_size=(1, 8), padding='same', data_format="channels_last"))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(meta["n_classes_"], kernel_constraint=max_norm(0.25)))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    # model.summary()

    return model


class EEGNet_8_2(KerasClassifier):
    def __init__(self,
                 loss,
                 optimizer=tf.keras.optimizers.Adam(),
                 epochs=200,
                 batch_size=128,
                 verbose=0,
                 random_state=42,
                 validation_split=0.2,
                 **kwargs, ):
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
        model.add(Conv2D(filters=8, kernel_size=(1, 64), use_bias=False, padding='same', data_format="channels_last"))
        model.add(BatchNormalization())

        model.add(DepthwiseConv2D(kernel_size=(self.X_shape_[1], 1), depth_multiplier=2, use_bias=False,
                                  depthwise_constraint=max_norm(1.), data_format="channels_last"))
        model.add(BatchNormalization())
        model.add(Activation(activation='elu'))
        model.add(AvgPool2D(pool_size=(1, 4), padding='same', data_format="channels_last"))
        model.add(Dropout(0.5))

        model.add(
            SeparableConv2D(filters=16, kernel_size=(1, 16), use_bias=False, padding='same',
                            data_format="channels_last"))
        model.add(BatchNormalization())
        model.add(Activation(activation='elu'))
        model.add(AvgPool2D(pool_size=(1, 8), padding='same', data_format="channels_last"))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(self.n_classes_, kernel_constraint=max_norm(0.25)))
        model.add(Activation('softmax'))
        model.compile(loss=compile_kwargs["loss"], optimizer=compile_kwargs["optimizer"])
        # model.summary()

        return model


funct_parser = {
    "tf.keras.callbacks.EarlyStopping()": tf.keras.callbacks.EarlyStopping(),  # Notice how this doesn't have ()
    "tf.keras.callbacks.ReduceLROnPlateau()": tf.keras.callbacks.ReduceLROnPlateau()  # Notice how this doesn't have ()
}


def custom_func(funct):
    my_func = funct_parser[str(funct)]  # This will cause my_func to be a copy of `np.sum`.
    return my_func()  # Call my_func


class EEGNet_8_2_Callbacks(KerasClassifier):
    def __init__(self,
                 loss,
                 optimizer=tf.keras.optimizers.Adam(),
                 epochs=200,
                 batch_size=128,
                 verbose=0,
                 random_state=42,
                 validation_split=0.2,
                 **kwargs, ):
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
        model.add(Conv2D(filters=8, kernel_size=(1, 64), use_bias=False, padding='same', data_format="channels_last"))
        model.add(BatchNormalization())

        model.add(DepthwiseConv2D(kernel_size=(self.X_shape_[1], 1), depth_multiplier=2, use_bias=False,
                                  depthwise_constraint=max_norm(1.), data_format="channels_last"))
        model.add(BatchNormalization())
        model.add(Activation(activation='elu'))
        model.add(AvgPool2D(pool_size=(1, 4), padding='same', data_format="channels_last"))
        model.add(Dropout(0.5))

        model.add(
            SeparableConv2D(filters=16, kernel_size=(1, 16), use_bias=False, padding='same',
                            data_format="channels_last"))
        model.add(BatchNormalization())
        model.add(Activation(activation='elu'))
        model.add(AvgPool2D(pool_size=(1, 8), padding='same', data_format="channels_last"))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(self.n_classes_, kernel_constraint=max_norm(0.25)))
        model.add(Activation('softmax'))
        model.compile(loss=compile_kwargs["loss"], optimizer=compile_kwargs["optimizer"])
        # model.summary()

        return model