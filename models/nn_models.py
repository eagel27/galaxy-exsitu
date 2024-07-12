import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras.regularizers import l2


def build_cnn_model(input_shape, mdn=False, weight_initializers=(None, None)):
    """
    Build a simple CNN.

    :param input_shape: A tuple corresponding to the shape of the input
    :param mdn: Indicates whether the model should return a distribution
    :param weight_initializers: The weight initializations for the dense and conv layers
    :return: The compiled model
    """

    dense_init, conv_init = weight_initializers
    if dense_init is None:
        dense_init = tf.keras.initializers.TruncatedNormal()
    if conv_init is None:
        conv_init = tf.keras.initializers.GlorotUniform(seed=None)

    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (5, 5), activation='relu',
               padding="same", kernel_initializer=conv_init)(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation='relu',
               padding="same", kernel_initializer=conv_init)(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation='relu',
               padding="same", kernel_initializer=conv_init)(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.001),
              kernel_initializer=dense_init)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dense(10, activation='relu', kernel_regularizer=l2(0.001),
              kernel_initializer=dense_init)(x)
    # model.add(Dropout(0.2))

    if not mdn:
        y = Dense(1, activation='linear')(x)
    else:
        x = Flatten()(x)
        x = Dense(tfp.layers.IndependentNormal.params_size(1), activation=None,
                  kernel_initializer=dense_init)(x)
        y = tfp.layers.IndependentNormal(1, tfd.Normal.sample)(x)

    model = Model(inputs=input_layer, outputs=y)
    model.summary()

    # compile CNN
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-6, clipnorm=1.)
    model.compile(optimizer=opt,
                  loss=loss_fn(mdn),
                  metrics=['mse'])

    return model


def load_saved_model(path, mdn=True, lr=1e-3, do_compile=True):
    opt = Adam(learning_rate=lr, decay=1e-6, clipnorm=1.)
    model = models.load_model(path, compile=False)
    if do_compile:
        model.compile(loss=loss_fn(mdn), optimizer=opt, metrics=['mse'])
    return model


def loss_fn(mdn=False):
    if mdn:
        return lambda x, rv_x: -rv_x.log_prob(x)
    return "mean_squared_error"
