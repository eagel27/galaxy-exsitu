import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import *
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow.keras.regularizers import l2


class FeatureExtractor(tf.keras.layers.Layer):

    def __init__(self, input_dim, weight_initializers=(None, None), name="feature_extractor", **kwargs):
        super(FeatureExtractor, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.weight_initializers = weight_initializers

    def build(self):
        dense_init, conv_init = self.weight_initializers
        if dense_init is None:
            dense_init = tf.keras.initializers.TruncatedNormal()
        if conv_init is None:
            conv_init = tf.keras.initializers.GlorotUniform(seed=None)

        inputs = tf.keras.Input(shape=self.input_dim, name='feature_inputs')

        x = layers.Conv2D(32, (5, 5), activation='relu', padding="same",
                          kernel_initializer=conv_init)(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization(axis=-1)(x)
        # x = Dropout(0.2)(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding="same",
                          kernel_initializer=conv_init)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization(axis=-1)(x)
        # x = Dropout(0.2)(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding="same",
                          kernel_initializer=conv_init)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization(axis=-1)(x)
        # x = Dropout(0.2)(x)
        outputs = layers.Flatten()(x)

        feature_extractor = tf.keras.Model([inputs], [outputs], name="feature_extractor")
        feature_extractor.summary()
        return feature_extractor


class RegressionPredictor(tf.keras.layers.Layer):

    def __init__(self, input_dim, weight_initializers=(None, None), name="regression_predictor", **kwargs):
        super(RegressionPredictor, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.weight_initializers = weight_initializers

    def build(self):
        dense_init, _ = self.weight_initializers
        if dense_init is None:
            dense_init = tf.keras.initializers.TruncatedNormal()

        inputs = tf.keras.Input(shape=(self.input_dim,), name='regressive_input')

        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001),
                         kernel_initializer=dense_init)(inputs)
        x = layers.Dense(10, activation='relu', kernel_regularizer=l2(0.001),
                         kernel_initializer=dense_init)(x)

        x = layers.Dropout(0.4)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(tfp.layers.IndependentNormal.params_size(1), activation=None,
                         kernel_initializer=dense_init)(x)
        x = tfp.layers.IndependentNormal(1, tfd.Normal.sample)(x)
        # x = layers.Dense(1, activation='linear', kernel_regularizer=l2(0.001),
        #                 kernel_initializer=dense_init)(x)

        regressor = tf.keras.Model([inputs], [x], name="regressive_predictor")
        regressor.summary()
        return regressor


class DomainPredictor(tf.keras.layers.Layer):

    def __init__(self, input_dim, weight_initializers=(None, None), name="domain_predictor", **kwargs):
        super(DomainPredictor, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.weight_initializers = weight_initializers

    def build(self):
        dense_init, _ = self.weight_initializers
        if dense_init is None:
            dense_init = tf.keras.initializers.TruncatedNormal()

        inputs = tf.keras.Input(shape=(self.input_dim,), name='domain_input')

        x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001),
                         kernel_initializer=dense_init)(inputs)
        x = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001),
                         kernel_initializer=dense_init)(x)
        x = layers.Dense(2, activation='softmax', kernel_regularizer=l2(0.001),
                         kernel_initializer=dense_init)(x)
        x = layers.Dropout(0.4)(x)

        domain_predictor = tf.keras.Model([inputs], [x], name="domain_predictor")
        domain_predictor.summary()
        return domain_predictor


class DomainAdaptationModel(tf.keras.Model):

    """
    Adapted from https://github.com/lancerane/Adversarial-domain-adaptation
    Based on https://arxiv.org/abs/1505.07818
    """
    def __init__(self, input_dim=(128, 128, 5), features_input_dim=16384, alpha=1, **kwargs):
        super(DomainAdaptationModel, self).__init__(**kwargs)
        self.feature_extractor = FeatureExtractor(input_dim=input_dim).build()
        self.regressive_predictor = RegressionPredictor(input_dim=features_input_dim).build()
        self.domain_predictor = DomainPredictor(input_dim=features_input_dim).build()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.regression_loss_tracker = tf.keras.metrics.Mean(name="regression_loss")
        self.regression_mse_tracker = tf.keras.metrics.MeanSquaredError(name="regression_mse")
        self.domain_loss_tracker = tf.keras.metrics.Mean(name="domain_prediction_loss")
        self.domain_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='d_train_accuracy')

        self.test_total_loss_tracker = tf.keras.metrics.Mean(name="test_total_loss")
        self.test_regression_loss_tracker = tf.keras.metrics.Mean(name="test_regression_loss")
        self.test_regression_mse_tracker = tf.keras.metrics.MeanSquaredError(name="test_regression_mse")
        self.test_domain_loss_tracker = tf.keras.metrics.Mean(name="test_domain_prediction_loss")

        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, decay=1e-6)
        self.f_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

        # self.mse_loss_object = tf.keras.losses.MeanSquaredError()
        self.domain_loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

        self.alpha = alpha

    @property
    def metrics(self):
        return [
            self.regression_loss_tracker,
            self.domain_loss_tracker,
            self.total_loss_tracker,
            self.test_regression_loss_tracker,
            self.test_domain_loss_tracker,
            self.test_total_loss_tracker,
        ]

    def test_step(self, da_dataset):
        da_dataset = da_dataset[0]
        features = self.feature_extractor(da_dataset['images'])
        predictions = self.regressive_predictor(features)
        label_loss = self.compiled_loss(da_dataset['labels'], predictions)
        # mse_label_loss = self.mse_loss_object(da_dataset['labels'], predictions)

        features = self.feature_extractor(da_dataset['domain_images'])
        domain_predictions = self.domain_predictor(features)
        domain_loss = self.domain_loss_object(da_dataset['domain_labels'], domain_predictions)

        total_loss = label_loss - self.alpha * domain_loss

        self.test_total_loss_tracker.update_state(total_loss)
        self.test_regression_loss_tracker.update_state(label_loss)
        self.test_regression_mse_tracker.update_state(da_dataset['labels'], predictions)
        self.test_domain_loss_tracker.update_state(domain_loss)

        return {
            "loss": self.test_total_loss_tracker.result(),
            "mse": self.test_regression_mse_tracker.result(),
            "dal": self.test_domain_loss_tracker.result()
        }

    def train_step(self, da_dataset):
        da_dataset = da_dataset[0]
        with tf.GradientTape(persistent=True) as tape:
            features = self.feature_extractor(da_dataset['images'])
            l_predictions = self.regressive_predictor(features)

            features = self.feature_extractor(da_dataset['domain_images'])
            d_predictions = self.domain_predictor(features)

            # mse_label_loss = self.mse_loss_object(da_dataset['labels'], l_predictions)
            label_loss = self.compiled_loss(da_dataset['labels'], l_predictions)
            domain_loss = self.domain_loss_object(da_dataset['domain_labels'], d_predictions)

            total_loss = label_loss - self.alpha * domain_loss
            # total_loss = label_loss

        f_gradients_on_label_loss = tape.gradient(label_loss, self.feature_extractor.trainable_variables)
        f_gradients_on_domain_loss = tape.gradient(domain_loss, self.feature_extractor.trainable_variables)

        f_gradients = [f_gradients_on_label_loss[i] - self.alpha * f_gradients_on_domain_loss[
            i] for i in range(len(f_gradients_on_domain_loss))]

        l_gradients = tape.gradient(label_loss, self.regressive_predictor.trainable_variables)

        self.f_optimizer.apply_gradients(zip(f_gradients + l_gradients,
                                             self.feature_extractor.trainable_variables +
                                             self.regressive_predictor.trainable_variables))

        ## Update the discriminator: Comment this bit to complete all updates in one step. Asynchronous updating
        ## seems to work a bit better, with better accuracy and stability, but may take longer to train
        with tf.GradientTape() as tape:
            features = self.feature_extractor(da_dataset['domain_images'])
            d_predictions = self.domain_predictor(features)
            domain_loss = self.domain_loss_object(da_dataset['domain_labels'], d_predictions)
        ####

        d_gradients = tape.gradient(domain_loss, self.domain_predictor.trainable_variables)
        d_gradients = [self.alpha * i for i in d_gradients]
        self.d_optimizer.apply_gradients(zip(d_gradients, self.domain_predictor.trainable_variables))

        # grads = tape.gradient(total_loss, self.trainable_weights)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.regression_loss_tracker.update_state(label_loss)
        self.regression_mse_tracker.update_state(da_dataset['labels'], l_predictions)
        self.domain_loss_tracker.update_state(domain_loss)
        self.domain_train_accuracy(da_dataset['domain_labels'], d_predictions)

        return {
            "loss": self.total_loss_tracker.result(),
            "r_loss": self.regression_loss_tracker.result(),
            "r_mse": self.regression_mse_tracker.result(),
            "da_loss": self.domain_loss_tracker.result(),
            "da_ac": self.domain_train_accuracy.result()
        }
