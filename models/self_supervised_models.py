import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2


class ProjectionHead(tf.keras.layers.Layer):

    def __init__(self, input_dim, weight_initializers=(None, None), name="projection_head", **kwargs):
        super(ProjectionHead, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.weight_initializers = weight_initializers
        self._name = name

    def build(self):
        dense_init, _ = self.weight_initializers
        if dense_init is None:
            dense_init = tf.keras.initializers.TruncatedNormal()

        inputs = tf.keras.Input(shape=(self.input_dim,), name=self._name + 'input')

        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001),
                         kernel_initializer=dense_init)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001),
                         kernel_initializer=dense_init)(x)

        projector = tf.keras.Model([inputs], [x], name=self._name + "_model")
        projector.summary()
        return projector


class ProjectionHeadRedo(tf.keras.layers.Layer):

    def __init__(self, input_dim, weight_initializers=(None, None), name="projection_head_redo", **kwargs):
        super(ProjectionHeadRedo, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.weight_initializers = weight_initializers
        self._name = name

    def build(self):
        dense_init, _ = self.weight_initializers
        if dense_init is None:
            dense_init = tf.keras.initializers.TruncatedNormal()

        inputs = tf.keras.Input(shape=(self.input_dim,), name=self._name + '_input')

        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001),
                         kernel_initializer=dense_init)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001),
                         kernel_initializer=dense_init)(x)

        projector = tf.keras.Model([inputs], [x], name=self._name + "_model")
        projector.summary()
        return projector


def byol_loss(p, z):
    p = tf.math.l2_normalize(p, axis=1)  # (2*bs, 128)
    z = tf.math.l2_normalize(z, axis=1)  # (2*bs, 128)

    similarities = tf.reduce_sum(tf.multiply(p, z), axis=1)
    return 2 - 2 * tf.reduce_mean(similarities)


class FeatureExtractor(tf.keras.layers.Layer):

    def __init__(self, input_dim, weight_initializers=(None, None), name="feature_extractor", **kwargs):
        super(FeatureExtractor, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.weight_initializers = weight_initializers
        self._name = name

    def build(self):
        dense_init, conv_init = self.weight_initializers
        if dense_init is None:
            dense_init = tf.keras.initializers.TruncatedNormal()
        if conv_init is None:
            conv_init = tf.keras.initializers.GlorotUniform(seed=None)

        inputs = tf.keras.Input(shape=self.input_dim, name=self._name + '_inputs')

        x = layers.Conv2D(64, (5, 5), activation='relu', padding="same",
                          kernel_initializer=conv_init)(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding="same",
                          kernel_initializer=conv_init)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(256, (3, 3), activation='relu', padding="same",
                          kernel_initializer=conv_init)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.GlobalAveragePooling2D()(x)

        feature_extractor = tf.keras.Model([inputs], [outputs], name=self._name + "_model")
        feature_extractor.summary()
        return feature_extractor


class RegressionPredictor(tf.keras.layers.Layer):

    def __init__(self, input_dim, weight_initializers=(None, None), name="regression_predictor",
                 output_dim=1, **kwargs):
        super(RegressionPredictor, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_initializers = weight_initializers

    def build(self):
        dense_init, _ = self.weight_initializers
        if dense_init is None:
            dense_init = tf.keras.initializers.TruncatedNormal()

        inputs = layers.Input(shape=(self.input_dim,), name='regressive_input')

        x = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001),
                         kernel_initializer=dense_init)(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(10, activation='relu', kernel_regularizer=l2(0.001),
                         kernel_initializer=dense_init)(x)
        y = layers.Dense(1, activation='linear', kernel_regularizer=l2(0.001),
                         kernel_initializer=dense_init, name='exsitu_f')(x)

        if self.output_dim > 1:
            y_major = layers.Dense(1, activation='linear', kernel_regularizer=l2(0.001),
                                   kernel_initializer=dense_init, name='major_rf')(x)
            y_minor = layers.Dense(1, activation='linear', kernel_regularizer=l2(0.001),
                                   kernel_initializer=dense_init, name='minor_rf')(x)
            y_vminor = layers.Dense(1, activation='linear', kernel_regularizer=l2(0.001),
                                    kernel_initializer=dense_init, name='vminor_rf')(x)
            regressor = tf.keras.Model([inputs], [y, y_major, y_minor, y_vminor],
                                       name="regressive_predictor")
        else:
            regressor = tf.keras.Model([inputs], [y], name="regressive_predictor")
        regressor.summary()
        return regressor


class BYOLModel(tf.keras.Model):
    """
    Adapted from https://github.com/garder14/byol-tensorflow2
    Based on https://arxiv.org/abs/2006.07733
    """
    def __init__(self, input_dim=(128, 128, 5), features_input_dim=256, alpha=1, output_dim=1, **kwargs):
        super(BYOLModel, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.feature_extractor = FeatureExtractor(input_dim=input_dim).build()
        self.regressive_predictor = RegressionPredictor(input_dim=features_input_dim,
                                                        output_dim=output_dim).build()
        self.projection_head = ProjectionHead(input_dim=features_input_dim).build()
        self.projection_head_redo = ProjectionHeadRedo(input_dim=128).build()

        self.feature_extractor_t = FeatureExtractor(input_dim=input_dim, name="target_feature_extractor").build()
        self.projection_head_t = ProjectionHead(input_dim=features_input_dim, name="target_projector").build()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.regression_loss_tracker = tf.keras.metrics.Mean(name="regression_loss")
        self.byol_loss_tracker = tf.keras.metrics.Mean(name="byol_prediction_loss")

        self.test_total_loss_tracker = tf.keras.metrics.Mean(name="test_total_loss")
        self.test_regression_loss_tracker = tf.keras.metrics.Mean(name="test_regression_loss")
        self.test_byol_loss_tracker = tf.keras.metrics.Mean(name="test_byol_prediction_loss")

        byol_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=10000,
            decay_rate=0.9)
        r_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=10000,
            decay_rate=0.9)

        self.byol_optimizer = tf.keras.optimizers.Adam(learning_rate=byol_lr_schedule)
        self.r_optimizer = tf.keras.optimizers.Adam(learning_rate=r_lr_schedule)

        self.mse_loss_object = tf.keras.losses.MeanSquaredError()
        self.byol_loss_object = byol_loss
        self.alpha = alpha

    @property
    def metrics(self):
        metrics = [
            self.regression_loss_tracker,
            self.byol_loss_tracker,
            self.total_loss_tracker,
            self.test_regression_loss_tracker,
            self.test_byol_loss_tracker,
            self.test_total_loss_tracker
        ]
        return metrics

    def test_step(self, dataset):
        x1 = dataset['image_aug1']
        x2 = dataset['image_aug2']
        y = dataset['exsitu_fraction']

        features_1 = self.feature_extractor(x1)
        projection_1 = self.projection_head(features_1)
        projection_redo_1 = self.projection_head_redo(projection_1)

        features_2 = self.feature_extractor(x2)
        projection_2 = self.projection_head(features_2)
        projection_redo_2 = self.projection_head_redo(projection_2)

        predictions = self.regressive_predictor(features_1)
        label_loss = self.mse_loss_object(y, predictions)

        features_t_1 = self.feature_extractor_t(x1)
        projection_t_1 = self.projection_head_t(features_t_1)

        features_t_2 = self.feature_extractor_t(x2)
        projection_t_2 = self.projection_head_t(features_t_2)

        p_online = tf.concat([projection_redo_1, projection_redo_2], axis=0)
        z_target = tf.concat([projection_t_2, projection_t_1], axis=0)

        b_loss = self.byol_loss_object(p_online, z_target)

        total_loss = self.alpha * label_loss + b_loss

        self.test_total_loss_tracker.update_state(total_loss)
        self.test_regression_loss_tracker.update_state(label_loss)
        self.test_byol_loss_tracker.update_state(b_loss)

        losses = {
            "loss": self.test_total_loss_tracker.result(),
            "mse": self.test_regression_loss_tracker.result(),
            "byol": self.test_byol_loss_tracker.result(),
            #  "dal": self.test_domain_loss_tracker.result()
        }
        return losses

    @tf.function
    def train_step(self, dataset):
        x1 = dataset['image_aug1']
        x2 = dataset['image_aug2']
        y = dataset['exsitu_fraction']

        # forward pass
        features_t_1 = self.feature_extractor_t(x1)
        projection_t_1 = self.projection_head_t(features_t_1)

        features_t_2 = self.feature_extractor_t(x2)
        projection_t_2 = self.projection_head_t(features_t_2)

        with tf.GradientTape(persistent=True) as tape:
            features_1 = self.feature_extractor(x1)
            projection_1 = self.projection_head(features_1)
            projection_redo_1 = self.projection_head_redo(projection_1)

            features_2 = self.feature_extractor(x2)
            projection_2 = self.projection_head(features_2)
            projection_redo_2 = self.projection_head_redo(projection_2)

            predictions = self.regressive_predictor(features_1)
            label_loss = self.mse_loss_object(y, predictions)

            p_online = tf.concat([projection_redo_1, projection_redo_2], axis=0)
            z_target = tf.concat([projection_t_2, projection_t_1], axis=0)

            b_loss = self.byol_loss_object(p_online, z_target)
            total_loss = self.alpha * label_loss + b_loss

        grads_byol = tape.gradient(total_loss, self.feature_extractor.trainable_variables)
        grads_byol2 = tape.gradient(b_loss, self.projection_head.trainable_variables)
        grads_byol_redo = tape.gradient(b_loss, self.projection_head_redo.trainable_variables)

        self.byol_optimizer.apply_gradients(zip(grads_byol + grads_byol2 + grads_byol_redo,
                                                self.feature_extractor.trainable_variables +
                                                self.projection_head.trainable_variables +
                                                self.projection_head_redo.trainable_variables))

        r_grads = tape.gradient(label_loss, self.regressive_predictor.trainable_variables)
        self.r_optimizer.apply_gradients(zip(r_grads, self.regressive_predictor.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.regression_loss_tracker.update_state(label_loss)
        self.byol_loss_tracker.update_state(b_loss)

        losses = {
            "loss": self.total_loss_tracker.result(),
            "r_loss": self.regression_loss_tracker.result(),
            "byol_loss": self.byol_loss_tracker.result(),
        }
        return losses

    def afterBatch(self, batch, logs):
        # Update target networks (exponential moving average of online networks)
        beta = 0.99
        f_target_weights = self.feature_extractor_t.get_weights()
        f_online_weights = self.feature_extractor.get_weights()
        for i in range(len(f_online_weights)):
            f_target_weights[i] = beta * f_target_weights[i] + (1 - beta) * f_online_weights[i]
        self.feature_extractor_t.set_weights(f_target_weights)

        g_target_weights = self.projection_head_t.get_weights()
        g_online_weights = self.projection_head.get_weights()
        for i in range(len(g_online_weights)):
            g_target_weights[i] = beta * g_target_weights[i] + (1 - beta) * g_online_weights[i]
        self.projection_head_t.set_weights(g_target_weights)

    def afterEpoch(self, epoch, logs):
        self.alpha += 0.5
