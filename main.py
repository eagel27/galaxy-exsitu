import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.client import device_lib
from contextlib import redirect_stdout
from constants import BATCHES, RESULTS_PATH, MDN, RUN_DIR, MODELS_RUN_PATH, DATASET_MAPPING, \
    DATASET_1D, INPUT_SHAPE, IGNORE_CHANNELS_MAPPING, MASK_RADIUS_MAPPING
from nn_data import input_fn_split, input_2d_cnn_fn_split, get_num_examples, \
    compute_prior_whole_dataset, input_plots, get_data
from nn_models import build_cnn_model, load_saved_model, loss_fn
from nn_results import GraphPlotter

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger("input_logger")


class CNNModel(object):

    def __init__(self, model_id, model_fn, ignore_channels=None, mask_radius=None,
                 per_channel=False, dataset_name='TNG', weight_initializers=None, ax=None):
        """ Initialize variables required for training and evaluation of model"""

        self.dataset_name = dataset_name
        self.model_id = model_id
        self.model_fn = model_fn
        self.ax = ax

        self.ignore_channels = ignore_channels
        self.mask_radius = mask_radius
        self.per_channel = per_channel
        self.weight_initializers = weight_initializers

        self.model = None
        self.autoencoder_model = None
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None
        self.len_ds_train = self.len_ds_val = self.len_ds_test = 0
        self.n_epochs = 0

        self.test_set_mimic = None
        self.ae = None
        self.test_other_mimic = None
        self.mask_reverse = False

        self.channels_str = '_'.join(str(i) for i in {0, 1, 2, 3, 4}.difference(
            set(self.ignore_channels or ())))

        path_name = '{}_{}_Channels_{}_MRadius_{}{}'.format(
            self.model_id, dataset_name, self.channels_str, mask_radius,
            '_per_channel' if self.per_channel else '')

        # Create directory for this run
        self.run_dir = os.path.join(RESULTS_PATH, RUN_DIR, path_name)
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

        self.plotter = GraphPlotter(self.run_dir, self.model_id)

        # Save model in dedicated save_models folder
        # (so that it is not copied along with the rest of the results)
        self.model_file_path = os.path.join(MODELS_RUN_PATH,
                                            'model_{}.h5'.format(path_name))

    def load_ds(self, mode, dataset_str):
        """
        Load the respective dataset by taking into account which channels to ignore
        and outside which mask radius to mask
        :param mode: 'train' or 'test' or 'validation'
        :param dataset_str: The dataset string identifying which dataset to load
        :return:
        """
        input_fn = input_2d_cnn_fn_split
        if DATASET_1D:
            input_fn = input_fn_split
        return input_fn(mode, dataset_str,
                        ignore_channels=self.ignore_channels,
                        mask_radius=self.mask_radius)

    def load_datasets(self, dataset_str='tng_dataset'):
        """
        Load the train, validation and test sets and plot some informative graphs
        :return:
        """

        self.ds_train = self.load_ds('train', dataset_str=dataset_str)
        self.ds_val = self.load_ds('validation', dataset_str=dataset_str)
        self.ds_test = self.load_ds('test', dataset_str=dataset_str)

        self.len_ds_train = get_num_examples('train', dataset_str=dataset_str)
        self.len_ds_val = get_num_examples('validation', dataset_str=dataset_str)
        self.len_ds_test = get_num_examples('test', dataset_str=dataset_str)
        log.debug(self.len_ds_test, self.len_ds_val, self.len_ds_train)

        # Plot some informative graphs for the input data of the CNN
        input_plots(self.ds_train, self.run_dir)

    def train_model(self):
        """
        Train the CNN model using the train set and check for
        overfitting with the validation set
        :return:
        """
        log.info('*************** TRAINING *******************')
        log.info('Running training with Channels: {} and '
                 'Radius: {}'.format(self.channels_str, self.mask_radius))

        tf.compat.v1.reset_default_graph()
        self.model = self.model_fn(INPUT_SHAPE, mdn=MDN, weight_initializers=self.weight_initializers)

        with open(self.run_dir + '/modelsummary_{}.txt'.format(self.model_id), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

        # Train model
        es = EarlyStopping(monitor='val_loss', patience=25)
        mc = ModelCheckpoint(filepath=self.model_file_path, monitor='val_loss', save_best_only=True)
        history = self.model.fit(self.ds_train,
                                 epochs=500,
                                 steps_per_epoch=self.len_ds_train // BATCHES,
                                 validation_steps=self.len_ds_val // BATCHES,
                                 validation_data=self.ds_val,
                                 callbacks=[es, mc],
                                 use_multiprocessing=True, workers=4)

        self.n_epochs = len(history.history['loss'])
        self.plotter.plot_training_graphs(history)

        self.model = self.load_saved_model()
        log.info('Evaluate with training set on best model containing {} examples'.format(self.len_ds_train))
        train_loss, train_mse = self.model.evaluate(self.ds_train,
                                                    steps=100, verbose=2)
        log.info('Best train Loss: {}, Best train MSE: {}'.format(train_loss, train_mse))

    def load_saved_model(self, lr=1e-3, do_compile=True):
        self.model = load_saved_model(self.model_file_path, mdn=True, lr=lr, do_compile=do_compile)
        return self.model

    def evaluate_model(self):
        """
        Evaluate the performance of the CNN using the test set
        :return:
        """
        self.model = self.load_saved_model()
        log.info('*************** EVALUATING *******************')
        log.info('Evaluate with test set containing {} examples'.format(self.len_ds_test))
        test_loss, test_mse = self.model.evaluate(self.ds_test, verbose=2)
        log.info('Test Loss: {}, Test MSE: {}'.format(test_loss, test_mse))

        with open(self.run_dir + "/Results.txt", "w") as result_file:
            result_file.write("Trained for epochs: %s\n\n"
                              "Test loss, MSE: %s %s" % (self.n_epochs, test_loss, test_mse))

        # Plot some informative graphs for the test data of the CNN
        input_plots(self.ds_test, self.run_dir, test=True)

        images, y_true = get_data(self.ds_test, batches=8)
        y_pred = self.model.predict(images).flatten()
        y_pred = np.array(y_pred)

        y_pred_distr = None
        if MDN:
            y_pred_distr = self.model(images)
            y_pred = y_pred_distr.mean().numpy().reshape(-1)

        self.plotter.plot_evaluation_results(y_true, y_pred, prior, y_pred_distr=y_pred_distr, mdn=MDN)
        self.plotter.plot_saliency_maps(self.model, images, y_true, y_pred)

    def cross_evaluate(self, dataset_str='eagle_dataset', cross_dir_suffix=''):
        """
        Cross-evaluate the performance of the CNN using the test set
        from the other simulation
        :return:
        """
        self.model = self.load_saved_model()
        log.info('*************** CROSS EVALUATING *******************')
        ds_test_other = self.load_ds('test', dataset_str)
        len_ds_test = get_num_examples('test', dataset_str=dataset_str)
        log.info('Evaluate with test set containing {} examples'.format(len_ds_test))

        test_loss, test_mse = self.model.evaluate(ds_test_other, verbose=2)
        log.info('Cross test Loss: {}, Cross test MSE: {}'.format(test_loss, test_mse))

        cross_run_dir = os.path.join(self.run_dir, 'Cross{}'.format(cross_dir_suffix))
        if not os.path.exists(cross_run_dir):
            os.makedirs(cross_run_dir)

        with open(cross_run_dir + "/Results_cross{}.txt".format(cross_dir_suffix), "w") as result_file:
            result_file.write("Trained for epochs: %s\n\n"
                              "Cross test loss, Cross MSE: %s %s" % (self.n_epochs, test_loss, test_mse))

        # Plot some informative graphs for the test data of the CNN
        input_plots(ds_test_other, cross_run_dir, test=True)

        images, y_true = get_data(ds_test_other, batches=8)
        y_pred = self.model.predict(images).flatten()
        y_pred = np.array(y_pred)

        y_pred_distr = None
        if MDN:
            y_pred_distr = self.model(images)
            y_pred = y_pred_distr.mean().numpy().reshape(-1)

        cross_plotter = GraphPlotter(cross_run_dir, self.model_id)
        cross_plotter.plot_evaluation_results(y_true, y_pred, prior, y_pred_distr=y_pred_distr, mdn=MDN)
        cross_plotter.plot_saliency_maps(self.model, images, y_true, y_pred)

    def run(self):
        """
        Load the datasets, train and evaluate results of model
        :return:
        """
        dataset_name_cross = 'EAGLE'
        if self.dataset_name == 'EAGLE':
            dataset_name_cross = 'TNG'

        dataset_main = DATASET_MAPPING[self.dataset_name]
        dataset_other = DATASET_MAPPING[dataset_name_cross]

        self.load_datasets(dataset_main)
        self.train_model()
        if self.dataset_name != 'BOTH':
            self.evaluate_model()
            self.cross_evaluate(dataset_other)
        else:
            self.cross_evaluate(DATASET_MAPPING['TNG'], cross_dir_suffix='_TNG')
            self.cross_evaluate(DATASET_MAPPING['EAGLE'], cross_dir_suffix='_EAGLE')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    log.info(device_lib.list_local_devices())
    plt.style.use('science')
    plt.rcParams.update({'font.size': 16})
    plt.rc('axes', titlesize=16, labelsize=16)

    zero_init = tf.keras.initializers.Zeros()
    ones_init = tf.keras.initializers.Ones()
    random_init = tf.keras.initializers.RandomUniform(minval=0, maxval=1, seed=666)
    random_centered_init = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=666)
    random_normal_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=666)
    random_normal_init2 = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=None)

    weight_initializers_all = [(random_normal_init2, random_normal_init2),
                               (random_normal_init2, random_normal_init2),
                               (random_normal_init2, None),
                               (None, random_normal_init2),
                               (None, None)]

    for simulation in ('TNG', 'EAGLE'):
        prior = compute_prior_whole_dataset(dataset=simulation, split='train')
        for ignore_channels in IGNORE_CHANNELS_MAPPING.keys():
            for mask_radius in MASK_RADIUS_MAPPING.keys():
                for i, weight_initializers in enumerate(weight_initializers_all):
                    with tf.device('/gpu:0'):
                        cnn_model = CNNModel(i, build_cnn_model,
                                             ignore_channels=ignore_channels,
                                             mask_radius=mask_radius, dataset_name=simulation,
                                             weight_initializers=weight_initializers)
                        cnn_model.run()
