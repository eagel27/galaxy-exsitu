import os.path
import pandas as pd
import numpy as np
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt

from utilities import calculate_corrected_y_pred_samples, plot_with_median, plot_with_error
from constants import IGNORE_CHANNELS_MAPPING, MASK_RADIUS_MAPPING, DATASET_MAPPING, MODELS_RUN_PATH,\
    DATASET_1D, ENSEMBLE_RUN_PATH, CNN_RESULTS_RUN_PATH


logging.basicConfig(format='%(asctime)s %(message)s',
                    level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger("input_logger")


def load_original_model(model_id, simulation, channels_str, mask_radius):
    from nn_models import load_saved_model
    model_path = os.path.join(MODELS_RUN_PATH,
                              'model_{}_{}_Channels_{}_MRadius_{}.h5'.format(
                                  model_id, simulation, channels_str, mask_radius))
    model = load_saved_model(model_path)
    return model


def load_predictions(train_simulation, model_id, ignore_channels, mask_radius,
                     dataset_main):
    from nn_data import input_fn_split, input_2d_cnn_fn_split
    channels_str = '_'.join(str(i) for i in {0, 1, 2, 3, 4}.difference(
            set(ignore_channels or ())))

    model = load_original_model(model_id, train_simulation, channels_str, mask_radius)

    input_fn = input_2d_cnn_fn_split
    if DATASET_1D:
        input_fn = input_fn_split

    ds_test_main = input_fn('test', dataset_main,
                            ignore_channels=ignore_channels,
                            mask_radius=mask_radius,
                            remove_object_id=False)

    data = ds_test_main.take(30)
    samples_list, y_true, obj_id, images = None, [], [], []
    for d in list(data):
        images.extend(d[0].numpy())
        y_true.extend(d[1].numpy())
        obj_id.extend(d[2].numpy())

        exsitu_pred_distr = model(d[0].numpy())

        samples = exsitu_pred_distr.sample(1000)
        if samples_list is not None:
            samples_list = np.hstack([samples_list, samples])
        else:
            samples_list = samples

    y_true_main = np.array(y_true)
    str_ids = np.array(obj_id)

    obj_ids = [int(obj_id.decode().split('_')[0]) for obj_id in str_ids]
    snap = [int(obj_id.decode().split('_')[1]) for obj_id in str_ids]
    return y_true_main, samples_list, obj_ids, snap


def calc_ensemble_predictions(train_simulation, test_simulation, ignore_channels, mask_radius,
                              cross, prior):
    merged = None
    channels_str = '_'.join(str(i) for i in {0, 1, 2, 3, 4}.difference(
        set(ignore_channels or ())))

    for model_id in range(5):
        true_values, samples, obj_ids, snap = load_predictions(train_simulation, model_id,
                                                               ignore_channels,
                                                               mask_radius,
                                                               test_simulation[0])
        if merged is None:
            merged = samples
        else:
            merged = np.concatenate([merged, samples], axis=0)

    sim_exsitu_f = true_values.flatten()
    pred_exsitu_f = np.mean(merged, axis=0).flatten()
    pred_exsitu_f_std = np.std(merged, axis=0).flatten()

    posterior, posterior_mode, corrected_posterior, \
        pred_exsitu_f_cor, _ = calculate_corrected_y_pred_samples(merged, prior)

    file_suffix = '{}{}{}'.format(train_simulation,
                                  '_{}'.format(test_simulation[1]) if train_simulation == 'BOTH' else '',
                                  '_CROSS' if cross else '')
    df = pd.DataFrame()
    df['Snapshot'] = snap
    df['GalaxyID'] = obj_ids
    df['ExSituFraction'] = pred_exsitu_f
    df['ExSituFraction_true'] = sim_exsitu_f
    df['ExSituFraction_std'] = pred_exsitu_f_std
    df['ExSituFraction_corr'] = pred_exsitu_f_cor
    df['ExSituFraction_map'] = posterior_mode

    save_path = ENSEMBLE_RUN_PATH
    df.to_hdf(
        save_path + '/all_{}_{}_{}.h5py'.format(file_suffix, channels_str, mask_radius),
        '/data')
    return


def ensemble_predictions():
    """
    Ensemble the predictions of all 5 trained models
    :return:
    """
    from nn_data import compute_prior_whole_dataset

    for cross in (False, True):
        for train_simulation in ('TNG', 'EAGLE'):
            # when training on both simulations, there is not a cross testing
            if train_simulation == 'BOTH' and cross is True:
                continue

            prior = compute_prior_whole_dataset(dataset=train_simulation, split='train')

            if not os.path.exists(ENSEMBLE_RUN_PATH):
                os.makedirs(ENSEMBLE_RUN_PATH)

            if train_simulation == 'BOTH':
                test_datasets = [(DATASET_MAPPING['TNG'], 'TNG100'),
                                 (DATASET_MAPPING['EAGLE'], 'EAGLE-L100')]
            elif (train_simulation == 'TNG') ^ cross:
                test_datasets = [(DATASET_MAPPING['TNG'], 'TNG100')]
            else:
                test_datasets = [(DATASET_MAPPING['EAGLE'], 'EAGLE-L100')]

            for test_dataset in test_datasets:
                for ignore_channels in IGNORE_CHANNELS_MAPPING.keys():
                    for mask_radius in MASK_RADIUS_MAPPING.keys():
                        calc_ensemble_predictions(train_simulation, test_dataset,
                                                  ignore_channels, mask_radius,
                                                  cross, prior)


def create_figure(f, test_sim_name, train_simulation):

    gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0)

    ax1 = f.add_subplot(gs[0, 0])
    ax2 = f.add_subplot(gs[1, 0], sharex=ax1)

    ax1.set_title('CNN trained on {}'.format(train_simulation))
    return ax1, ax2


def config_figure(ax1, ax2):
    ax1.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'k--')
    ax2.plot(np.arange(0, 1.1, 0.1), [0] * 11)
    ax2.plot(np.arange(0, 1.1, 0.1), [0.1] * 11, 'k--', alpha=0.3)
    ax2.plot(np.arange(0, 1.1, 0.1), [-0.1] * 11, 'k--', alpha=0.3)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.1, 1.1)
    #ax1.set_xticks([])
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_yticks(ax1.get_yticks()[1:-1])
    ax1.set_ylabel('Predictions', fontsize=14)
    ax1.legend(loc='upper left')

    #ax2.set_xticks(ax2.get_xticks())

    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.4, 0.4)
    ax2.set_xlabel('True Values', fontsize=14)
    ax2.set_ylabel(r'$\epsilon$', fontsize=14)

    return ax1, ax2


def plot_predictions(test_set_Y, Y_pred, y_std, test_sim, train_sim, model_id):
    f_channel = plt.figure(figsize=(4, 5))
    ax1_channel, ax2_channel = create_figure(f_channel, test_sim, train_sim)
    plot_with_error(test_set_Y[:128], Y_pred[:128], y_std[:128], ax1_channel,
                     color='blue')

    plot_with_median(test_set_Y, Y_pred - test_set_Y, ax2_channel,
                     color1='blue', label=None, percentiles=(16, 84), apply_log=False)

    config_figure(ax1_channel, ax2_channel)
    #plt.show()
    save_path = CNN_RESULTS_RUN_PATH

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    f_channel.savefig(os.path.join(save_path,
                                   '{}_on_{}_model_{}'.format(train_sim, test_sim, model_id)))


def plot_ensembled_predictions():
    """ Plot 1-1 plots to check accuracy of models """
    if not os.path.exists(CNN_RESULTS_RUN_PATH):
        os.makedirs(CNN_RESULTS_RUN_PATH)

    load_path = ENSEMBLE_RUN_PATH

    for cross in (False, True):
        for train_simulation in ('TNG', 'EAGLE'):
            # when training on both simulations, there is not a cross testing
            if train_simulation == 'BOTH' and cross is True:
                continue

            same_data = pd.read_hdf(os.path.join(
                load_path,
                'all_{}_0_1_2_3_4_None.h5py'.format(train_simulation)))

            cross_data = pd.read_hdf(os.path.join(
                load_path,
                'all_{}_CROSS_0_1_2_3_4_None.h5py'.format(train_simulation)))

            plot_predictions(same_data['ExSituFraction_true'],
                             same_data['ExSituFraction_map'], same_data['ExSituFraction_std'],
                             train_simulation, train_simulation, 'All')

            plot_predictions(cross_data['ExSituFraction_true'],
                             cross_data['ExSituFraction_map'], cross_data['ExSituFraction_std'],
                             'EAGLE' if train_simulation == 'TNG' else 'TNG',
                             train_simulation, 'All')


if __name__ == '__main__':
    plt.style.use('science')
    plt.rcParams.update({'font.size': 16})
    plt.rc('axes', titlesize=16, labelsize=16)

    ensemble_predictions()
    #plot_ensembled_predictions()
