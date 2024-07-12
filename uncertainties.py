import os
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib

from constants import UNCERTAINTIES_RUN_PATH, MODELS_RUN_PATH, DATASET_1D, DATASET_MAPPING
from dataloader.dataloader import input_fn_split, input_2d_cnn_fn_split, get_data
from models.nn_models import load_saved_model
from utilities import plot_with_median

logging.basicConfig(format='%(asctime)s %(message)s',
                    level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger("input_logger")


def get_data_model(model_id, dataset_str, ignore_channels, mask_radius):
    """
    Get the trained model and the input and outputs to recover coverage probabilities
    :param model_id: The model_id
    :param dataset_str: The training dataset string
    :param ignore_channels: If and which channels have been ignored during training (or ())
    :param mask_radius: The radius of the inputs that has been mased during training (or None)
    :return:
    """
    channels_str = '_'.join(str(i) for i in {0, 1, 2, 3, 4}.difference(
                    set(ignore_channels or ())))
    model_path = os.path.join(MODELS_RUN_PATH,
                              'model_{}_{}_Channels_{}_MRadius_{}.h5'.format(
                                  model_id, simulation, channels_str, mask_radius))

    model = load_saved_model(model_path)
    images, y_true = get_data(dataset_str, batches=15)
    return images, y_true, model


def get_coverage_probabilities(data_x, data_y, model):
    """
    Compute the coverage probabilities of the provided model
    on the test dataset
    :param data_x: The data inputs
    :param data_y: The data true values
    :param model: The trained probabilistic model that predicts distributions
    :return: (x, y) a tuple of lists corresponding to a list of probability volumes
    and the corresponding percentage of true values in that volume.
    """
    y_pred_distr = model(data_x)
    y_pred = y_pred_distr.mean().numpy().reshape(-1)
    y_pred_std = y_pred_distr.stddev().numpy().reshape(-1)

    errors = np.absolute(data_y - y_pred)
    x, y = [], []
    for sigma_times in np.arange(0, 3, 0.01):
        how_many = np.count_nonzero(errors <= sigma_times * y_pred_std)
        y.append(how_many / data_y.shape[0])
        x.append(math.erf(sigma_times / math.sqrt(2)))

    return x, y


def init_uncertainty_plot():
    """
    Init the coverage plot. Use two panels for both trainings
    :return:
    """
    plt.close()
    f, ax = plt.subplots(2, 1, sharex='col', sharey=True, figsize=(6, 10))
    return f, ax


def config_uncertainty_plot(ax, simulation, test_set_sim):
    """
    Config the coverage plot and add text for marking
    overconfident and conservative regions
    :return:
    """
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=30)
    ax.set_xlabel('Percentage of probability volume')
    ax.set_ylabel('Percentage of true values in volume')
    ax.set_title('Probability Coverage when ' 
                 'trained on {}'.format(simulation))
    ax.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
    ax.text(0.3, 0.9, '$\it{Conservative}$',
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    ax.text(0.7, 0.1, '$\it{Overconfident}$',
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    return ax


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    log.info(device_lib.list_local_devices())
    plt.style.use('science')
    plt.rcParams.update({'font.size': 30})
    plt.rc('axes', titlesize=25, labelsize=25)

    model_id = 6
    ignore_channels = (3, 4)
    mask_radius = 16

    input_fn = input_2d_cnn_fn_split
    if DATASET_1D:
        input_fn = input_fn_split
    
    if not os.path.exists(UNCERTAINTIES_RUN_PATH):
        os.mkdir(UNCERTAINTIES_RUN_PATH)
    
    f, ax = init_uncertainty_plot()
    for simulation in ('TNG', 'EAGLE'):
        for i, cross in enumerate((False, True)):
            test_set_sim = 'TNG'
            if (simulation == 'EAGLE') ^ cross:
                test_set_sim = 'EAGLE'
            
            dataset_str = DATASET_MAPPING[test_set_sim]

            suffix = '_cross' if cross else ''
            if simulation == 'BOTH':
                suffix = '_EAGLE' if cross else '_TNG'

            ds_test = input_fn('test', dataset_str, 
                               ignore_channels=ignore_channels, 
                               mask_radius=mask_radius)

            x, y = [], []
            for model_id in range(5):
                data_x, data_y, model = get_data_model(model_id, ds_test, ignore_channels, mask_radius)
                x1, y1 = get_coverage_probabilities(data_x, data_y, model)
                x.extend(x1)
                y.extend(y1)

            x = np.array(x)
            y = np.array(y)
            plot_with_median(x, y, ax[i], label='Test on ' + test_set_sim,
                             color1='blue' if test_set_sim == 'TNG' else 'green')

        f.savefig(os.path.join(UNCERTAINTIES_RUN_PATH,
                               'coverage_{}_{}_{}.png'.format(simulation, 
                                   'MK' if ignore_channels == (3, 4) else 'All', 
                                   '1Re' if mask_radius == 16 else '4Re',
                                   )))
        plt.close()


