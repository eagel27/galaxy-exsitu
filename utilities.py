from constants import TNG_SNAPSHOT
import os
import re
import numpy as np
from scipy.integrate import simps
import pandas as pd
import pickle
from constants import TNG_DATASET_SPLITS, EAGLE_DATASET_SPLITS, DATASET_1D_RUN_PATH


def plot_with_median(data_x, data_y, ax, color1='darkblue', color2=None,
                     label=None, percentiles=(16, 84), total_bins=20, apply_log=False):
    """
    Plot the running media of the data_x, data_y data with requested percentiles.
    :param data_x: The data inputs
    :param data_y: The data outputs
    :param ax: The matplotlib ax to plot on
    :param color1: The color of the median line
    :param color2: The color of the shaded regions of the percentiles
    :param label: The label that should be displayed on the legend
    :param percentiles: The percentiles to plot along with the median
    :param total_bins: The number of bins to digitize the data. Increase to increase detail.
    :param apply_log: Whether the x-axis should be in log
    :return:
    """

    if color2 is None:
        color2 = color1

    if apply_log:
        bins = np.geomspace(data_x.min(), data_x.max(), total_bins)
    else:
        bins = np.linspace(data_x.min(), data_x.max(), total_bins)

    delta = bins[1] - bins[0]
    idx = np.digitize(data_x, bins)
    running_median = [np.nanmedian(data_y[idx == k]) for k in range(total_bins)]
    running_prc_low = [np.nanpercentile(data_y[idx == k], percentiles[0])
                       for k in range(total_bins)]
    running_prc_high = [np.nanpercentile(data_y[idx == k], percentiles[1])
                        for k in range(total_bins)]

    if percentiles:
        ax.plot(bins-delta/2, running_median, color=color1, lw=2, alpha=.8, label=label)
        ax.fill_between(bins - delta / 2, running_prc_low, running_median, facecolor=color2, alpha=0.1)
        ax.fill_between(bins - delta / 2, running_prc_high, running_median, facecolor=color2, alpha=0.1)
    else:
        ax.plot(bins - delta / 2, running_median, color=color1, linestyle='--', lw=2, alpha=.8,  label=label)

    if apply_log:
       ax.set_xscale('symlog')


def plot_with_error(data_x, data_y, err, ax, color='blue'):
    """

    :param data_x:
    :param data_y:
    :param err:
    :param ax:
    :param color:
    :return:
    """
    ax.errorbar(data_x, data_y, yerr=err, linestyle="None", fmt='o',
                capsize=3, color=color, capthick=0.5)


def load_splits_file(splits_file):
    with open(splits_file, 'rb') as f:
        splits = pickle.load(f)
    return splits


def load_split_df(splits_file, mode='train'):
    splits = load_splits_file(splits_file)
    split_df = pd.DataFrame(columns=['GalaxyID', 'Snapshot'])
    if mode == 'all':
        modes = ['train', 'test', 'validation']
    else:
        modes = [mode]
    for m in modes:
        split = pd.DataFrame(splits[m], columns=['GalaxyID', 'Snapshot'])
        split_df = pd.concat([split_df, split])
    return split_df


def load_all(simulation):
    try:
        all_sim = pd.read_hdf(os.path.join(DATASET_1D_RUN_PATH,
                              'dataset_{}.hdf5'.format(simulation)),
                              '/data')
    except:
        all_sim = pd.read_hdf(os.path.join(DATASET_1D_RUN_PATH,  # old format
                              'properties_{}.h5py'.format(simulation)),
                              '/data')

    all_sim['Simulation'] = simulation.upper()
    #all_sim = all_sim.loc[all_sim.index.repeat(all_sim.Aligns)]
    all_sim.loc[all_sim.Gas_Mass == 0.00, 'Gas_Mass'] = 0.001
    all_sim.loc[all_sim.BH_Mass == 0.00, 'BH_Mass'] = 0.001
    #all_sim['Metallicity'] = np.log10(all_sim.Metallicity)
    return all_sim


def load_mode(simulation, mode):
    split_df = load_split_df(TNG_DATASET_SPLITS if simulation in ('TNG', 'tng')
                             else EAGLE_DATASET_SPLITS,
                             mode)
    all_sim_df = load_all(simulation)
    split_df = split_df.merge(all_sim_df)
    return split_df


def calculate_corrected_y_pred_samples(samples, prior):
    """ Calculate the MAP value along with the corrected from prior prediction """
    x = np.linspace(0.01, 0.99, 50)

    logps = []
    for i in range(samples.shape[1]):
        a = np.histogram(samples[:, i], bins=np.append(x, [1]))[0]
        logps.append(a)

    logps = np.stack(logps)
    logps = logps.transpose()
    logps_mode = x[np.exp(logps).argmax(axis=0)]

    corrected_posterior = logps / (prior.pdf(x).reshape((-1, 1)))
    # corrected_posterior[corrected_posterior <= 0] = 1e-2

    y_pred_prior_mean = (simps(x.reshape((-1, 1)) * corrected_posterior, x, axis=0) /
                         simps(corrected_posterior, x, axis=0))

    y_pred_mode = x[np.exp(corrected_posterior).argmax(axis=0)]

    return logps, logps_mode, corrected_posterior, y_pred_prior_mean, y_pred_mode


def list_all_dir_files(dir_path):
    """ List all files in a given directory """
    dir_files = [f for f in os.listdir(dir_path)
                 if os.path.isfile(os.path.join(dir_path, f))]
    return dir_files


def get_galaxy_id_from_fits_filename(filename):
    regex = r'moments_TNG100-1_{}_(\d+)_stars_i\d__\d+.fits'.format(TNG_SNAPSHOT)
    galaxy_id = int(re.match(regex, filename).groups(0)[0])
    return galaxy_id


def get_galaxy_ids(dir_path):
    filenames = list_all_dir_files(dir_path)
    regex = r'moments_TNG100-1_{}_(\d+)_stars_i\d__\d+.fits'.format(TNG_SNAPSHOT)
    galaxy_ids = set(re.match(regex, f).groups(0)[0] for f in filenames)
    return galaxy_ids


def log10(x):
    import tensorflow as tf
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
