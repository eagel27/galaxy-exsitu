import os.path

import matplotlib.pyplot as plt
import copy
import os 

import numpy as np
import pandas as pd

from preprocess import *
from constants import DATASET_1D, DATASET_MAPPING, GRADIENTS_RUN_PATH, INPUT_CORRELATION_PATH_RUN_PATH, \
    NORM_MAPPING, GRADIENTS_MAPPING


def presentation_plot(save_dir, images, labels):
    for i in range(3):
        fig, axes = plt.subplots(nrows=2, ncols=1)

        data1 = images[0][:, :, i]
        data2 = images[1][:, :, i]
        if i == 0:
            data1 = tf.math.log(data1).numpy()
            data2 = tf.math.log(data2).numpy()

        # find minimum of minima & maximum of maxima
        minmin = np.min([np.min(data1), np.min(data2)])
        maxmax = np.max([np.max(data1), np.max(data2)])

        im1 = axes[0].imshow(data1, vmin=minmin, vmax=maxmax, aspect='auto')
        im2 = axes[1].imshow(data2, vmin=minmin, vmax=maxmax, aspect='auto')
        # add space for colour bar
        fig.subplots_adjust(right=0.4)
        cbar_ax = fig.add_axes([0.45, 0.15, 0.04, 0.7])
        fig.colorbar(im2, cax=cbar_ax)
        plt.savefig(save_dir + '/map_{}'.format(i))


def plot_original_maps(save_dir, images, labels, obj_ids, test=False):
    # Plot 3 images on all 5 features
    channels = images[0].shape[-1]
    fig, big_axes = plt.subplots(figsize=(20.0, 12.0), nrows=3, ncols=1, sharey=True)

    for row, big_ax in enumerate(big_axes):
        big_ax.set_title('Ex-situ fraction: %.3f, Object: %s' % (labels[row], obj_ids[row]), rotation=0)

        # Turn off axis lines and ticks of the big subplot
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

        for i in range(channels):
            ax = fig.add_subplot(3, channels, channels*row + i + 1)
            ax.imshow(images[row][:, :, i], cmap="jet")

    fig.set_facecolor('w')
    plt.tight_layout()
    plt.savefig(save_dir + '/{}Original maps.png'.format('Test ' if test else ''))
    plt.close()


def plot_galaxy_histograms(save_dir, images, test=False):
    for im_i in range(3):
        image = images[im_i]
        print("Dimensions of the Input are:", image.shape)
        channels = image.shape[-1]
        fig, axs = plt.subplots(channels, 2, figsize=(12.0, 12.0))
        for i in range(channels):
            im = axs[i, 0].imshow(image[:, :, i], cmap="jet")
            axs[i, 1].hist(image[:, :, i].flatten())
        plt.savefig(save_dir +'/{}Maps - Histograms_{}.png'.format('Test ' if test else '', im_i))
        plt.close()


def plot_preprocessing_histograms(save_dir, image):
    # Plot hist on normalized and standarized images
    for i in range(5):
        fig, big_axes = plt.subplots(figsize=(8.0, 16.0), nrows=4, ncols=1, sharey=True)

        titles = ['Original', 'Per Image Standardized', 'Per Channel Standardized', 'Per Channel Standardized - Log']
        for row, big_ax in enumerate(big_axes):
            big_ax.set_title(titles[row], rotation=0)

            # Turn off axis lines and ticks of the big subplot
            # obs alpha is 0 in RGBA string!
            big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
            # removes the white frame
            big_ax._frameon = False

            im = copy.deepcopy(image[:, :, i])

            if row == 1:
                im = per_image_standardization(im, i)
                im = im.numpy()
            elif row == 2:
                im = per_channel_standardization(im, i, dataset_str='tng_dataset', logged=False)
                im = im.numpy()
            elif row == 3:
                im = per_channel_standardization(im, i, dataset_str='tng_dataset', logged=False)
                im = im.numpy()

            ax = fig.add_subplot(4, 2, 2 * row + 1)
            ax.imshow(im, cmap="jet")

            ax = fig.add_subplot(4, 2, 2 * row + 2)
            ax.hist(im.flatten())

        fig.set_facecolor('w')
        plt.tight_layout()
        plt.savefig(save_dir + '/Preprocessing_{}.png'.format(i))
        plt.close()


def plot_correlation_central(save_dir, images, y_true, test=False):
    channels = images[0].shape[-1]
    for channel in range(channels):
        central_pixels = []
        for im in images:
            central_pixel = im[64, 64, channel]
            central_pixels.append(central_pixel)

        plt.scatter(y_true, central_pixels)
        plt.xlabel('Ex-situ fraction')
        plt.ylabel('Central Pixel Value of {} Channel'.format(channel))
        plt.savefig(save_dir + '/{}Correlation_central_pixel_channel_{}'.format('Test ' if test else '',
                                                                                channel))
        plt.close()


def plot_correlation_all(images, y_true, simulation, test=False, statistic='mean', normalize=True):
    channels = images[0].shape[-1]
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(60, 10))

    channels_labels = ['Mass', 'Velocity', 'Velocity $\sigma$', 'Metallicity', 'Age']
    df = pd.DataFrame(columns=['ExsituF'] + channels_labels)
    df['ExsituF'] = y_true
    for channel in range(channels):
        stat_list = []
        for im in images:
            stat = getattr(np, statistic)(im[:, :, channel])
            stat_list.append(stat)
        ax[channel].scatter(y_true, stat_list)
        ax[channel].set_xlabel('Ex-situ fraction')
        ax[channel].set_ylabel('{} Pixel Value of {}'.format(statistic.capitalize(), channels_labels[channel]))
        df[channels_labels[channel]] = stat_list

    df.to_hdf(os.path.join(INPUT_CORRELATION_PATH_RUN_PATH,
                           'correlations_{}_{}_norm_{}.h5py'.format(simulation,
                                                                    statistic,
                                                                    normalize)),
              '/data')

    save_dir = os.path.join(INPUT_CORRELATION_PATH_RUN_PATH,
                            '{}_{}'.format(simulation, normalize))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    plt.tight_layout()
    fig.savefig(save_dir + '/{}Correlation_{}_pixels'.format('Test ' if test else '', statistic))
    plt.close()


def plot_correlation_nonzero(save_dir, images, y_true, test=False, statistic='mean'):
    channels = images[0].shape[-1]
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(60, 10))

    channels_labels = ['Mass', 'Velocity', 'Velocity $\sigma$', 'Metallicity', 'Age']
    for channel in range(channels):
        stat_list = []
        for im in images:
            ch_im = im[:, :, channel]
            corner_pix_values = [ch_im[0, 0], ch_im[127, 0], ch_im[0, 127], ch_im[127, 127]]
            norm_zero_value = max(corner_pix_values, key=corner_pix_values.count)
            mask = (im[:, :, channel] != norm_zero_value)
            if statistic == 'count':
                stat = np.count_nonzero(mask)
            else:
                nonzero_values = im[:, :, channel][mask]
                stat = getattr(np, statistic)(nonzero_values)
            stat_list.append(stat)
        ax[channel].scatter(y_true, stat_list)
        ax[channel].set_xlabel('Ex-situ fraction')
        ax[channel].set_ylabel('{} of nonzero Pixel Values of {}'.format(statistic.capitalize(),
                                                                         channels_labels[channel]))
    plt.tight_layout()
    fig.savefig(save_dir + '/{}Correlation_nonzero_{}_pixels'.format('Test ' if test else '', statistic))
    plt.close()


def plot_correlation_gradients(images, y_true, simulation, test=False, mask_radius=None, mask_radius_reverse=None):
    channels = images[0].shape[-1]
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(60, 10))

    channels_labels = ['Mass', 'Velocity', 'Velocity $\sigma$', 'Metallicity', 'Age']
    df = pd.DataFrame(columns=['ExsituF'] + channels_labels)
    df['ExsituF'] = y_true
    for channel in range(channels):
        stat_list = []
        for im in images:
            ch_im = im[:, :, channel]
            corner_pix_values = [ch_im[0, 0], ch_im[127, 0], ch_im[0, 127], ch_im[127, 127]]
            norm_zero_value = max(corner_pix_values, key=corner_pix_values.count)
            stars_mask = (ch_im != norm_zero_value)
            #ch_im = ch_im[mask]
            height, width = ch_im.shape
            if mask_radius is None:
                central_mask = create_circular_mask(width, height, radius=8) & stars_mask
            else:
                central_mask_1 = create_circular_mask(width, height, radius=mask_radius - 2)
                central_mask_2 = create_circular_mask(width, height, radius=mask_radius + 2)
                central_mask = central_mask_2 & (~central_mask_1) & stars_mask

            if mask_radius_reverse is None:
                outskirts_mask = (~create_circular_mask(width, height, radius=48)) & stars_mask
            else:
                outskirts_mask_1 = create_circular_mask(width, height, radius=mask_radius_reverse-2)
                outskirts_mask_2 = create_circular_mask(width, height, radius=mask_radius_reverse+2)
                outskirts_mask = outskirts_mask_2 & (~outskirts_mask_1) & stars_mask
            
            central_mean = np.nanmean(ch_im[central_mask])
            outskirts_mean = np.nanmean(ch_im[outskirts_mask])
            stat = outskirts_mean - central_mean

            stat_list.append(stat)

        df[channels_labels[channel]] = stat_list
        ax[channel].scatter(y_true, stat_list)
        ax[channel].set_xlabel('Ex-situ fraction')
        ax[channel].set_ylabel('Gradients between central and outskirt pixel Values of {}'.format(
                                                                         channels_labels[channel]))

    df.to_hdf(os.path.join(GRADIENTS_RUN_PATH, 'gradients_{}_{}_{}.h5py'.format(
        simulation, mask_radius, mask_radius_reverse)), '/data')

    save_dir = os.path.join(GRADIENTS_RUN_PATH,
                            '{}_{}_{}'.format(simulation, mask_radius, mask_radius_reverse))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    plt.tight_layout()
    fig.savefig(save_dir + '/{}Correlation_gradient_pixels'.format('Test ' if test else ''))
    plt.close()
    
    stars = np.copy(ch_im)
    stars[~stars_mask] = norm_zero_value
    plt.imshow(stars)
    plt.savefig(save_dir + '/{}gradient_pixels_stars'.format('Test ' if test else ''))
    plt.close()

    outskirts_im = np.copy(ch_im)
    outskirts_im[~outskirts_mask] = norm_zero_value
    plt.imshow(outskirts_im)
    plt.savefig(save_dir + '/{}gradient_pixels_outskirts'.format('Test ' if test else ''))
    plt.close()

    central_im = np.copy(ch_im)
    central_im[~central_mask] = norm_zero_value
    plt.imshow(central_im)
    plt.savefig(save_dir + '/{}gradient_pixels_central'.format('Test ' if test else ''))
    plt.close()


def save_gradients():
    for simulation in ('TNG', 'EAGLE'):
        for mask_radius, mask_reverse in GRADIENTS_MAPPING.keys():
            sim_dataset = input_fn('train', DATASET_MAPPING[simulation],
                                   ignore_channels=(),
                                   mask_radius=None,
                                   per_channel=False)
            images, y_true = get_data(sim_dataset, batches=30)
            if not os.path.exists(GRADIENTS_RUN_PATH):
                os.mkdir(GRADIENTS_RUN_PATH)
            plot_correlation_gradients(images, y_true, simulation,
                                       mask_radius_reverse=mask_reverse,
                                       mask_radius=mask_radius)


def save_correlations():
    for simulation in ('TNG', 'EAGLE'):
        for normalize in NORM_MAPPING.keys():
            sim_dataset = input_fn('train', DATASET_MAPPING[simulation],
                                   ignore_channels=(),
                                   mask_radius=None,
                                   per_channel=False,
                                   normalize=normalize)
            images, y_true = get_data(sim_dataset, batches=30)
            if not os.path.exists(INPUT_CORRELATION_PATH_RUN_PATH):
                os.mkdir(INPUT_CORRELATION_PATH_RUN_PATH)

            for statistic in ('mean', 'sum'):
                plot_correlation_all(images, y_true, simulation,
                                     normalize=normalize,
                                     statistic=statistic)


if __name__ == '__main__':
    from nn_data import input_2d_cnn_fn_split, input_fn_split, get_data
    input_fn = input_2d_cnn_fn_split
    if DATASET_1D:
        input_fn = input_fn_split

    save_gradients()
    save_correlations()

