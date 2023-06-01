import os
import scipy
import tensorflow_datasets as tfds
import pandas as pd
import pickle
import tensorflow as tf
import numpy as np

from input_graphs import plot_original_maps, plot_galaxy_histograms
from constants import (BATCHES, RESULTS_PATH, TNG_DATASET_SPLITS, EAGLE_DATASET_SPLITS, INPUT_SHAPE,
                       TNG_BALANCED_DATASET_ALIGNS_PATH, EAGLE_BALANCED_DATASET_ALIGNS_PATH)
from preprocess import per_channel_standardization, per_image_standardization, mask_whole_image, mask_outer_radius


rng = tf.random.Generator.from_seed(123, alg='philox')


def augment(image, label, object_id):
    """
    Augment the image by doing a random flip left/right and up/down
    :param image: The image of the 5 maps
    :param label: The label -- exsitu fraction we will use as output
    :param object_id: The object_id of this galaxy
    :return: a similar tuple but with the augmented image instead
    """
    seed = rng.make_seeds(2)[0]
    image = tf.image.stateless_random_flip_left_right(image, seed)
    image = tf.image.stateless_random_flip_up_down(image, seed)
    # image = tf.numpy_function(lambda img: tf.keras.preprocessing.image.random_zoom(
    #        img, (0.6, 0.6), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest',
    #        ), [image], tf.float32)

    return image, label, object_id


def channels_last(example):
    """
    Reshape the image of the example so that channels are last
    :param example: A dictionary containing one example of the dataset
    :return: A tuple of image, output, object_id
    """
    nn_output = example['exsitu_fraction']
    nn_input = example['image']

    # Reshape input from (5, 128, 128) -> (128, 128, 5)
    nn_input = tf.transpose(nn_input, (1, 2, 0))
    return nn_input, nn_output, example['object_id']


def preprocessing(image, ignore_channels=None, mask_radius=None, dataset_str=None,
                  per_channel=False, normalize=True):
    """
    Apply the required preprocessing to the 2D input maps according to the arguments passed.
    We ignore channels by masking the whole image to zero values.
    We decrease the aperture by masking pixels outside a certain radius.
    All images are normalized either individually or per channel.

    :param image: The 5 maps of each galaxy (mass, velocity, velocity disp, metallicity, age)
    :param ignore_channels: A tuple containing the indexes of the channels we want to mask
    :param mask_radius: A float specifying the radius that all pixels outside will be masked
    :param dataset_str: The dataset_str of the dataset
    (to resolve the median/sigma in case a per channel normalization is requested)
    :param per_channel: If true, the median/sigma of the whole channel are used for standardizing
    the channel maps. If false (default) every map is standardized individually.
    :param normalize: Indicates whether we should normalize the maps.
    :return: the processed image
    """
    channels_processed = []
    for i in range(INPUT_SHAPE[2]):
        channel_input = image[:, :, i]

        if ignore_channels and i in ignore_channels:
            channel_input_norm = mask_whole_image(channel_input)
        else:
            channel_input_norm = channel_input
            if normalize:
                if per_channel:
                    channel_input_norm = per_channel_standardization(channel_input_norm, i,
                                                                     dataset_str, logged=False)
                else:
                    channel_input_norm = per_image_standardization(channel_input_norm, i,
                                                                   logged=False, mode='norm')
            if mask_radius is not None:
                channel_input_norm = mask_outer_radius(channel_input_norm, mask_radius)

        channels_processed.append(channel_input_norm)

    image_processed = tf.stack(channels_processed, axis=2)
    return image_processed


def input_fn_split(mode='train', dataset_str='tng_dataset',
                   batch_size=BATCHES, with_info=False,
                   ignore_channels=None, mask_radius=None,
                   normalize=True, per_channel=False, remove_object_id=True):
    """
    Loads datasets from already split version.
    mode: 'train' or 'test' or 'validation'
    """

    shuffle = mode in ('train', 'validation')
    dataset = tfds.load(
        dataset_str,
        split=mode,
        shuffle_files=shuffle
    )

    if shuffle:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10000)

    dataset = dataset.map(channels_last, num_parallel_calls=tf.data.AUTOTUNE)
    # Apply data preprocessing
    dataset = dataset.map(lambda x, y, z: (preprocessing(x, ignore_channels,
                                                         mask_radius, dataset_str,
                                                         per_channel, normalize),
                                           y,
                                           z),
                          num_parallel_calls=tf.data.AUTOTUNE)

    if mode == 'train':
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    if remove_object_id:
        dataset = dataset.map(lambda x, y, z: (x, y))

    dataset = dataset.batch(batch_size, drop_remainder=True)
    # fetch next batches while training current one (-1 for autotune)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def input_2d_cnn_fn_split(mode='train', dataset_str='tng2d_dataset',
                          batch_size=BATCHES, with_info=False,
                          ignore_channels=None, mask_radius=None,
                          per_channel=False, for_ae=False, normalize=True,
                          remove_object_id=True):
    """
    Loads datasets from already split version.
    mode: 'train' or 'test' or 'validation'
    """

    shuffle = mode in ('train', 'validation')

    if dataset_str == 'both':
        dataset = None
        for dataset_str in ('tng2d_dataset', 'eagle2d_dataset'):
            dataset_single = tfds.load(
                dataset_str,
                split=mode,
                shuffle_files=shuffle
            )
            if dataset is not None:
                dataset = dataset.concatenate(dataset_single)
            else:
                dataset = dataset_single
    else:
        dataset = tfds.load(
            dataset_str,
            split=mode,
            shuffle_files=shuffle
        )

    if shuffle:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(10000)

    dataset = dataset.map(channels_last, num_parallel_calls=tf.data.AUTOTUNE)

    # Apply data preprocessing
    dataset = dataset.map(lambda x, y, z: (preprocessing(x, ignore_channels,
                                                         mask_radius, dataset_str, per_channel, normalize),
                                           y,
                                           z),
                          num_parallel_calls=tf.data.AUTOTUNE)

    if mode == 'train':
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    if remove_object_id:
        dataset = dataset.map(lambda x, y, z: (x, y))

    dataset = dataset.batch(batch_size, drop_remainder=True)
    # fetch next batches while training current one (-1 for autotune)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_num_examples(mode='train', dataset_str='tng_dataset'):
    if dataset_str == 'both':
        iterate_datasets = ('tng2d_dataset', 'eagle2d_dataset')
    else:
        iterate_datasets = (dataset_str,)

    num_examples = 0
    for dataset_str in iterate_datasets:
        builder = tfds.builder(dataset_str)
        splits = builder.info.splits
        num_examples += splits[mode].num_examples
    return num_examples


def get_data(dataset, batches=10, get_obj_id=False):
    data = dataset.take(batches)
    images, y_true, obj_ids = [], [], []
    for d in list(data):
        images.extend(d[0].numpy())
        y_true.extend(d[1].numpy())
        if get_obj_id:
            obj_ids.extend(d[2].numpy())
    images = np.stack(images)
    y_true = np.array(y_true)
    obj_ids = np.array(obj_ids)

    if get_obj_id:
        return images, y_true, obj_ids
    return images, y_true



def get_data_with_objid(dataset, batches=10):
    data = dataset.take(batches)
    images, y_true, obj_id = [], [], []
    for d in list(data):
        images.extend(d[0].numpy())
        y_true.extend(d[1].numpy())
        obj_id.extend(d[2].numpy())

    images = np.stack(images)
    y_true = np.array(y_true)
    obj_id = np.array(obj_id)

    return images, y_true, obj_id


def get_catalog_info_whole_dataset(dataset='TNG', split=None):
    # Read spilts from file
    splits_file = TNG_DATASET_SPLITS
    aligns_path = TNG_BALANCED_DATASET_ALIGNS_PATH
    if dataset != 'TNG':
        splits_file = EAGLE_DATASET_SPLITS
        aligns_path = EAGLE_BALANCED_DATASET_ALIGNS_PATH
    with open(splits_file, 'rb') as f:
        galaxy_splits = pickle.load(f)

    split_ids = galaxy_splits[split]
    balanced_data = pd.read_hdf(aligns_path, '/data')

    exsitu_fractions = []
    for (galaxy_id, snap) in split_ids:
        galaxy = balanced_data.loc[(balanced_data.GalaxyID == galaxy_id) &
                                   (balanced_data.Snapshot == snap)]
        exsitu_fraction = float(galaxy['ExSitu_Fraction'].item())
        alignments = int(galaxy['Num_of_Aligns'].item())
        exsitu_fractions.extend([exsitu_fraction] * alignments)

    return exsitu_fractions


def compute_prior_whole_dataset(dataset='TNG', split=None):
    if dataset != 'BOTH':
        f = get_catalog_info_whole_dataset(dataset=dataset, split=split)
    else:
        f1 = get_catalog_info_whole_dataset(dataset='TNG', split=split)
        f2 = get_catalog_info_whole_dataset(dataset='EAGLE', split=split)
        f = f1 + f2
    hist = np.histogram(f, bins=50)

    # Do not allow zero bin in hist!!
    non_zero = [i if i > 0 else 1 for i in hist[0]]
    hist = (non_zero, hist[1])
    prior = scipy.stats.rv_histogram(hist)
    plt.hist(f, 100, density=True)
    x = np.linspace(0.01, 0.99, 50)
    plt.plot(x, prior.pdf(x))
    plt.savefig('Prior.png')
    plt.close()
    return prior



def input_plots(ds_set, save_dir, test=False, get_obj_ids=False):
    r = get_data(ds_set, batches=20, get_obj_id=get_obj_ids)
    if get_obj_ids:
        images, y_true, obj_ids = r
    else:
        images, y_true = r
        obj_ids = np.zeros(len(y_true))

    plot_original_maps(save_dir, images, y_true, obj_ids, test=test)
    plot_galaxy_histograms(save_dir, images, test=test)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.style.use('science')
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', titlesize=15, labelsize=15)
    ds_test = input_fn_split('test', remove_object_id=False)

    len_ds_train = get_num_examples('train')
    len_ds_val = get_num_examples('validation')
    len_ds_test = get_num_examples('test')

    print(len_ds_test, len_ds_val, len_ds_train)
    save_dir = os.path.join(RESULTS_PATH, 'Input_Plots', 'TNG100')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    input_plots(ds_test, save_dir, get_obj_ids=True)
