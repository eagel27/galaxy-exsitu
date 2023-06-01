import os.path
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm
from itertools import chain
from constants import MEAN_STD_ARRAYS_PATH


def read_dataset(dataset_str="accretion_dataset", only_train=True):
    """ Read the whole dataset (all the splits) """
    if only_train:
        return list(tfds.load(dataset_str, split='train'))

    splits = tfds.load(dataset_str)
    dataset = list(chain(splits['train'],
                         splits['validation'],
                         splits['test']))
    return dataset


def calculate_standardize_values(dataset_str, only_train=True):
    """ Calculate the mean and STD arrays
    per channel for the whole dataset """

    dataset = read_dataset(dataset_str)

    # Mass, Vel, Vel_disp, Log_mass
    pixel_mean = np.zeros(5)
    pixel_std = np.zeros(5)

    k = 1
    for data in tqdm(dataset, "Computing mean/std", len(dataset), unit="samples"):
        image = data['image']
        image = np.array(image).reshape((5, 128*128))
        image = image.transpose().reshape((128, 128, 5))

        #log_mass = np.copy(image[:, :, 0])
        #log_mass[np.where(log_mass <= 0)] = 1e-10
        #log_mass = np.log10(log_mass).reshape((-1, 1))

        pixels = image.reshape((-1, image.shape[2]))
        #pixels = np.concatenate((pixels, log_mass), axis=1)

        for pixel in pixels:
            diff = pixel - pixel_mean
            pixel_mean += diff / k
            pixel_std += diff * (pixel - pixel_mean)
            k += 1

    pixel_std = np.sqrt(pixel_std / (k - 2))

    if not os.path.exists(MEAN_STD_ARRAYS_PATH):
        os.makedirs(MEAN_STD_ARRAYS_PATH)

    np.save(os.path.join(MEAN_STD_ARRAYS_PATH,
                         'mean_{}{}.npy'.format(dataset_str, '_train' if only_train else '')), pixel_mean)
    np.save(os.path.join(MEAN_STD_ARRAYS_PATH,
                         'std_{}{}.npy'.format(dataset_str, '_train' if only_train else '')), pixel_std)


def retrieve_mean_std_values(dataset_str, only_train=True):
    """ Retrieve the mean and STD arrays containing
    the mean and std per channel for the whole dataset """

    pixel_mean = np.load(os.path.join(MEAN_STD_ARRAYS_PATH,
                                      'mean_{}{}.npy'.format(dataset_str, '_train' if only_train else '')))
    pixel_std = np.load(os.path.join(MEAN_STD_ARRAYS_PATH,
                                     'std_{}{}.npy'.format(dataset_str, '_train' if only_train else '')))

    return pixel_mean, pixel_std


if __name__ == '__main__':
    dataset_str = 'tng_dataset'
    calculate_standardize_values(dataset_str)
    print(retrieve_mean_std_values(dataset_str))
