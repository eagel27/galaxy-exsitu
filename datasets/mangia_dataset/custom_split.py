import copy
import random
import h5py
import pickle
from collections import Counter
import pandas as pd
import constants as const
import numpy as np
import matplotlib.pyplot as plt


def split_galaxy_ids(cat_file, splits):
    """
    Split the galaxy ids that exist in the file provided
    to the requested splits according to the percentages.

    :param cat_file: The catalog file containing the ids that you wish to split
    :param splits: Ordered dictionary holding as keys the required splits and
    as values the percentage of every split
        e.g {'train': train_percentage}

    :return: Dictionary holding the galaxy ids of each split
        e.g {'train': train_galaxy_ids}
    """

    with open(const.TREES_PATH, 'rb') as fb:
        trees = np.array(pickle.load(fb))

    df = pd.read_csv(cat_file, index_col=0)

    galaxy_splits = {'train': set(), 'validation': set(), 'test': set()}
    galaxy_ids = set([(int(row['snapshot']), int(row['subhalo_id'])) for i, row in df.iterrows()])

    galaxy_splits = {}
    remaining_ids = copy.deepcopy(galaxy_ids)
    for split, percentage in splits.items():
        split_ids = set(random.sample(remaining_ids, int(len(galaxy_ids) * percentage)))
        galaxy_splits[split] = split_ids
        remaining_ids -= split_ids

    # Include the remaining ids (if any) to last split
    if remaining_ids:
        galaxy_splits[split].update(remaining_ids)
    return galaxy_splits


def create_custom_split(splits):
    # USE EXACTLY THE SAME SPLITS AS regina_mock_original_dataset
    with open(const.MANGIA_DATASET_SPLITS, 'rb') as f:
        galaxy_splits = pickle.load(f)
    #galaxy_ids_file = const.MOCK_INFO_PATH
    #galaxy_splits = split_galaxy_ids(galaxy_ids_file, splits)
    #print(galaxy_splits)
    return galaxy_splits


if __name__ == '__main__':
    splits = const.SPLITS
    splits = create_custom_split(splits)

    df = pd.read_csv(const.MOCK_INFO_PATH, index_col=0)
    for split_name, split in splits.items():
        exsitu_f_list = []
        for (snap, gid) in split:
            exsitu_f = df[(df.snapshot == int(snap)) & (df.subhalo_id == int(gid))].iloc[0].ExSitu_F
            exsitu_f_list.append(exsitu_f)
        plt.hist(exsitu_f_list)
        plt.title('%s samples: %d' % (split_name, len(exsitu_f_list)))
        plt.savefig('%s.png' % split_name)
        plt.close()




