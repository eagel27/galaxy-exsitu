import copy
import random
import h5py
import pickle
from collections import Counter

import constants as const


def split_galaxy_ids(maps_file, splits):
    """
    Split the galaxy ids that exist in the maps file provided
    to the requested splits according to the percentages.

    :param maps_file: The maps file that you wish to split
    :param splits: Ordered dictionary holding as keys the required splits and
    as values the percentage of every split
        e.g {'train': train_percentage}

    :return: Dictionary holding the galaxy ids of each split
        e.g {'train': train_galaxy_ids}
    """

    with h5py.File(maps_file, 'r') as f:
        g_keys = f.keys()
        galaxy_snap_ids = [tuple(map(int, g_key.split('_')[:2])) for g_key in g_keys]
        alignments = Counter(galaxy_snap_ids)

    galaxies_per_alignment = {}

    for key, value in alignments.items():
        galaxies_per_alignment.setdefault(value, set()).add(key)

    galaxy_splits = {'train': set(), 'validation': set(), 'test': set()}
    for alignment, galaxies_ids in galaxies_per_alignment.items():
        remaining_ids = copy.deepcopy(galaxies_ids)
        for split, percentage in splits.items():
            split_ids = set(random.sample(remaining_ids, round(len(galaxies_ids) * percentage)))
            galaxy_splits[split].update(split_ids)
            remaining_ids -= split_ids

        # Include the remaining ids (if any) to last split
        if remaining_ids:
            galaxy_splits[split].update(remaining_ids)

    # Write spilts in file
    with open(const.EAGLE_DATASET_SPLITS, 'wb') as f:
        pickle.dump(galaxy_splits, f)

    return galaxy_splits


def create_custom_split(splits):
    maps_dir = const.MAPS_PATH

    galaxy_splits = split_galaxy_ids(maps_dir, splits)
    return galaxy_splits


if __name__ == '__main__':
    splits = const.SPLITS
    create_custom_split(splits)



