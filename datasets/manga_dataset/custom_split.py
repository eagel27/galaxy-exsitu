import copy
import random
from astropy.table import Table

import constants as const


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

    df = Table.read(cat_file, format='fits').to_pandas()
    galaxy_ids = set([i.decode("utf-8") for i in df['mangaid'].unique()])

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
    galaxy_ids_file = const.MANGA_INFO_PATH
    galaxy_splits = split_galaxy_ids(galaxy_ids_file, splits)
    return galaxy_splits


if __name__ == '__main__':
    splits = const.SPLITS
    create_custom_split(splits)



