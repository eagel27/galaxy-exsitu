"""tng_dataset dataset."""
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import h5py
import pandas as pd

import constants as const
import custom_split as split
from scipy.ndimage import gaussian_filter

_DESCRIPTION = """
Maps of TNG100 galaxy dataset along with their stellar assembly data.
Data is extracted from:
 - H5PY file with 6 maps for each galaxy 
 (stellar mass, velocity, velocity dispersion, metallicity, formation time, age) 
 - H5PY file of stellar assembly 
  (/home/eirini/Documents/PhD/Data/TNG100-1/postprocessing/stellar_assembly/galaxies_099.hdf5)


Preprocessing:
 - Decide which maps to use (maybe not use formation time - repeated by age)
 - Log mass & metallicity maps
 - Remove any non finite values
 - Clip values when required
"""

# TODO(tng_dataset): BibTeX citation
_CITATION = """
"""


class TngDataset(tfds.core.GeneratorBasedBuilder):
    """ DatasetBuilder for tng_dataset dataset. """

    VERSION = tfds.core.Version('3.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '2.0.0': 'Less strict limits on clipping',
      '3.0.0': 'Keep only ex-situ fractions < 0.85'
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """ Returns the dataset metadata. """
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
              # These are the features of your dataset like images, labels ...
              'image': tfds.features.Tensor(shape=(len(const.MAP_NAMES), 128, 128), dtype=tf.float32),
              'exsitu_fraction': tf.float32,
              'object_id': tf.string,
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'exsitu_fraction'),  # Set to `None` to disable
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl):
        """ Returns generators according to split """
        split_ids = split.create_custom_split(const.SPLITS)

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=dict(
                    split_ids=split_ids['train'],
                )),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs=dict(
                    split_ids=split_ids['validation'],
                )),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs=dict(
                    split_ids=split_ids['test'],
                )),
        ]

    def _generate_examples(self, split_ids):
        """ Yields examples. """

        maps_data = h5py.File(const.MAPS_PATH, 'r')
        maps_data_keys = maps_data.keys()
        balanced_data = pd.read_hdf(const.TNG_BALANCED_DATASET_ALIGNS_PATH, '/data')

        i = 0
        sigma = 0.6
        for (galaxy_id, snap) in split_ids:
            try:

                exsitu_fraction = balanced_data.loc[(balanced_data.GalaxyID == galaxy_id) &
                                                    (balanced_data.Snapshot == snap)]['ExSitu_Fraction'].item()

                exsitu_fraction = np.float32(exsitu_fraction)
                
                if exsitu_fraction >= 0.85:
                    continue

                galaxy_views = [m_key for m_key in maps_data_keys
                                if m_key.startswith('{}_{}'.format(galaxy_id, snap))]

                for galaxy_view in galaxy_views:
                    image = np.zeros((len(const.MAP_NAMES), 128, 128))
                    for map_i, map_name in enumerate(const.MAP_NAMES):
                        mmap = maps_data[galaxy_view + '/MAPS/{}_map'.format(map_name)][...]
                        map_log = const.MAP_LOGS[map_i]
                        map_clip = const.MAP_CLIPS[map_i]
                        if map_log:
                            mmap[mmap == 0] = 1
                            mmap = np.log10(mmap)
                        if map_clip:
                            # velocity special case, limits should be symmetrical
                            if map_i == 1:
                                lim = min(min(abs(np.nanmin(mmap)), abs(np.nanmax(mmap))), map_clip[1])
                                map_clip = (-lim, lim)
                            mmap = mmap.clip(*map_clip)

                        # Do not allow NaN or inf values
                        mmap[~np.isfinite(mmap)] = 0
                        mmap = gaussian_filter(mmap, sigma=sigma)
                        image[map_i, :, :] = mmap

                    i += 1
                    # Yield with i because in our case object_id will be the same for the 4 different projections
                    yield i, {'image': image.astype("float32"),
                              'exsitu_fraction': exsitu_fraction.astype("float32"),
                              'object_id': '{}_{}'.format(galaxy_id, snap)}
            except Exception as e:
                print("Galaxy id not added to the dataset: object_id=", galaxy_id,
                      "ex situ fraction ", exsitu_fraction, e)

        maps_data.close()
