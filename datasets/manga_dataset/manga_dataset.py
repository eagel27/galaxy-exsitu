"""manga__dataset dataset."""
import os.path

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

from astropy.io import fits
from astropy.table import Table

import custom_split as split
import constants as const


_DESCRIPTION = """
MaNGA galaxy dataset
"""

# TODO(manga_dataset): BibTeX citation
_CITATION = """
"""

size = 71


def crop_center(img, cropx, cropy):
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


class MangaDataset(tfds.core.GeneratorBasedBuilder):
    """ DatasetBuilder for manga_dataset dataset. """

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """ Returns the dataset metadata. """
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
              # These are the features of your dataset like images, labels ...
              'image': tfds.features.Tensor(shape=(size, size, len(const.MAP_NAMES)), dtype=tf.float32),
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
                name=tfds.Split.TEST,
                gen_kwargs=dict(
                    split_ids=split_ids['test'],
                )),
        ]

    def _generate_examples(self, split_ids):
        """ Yields examples. """

        df_info = Table.read(const.MANGA_INFO_PATH, format='fits').to_pandas()

        i = 0
        for index, row in df_info.iterrows():
            galaxy_id = row['mangaid'].decode("utf-8") 

            if galaxy_id not in split_ids:
                continue
                
            exsitu_fraction = 0  # It is unknown !
            plate = str(row['plate'])
            plateifu = row['plateifu'].decode("utf-8") 

            file_path = os.path.join(plate,
                                     'manga-{}.Pipe3D.cube.fits.gz'.format(plateifu))

            try:
                exsitu_fraction = np.float32(exsitu_fraction)
                maps_file_path = os.path.join(const.MANGA_MAPS_PATH, file_path)
                maps_file = fits.open(maps_file_path)

                mean_intensity = maps_file[1].data[3, :, :]
                noise = maps_file[1].data[4, :, :]
                noisy_pixels = np.divide(mean_intensity, noise) < 5

                image = np.zeros((size, size, len(const.MAP_NAMES)))
            
                for map_i, map_name in enumerate(const.MAP_NAMES):
                    mmap = maps_file[1].data[const.MAP_CHANNEL_ID[map_i], :, :]
                    if map_i == 1:
                        central_vel = np.mean(mmap[mmap.shape[0]//2 - 1:mmap.shape[0]//2 + 1,
                                              mmap.shape[1]//2 - 1:mmap.shape[1]//2 + 1])
                        mmap -= central_vel
                        mass_mmap = maps_file[1].data[const.MAP_CHANNEL_ID[0], :, :]
                        mmap[mass_mmap == 0] = np.nan
                    if map_i == 4:
                        mmap = np.divide(10 ** mmap, 10 ** 9)
                  
                    # mask noisy bins
                    mmap[noisy_pixels] = 0
                    
                    # Do not allow NaN or inf values
                    mmap[~np.isfinite(mmap)] = 0
                    
                    if mmap.shape[0] > size:
                        mmap = crop_center(mmap, size, size)
                    else:
                        diff = size - mmap.shape[0]
                        mmap = np.pad(mmap, ((diff//2, round(diff/2)),
                                             (diff//2, round(diff/2))))

                    if mmap.shape != (size, size):
                        mmap = np.pad(mmap, ((1, 0), (1, 0)))

                    image[:, :, map_i] = mmap

                i += 1
                # Yield with i because in our case object_id will be the same for the 4 different projections
                yield i, {'image': image.astype("float32"),
                          'exsitu_fraction': exsitu_fraction.astype("float32"),
                          'object_id': '{}_{}'.format(galaxy_id, plate)}
            
                maps_file.close()
            
            except Exception as e:
                print("Galaxy id not added to the dataset: object_id=", galaxy_id,
                      "ex situ fraction ", exsitu_fraction, e)

