"""mangia_dataset dataset."""
import os.path

import random
from scipy.ndimage import gaussian_filter, zoom
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
import custom_split as split
import constants as const
from scipy.ndimage import gaussian_filter
from astropy.io import fits

_DESCRIPTION = """
Mock Maps from MANGIA (TNG50) galaxy dataset --> but balanced.
Here we do not rescale to 4Re, we just pad to 71x71 pixels.
"""

# TODO(mangia_dataset): BibTeX citation
_CITATION = """
"""

MASK_MAPPING = {
    19: 0,
    37: 1,
    61: 2,
    91: 3,
    127: 4
}

size = 71


def augment_image(image):
    """ Produce a random augmentation of the provided image """
    img_size = image.shape[0]
    # Random cropping
    h_crop = tf.cast(tf.random.uniform(shape=[], minval=img_size-10,
                                       maxval=img_size+10, dtype=tf.int32), tf.float32)
    w_crop = h_crop * tf.random.uniform(shape=[], minval=0.8, maxval=1.0)
    h_crop, w_crop = tf.cast(h_crop, tf.int32), tf.cast(w_crop, tf.int32)
    opposite_aspectratio = tf.random.uniform(shape=[])
    if opposite_aspectratio < 0.5:
        h_crop, w_crop = w_crop, h_crop
    
    if h_crop > img_size or w_crop > img_size:
        image = tf.image.resize_with_pad(image, h_crop, w_crop)
    else:
        image = tf.image.random_crop(image, size=[h_crop, w_crop, image.shape[-1]])

    # Horizontal flipping
    horizontal_flip = tf.random.uniform(shape=[])
    if horizontal_flip < 0.5:
        image = tf.image.random_flip_left_right(image)

    # Resizing to original size
    image = tf.image.resize(image, size=[img_size, img_size])
    return image.numpy()


class MangiaDataset(tfds.core.GeneratorBasedBuilder):
    """ DatasetBuilder for mangia_dataset dataset. """

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
                name=tfds.Split.TRAIN,
                gen_kwargs=dict(
                    split_ids=split_ids['train'],
                    name='train'
                )),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs=dict(
                    split_ids=split_ids['validation'],
                    name='validation'
                )),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs=dict(
                    split_ids=split_ids['test'],
                    name='test'
                )),
        ]

    def _generate_examples(self, split_ids, name):
        """ Yields examples. """
        
        import resource
        low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

        df_info = pd.read_csv(const.MOCK_INFO_PATH, index_col=0)
        df_info['Extra'] = False
        
        if name == 'train':
            df_info_extra = pd.read_csv(const.MOCK_EXTRA_INFO_PATH, index_col=0)
            df_info_extra['Extra'] = True
            df_info = pd.concat([df_info_extra, df_info], ignore_index=True)

        ifu_masks = np.load(const.MOCK_IFU_MASKS)

        i = 0
        for index, row in df_info.iterrows():
            galaxy_id = int(row['subhalo_id'])
            snapshot = int(row['snapshot'])
            extra = row['Extra']

            if (snapshot, galaxy_id) not in split_ids and not extra:
                continue
            
            exsitu_fraction = row['ExSitu_F']
            
            if not np.isfinite(exsitu_fraction):
                continue

            view = int(row['view'])
            fibers = 127
            file_name = 'TNG50-{}-{}-{}-{}.cube_maps.fits'.format(snapshot, galaxy_id, view, fibers)
            if extra:
                file_name = 'ilust-{}-{}-{}-{}.cube.SSP.cube.fits.gz'.format(snapshot, galaxy_id, view, fibers)

            IFU = int(row['manga_ifu_dsn'])
            mask = ifu_masks[MASK_MAPPING[IFU]]
            realisations = int(row['Num_of_Aligns'])

            maps_file_path = os.path.join(const.MOCK_MAPS_PATH, file_name)
            if extra:
                maps_file_path = os.path.join(const.MOCK_EXTRA_MAPS_PATH, file_name)

            if not os.path.exists(maps_file_path):
                print('Maps file {} does not exist !'.format(maps_file_path))
                continue
            
            image = np.zeros((size, size, len(const.MAP_NAMES)))
            try:
                with fits.open(maps_file_path) as maps_file:
                    if extra:
                        data = maps_file[0].data
                    else:
                        data = maps_file[4].data

                mean_intensity = data[3, :, :]
                noise = data[4, :, :]
                noisy_pixels = np.divide(mean_intensity, noise) < 5
                
                for map_i, map_name in enumerate(const.MAP_NAMES):
                    mmap = data[const.MAP_CHANNEL_ID[map_i], :, :]

                    if map_i == 1:
                        central_vel = np.mean(mmap[mmap.shape[0]//2 - 1:mmap.shape[0]//2 + 1,
                                                   mmap.shape[1]//2 - 1:mmap.shape[1]//2 + 1])
                        mmap -= central_vel
                        mass_mmap = data[const.MAP_CHANNEL_ID[0], :, :]
                        mmap[mass_mmap == 0] = np.nan
                    if map_i == 4:
                        mmap = np.divide(10 ** mmap, 10 ** 9)

                    # mask to IFU
                    mmap[~mask] = 0

                    # mask noisy bins
                    mmap[noisy_pixels] = 0

                    # Do not allow NaN or inf values
                    mmap[~np.isfinite(mmap)] = 0

                    diff = size - mmap.shape[0]
                    mmap = np.pad(mmap, ((diff//2, round(diff/2)),
                                         (diff//2, round(diff/2))))
                    
                    # Do not allow NaN or inf values
                    diff = size - intr_mmap.shape[0]
                    intr_mmap = np.pad(intr_mmap, ((diff//2, round(diff/2)),
                                                   (diff//2, round(diff/2))))
                    image[:, :, map_i] = mmap

                i += 1
                # Yield with i because in our case object_id will be the same for the 4 different projections
                yield i, {'image': image.astype("float32"),
                          'exsitu_fraction': exsitu_fraction,
                          'object_id': '{}_{}'.format(galaxy_id, snapshot)}

                if name in ('train', 'validation'):
                    for j in range(realisations//2 or 1):
                        image_aug = augment_image(image)
    
                        i += 1
                        # Yield with i because in our case object_id will be the same for the 4 different projections
                        yield i, {'image': image_aug.astype("float32"),
                                  'exsitu_fraction': exsitu_fraction,
                                  'object_id': '{}_{}'.format(galaxy_id, snapshot)}
                        
            except Exception as e:
                print("Galaxy id not added to the dataset: object_id=", galaxy_id,
                      "ex situ fraction ", exsitu_fraction, e)
            finally:
                maps_file.close()
