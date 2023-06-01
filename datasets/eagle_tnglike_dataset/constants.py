from collections import OrderedDict
import os

DEBUG = False

EAGLE_SNAPSHOT = 28
SPLITS = OrderedDict({
    'train': 0.75,
    'validation': 0.1,
    'test': 0.15
})

MAP_NAMES = ('mass', 'vel', 'vel_disp', 'metal', 'age')
MAP_LOGS = (True, False, False, True, False)
MAP_CLIPS = (None, (-250, 250), (None, 350), (-1, 1), None)

if DEBUG:
    BASE_PATH = '/home/eirini/Documents/PhD/'
else:
    BASE_PATH = '/net/diva/scratch1/eirinia/projects'


EAGLE_SIM = 'L68n1504FP'
MAPS_PATH = os.path.join(BASE_PATH, 'Results/Datasets/Maps/EAGLE_TNGlike/'
                                    '{}/maps_EAGLE_TNGlike_balanced.h5py'.format(EAGLE_SIM))
DATA_BASE_PATH = os.path.join(BASE_PATH, 'Data')
EAGLE_DATA_BASE_PATH = os.path.join(DATA_BASE_PATH, 'EAGLE_TNGlike/{}'.format(EAGLE_SIM))
EAGLE_POSTPROCESSING_PATH = os.path.join(EAGLE_DATA_BASE_PATH, 'postprocessing')
EAGLE_BALANCED_DATASET_ALIGNS_PATH = os.path.join(EAGLE_POSTPROCESSING_PATH,
                                                  'balanced_dataset/balanced_alignments.hdf5')
EAGLE_DATASET_SPLITS = os.path.join(EAGLE_POSTPROCESSING_PATH, 'balanced_dataset/dataset_splits_1D.pkl')
