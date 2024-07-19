from collections import OrderedDict
import os

DEBUG = False

TNG_SNAPSHOT = 99
SPLITS = OrderedDict({
    'train': 0.75,
    'validation': 0.1,
    'test': 0.15
})

MAP_NAMES = ('mass', 'vel', 'vel_disp', 'metal', 'age')
MAP_CHANNEL_ID = (18, 13, 15, 9, 6)
MAP_LOGS = (True, False, False, True, False)
MAP_CLIPS = (None, (-200, 200), (None, 250), (-1, 1), None)

INTR_MAP_CHANNEL_ID = (1, 2, 7, 12, 14)

if DEBUG:
    BASE_PATH = '/home/eirini/Documents/PhD/'
    TREES_PATH = '/home/eirini/Documents/PhD/Data/MaNGIA/mangia_trees'
    MOCK_INFO_PATH = '/home/eirini/Documents/PhD/Data/MaNGIA/info_mangia_updated.csv'
else:
    BASE_PATH = '/net/diva/scratch1/eirinia/projects'
    TREES_PATH = '/net/diva/scratch1/eirinia/projects/Data/IllustrisTNG/TNG50-1/MaNGIA/mangia_trees'
    MOCK_INFO_PATH = '/net/diva/scratch1/eirinia/projects/Data/IllustrisTNG/TNG50-1/MaNGIA/info_mangia_updated.csv'

MOCK_MAPS_PATH = '/net/diva/scratch1/eirinia/projects/Data/IllustrisTNG/TNG50-1/MaNGIA/maps'
MOCK_EXTRA_MAPS_PATH = '/net/diva/scratch1/eirinia/projects/Data/IllustrisTNG/TNG50-1/MaNGIA/maps/Extra'
MOCK_INFO_PATH = '/net/diva/scratch1/eirinia/projects/Data/IllustrisTNG/TNG50-1/MaNGIA/info_mangia_aligns.csv'
MOCK_EXTRA_INFO_PATH = '/net/diva/scratch1/eirinia/projects/Data/IllustrisTNG/TNG50-1/' \
                       'MaNGIA/info_mangia_extra_aligns.csv'

MOCK_IFU_MASKS = '/scratch/eirinia/projects/Data/IllustrisTNG/TNG50-1/ifu_postproc_masks.npy'

DATA_BASE_PATH = os.path.join(BASE_PATH, 'Data')
MANGIA_DATA_BASE_PATH = os.path.join(DATA_BASE_PATH, 'MaNGIA')
MANGIA_POSTPROCESSING_PATH = os.path.join(MANGIA_DATA_BASE_PATH, 'postprocessing')
MANGIA_DATASET_SPLITS = os.path.join(MANGIA_POSTPROCESSING_PATH, 'balanced_dataset/dataset_splits.pkl')
