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


STELLAR_ASSEMBLY_PATH = os.path.join(BASE_PATH, 'Data/EAGLE/RefL0100N1504/'
                                                'postprocessing/stellar_assembly/exsitu_{}.h5py')

MAPS_PATH = os.path.join(BASE_PATH, 'Results/Datasets/Maps/EAGLE/'
                                    'RefL0100N1504/maps_RefL0100N1504_balanced.h5py')

STELLAR_ASSEMBLY_FILE_PATH = STELLAR_ASSEMBLY_PATH.format('0' + str(EAGLE_SNAPSHOT))

EAGLE_SIM = 'RefL0100N1504'
DATA_BASE_PATH = os.path.join(BASE_PATH, 'Data')
EAGLE_DATA_BASE_PATH = os.path.join(DATA_BASE_PATH, 'EAGLE/{}'.format(EAGLE_SIM))
EAGLE_POSTPROCESSING_PATH = os.path.join(EAGLE_DATA_BASE_PATH, 'postprocessing')
EAGLE_BALANCED_DATASET_ALIGNS_PATH = os.path.join(EAGLE_POSTPROCESSING_PATH,
                                                  'balanced_dataset/balanced_alignments.hdf5')
EAGLE_DATASET_SPLITS = os.path.join(EAGLE_POSTPROCESSING_PATH, 'balanced_dataset/dataset_splits_1D.pkl')
