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
MAP_LOGS = (True, False, False, True, False)
MAP_CLIPS = (None, (-250, 250), (None, 350), (-1, 1), None)

if DEBUG:
    BASE_PATH = '/home/eirini/Documents/PhD/'
else:
    BASE_PATH = '/net/diva/scratch1/eirinia/projects'


STELLAR_ASSEMBLY_PATH = os.path.join(BASE_PATH, 'Data/IllustrisTNG/TNG100-1/'
                                                'postprocessing/stellar_assembly/galaxies_{}.hdf5')

MAPS_PATH = os.path.join(BASE_PATH, 'Results/Datasets/Maps/IllustrisTNG/'
                                    'TNG100-1/maps_TNG100-1_balanced.h5py')

STELLAR_ASSEMBLY_FILE_PATH = STELLAR_ASSEMBLY_PATH.format('0' + str(TNG_SNAPSHOT))

TNG_SIM = 'TNG100-1'
DATA_BASE_PATH = os.path.join(BASE_PATH, 'Data')
TNG_DATA_BASE_PATH = os.path.join(DATA_BASE_PATH, 'IllustrisTNG/{}'.format(TNG_SIM))
TNG_POSTPROCESSING_PATH = os.path.join(TNG_DATA_BASE_PATH, 'postprocessing')
TNG_BALANCED_DATASET_ALIGNS_PATH = os.path.join(TNG_POSTPROCESSING_PATH, 'balanced_dataset/balanced_alignments.hdf5')
TNG_DATASET_SPLITS = os.path.join(TNG_POSTPROCESSING_PATH, 'balanced_dataset/dataset_splits_1D.pkl')
