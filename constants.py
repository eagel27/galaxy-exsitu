import os

DEBUG = True

IGNORE_CHANNELS_MAPPING = {
    (): {
        'color': 'darkblue',
        'label': 'Predict with Mass, Kinematics, Age, Metallicity'
    },
    (0, 1, 2): {
        'color': 'brown',
        'label': 'Predict only with Age, Metallicity'
    },
    (3, 4): {
        'color': 'orange',
        'label': 'Predict only with Mass, Kinematics'
    }
}


MASK_RADIUS_MAPPING = {
    None: {
        'color': 'darkblue',
        'label': 'Predict with 4Re'
    },
    32: {
        'color': 'green',
        'label': r'Predict only with 2Re'
    },
    16: {
        'color': 'red',
        'label': r'Predict only with 1Re'
    }
}

CHANNELS_MAPPING_BBOX = {
    '0_1_2_3_4': {
        'color': 'darkblue',
        'label': r'All'
    },
    '0_1_2': {
        'color': 'green',
        'label': 'M*, V*, $\sigma$*'
    },
    '3_4': {
        'color': 'red',
        'label': 'Age*, Z*'
    }
}

MASK_RADIUS_MAPPING_BBOX = {
    None: {
        'color': 'darkblue',
        'label': '4 HMSR'
    },
    32: {
        'color': 'green',
        'label': r'2 HMSR'
    },
    16: {
        'color': 'red',
        'label': r'1 HMSR'
    }
}

GRADIENTS_MAPPING = {
    (None, 16): r'HMSR/4 - HMSR/2',
    #(16, 48): r'Re/2 - 1Re',
    (None, None): r'HMSR/4 - 2HMSR',
    (32, None): r'HMSR - 2HMSR'
}

NORM_MAPPING = {
    False: 'Original Images',
    True: 'Normalized Images',
}

INPUT_SHAPE = (128, 128, 5)
TNG_SNAPSHOT = 99
BATCHES = 128
MDN = True

if DEBUG:
    BASE_PATH = '/home/eirini/Documents/PhD/'
else:
    BASE_PATH = '/net/diva/scratch1/eirinia/projects'


RESULTS_PATH = os.path.join(BASE_PATH, 'Results')


# The path in Deimos for the high resolution maps as created by (Maan,Bottrell+)
# Maps include: Stellar density, velocity, velocity dispersion
KINETIC_MAPS_PATH = '/scratch/reginas/TNG_Hani'

# The path in Deimos for the resampled high resolution data
# that will be used as input to the NN
TNG_INPUT_MAPS_PATH = '/scratch/eirinia/projects/GalaxyAccretion/Input/TNG100-1'

if DEBUG:
    STELLAR_ASSEMBLY_PATH = '/home/eirini/Documents/PhD/Data/IllustrisTNG/TNG100-1/' \
                            'postprocessing/stellar_assembly/galaxies_{}.hdf5'

    KINETIC_MAPS_PATH = '/home/eirini/Documents/PhD/Data/IllustrisTNG/TNG100-1/ResampledMaps/'

    RESULTS_PATH = '/home/eirini/Documents/PhD/Results/Neural Networks/CNNs'
    DATASET_RESULTS_PATH = '/home/eirini/Documents/PhD/Results/Datasets'
else:
    STELLAR_ASSEMBLY_PATH = '/net/diva/scratch1/eirinia/projects/Data/IllustrisTNG/' \
                            'TNG100-1/postprocessing/stellar_assembly/galaxies_{}.hdf5'

    KINETIC_MAPS_PATH = '/net/diva/scratch1/eirinia/projects/GalaxyAccretion/' \
                        'TNG100-1/ResampledInput'

    RESULTS_PATH = '/scratch/eirinia/projects/Results'
    DATASET_RESULTS_PATH = '/scratch/eirinia/projects/Results/Datasets'


KINETIC_MAPS_SNAPSHOT_DIR = os.path.join(KINETIC_MAPS_PATH, '0' + str(TNG_SNAPSHOT))
KINETIC_MAPS_SPLIT_SNAPSHOT_DIR = os.path.join(KINETIC_MAPS_PATH, '0' + str(TNG_SNAPSHOT) + '_split')

TNG_INPUT_SNAPSHOT_DIR = os.path.join(TNG_INPUT_MAPS_PATH, '0' + str(TNG_SNAPSHOT))
STELLAR_ASSEMBLY_FILE_PATH = STELLAR_ASSEMBLY_PATH.format('0' + str(TNG_SNAPSHOT))

RESAMPLE_ZOOM = 0.25
RESAMPLE_ORDER = 3

MEAN_STD_ARRAYS_PATH = os.path.join(KINETIC_MAPS_PATH, 'Mean_std_arrays')

# Experiment values
IGNORE_CHANNELS = ()
MASK_RADIUS = 100

# Change these for different runs
EAGLE_MATCH_TNG = True
RUN_DIR = 'EAGLE_TNGlike'

TNG_DIR = 'IllustrisTNG'

if EAGLE_MATCH_TNG:
    DATASET_1D = True   # Use 1D dataset for EAGLE-TNGlike data (2D not available at this point)
    EAGLE_SIM = 'L68n1504FP'
    DATASET_MAPPING = {
        'TNG': 'tng_dataset',
        'EAGLE': 'eagle_tng_like_dataset',
        'ΒΟΤΗ': 'both'
    }
    EAGLE_DIR = 'EAGLE_TNGlike'
else:
    DATASET_1D = False
    DATASET_MAPPING = {
        'TNG': 'tng2d_dataset',
        'EAGLE': 'eagle2d_dataset',
        'ΒΟΤΗ': 'both'
    }
    EAGLE_DIR = 'EAGLE'
    EAGLE_SIM = 'RefL0100N1504'

if DEBUG:
    BASE_PATH_MAPS = '/home/eirini/Documents/PhD/'
else:
    BASE_PATH_MAPS = '/net/diva/scratch1/eirinia/projects'

little_h = 0.6774
SPLITS_FILENAME = 'dataset_splits{}.pkl'.format('_1D' if DATASET_1D else '')

DATA_BASE_PATH = os.path.join(BASE_PATH_MAPS, 'Data')
EAGLE_DATA_BASE_PATH = os.path.join(DATA_BASE_PATH, '{}/{}'.format(EAGLE_DIR, EAGLE_SIM))
EAGLE_POSTPROCESSING_PATH = os.path.join(EAGLE_DATA_BASE_PATH, 'postprocessing')
EAGLE_BALANCED_DATASET_ALIGNS_PATH = os.path.join(EAGLE_POSTPROCESSING_PATH,
                                                  'balanced_dataset/balanced_alignments.hdf5')
EAGLE_DATASET_SPLITS = os.path.join(EAGLE_POSTPROCESSING_PATH, 'balanced_dataset', SPLITS_FILENAME)

TNG_SNAP = '099'
TNG_SIM = 'TNG100-1'
DATA_BASE_PATH = os.path.join(BASE_PATH_MAPS, 'Data')
TNG_DATA_BASE_PATH = os.path.join(DATA_BASE_PATH, '{}/{}'.format(TNG_DIR, TNG_SIM))
TNG_POSTPROCESSING_PATH = os.path.join(TNG_DATA_BASE_PATH, 'postprocessing')
TNG_BALANCED_DATASET_ALIGNS_PATH = os.path.join(TNG_POSTPROCESSING_PATH,
                                                'balanced_dataset/balanced_alignments.hdf5')
TNG_DATASET_SPLITS = os.path.join(TNG_POSTPROCESSING_PATH, 'balanced_dataset', SPLITS_FILENAME)
TNG_STELLAR_ASSEMBLY_PATH_BASE = os.path.join(TNG_POSTPROCESSING_PATH, 'stellar_assembly')
TNG_STELLAR_ASSEMBLY_PATH = os.path.join(TNG_STELLAR_ASSEMBLY_PATH_BASE, 'galaxies_{}.hdf5')

MEAN_STD_ARRAYS_PATH = os.path.join(DATA_BASE_PATH, 'Mean_std_arrays')


MODELS_PATH = os.path.join(RESULTS_PATH, 'saved_models')
MODELS_RUN_PATH = os.path.join(MODELS_PATH, RUN_DIR)

ENSEMBLE_SAVE_PATH = os.path.join(RESULTS_PATH, 'saved_results')
ENSEMBLE_RUN_PATH = os.path.join(ENSEMBLE_SAVE_PATH, RUN_DIR)

GRADIENTS_PATH = os.path.join(RESULTS_PATH, 'Gradients_df')
GRADIENTS_RUN_PATH = os.path.join(GRADIENTS_PATH, RUN_DIR)

INPUT_CORRELATION_PATH = os.path.join(RESULTS_PATH, 'Correlations')
INPUT_CORRELATION_PATH_RUN_PATH = os.path.join(INPUT_CORRELATION_PATH, RUN_DIR)

UNCERTAINTIES_PATH = os.path.join(RESULTS_PATH, 'Uncertainties')
UNCERTAINTIES_RUN_PATH = os.path.join(UNCERTAINTIES_PATH, RUN_DIR)

DATASET_1D_PATH = os.path.join(DATASET_RESULTS_PATH, 'Datasets_1D')
DATASET_1D_RUN_PATH = DATASET_1D_PATH
if EAGLE_MATCH_TNG:
    DATASET_1D_RUN_PATH = os.path.join(DATASET_1D_PATH, RUN_DIR)

CORRELATIONS_1D_RESULTS_PATH = os.path.join(RESULTS_PATH, '1D_Correlations')
CORRELATIONS_1D_RESULTS_RUN_PATH = CORRELATIONS_1D_RESULTS_PATH
if EAGLE_MATCH_TNG:
    CORRELATIONS_1D_RESULTS_RUN_PATH = os.path.join(CORRELATIONS_1D_RESULTS_PATH, RUN_DIR)


CATALOG_RESULTS_PATH = os.path.join(RESULTS_PATH, 'Catalog_Results_Lim')
CATALOG_RESULTS_RUN_PATH = os.path.join(CATALOG_RESULTS_PATH, RUN_DIR)
CATALOG_MODELS_PATH = os.path.join(CATALOG_RESULTS_RUN_PATH, 'Models')

CNN_RESULTS_PATH = os.path.join(RESULTS_PATH, 'CNN')
CNN_RESULTS_RUN_PATH = os.path.join(CNN_RESULTS_PATH, RUN_DIR)

UMAPS_RESULTS_PATH = os.path.join(RESULTS_PATH, 'UMAPs')
UMAPS_RESULTS_RUN_PATH = os.path.join(UMAPS_RESULTS_PATH, RUN_DIR)
UMAPS_RESULTS_RUN_PATH_DIVA = os.path.join(BASE_PATH_MAPS, 'Results',
                                           'UMAPs', RUN_DIR)

RUN_DIVA = True
if RUN_DIVA:
    UMAPS_RESULTS_RUN_PATH = UMAPS_RESULTS_RUN_PATH_DIVA

RESULTS_DA_PATH = os.path.join(RESULTS_PATH, 'Domain_Adaptation')
MODELS_DA_RUN_PATH = os.path.join(MODELS_PATH, 'Domain_Adaptation', RUN_DIR)
