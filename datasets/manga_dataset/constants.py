from collections import OrderedDict
import os

DEBUG = False

TNG_SNAPSHOT = 99
SPLITS = OrderedDict({
    'test': 1
})

MAP_NAMES = ('mass', 'vel', 'vel_disp', 'metal', 'age')
MAP_CHANNEL_ID = (18, 13, 15, 9, 6)
MAP_LOGS = (True, False, False, True, False)
MAP_CLIPS = (None, (-250, 250), (None, 350), (-1, 1), None)

if DEBUG:
    BASE_PATH = '/home/eirini/Documents/PhD/'
else:
    BASE_PATH = '/net/diva/scratch1/eirinia/projects'


MANGA_MAPS_PATH = '/scratch/eirinia/projects/Data/MaNGA/v3_1_1'
MANGA_INFO_PATH = os.path.join(MANGA_MAPS_PATH, 'SDSS17Pipe3D_v3_1_1.fits')
