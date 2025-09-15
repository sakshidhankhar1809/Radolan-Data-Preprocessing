import os
import numpy as np
from utils import crop_region, fetch_and_process, BASE_URL
import pandas as pd
import random
import h5py
from itertools import islice
#%%

latlon_szg = np.array([10.446078, 52.150786])
margarethenklippe = np.array([10.3726, 51.8826])
Rammelsberghaus = np.array([10.418889, 51.890278])
gottingen = np.array([9.935556, 51.533889])
greene = np.array([9.9399, 51.8569])
derenburg = np.array([10.9107, 51.8714])
# crop_region(2003, 2017, gottingen, 3, 4, 'gottingen', resolution=15, save_path='./data/gottingen', save=True)
# crop_region(2003, 2017, margarethenklippe, 3, 4, 'Margarethenklippe', resolution=15, save_path='./data/Margarethenklippe', save=True)
# crop_region(2003, 2017, Rammelsberghaus, 3, 4, 'Rammelsberghaus', resolution=15, save_path='./data/Rammelsberghaus', save=True)
# crop_region(2004, 2017, greene, 71, 71, 'Greene', resolution=15, save_path='./data/Greene', save=True)
crop_region(2004, 2004, derenburg, 71, 71, 'Derenburg', resolution=15, save_path='./data/Derenburg', save=True)
