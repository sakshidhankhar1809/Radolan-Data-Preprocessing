import os
import numpy as np
from utils import crop_region, fetch_and_process, BASE_URL
import pandas as pd
import random
import h5py
from itertools import islice
#%%

gottingen = np.array([9.935556, 51.533889])
crop_region(2003, 2017, gottingen, 3, 4, 'gottingen', resolution=15, save_path='./data/gottingen', save=True)


