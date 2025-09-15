import os
import numpy as np
from utils import crop_region, pp, fetch_and_process, BASE_URL
import pandas as pd
import random
import h5py
from itertools import islice
#%%
PATH = './data/YW2017.002_201707_asc/YW2017.002_20170726_asc/'

list_files = os.listdir(PATH)
rnd_idx = random.randint(0, len(list_files))

latlon_muenchen = (11.576124, 48.137154)
latlon_goslar = np.array([10.4290, 51.9060])
latlon_szg = np.array([10.446078, 52.150786])
latlon_goe = np.array([9.9158, 51.5413])
# pp = pp(latlon_goe, 7, 9, 'Göttingen', 1100, 900)

# pp = pp(latlon_goslar, 3, 4, 'Goslar', 1100, 900)
# mask_idx = pp.mask_region()
# data = np.loadtxt(os.path.join(PATH, list_files[rnd_idx]) , skiprows=6)
# pp.plot_mask(data)

total_df = crop_region(2003, 2023, latlon_goe, 3, 4, 'Göttingen', resolution=15, save_path='./data/Goettingen', save=True)

