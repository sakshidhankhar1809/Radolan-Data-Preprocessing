#%%
import gzip
import io
import os
import tarfile
import time
from multiprocessing import Pool, cpu_count

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import wradlib as wrl
from osgeo import osr
from tqdm import tqdm

# from multiprocessing import Pool, cpu_count


BASE_URL = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/5_minutes/radolan/reproc/2017_002/asc/"

dim_x = 1100
dim_y = 900


###############################################################################

# def download(url, destination_path):
#     """
#     This function Downloads the radolan data and store on the specified path
#     params:
#         url: str
#         destination_path: str
#     """
#     if not os.path.exists(destination_path):
#         os.makedirs(destination_path)
    
#     filename = url.split('/')[-1].replace(" ", "_")
#     file_path = os.path.join(destination_path, filename)
#     r = requests.get(url, file_path)
#     if r.ok:
#         print("saving to: ",os.path.abspath(file_path))
#         with open(file_path, 'wb') as f:
#             for chunk in r.iter_content(chunk_size=1024*8):
#                 if chunk:
#                     f.write(chunk)
#                     f.flush()
#                     os.fsync(f.fileno())
#     else:
#         print("Error occured", r.status_code, r.text)


# def fetch_from_dwd(start_year, end_year):
#     """
#     This function will fetch the data for the desired year.
#     params:
#         year: int
#     """
#     for year in range(start_year, end_year+1):
#         base_dir = os.path.join(os.getcwd(), "data", f"{str(year)}/")
#         if not os.path.exists(base_dir):
#             os.makedirs(base_dir)
#         for i in range(1,13):
#             file_name = f"YW2017.002_{year}" + f"{i:02}" + "_asc.tar"
#             url = BASE_URL + str(year) + "/" + file_name 
#             # url = urljoin(BASE_URL, str(year), file_name)
#             month_path = os.path.join(base_dir, str(i))
#             if not os.path.exists(month_path):
#                 os.makedirs(month_path)
            
#             download(url, month_path)
            
#             # Extracting the .tar files includeing each days of a month
#             tar_path = os.path.join(month_path, file_name)

#             with tarfile.open(tar_path, 'r') as tar:
#                 tar.extractall(path=month_path)
#             print(f"deleting {file_name}")
#             os.remove(tar_path)
                
#             # Extracting the ".tar.gz" files for each day
#             for i, dirname in enumerate(os.listdir(month_path)):
#                 gz_dir = os.path.join(month_path, str(i+1))
#                 if not os.path.exists(gz_dir):
#                     os.makedirs(gz_dir)
#                     gz_path = os.path.join(month_path, dirname)
#                 with tarfile.open(gz_path, 'r:gz') as gz_tar:
#                     gz_tar.extractall(path=gz_dir)
#                 os.remove(gz_path)


def check_supplement(year, month, day, hh, MM):
    """
    This function checks if the current date and time is a missed timestampt
    
    """
    status = None
    hour_range = [i for i in range(0, 6)]
    if (year == 2018) and (month == 1) and (day == 1) and (hh in hour_range):
        if hh == 5 and (MM > 45):
            status = False
        else:
            status = True
    return status

def get_supplement(file_name):
    """
    A helper function for handeling missed timestampts for the date 2018-01-01
    from time 00:00 to 05:45. It automatically handels missing .asc files, since
    the actual date files contain all NaN values, lets say ´-9´.
    """
    url = BASE_URL + 'supplement/YW2017.002_Supplement_asc.tar.gz'
    response = requests.get(url)
    response.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(response.content)) as file:
        list_name = file.getmembers()
        final_file = file.extractfile(file_name)
        final_content = final_file.read().decode('ascii')
    return final_content
    
def fetch_and_process(year, base_url, mask_idx, save_path, resolution=5, save=None):
    """
    Function for retriving the Radolan files w.o storing them locally.
    param:
        year: int (Year for which data is being retrieved)
        url: string (Base URL for the year)
        mask_idx: array (Indices for the region of interest)
        save_path: string (Path where data should be saved)
        resolution: int (Timesteps that should be accumulated over)
    """
    year_data = []
    base_dir = os.path.join(save_path, str(year))
    if not os.path.exists(base_dir) and save == True:
        os.makedirs(base_dir)
    for month in range(1, 13):
        month_dir = os.path.join(base_dir, str(month))
        if not os.path.exists(month_dir) and save == True:
            os.makedirs(month_dir)
            
        url = base_url + f'YW2017.002_{year}{month:02}_asc.tar'
        response = requests.get(url)
        response.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(response.content)) as outer_tar:
            for dd in tqdm(range(1, len(outer_tar.getmembers()) + 1)):
                day_dir = os.path.join(month_dir, str(dd))
                if not os.path.exists(day_dir) and save == True:
                    os.makedirs(day_dir)
                gzipped_file = outer_tar.extractfile(f'YW2017.002_{year}{month:02}{dd:02}_asc.tar.gz')
                sum_ = None
                with gzip.open(gzipped_file) as gz:
                    with tarfile.open(fileobj=io.BytesIO(gz.read())) as inner_tar:
                        for hh in range(0, 24):
                            for MM in range(0, 60, 5):
                                file_name = f'YW_2017.002_{year}{month:02}{dd:02}_{hh:02}{MM:02}.asc'
                                if check_supplement(year, month, dd, hh, MM):
                                    final_content = get_supplement(file_name)
                                else:
                                    final_file = inner_tar.extractfile('YW_2017.002_' + f'{year}{month:02}{dd:02}_{hh:02}{MM:02}.asc')
                                    final_content = final_file.read().decode('ascii')
                                data = np.ma.masked_equal(np.loadtxt(io.StringIO(final_content), skiprows=6), -9)
                                data = np.flip(data, axis=0)
                                # data = data[mask_idx[0, 1]:mask_idx[1, 1], mask_idx[1, 0]:mask_idx[0, 0]]
                                data = data[mask_idx[0, 0]:mask_idx[1, 0], mask_idx[0, 1]:mask_idx[1, 1]]
                                if sum_ is None:
                                    sum_ = np.zeros_like(data)
                                sum_ += np.maximum(data, 0)
                                if (MM+5) % resolution == 0:
                                    if save==True:
                                        file_name = file_name + 'aggregated'
                                        file_path = os.path.join(day_dir, file_name)
                                        np.savez_compressed(file_path + '.npz', data=sum_)
                                    # year_df = pd.concat([year_df, pd.DataFrame([sum.flatten()])], ignore_index=True)
                                    year_data.append(sum_.flatten())
                                    # year_data.append(sum_) # Changed from flatten to 2D arrays for each row
                                    sum_ = np.zeros_like(data)

              
            print(f'Month: {month} has been processed! and the shape of pandas DataFrame is{len(year_data)}')
    year_data_np = np.array(year_data)
    if save == True:
        # idx = pd.date_range(f'{year}', f'{year+1}', freq=f'{resolution}min')[:-1]
        dataframe = pd.concat([pd.DataFrame(year_data)], axis=0)
        # dataframe.index = idx
        dataframe.to_csv(base_dir + f'/{year}.csv')
            
    # year_df.to_hdf(os.path.join(base_dir, f'{year}_hdf.h5'), key=f'{year}_df', mode='w')
    return year_data_np

# def process_file(file_path, mask_idx, save_path):
#     """
#     Function to process each file.
#     It loads, masks, and saves the processed data.
#     """
#     data = np.loadtxt(file_path, skiprows=6)
#     data = data[mask_idx[0, 0]:mask_idx[1, 0], mask_idx[0, 1]:mask_idx[1, 1]]
#     os.remove(file_path)
#     np.savez_compressed(save_path, data=data)


# def process_folder(folder_path, mask_idx):
#     """
#     Function to process all files within a folder sequentially.
#     """
#     files = [os.path.join(folder_path, file) for file in sorted(os.listdir(folder_path))]
#     for file_path in files:
#         process_file(file_path, mask_idx)


def save_df(index, PATH):
    if not os.path.exists(PATH): 
        raise FileNotFoundError(f"{PATH} dose not exist!")
    final_array = []
    
    for folder in sorted(os.listdir(PATH)):
        print(f"{folder} is being processed")
        file_path = os.path.join(PATH, folder, f'{folder}_data.h5')
        with h5py.File(file_path, 'r') as hdf:
            a_group_key = list(hdf.keys())[0]
            ds_array = hdf[a_group_key][()]
            final_array.append(ds_array)
        os.remove(file_path)
        os.removedirs(os.path.join(PATH, f"{folder}"))

    final_df = pd.DataFrame(np.concat(final_array), index=index)
    with h5py.File(PATH + 'total_file.h5', 'w') as hdf:
        hdf.create_dataset('total_data', data=final_df, compression='gzip')
                    


def crop_region(start_year, end_year, latlon, mask_height, mask_width, city_name, resolution, save_path, save=None):
    start = time.time()
    if not isinstance(start_year, int) or not isinstance(end_year, int):
        raise ValueError("Year must be an Integer!")
    # if start_year >= end_year:
    #     raise ValueError("Start Year should be greater thatn End Year")
    if save_path is None:
        raise TypeError("Path has not been given")
     
    city_obj = preProcess(latlon, mask_height, mask_width, city_name, dim_x, dim_y)
    city_idx = city_obj.mask_region()

    years = list(range(start_year, end_year + 1))
    print(years)
    with Pool(processes=min(len(years), cpu_count())) as pool:
        year_dfs = pool.starmap(
            fetch_and_process, 
            [(yyyy, BASE_URL + f'{yyyy}/', city_idx, save_path, resolution, save) for yyyy in years]
            )

    total_array = np.concatenate(year_dfs, axis=0)
    time_idx = pd.date_range(str(start_year), str(end_year+1), freq=f'{resolution}min')[:-1]
    total_df = pd.DataFrame(total_array) 
    total_df = pd.DataFrame(total_array, index=time_idx)
 
    # save_df(time_idx, save_path)
    total_df.to_csv(save_path + '/total.csv')
    
    return total_df
    
dwd_string = (
    "+proj=stere +lat_0=90 +lon_0=10  +lat_ts=90 +k=0.93301270189 "
    "+x_0=0 +y_0=0 +a=6370040 +b=6370040 +to_meter=1000 +no_defs"
)

class preProcess:
    """
    This class is the implementation for handeling the raw ASCII data form
    DWD service. To crop the data for the region of intrest on the map of
    Germany, the mathod ′mask_region()′ can be used.
    params:
        path: string (path to the directory where the DWD data stored)
        dim_x: int (first diminsion of ASC file. Usually 1100)
        dim_y: int (second diminsion of ASC file. Usually 900)
        latlon: np.dtype (this is the numpy array containing the lon and la for
                          )
        mask_with: int (size of mask)'
        city_name: string (name of target location)
    """
    def __init__(
            self,
            latlon: np.dtype,
            mask_width: int,
            mask_height: int,
            city_name: str,
            dim_x: int,
            dim_y: int,
            path=None
            ) -> None:
        
        self.x_shape = dim_x
        self.y_shape = dim_y
        self.latlon = latlon
        self.mask_width = mask_width
        self.mask_heigth = mask_height
        self.proj_stereo = wrl.georef.create_osr("dwd-radolan")
        # self.proj_stereo = wrl.georef.projstr_to_osr(dwd_string)
        self.proj_wgs = osr.SpatialReference()
        self.proj_wgs.ImportFromEPSG(4326)
        self.radolan_grid_xy = wrl.georef.get_radolan_grid(dim_x, dim_y, wgs84=True, mode='radolan')
        self.path = path
        self.city_name = city_name
        
    def load_data(self):
        pass
    
    @staticmethod
    def get_metadata(path, file):
        with open(path, 'r') as f:
            header = [next(f) for _ in range(6)]
            
        metadata = {}
        for line in header:
            key, value = line.strip().split()
            metadata[key] = float(value)

        return metadata
    
    
    def wsg_to_stereo(self, coords):
        xy_coord = wrl.georef.reproject(coords,  
            src_crs=self.proj_wgs, 
            trg_crs=self.proj_stereo
            )
        return xy_coord
    
    def stereo_to_wgs(self, coords):
        ll_coord = wrl.georef.reproject(
            coords,
            src_crs = self.proj_stereo,
            trg_crs = self.proj_wgs
        )
        return ll_coord
    
    def find_center(self, radolan_xy, center):
        """
        This method finds the nearest grid point for the given ′center′ point.
        parasm:
            center: np.ndarray (the numpy.array of given target point)
            x: np.ndaray (numpy array for the XX grid points)
            y: np.ndarray (numpy array for the YY grid points)
        """
        lat_diff = radolan_xy[:, :, 0] - center[0]
        lon_diff = radolan_xy[:, :, 1] - center[1]
        dist = lat_diff**2 + lon_diff**2
        i_center, j_center= np.unravel_index(np.argmin(dist), dist.shape)
        
        return np.array([i_center, j_center])
    
    def mask_region(self):
        """
        This method applies a mask over desired region on the Germany's map. 
        param:
            center: numpy.ndarray
        """
        # center_xy = self.wsg_to_stereo(self.latlon)
        center_points = self.find_center(self.radolan_grid_xy, self.latlon)
        
        i_start = max(center_points[0] - self.mask_width, 0)
        i_end = min(center_points[0] + self.mask_width + 1, self.x_shape)
        j_start = max(center_points[1] - self.mask_heigth, 0)
        j_end = min(center_points[1] + self.mask_heigth + 1, self.y_shape)

        # masked_region = data[i_start:i_end, j_start:j_end]
        self.mask_idx = np.array([[i_start, j_start],
                         [i_end, j_end]])
        
        return self.mask_idx
    
    def plot_mask(self, data):
        self.target_xy = self.wsg_to_stereo(self.latlon)

        x1 = self.radolan_grid_xy[:, :, 0]
        y1 = self.radolan_grid_xy[:, :, 1]
        
        self.mask_idx = self.mask_region()
        
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x1, y1, data, shading='auto')
        plt.colorbar(shrink=0.75)
        plt.plot(self.target_xy[0], self.target_xy[1], 'ro', markersize=5, label=self.city_name)

        # Highlight the square region on the plot
        plt.gca().add_patch(plt.Rectangle((x1[self.mask_idx[0,0], self.mask_idx[0,1]], y1[self.mask_idx[0, 0], self.mask_idx[0, 1]]), 
                                        x1[self.mask_idx[1,0]-1, self.mask_idx[1, 1]-1] - x1[self.mask_idx[0, 0], self.mask_idx[0, 1]], 
                                        y1[self.mask_idx[1, 0]-1, self.mask_idx[1, 1]-1] - y1[self.mask_idx[0 ,0], self.mask_idx[0, 1]], 
                                        fill=False, edgecolor='blue', linewidth=2, label=f'{self.mask_width }x{self.mask_heigth} Mask'))

        plt.legend()
        plt.show()
        















