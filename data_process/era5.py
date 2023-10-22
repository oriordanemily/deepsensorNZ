#%% 

import os
from typing import Literal
import glob

import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ProcessERA5:
    def __init__(self) -> None:
        self.path = 'data/ftp.bodekerscientific.com/Greg/ForRisa2'
        self.names = {
            'precipitation': {
                'folder': f'{self.path}/total_precipitation_hourly',
                'var_name': 'precipitation',
            },
            'temperature': {
                'folder': f'{self.path}/2m_temperature',
                'var_name': 't2m',
            },
        }


    def get_ds(self, 
               var:Literal['precipitation', 'temperature'],
               ):
        filenames = self.get_filenames(var)
        return xr.open_mfdataset(filenames)

    
    def ds_to_da(self,
                 ds,
                 var:Literal['precipitation', 'temperature'],
                 ):
        return ds[self.names[var]['var_name']]
    

    def get_ds_year(self,
                    var:Literal['precipitation', 'temperature'],
                    year:int,
                    ):
        filenames = self.get_filenames_year(var, year)
        return xr.open_mfdataset(filenames)


    def get_filenames(self,
                      var:Literal['precipitation', 'temperature'],
                      ):
        if var == 'temperature':
            filenames = glob.glob(f'{self.names[var]["folder"]}/*/*.nc')
        elif var == 'precipitation':  
            filenames = glob.glob(f'{self.names[var]["folder"]}/*.nc')
        else:
            raise ValueError(f'var={var} not recognised')
        return filenames


    def get_filenames_year(self,
                           var:Literal['precipitation', 'temperature'],
                           year:int,
                           ):
        if var == 'temperature':
            filenames = glob.glob(f'{self.names[var]["folder"]}/{year}/*.nc')
        elif var == 'precipitation':
            filenames = [f'{self.names[var]["folder"]}/{fname}' for fname in os.listdir(f'{self.names["precipitation"]["folder"]}') if str(year) in fname]
        else:
            raise ValueError(f'var={var} not recognised')
        return filenames
        




# #%% 

# era5 = ProcessERA5()
# var = 'temperature'
# ds = era5.get_ds(var)


# #%% test 
# #ds = xr.open_mfdataset(f'{era5.names[var]}')

# oswalk = list(os.walk(f'{era5.names[var]["folder"]}'))
# subdirs = [f'{oswalk[0][0]}/{i}' for i in oswalk[0][1]]

# ds1 = xr.open_mfdataset(f'{subdirs[0]}/*.nc')
# ds = xr.open_mfdataset([f'{subdir}/*.nc' for subdir in subdirs])

# # way 1
# # ds_list = [xr.open_mfdataset(f'{subdir}/*.nc') for subdir in subdirs]
# # ds = xr.concat(ds_list, dim='time')

# #%% load era5 data

# # #way 2
# import glob
# # Get a list of all .nc files available in different folders
# filenames = glob.glob(f'{era5.names[var]["folder"]}/*/*.nc')
# dsmerged = xr.open_mfdataset(filenames)

# # precip

# var = 'precipitation'
# filenames = glob.glob(f'{era5.names[var]["folder"]}/*.nc')

# #%% 

