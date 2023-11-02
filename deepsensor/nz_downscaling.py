#%% 
import logging
logging.captureWarnings(True)

import os

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import seaborn as sns

import deepsensor.torch
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.model.convnp import ConvNP
from deepsensor.active_learning.algorithms import GreedyAlgorithm
from deepsensor.active_learning.acquisition_fns import Stddev
from deepsensor.train.train import train_epoch, set_gpu_default_device
from deepsensor.data.utils import construct_x1x2_ds

from data_process.era5 import ProcessERA5
from data_process.stations import ProcessStations

crs = ccrs.PlateCarree()

#%% 


use_gpu = True
if use_gpu:
    set_gpu_default_device()

#%% 

# # Load raw data
# era5_raw_ds = xr.open_mfdataset('../../germany/era5/*.nc')
# era5_raw_ds = era5_raw_ds["t2m"]
# era5_raw_ds = era5_raw_ds - 273.15
# # Coarsen to CMIP6-ish resolution
# print(era5_raw_ds.shape)
# era5_raw_ds = era5_raw_ds.coarsen(lat=5, lon=5, boundary="trim").mean()
# era5_raw_ds = era5_raw_ds.load()
# print(era5_raw_ds.shape)

#%% get ERA5

#var = 'precipitation'
var = 'temperature'
year = 2000

era5 = ProcessERA5()
#ds = era5.get_ds('temperature')
ds = era5.get_ds_year(var, year)
da = era5.ds_to_da(ds, var)
#era5_raw_ds = da.coarsen(latitude=5, longitude=5, boundary="trim").mean()
#era5_raw_ds = era5_raw_ds.load()

stations = ProcessStations()
all_stations = stations.get_list_all_stations(var)
print(len(all_stations))

i = 0
ds = xr.open_mfdataset(f'{stations.get_var_path(var)}/{all_stations[i]}')
ds.sel(time='2000')

#%% plot era5 snapshot

print(da)
# fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs))
# da.isel(time=0).load().plot()
# #ax.add_feature(cf.BORDERS)
# ax.coastlines()

minlon = np.array(da['longitude'].min())
maxlon = np.array(da['longitude'].max())
minlat = np.array(da['latitude'].min())
maxlat = np.array(da['latitude'].max())

#print(da)
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs), figsize=(10, 12))
proj = ccrs.PlateCarree()
#ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.coastlines()
ax.set_xlim(minlon, maxlon)
ax.set_ylim(minlat, maxlat)
da.isel(time=0).plot()
#ax.add_feature(cf.BORDERS)

#%% plot stations




