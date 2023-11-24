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
from tqdm import tqdm

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

var = 'temperature'
#var = 'precipitation'
year = 2000

era5 = ProcessERA5()
#ds = era5.get_ds('temperature')
ds_era = era5.get_ds_year(var, year)
da_era = era5.ds_to_da(ds_era, var)
#era5_raw_ds = da.coarsen(latitude=5, longitude=5, boundary="trim").mean()
#era5_raw_ds = era5_raw_ds.load()

#%% get list of lon lat of stations that have year (2000)

stations = ProcessStations()
all_stations = stations.get_list_all_stations(var)
print(len(all_stations))

#year = '2000'
lons = []
lats = []

stations_available_for_year = []
for i in tqdm(range(len(all_stations))):
    ds_station = stations.get_station_ds(var, i_station=i)
    da_station = stations.get_da_from_ds(ds_station, var)
    #da = stations.get_station_da(var, i_station=i)
    try:
        da_station = da_station.sel(time=str(year))
        stations_available_for_year.append(i)
        lons.append(ds_station.longitude.values)
        lats.append(ds_station.latitude.values)
    except:
        pass

# stations_available_for_year
# 1,8,10,11,12 15 20,22, 

example_station = 1
ds_station = stations.get_station_ds(var, i_station=example_station)
da_station = stations.get_da_from_ds(ds_station, var)
#da_station = stations.get_station_da(var, i_station=example_station)
da_station = da_station.sel(time=str(year))

#%% 

#%% plot era5 snapshot

# print(ds_station)

lon_lat_tuples = list(zip(lons, lats))
# fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs))
# da.isel(time=0).load().plot()
# #ax.add_feature(cf.BORDERS)
# ax.coastlines()

minlon = np.array(ds_era['longitude'].min())
maxlon = np.array(ds_era['longitude'].max())
minlat = np.array(ds_era['latitude'].min())
maxlat = np.array(ds_era['latitude'].max())

lon_lim=(minlon, maxlon)
lat_lim=(minlat, maxlat)

stations.plot_points(lon_lat_tuples, lon_lim=(minlon, maxlon), lat_lim=(minlat, maxlat))

#%% debug 
marker_size = 30

proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.coastlines()
da_era.isel(time=0).plot()
if lon_lim is not None and lat_lim is not None:
    ax.set_xlim(lon_lim)
    ax.set_ylim(lat_lim)
ax.gridlines(draw_labels=True, crs=proj)
for lon, lat in lon_lat_tuples:
    ax.scatter(lon, lat, color='red', marker='o', s=marker_size)
plt.show()


#%% 
####  works 

# #print(da)
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs), figsize=(10, 12))
proj = ccrs.PlateCarree()
#ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.coastlines()
ax.set_xlim(minlon, maxlon)
ax.set_ylim(minlat, maxlat)
#ax.set_xlim(161, 183)
#ax.set_ylim(-31, -50)
da_era.isel(time=0).plot()
#ax.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]], ccrs.PlateCarree())
# #ax.set_extent([161, 183, -50, -31], ccrs.PlateCarree())

# #ax.add_feature(cf.BORDERS)

#%% plot stations




#%% 


