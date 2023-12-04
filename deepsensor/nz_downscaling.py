#%% 

import logging
logging.captureWarnings(True)
import os

import xarray as xr
import pandas as pd
import numpy as np
import scipy
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

use_gpu = True
if use_gpu:
    set_gpu_default_device()

f_topography = 'data/topography/nz_elevation_100m.nc'

#%% load ERA5

var = 'temperature'
#var = 'precipitation'
year = 2000

era5 = ProcessERA5()
#ds = era5.get_ds('temperature')
ds_era = era5.get_ds_year(var, year)
da_era = era5.ds_to_da(ds_era, var)
#era5_raw_ds = da.coarsen(latitude=5, longitude=5, boundary="trim").mean()
#era5_raw_ds = era5_raw_ds.load()

## * get list of lon lat of stations that have year (2000)

stations = ProcessStations()
station_paths = stations.get_path_all_stations(var)

#all_stations = stations.get_list_all_stations(var)
#print(len(all_stations))

#year = '2000'
lons = []
lats = []
stations_available_for_year = []
p = station_paths[0]
lon_lat_tuples = []

#%% 

for p in tqdm(station_paths):
    ds_station = stations.get_station_ds(filepath=p)
    da_station = stations.get_da_from_ds(ds_station, var)
    try:
        da_station = da_station.sel(time=str(year))
        stations_available_for_year.append(p)
        lon_lat_tuples.append((ds_station.longitude.values, ds_station.latitude.values))
        # lons.append(ds_station.longitude.values)
        # lats.append(ds_station.latitude.values)
    except:
        pass

# stations_available_for_year
# 1,8,10,11,12 15 20,22, 
# ['data/nz/ScreenObs/1002.nc','data/nz/ScreenObs/10617.nc','data/nz/ScreenObs/1087.nc']

# load example station
example_st = ['data/nz/ScreenObs/1002.nc','data/nz/ScreenObs/10617.nc','data/nz/ScreenObs/1087.nc']
ds_station = stations.get_station_ds(variable=var, filepath=example_st[0])
da_station = stations.get_da_from_ds(ds_station, var)
da_station = da_station.sel(time=str(year))

#%% plot era5 snapshot with stations

minlon = np.array(ds_era['longitude'].min())
maxlon = np.array(ds_era['longitude'].max())
minlat = np.array(ds_era['latitude'].min())
maxlat = np.array(ds_era['latitude'].max())

lon_lim=(minlon, maxlon)
lat_lim=(minlat, maxlat)

marker_size = 30
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs), figsize=(10, 12))
proj = ccrs.PlateCarree()
ax.coastlines()
ax.set_xlim(minlon, maxlon)
ax.set_ylim(minlat, maxlat)
da_era.isel(time=0).plot()
for lon, lat in lon_lat_tuples:
    ax.scatter(lon, lat, color='red', marker='o', s=marker_size)

#%% 

# coarsen era5
da_era_coarse = da_era.coarsen(latitude=5, longitude=5, boundary="trim").mean()
print(da_era_coarse.shape)

# load elevation 
ds_elev = xr.load_dataset(f_topography)
ds_elev_coarse = ds_elev.coarsen(latitude=5, longitude=5, boundary="trim").mean()
print(ds_elev_coarse)

#%% 

hires_aux_raw_ds = ds_elev_coarse

# Compute Topographic Position Index from elevation data

# Resolutions in coordinate values along the spatial row and column dimensions
#   Here we assume the elevation is on a regular grid, so the first difference
#   is equal to all others.
coord_names = list(hires_aux_raw_ds.dims)
resolutions = np.array(
    [np.abs(np.diff(hires_aux_raw_ds.coords[coord].values)[0]) for coord in coord_names])

for window_size in [.1, .05, .025]:
    smoothed_elev_da = hires_aux_raw_ds['elevation'].copy(deep=True)

    # Compute gaussian filter scale in terms of grid cells
    scales = window_size / resolutions

    smoothed_elev_da.data = scipy.ndimage.gaussian_filter(smoothed_elev_da.data, sigma=scales, mode='nearest')

    TPI_da = hires_aux_raw_ds['elevation'] - smoothed_elev_da
    
    hires_aux_raw_ds[f"TPI_{window_size}"] = TPI_da

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs))
    TPI_da.plot(ax=ax)
    ax.add_feature(cf.BORDERS)
    ax.coastlines()

#%% 




#%% 
