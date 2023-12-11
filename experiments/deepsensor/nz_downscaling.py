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

from nzdownscale.dataprocess.era5 import ProcessERA5
from nzdownscale.dataprocess.stations import ProcessStations
from nzdownscale.dataprocess.topography import ProcessTopography
from nzdownscale.dataprocess.utils import DataProcess, PlotData
from nzdownscale.dataprocess.config import DATA_PATHS

crs = ccrs.PlateCarree()

use_gpu = True
if use_gpu:
    set_gpu_default_device()

#%% settings

var = 'temperature'
#var = 'precipitation'
year = 2000

dataprocess = DataProcess()

#%% load elevation 

top = ProcessTopography()
ds_elev = top.open_ds()

#%% load ERA5

era5 = ProcessERA5()
ds_era = era5.load_ds(var, year)
da_era = era5.ds_to_da(ds_era, var)

#%% load stations (covering year 2000)

stations = ProcessStations()
station_paths = stations.get_path_all_stations(var)
df = stations.get_metadata_df(var)

df_filtered = df[(df['start_year']<year) & (df['end_year']>=year)]
station_paths_filtered = list(df_filtered.index)
print(len(station_paths_filtered))

#%% plot era5 snapshot with stations

nzplot = PlotData()
ax = nzplot.nz_map_with_coastlines()
da_era.isel(time=0).plot()
ax = stations.plot_stations(df, ax)
plt.plot()

#%% Coarsen ERA5 and topography

coarsen_factor = 5
da_era_coarse = era5.coarsen_da(da_era, coarsen_factor)

nzplot = PlotData()
ax = nzplot.nz_map_with_coastlines()
da_era_coarse.isel(time=0).plot()
plt.plot()

#da_elev_coarse = top.coarsen_da(da_elev, coarsen_factor)
#da_elev = top.ds_to_da(ds_elev)
ds_elev_coarse = top.coarsen_da(ds_elev, coarsen_factor)
da_elev_coarse = top.ds_to_da(ds_elev_coarse)

nzplot = PlotData()
ax = nzplot.nz_map_with_coastlines()
da_elev_coarse.plot()
plt.show()

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
