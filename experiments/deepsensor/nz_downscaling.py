#%% 

import logging
logging.captureWarnings(True)
import os
import time

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

from nzdownscale.dataprocess import era5, stations, topography, utils, config

# from nzdownscale.dataprocess.era5 import ProcessERA5
# from nzdownscale.dataprocess.stations import ProcessStations
# from nzdownscale.dataprocess.topography import ProcessTopography
# from nzdownscale.dataprocess.utils import DataProcess, PlotData
# from nzdownscale.dataprocess.config import DATA_PATHS


#%% Settings

# Variables
var = 'temperature'
# var = 'precipitation'
years = [2000, 2001]  # test

# plotting
crs = ccrs.PlateCarree()

# GPU 
use_gpu = True
if use_gpu:
    set_gpu_default_device()

dataprocess = utils.DataProcess()

#%% load elevation 

process_top = topography.ProcessTopography()
ds_elev = process_top.open_ds()

#%% load ERA5

process_era = era5.ProcessERA5()
ds_era = process_era.load_ds(var, years)
da_era = process_era.ds_to_da(ds_era, var)

#%% load stations (covering specified years)

process_stations = stations.ProcessStations()
station_paths = process_stations.get_path_all_stations(var)
df = process_stations.get_metadata_df(var)

df_filtered = df[(df['start_year']<years[0]) & (df['end_year']>=years[-1])]
station_paths_filtered = list(df_filtered.index)
print(len(station_paths_filtered))

#%% plot era5 snapshot with stations

nzplot = utils.PlotData()
ax = nzplot.nz_map_with_coastlines()
da_era.isel(time=0).plot()
ax = process_stations.plot_stations(df, ax)
plt.plot()

#%% Coarsen and plot (ERA5 and topography)

coarsen_factor = 10
da_era_coarse = process_era.coarsen_da(da_era, coarsen_factor)
latres = dataprocess.resolution(da_era_coarse, 'latitude')
lonres = dataprocess.resolution(da_era_coarse, 'longitude')

nzplot = utils.PlotData()
ax = nzplot.nz_map_with_coastlines()
da_era_coarse.isel(time=0).plot()
plt.title(f'Coarsened ERA5\nLat res: {latres:.4f} degrees, lon res: {lonres:.4f} degrees')
plt.plot()

#da_elev_coarse = top.coarsen_da(da_elev, coarsen_factor)
#da_elev = top.ds_to_da(ds_elev)
coarsen_factor = 5
ds_elev_coarse = process_top.coarsen_da(ds_elev, coarsen_factor)
da_elev_coarse = process_top.ds_to_da(ds_elev_coarse)
latres = dataprocess.resolution(ds_elev_coarse, 'latitude')
lonres = dataprocess.resolution(ds_elev_coarse, 'longitude')

nzplot = utils.PlotData()
ax = nzplot.nz_map_with_coastlines()
da_elev_coarse.plot()
plt.title(f'Coarsened topography\nLat res: {latres:.4f} degrees, lon res: {lonres:.4f} degrees')
plt.show()

#%% 
# Compute Topographic Position Index from elevation data
PLOT = True

# Resolutions in coordinate values along the spatial row and column dimensions. Here we assume the elevation is on a regular grid, so the first difference is equal to all others.

highres_aux_raw_ds = ds_elev_coarse


coord_names = list(highres_aux_raw_ds.dims)
resolutions = np.array(
    [np.abs(np.diff(highres_aux_raw_ds.coords[coord].values)[0]) for coord in coord_names])

for window_size in [.1, .05, .025]:
    smoothed_elev_da = highres_aux_raw_ds['elevation'].copy(deep=True)

    # Compute gaussian filter scale in terms of grid cells
    scales = window_size / resolutions

    smoothed_elev_da.data = scipy.ndimage.gaussian_filter(smoothed_elev_da.data, sigma=scales, mode='nearest')

    TPI_da = highres_aux_raw_ds['elevation'] - smoothed_elev_da
    
    highres_aux_raw_ds[f"TPI_{window_size}"] = TPI_da

    if PLOT:
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs))
        TPI_da.plot(ax=ax)
        ax.add_feature(cf.BORDERS)
        ax.coastlines()
        plt.show()

#%% Low resolution topography 

coarsen_factor = 20
aux_raw_ds = process_top.coarsen_da(ds_elev_coarse, coarsen_factor)
aux_raw_ds = process_top.ds_to_da(aux_raw_ds)
latres = dataprocess.resolution(aux_raw_ds, 'latitude')
lonres = dataprocess.resolution(aux_raw_ds, 'longitude')

nzplot = utils.PlotData()
ax = nzplot.nz_map_with_coastlines()
aux_raw_ds.plot()
plt.title(f'Low-res topography\nLat res: {latres:.4f} degrees, lon res: {lonres:.4f} degrees')
plt.show()

#%% 

# Print resolution of lowres and highres elevation data
print(f"Lowres lat resolution: {dataprocess.resolution(aux_raw_ds, 'latitude'):.4f} degrees")
print(f"Lowres lon resolution: {dataprocess.resolution(aux_raw_ds, 'longitude'):.4f} degrees")
print(f"Highres lat resolution: {dataprocess.resolution(highres_aux_raw_ds, 'latitude'):.4f} degrees")
print(f"Highres lon resolution: {dataprocess.resolution(highres_aux_raw_ds, 'longitude'):.4f} degrees")
print(f"Original lat resolution: {dataprocess.resolution(ds_elev, 'latitude'):.4f} degrees")
print(f"Original lon resolution: {dataprocess.resolution(ds_elev, 'longitude'):.4f} degrees")

#%% 

# Slice era5 data to elevation data's spatial extent

top_ds = aux_raw_ds
era_ds = da_era_coarse

top_min_lat =  top_ds['latitude'].min()
top_max_lat = top_ds['latitude'].max()
top_min_lon = top_ds['longitude'].min()
top_max_lon = top_ds['longitude'].max()

era5_raw_ds = era_ds.sel(latitude=slice(top_max_lat, top_min_lat), longitude=slice(top_min_lon, top_max_lon))

era5_raw_ds.isel(time=0).plot()

#%% 

station_paths = station_paths_filtered
df_list = []
path = station_paths[0]
for path in tqdm(station_paths):
    df = process_stations.load_station_df(path, var, daily=True)
    df_list.append(df)
df = pd.concat(df_list)
station_raw_df = df.reset_index().set_index(['time', 'latitude', 'longitude']).sort_index()

### filter years
station_raw_df_ = station_raw_df.reset_index()
station_raw_df_ = station_raw_df_[(station_raw_df_['time']>=str(years[0])) & (station_raw_df_['time']<=f'{str(years[-1])}-12-31')]
station_raw_df = station_raw_df_.set_index(['time', 'latitude', 'longitude']).sort_index()
###

print(station_raw_df)

#%% Normalise and preprocess data

data_processor = DataProcessor(x1_name="latitude", x1_map=(era5_raw_ds["latitude"].min(), era5_raw_ds["latitude"].max()), x2_name="longitude", x2_map=(era5_raw_ds["longitude"].min(), era5_raw_ds["longitude"].max()))
era5_ds, station_df = data_processor([era5_raw_ds, station_raw_df])

aux_ds, hires_aux_ds = data_processor([aux_raw_ds, highres_aux_raw_ds], method="min_max")
print(data_processor)

# Generate auxiliary dataset of x1/x2 coordinates to break translation equivariance in the model's CNN
# to enable learning non-stationarity
x1x2_ds = construct_x1x2_ds(aux_ds)
aux_ds['x1_arr'] = x1x2_ds['x1_arr']
aux_ds['x2_arr'] = x1x2_ds['x2_arr']
aux_ds

# should we add 2d circular day of year variable?
# see collapsed cell https://tom-andersson.github.io/deepsensor/user-guide/convnp.html#initialising-a-convnp

#%% Set up task loader

task_loader = TaskLoader(context=[era5_ds, aux_ds], target=station_df, aux_at_targets=hires_aux_ds)
print(task_loader)

train_start = '2000-01-01'
train_end = '2000-12-31'
val_start = '2001-01-01'
val_end = '2001-12-31'

train_dates = era5_raw_ds.sel(time=slice(train_start, train_end)).time.values
val_dates = era5_raw_ds.sel(time=slice(val_start, val_end)).time.values

train_tasks = []
for date in tqdm(train_dates[::2], desc="Loading train tasks..."):
    task = task_loader(date, context_sampling="all", target_sampling="all")
    train_tasks.append(task)

val_tasks = []
for date in tqdm(val_dates, desc="Loading val tasks..."):
    task = task_loader(date, context_sampling="all", target_sampling="all")
    val_tasks.append(task)

print("Loading Dask arrays...")
tic = time.time()
task_loader.load_dask()
print(f"Done in {time.time() - tic:.2f}s")

#%% Inspect train task

train_tasks[0]
#%%

# Set up model
model = ConvNP(data_processor, task_loader, unet_channels=(64,)*4, 
                likelihood="gnp", internal_density=100)

# Print number of parameters to check model is not too large for GPU memory
_ = model(val_tasks[0])
print(f"Model has {deepsensor.backend.nps.num_params(model.model):,} parameters")


#%% Plot context encoding

fig = deepsensor.plot.context_encoding(model, train_tasks[0], task_loader)

#%%
fig = deepsensor.plot.task(train_tasks[0], task_loader)
plt.show()

#%%

fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(projection=crs))
ax.coastlines()
ax.add_feature(cf.BORDERS)

minlon = config.PLOT_EXTENT['all']['minlon']
maxlon = config.PLOT_EXTENT['all']['maxlon']
minlat = config.PLOT_EXTENT['all']['minlat']
maxlat = config.PLOT_EXTENT['all']['maxlat']
ax.set_extent([minlon, maxlon, minlat, maxlat], crs)
# ax = nzplot.nz_map_with_coastlines()

deepsensor.plot.offgrid_context(ax, val_tasks[0], data_processor, task_loader, plot_target=True, add_legend=True, linewidths=0.5)
plt.show()
# fig.savefig("train_stations.png", bbox_inches="tight")

#%% Train

import lab as B
def compute_val_loss(model, val_tasks):
    val_losses = []
    for task in val_tasks:
        val_losses.append(B.to_numpy(model.loss_fn(task, normalise=True)))
    return np.mean(val_losses)

n_epochs = 30
train_losses = []
val_losses = []

val_loss_best = np.inf

for epoch in tqdm(range(n_epochs)):
    batch_losses = train_epoch(model, train_tasks)
    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)

    val_loss = compute_val_loss(model, val_tasks)
    val_losses.append(val_loss)

    if val_loss < val_loss_best:
        import torch
        import os
        val_loss_best = val_loss
        folder = "models/downscaling/"
        if not os.path.exists(folder): os.makedirs(folder)
        torch.save(model.model.state_dict(), folder + f"model.pt")

    # print(f"Epoch {epoch} train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}")

#%%


