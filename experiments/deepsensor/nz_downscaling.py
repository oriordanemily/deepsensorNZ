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
from scipy.ndimage import gaussian_filter
import torch
import lab as B
import wandb

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
from nzdownscale.dataprocess.config import LOCATION_LATLON

#%%
wandb_log = True
if wandb_log:
    wandb.login()
#c253beb4357c18f8e7ad7cc3745e4fdb9d16c9b0

#%% Settings

# Variables
var = 'temperature'
# var = 'precipitation'
years = [2000, 2001, 2002, 2003, 2004, 2005]  
daily = True 

# plotting
crs = ccrs.PlateCarree()

# GPU 
use_gpu = True
if use_gpu:
    # set_gpu_default_device()
    torch.set_default_device("cuda")
    torch.cuda.set_device(2)
    B.set_global_device("cuda:2")

dataprocess = utils.DataProcess()

#%% load elevation 

process_top = topography.ProcessTopography()
ds_elev = process_top.open_ds()

#%% load ERA5

process_era = era5.ProcessERA5()
ds_era = process_era.load_ds(var, years, daily=daily)
da_era = process_era.ds_to_da(ds_era, var)

#%% load stations (covering specified years)

process_stations = stations.ProcessStations()
station_paths = process_stations.get_path_all_stations(var)
df = process_stations.get_metadata_df(var)

# this was only using stations that have data across all years
# have changed to use stations covering any part of the years specified
# check this with Risa
# df_filtered = df[(df['start_year']<years[-1]) & (df['end_year']>=years[0])]
df_filtered = df[(df['start_year']<years[0]) & (df['end_year']>=years[-1])]
station_paths_filtered = list(df_filtered.index)
print(len(station_paths_filtered))

#%% plot era5 snapshot with stations

nzplot = utils.PlotData()
ax = nzplot.nz_map_with_coastlines()
da_era.isel(time=0).plot()
ax = process_stations.plot_stations(df, ax)
plt.title('ERA5 with stations');
# plt.plot()

#%% Coarsen and plot (ERA5 and topography)
# note that this was just coarsened to make the model fit into memory 
# may want to change this down the line

# ERA5 = 0.1 degrees (~10km)
coarsen_factor_era = 1
if coarsen_factor_era == 1:
    da_era_coarse = da_era
else:
    da_era_coarse = process_era.coarsen_da(da_era, coarsen_factor_era)

# da_era_coarse = da_era_coarse.fillna(0)

latres_era = dataprocess.resolution(da_era_coarse, 'latitude')
lonres_era = dataprocess.resolution(da_era_coarse, 'longitude')

nzplot = utils.PlotData()
ax = nzplot.nz_map_with_coastlines()
da_era_coarse.isel(time=0).plot()
str_coarsened = 'Coarsened ' if coarsen_factor_era != 1 else ' '
plt.title(f'{str_coarsened}ERA5\n'
          f'Lat res: {latres_era:.4f} degrees, '
          f'lon res: {lonres_era:.4f} degrees')
plt.plot()

# Topography = 0.002 degrees (~200m)
coarsen_factor_topo = 5 #5
if coarsen_factor_topo == 1:
    ds_elev_coarse = ds_elev
else:
    ds_elev_coarse = process_top.coarsen_da(ds_elev, coarsen_factor_topo)

#fill all nan values with 0 to avoid training error
ds_elev_coarse = ds_elev_coarse.fillna(0)
da_elev_coarse = process_top.ds_to_da(ds_elev_coarse)

latres_topo = dataprocess.resolution(ds_elev_coarse, 'latitude')
lonres_topo = dataprocess.resolution(ds_elev_coarse, 'longitude')

nzplot = utils.PlotData()
ax = nzplot.nz_map_with_coastlines()
da_elev_coarse.plot()
str_coarsened = 'Coarsened ' if coarsen_factor_topo != 1 else ''
plt.title(f'{str_coarsened}topography\n'
          f'Lat res: {latres_topo:.4f} degrees, '
          f'lon res: {lonres_topo:.4f} degrees')
plt.show()

#%% Compute Topographic Position Index (TPI) from elevation data
# TPI helps us distinguish topo features, e.g. hilltop, valley, ridge...
PLOT_TPI = False
highres_aux_raw_ds = ds_elev_coarse

# Calculate the lat and lon resolutions in the elevation dataset
# Here we assume the elevation is on a regular grid,
# so the first difference is equal to all others.
# This may not be a fair assumption... 
coord_names = list(highres_aux_raw_ds.dims)
resolutions = np.array(
    [np.abs(np.diff(highres_aux_raw_ds.coords[coord].values)[0]) 
     for coord in coord_names])

for window_size in [.1, .05, .025]:
    smoothed_elev_da = highres_aux_raw_ds['elevation'].copy(deep=True)

    # Compute gaussian filter scale in terms of grid cells
    scales = window_size / resolutions

    smoothed_elev_da.data = gaussian_filter(smoothed_elev_da.data, 
                                            sigma=scales, 
                                            mode='constant', 
                                            cval=0)

    TPI_da = highres_aux_raw_ds['elevation'] - smoothed_elev_da
    highres_aux_raw_ds[f"TPI_{window_size}"] = TPI_da

    if PLOT_TPI:

        minlon = config.PLOT_EXTENT['all']['minlon']
        maxlon = config.PLOT_EXTENT['all']['maxlon']
        minlat = config.PLOT_EXTENT['all']['minlat']
        maxlat = config.PLOT_EXTENT['all']['maxlat']

        # fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs))
        # ax.set_extent([minlon, maxlon, minlat, maxlat], crs)
        ax = nzplot.nz_map_with_coastlines()
        TPI_da.plot(ax=ax)
        ax.add_feature(cf.BORDERS)
        ax.coastlines()
        ax.set_title(f'TPI with window size {window_size}, scales = {scales}')
        plt.show()

#%% Low resolution topography 
# Q for Risa: why do we input this too - model learns difference in res?

coarsen_factor = int(latres_era/latres_topo) #changed this to match era5 resolution
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
print(f"Lowres topo lat resolution: "
      f"{dataprocess.resolution(aux_raw_ds, 'latitude'):.4f} degrees")
print(f"Lowres topo lon resolution: "
      f"{dataprocess.resolution(aux_raw_ds, 'longitude'):.4f} degrees")
print(f"Highres topo lat resolution: "
      f"{dataprocess.resolution(highres_aux_raw_ds, 'latitude'):.4f} degrees")
print(f"Highres topo lon resolution: "
      f"{dataprocess.resolution(highres_aux_raw_ds, 'longitude'):.4f} degrees")
#%% 

# Slice era5 data to elevation data's spatial extent
top_min_lat = highres_aux_raw_ds['latitude'].min()
top_max_lat = highres_aux_raw_ds['latitude'].max()
top_min_lon = highres_aux_raw_ds['longitude'].min()
top_max_lon = highres_aux_raw_ds['longitude'].max()

era5_raw_ds = da_era_coarse.sel(latitude=slice(top_max_lat, top_min_lat), 
                         longitude=slice(top_min_lon, top_max_lon))

nzplot = utils.PlotData()
ax = nzplot.nz_map_with_coastlines()
era5_raw_ds.isel(time=0).plot()
ax.set_title('ERA5 with topography extent');

#%% 

station_paths = station_paths_filtered
df_list = []
for path in tqdm(station_paths):
    df = process_stations.load_station_df(path, var, daily=daily)
    df_list.append(df)
print('Concatenating station data...')
df = pd.concat(df_list)
df.index 
station_raw_df = df.reset_index().set_index(['time', 
                                             'latitude', 
                                             'longitude']).sort_index()

### filter years
station_raw_df_ = station_raw_df.reset_index()
station_raw_df_ = station_raw_df_[(station_raw_df_['time']>=str(years[0])) &
                        (station_raw_df_['time']<=f'{str(years[-1])}-12-31')]
station_raw_df = station_raw_df_.set_index(['time', 
                                            'latitude', 
                                            'longitude']).sort_index()
###

station_raw_df

#%% Normalise and preprocess data

#changed the maps here to use the high res topo data as that seems to have the
#largest extent
data_processor = DataProcessor(x1_name="latitude", 
                               x1_map=(era5_raw_ds["latitude"].min(), era5_raw_ds["latitude"].max()),
                            #    x1_map=(highres_aux_raw_ds["latitude"].min(), 
                            #            highres_aux_raw_ds["latitude"].max()),
                               x2_name="longitude", 
                               x2_map = (era5_raw_ds["longitude"].min(), era5_raw_ds["longitude"].max()))
                            #    x2_map=(highres_aux_raw_ds["longitude"].min(), 
                                    #    highres_aux_raw_ds["longitude"].max()))

# compute normalisation parameters
era5_ds, station_df = data_processor([era5_raw_ds, station_raw_df]) #meanstd
aux_ds, hires_aux_ds = data_processor([aux_raw_ds, highres_aux_raw_ds], method="min_max") #minmax
print(data_processor)

TEST_NORM = False
if TEST_NORM: 
    for ds, raw_ds, ds_name in zip([era5_ds, aux_ds, hires_aux_ds], 
                [era5_raw_ds, aux_raw_ds, highres_aux_raw_ds], 
                ['ERA5', 'Topography', 'Topography (high res)']):
        ds_unnormalised = data_processor.unnormalise(ds)
        xr.testing.assert_allclose(raw_ds, ds_unnormalised, atol=1e-3)
        print(f"Unnormalised {ds_name} matches raw data")

    station_df_unnormalised = data_processor.unnormalise(station_df)
    pd.testing.assert_frame_equal(station_raw_df, station_df_unnormalised)
    print(f"Unnormalised station_df matches raw data")

#%% 
# Generate auxiliary dataset of x1/x2 coordinates to break translation 
# equivariance in the model's CNN to enable learning non-stationarity
x1x2_ds = construct_x1x2_ds(aux_ds)
aux_ds['x1_arr'] = x1x2_ds['x1_arr']
aux_ds['x2_arr'] = x1x2_ds['x2_arr']
aux_ds

# should we add 2d circular day of year variable? see collapsed cell 
# https://tom-andersson.github.io/deepsensor/user-guide/convnp.html#initialising-a-convnp

#%% Set up task loader

task_loader = TaskLoader(context=[era5_ds, aux_ds],
                        target=station_df, 
                        aux_at_targets=hires_aux_ds)
print(task_loader)
#%% 
train_start = f'{years[0]}-01-01'
train_end = f'{years[-2]}-12-31'
val_start = f'{years[-1]}-01-01'
val_end = f'{years[-1]}-12-31'

print('Training period:', train_start, 'to', train_end)
print('Validation period:', val_start, 'to', val_end)

train_dates = era5_raw_ds.sel(time=slice(train_start, train_end)).time.values
val_dates = era5_raw_ds.sel(time=slice(val_start, val_end)).time.values

train_tasks = []
# only loaded every other date to speed up training for now
for date in tqdm(train_dates[::2], desc="Loading train tasks..."):
    task = task_loader(date, context_sampling="all", target_sampling="all").remove_context_nans().remove_target_nans()
    train_tasks.append(task)

val_tasks = []
for date in tqdm(val_dates[::2], desc="Loading val tasks..."):
    task = task_loader(date, context_sampling="all", target_sampling="all").remove_context_nans().remove_target_nans()
    val_tasks.append(task)

print("Loading Dask arrays...")
tic = time.time()
task_loader.load_dask()
print(f"Done in {time.time() - tic:.2f}s")

# #%% Inspect train task
# for i in range(100):
#     print(train_tasks[i]['X_t'][0].shape)

# #%% Inspect val task
# for i in range(100):
#     print(val_tasks[i]['X_t'][0].shape)

#%%

# Set up model
model = ConvNP(data_processor,
               task_loader, 
               unet_channels=(64,)*4, 
               likelihood="gnp",
               internal_density=500) 
#internal density edited to make model fit into memory -
# may want to adjust down the line


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

deepsensor.plot.offgrid_context(ax, train_tasks[0], data_processor, task_loader, plot_target=True, add_legend=True, linewidths=0.5)
plt.show()
# fig.savefig("train_stations.png", bbox_inches="tight")

#%% Train

import lab as B
def compute_val_loss(model, val_tasks):
    val_losses = []
    for task in val_tasks:
        val_losses.append(B.to_numpy(model.loss_fn(task, normalise=True)))
        val_losses_not_nan = [arr for arr in val_losses if~ np.isnan(arr)]
    print('Number of val_losses: ', len(val_losses), 'number of val_losses_not_nan: ', len(val_losses_not_nan))
    return np.mean(val_losses_not_nan)

def split_by_number_of_targets(tasks):
    numtargets_tasks_dict = {}
    for t_task in tasks:
        num_target = t_task['X_t'][0].shape[1]
        if num_target not in numtargets_tasks_dict.keys():
            numtargets_tasks_dict[num_target] = []
        numtargets_tasks_dict[num_target].append(t_task)
    return numtargets_tasks_dict
#%%
n_epochs = 10
train_losses = []
val_losses = []

if wandb_log:
    run = wandb.init(project='nz-deepsensor-temp',
                    config={
                        "n_epochs": n_epochs,
                    })
val_loss_best = np.inf

model_filename = "model_nosea_2000_2005_daily_test.pt"

split_by_targets = False
# when trying to run with batch_size arg in train_epoch, it said we had to 
# have equal target size when using batch_size. so have split tasks by number
# of targets and train through each of those as a batch separately
if split_by_targets:
    train_tasks_dict = split_by_number_of_targets(train_tasks)
    val_tasks_dict = split_by_number_of_targets(val_tasks)

for epoch in tqdm(range(n_epochs)):
    if split_by_targets:
        batch_losses = []
        for _, tasks in train_tasks_dict.items():
            print(len(tasks))
            batch_losses.extend(train_epoch(model, tasks))
    else:
        batch_losses = train_epoch(model, train_tasks)
    batch_losses_not_nan = [arr for arr in batch_losses if~ np.isnan(arr)]
    print(f'Number of batch_losses: {len(batch_losses)}, number of batch_losses_not_nan: {len(batch_losses_not_nan)}')
    train_loss = np.mean(batch_losses_not_nan)
    train_losses.append(train_loss)

    val_loss = compute_val_loss(model, val_tasks)
    val_losses.append(val_loss)

    if wandb_log:
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})
    if val_loss < val_loss_best:
        import os
        val_loss_best = val_loss
        folder = "models/downscaling/"
        if not os.path.exists(folder): os.makedirs(folder)
        torch.save(model.model.state_dict(), folder + model_filename)
    print(f"Epoch {epoch} train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}")

with open(folder + f"train_losses_{model_filename}.txt", "w") as f:
    f.write(str(train_losses))
with open(folder + f"val_losses_{model_filename}.txt", "w") as f:
    f.write(str(val_losses))


#%% Use this for a trained model
folder = "models/downscaling/"
model.model.load_state_dict(torch.load(folder + model_filename))
    
#%%
# Plot training and validation losses
fig, ax = plt.subplots(1, 1)

with open(folder + f"train_losses_{model_filename}.txt", "r") as f:
    train_losses = eval(f.read())
with open(folder + f"val_losses_{model_filename}.txt", "r") as f:
    val_losses = eval(f.read())

ax.plot(train_losses, label="train")
ax.plot(val_losses, label="val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
plt.show()
#%% Look at some of the validation data

date = "2005-02-25"
test_task = task_loader(date, ["all", "all"], seed_override=42)
pred = model.predict(test_task, X_t=era5_raw_ds, resolution_factor=20)


fig = deepsensor.plot.prediction(pred, date, data_processor, task_loader, 
                                 test_task, crs=ccrs.PlateCarree())

#%%



def gen_test_fig(era5_raw_ds=None, mean_ds=None, std_ds=None, samples_ds=None, add_colorbar=False, var_clim=None, std_clim=None, var_cbar_label=None, std_cbar_label=None, fontsize=None, figsize=(15, 5)):
    if var_clim is None:
        vmin = np.array(era5_raw_ds.min())
        vmax = np.array(era5_raw_ds.max())
    else:
        vmin, vmax = var_clim

    if std_clim is None and std_ds is not None:
        std_vmin = np.array(std_ds.min())
        std_vmax = np.array(std_ds.max())
    elif std_clim is not None:
        std_vmin, std_vmax = std_clim
    else:
        std_vmin = None
        std_vmax = None

    ncols = 0
    if era5_raw_ds is not None:
        ncols += 1
    if mean_ds is not None:
        ncols += 1
    if std_ds is not None:
        ncols += 1
    if samples_ds is not None:
        ncols += samples_ds.shape[0]

    fig, axes = plt.subplots(1, ncols, subplot_kw=dict(projection=crs), figsize=figsize)

    axis_i = 0
    if era5_raw_ds is not None:
        ax = axes[axis_i]
        # era5_raw_ds.sel(lat=slice(mean_ds["lat"].min(), mean_ds["lat"].max()), lon=slice(mean_ds["lon"].min(), mean_ds["lon"].max())).plot(ax=ax, cmap="jet", vmin=vmin, vmax=vmax, add_colorbar=False)
        era5_raw_ds.plot(ax=ax, cmap="jet", vmin=vmin, vmax=vmax, add_colorbar=add_colorbar, cbar_kwargs=dict(label=var_cbar_label))
        ax.set_title("ERA5", fontsize=fontsize)

    if mean_ds is not None:
        axis_i += 1
        ax = axes[axis_i]
        mean_ds.plot(ax=ax, cmap="jet", vmin=vmin, vmax=vmax, add_colorbar=add_colorbar, cbar_kwargs=dict(label=var_cbar_label))
        ax.set_title("ConvNP mean", fontsize=fontsize)

    if samples_ds is not None:
        for i in range(samples_ds.shape[0]):
            axis_i += 1
            ax = axes[axis_i]
            samples_ds.isel(sample=i).plot(ax=ax, cmap="jet", vmin=vmin, vmax=vmax, add_colorbar=add_colorbar, cbar_kwargs=dict(label=var_cbar_label))
            ax.set_title(f"ConvNP sample {i+1}", fontsize=fontsize)

    if std_ds is not None:
        axis_i += 1
        ax = axes[axis_i]
        std_ds.plot(ax=ax, cmap="Greys", add_colorbar=add_colorbar, vmin=std_vmin, vmax=std_vmax, cbar_kwargs=dict(label=std_cbar_label))
        ax.set_title("ConvNP std dev", fontsize=fontsize)

    for ax in axes:
        ax.add_feature(cf.BORDERS)
        ax.coastlines()
    return fig, axes

pred_db = pred['dry_bulb']

fig, axes = gen_test_fig(
    era5_raw_ds.sel({'time':date}),
    pred_db["mean"].sel({'time':date}),
    pred_db["std"].sel({'time':date}),
    add_colorbar=True,
    var_cbar_label="2m temperature [°C]",
    std_cbar_label="std dev [°C]",
    std_clim=(None, 2),
    figsize=(20, 20/3)
)



# %%
from datetime import timedelta

location = "napier"
if location not in LOCATION_LATLON:
    raise ValueError(f"Location {location} not in LOCATION_LATLON, please set X_t manually")
X_t = LOCATION_LATLON[location]
dates = pd.date_range("2005-07-01 00:00:00", "2005-08-30 00:00:00", freq=timedelta(hours=1))
station_raw_df
# %%
locs = set(zip(station_raw_df.reset_index()["latitude"], station_raw_df.reset_index()["longitude"]))
locs
# %%
# Find closest station to desired target location
X_station_closest = min(locs, key=lambda loc: np.linalg.norm(np.array(loc) - X_t))
X_t = np.array(X_station_closest)#.reshape(2, 1)
X_t
# %%
# As above but zooming in
lat_slice = slice(X_t[0] + 2, X_t[0] - 2)
lon_slice = slice(X_t[1] - 2, min(X_t[1] + 2, 180))
fig, axes = gen_test_fig(
    era5_raw_ds.isel(time=0).sel(latitude=lat_slice, longitude=lon_slice),
    pred_db["mean"].sel(latitude=lat_slice, longitude=lon_slice),
    pred_db["std"].sel(latitude=lat_slice, longitude=lon_slice),
    add_colorbar=True,
    # var_clim=(10, -5),
    var_cbar_label="2m temperature [°C]",
    std_cbar_label="std dev [°C]",
    std_clim=(None, 2),
)
# Plot X_t
for ax in axes:
    ax.scatter(X_t[1], X_t[0], marker="s", color="black", transform=crs, s=10**2, facecolors='none', linewidth=2)


# %%
# Get station target data
station_closest_df = station_raw_df.reset_index().set_index(["latitude", "longitude"]).loc[X_station_closest].set_index("time")
new_dates = station_closest_df.index.intersection(dates).tolist()
station_closest_df = station_closest_df.loc[station_closest_df.index.isin(new_dates)]
station_closest_df
# %%
# Plot location of X_t on map using cartopy
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs), figsize=(20, 20))
pred_db['mean'].plot(ax=ax, cmap="jet")
ax.coastlines()
ax.add_feature(cf.BORDERS)
ax.scatter(X_t[1], X_t[0], transform=crs, color="black", marker="*", s=200)
# Plot station locations
ax.scatter([loc[1] for loc in locs], [loc[0] for loc in locs], transform=crs, color="red", marker=".")
# ax.set_extent([6, 15, 47.5, 55])

# %%
era5_raw_df = era5_raw_ds.sel(latitude=X_t[0], longitude=X_t[1], method="nearest").to_dataframe()
era5_raw_df = era5_raw_df.loc[new_dates]
era5_raw_df
#%%
test_tasks = task_loader(new_dates, "all")
preds = model.predict(test_tasks, X_t=era5_raw_ds, resolution_factor=2)
preds_db = preds['dry_bulb']


#%%
start = 0
end = 50
# Plot
sns.set_style("white")
fig, ax = plt.subplots(1, 1, figsize=(7*.9, 3*.9))
convnp_mean = preds_db["mean"].sel(latitude=X_t[0], longitude=X_t[1], method='nearest').values.astype('float')[start:end]
ax.plot(convnp_mean, label="ConvNP", marker="o", markersize=3)
stddev = preds_db["std"].sel(latitude=X_t[0], longitude=X_t[1], method='nearest').values.astype('float')[start:end]
# Make 95% confidence interval
ax.fill_between(range(len(convnp_mean)), convnp_mean - 2 * stddev, convnp_mean + 2 * stddev, alpha=0.25, label="ConvNP 95% CI")
era5_vals = era5_raw_df["t2m"].values.astype('float')[start:end]
ax.plot(era5_vals, label="ERA5", marker="o", markersize=3)
# Plot true station data
ax.plot(station_closest_df["dry_bulb"].values.astype('float')[start:end], label="Station", marker="o", markersize=3)
# Add legend
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=4, mode="expand", borderaxespad=0)
ax.set_xlabel("Time")
ax.set_ylabel("2m temperature [°C]")
ax.set_xticks(range(len(era5_raw_df))[start:end:14])
ax.set_xticklabels(era5_raw_df.index[start:end:14].strftime("%Y-%m-%d"), rotation=15)
ax.set_title(f"ConvNP prediction for {location}", y=1.15)
# %%
