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
# da_era_coarse = da_era_coarse.fillna(0)
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
#fill all nan values with 0 to avoid training error
ds_elev_coarse = ds_elev_coarse.fillna(0)
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

    smoothed_elev_da.data = scipy.ndimage.gaussian_filter(smoothed_elev_da.data, sigma=scales, mode='constant', cval=0)

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
model = ConvNP(data_processor, task_loader, unet_channels=(64,)*4, likelihood="gnp", internal_density=50) #internal density edited to make model fit into memory - may want to adjust down the line


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
        val_losses_not_nan = [arr for arr in val_losses if~ np.isnan(arr)]
    return np.mean(val_losses_not_nan)

n_epochs = 30
train_losses = []
val_losses = []

val_loss_best = np.inf

for epoch in tqdm(range(n_epochs)):
    batch_losses = train_epoch(model, train_tasks)
    batch_losses_not_nan = [arr for arr in batch_losses if~ np.isnan(arr)]
    train_loss = np.mean(batch_losses_not_nan)
    train_losses.append(train_loss)

    val_loss = compute_val_loss(model, val_tasks)
    val_losses.append(val_loss)

    if val_loss < val_loss_best:
        import torch
        import os
        val_loss_best = val_loss
        folder = "models/downscaling/"
        if not os.path.exists(folder): os.makedirs(folder)
        torch.save(model.model.state_dict(), folder + f"model_nosea.pt")

    print(f"Epoch {epoch} train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}")


#%% Use this for a trained model
    
model.model.load_state_dict(torch.load(folder + f"model_nosea.pt"))
    

#%% Look at some of the validation data

date = "2001-06-25"
test_task = task_loader(date, ["all", "all"], seed_override=42)
pred = model.predict(test_task, X_t=era5_raw_ds, resolution_factor=2)


fig = deepsensor.plot.prediction(pred, date, data_processor, task_loader, test_task, crs=ccrs.PlateCarree())

#%%


def gen_test_fig(era5_raw_ds=None, mean_ds=None, std_ds=None, samples_ds=None, add_colorbar=False, var_clim=None, std_clim=None, var_cbar_label=None, std_cbar_label=None, fontsize=None, figsize=(15, 5)):
    if var_clim is None:
        vmin = np.array(mean_ds.min())
        vmax = np.array(mean_ds.max())
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
    era5_raw_ds.isel(time=0), 
    pred_db["mean"],
    pred_db["std"],
    add_colorbar=True,
    var_cbar_label="2m temperature [°C]",
    std_cbar_label="std dev [°C]",
    std_clim=(None, 2),
    figsize=(20, 20/3)
)



# %%

X_t = np.array([[-41.2924, 174.7787]]).T #wellington
dates = pd.date_range("2001-09-01", "2001-10-31")
station_raw_df
# %%
locs = set(zip(station_raw_df.reset_index()["latitude"], station_raw_df.reset_index()["longitude"]))
locs
# %%
# Find closest station to desired target location
X_station_closest = min(locs, key=lambda loc: np.linalg.norm(np.array(loc) - X_t.T))
X_t = np.array(X_station_closest).reshape(2, 1)
X_t
# %%
# As above but zooming in
fig, axes = gen_test_fig(
    era5_raw_ds.isel(time=0).sel(latitude=slice(-39, -43), longitude=slice(173, 176)),
    pred_db["mean"].sel(latitude=slice(-39, -43), longitude=slice(173, 176)),
    pred_db["std"].sel(latitude=slice(-39, -43), longitude=slice(173, 176)),
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
station_closest_df = station_raw_df.reset_index().set_index(["latitude", "longitude"]).loc[X_station_closest].set_index("time").loc[dates]
station_closest_df
# %%
# Plot location of X_t on map using cartopy
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs), figsize=(20, 20))
pred_db['mean'].plot(ax=ax, cmap="jet")
ax.coastlines()
ax.add_feature(cf.BORDERS)
ax.scatter(X_t[1], X_t[0], transform=crs, color="red", marker="x")
# Plot station locations
ax.scatter([loc[1] for loc in locs], [loc[0] for loc in locs], transform=crs, color="black", marker="x")
# ax.set_extent([6, 15, 47.5, 55])
# %%

era5_raw_df = era5_raw_ds.sel(latitude=-41.2924, longitude=174.7787, method="nearest").to_dataframe()
era5_raw_df = era5_raw_df.loc[dates]
era5_raw_df
#%%
test_tasks = task_loader(dates, "all")
preds = model.predict(test_tasks, X_t=era5_raw_ds, resolution_factor=2)
preds_db = preds['dry_bulb']


#%%


# Plot
sns.set_style("white")
fig, ax = plt.subplots(1, 1, figsize=(7*.9, 3*.9))
convnp_mean = preds_db["mean"].values.astype('float')[:, 0, 0]
ax.plot(convnp_mean, label="ConvNP", marker="o", markersize=3)
stddev = preds_db["std"].values.astype('float')
# Make 95% confidence interval
# ax.fill_between(range(len(convnp_mean)), convnp_mean - 2 * stddev, convnp_mean + 2 * stddev, alpha=0.25)#, label="ConvNP 95% CI")
era5_vals = era5_raw_df["t2m"].values.astype('float')
ax.plot(era5_vals, label="ERA5", marker="o", markersize=3)
# Plot true station data
ax.plot(station_closest_df["dry_bulb"].values.astype('float'), label="Station", marker="o", markersize=3)
# Add legend
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=3, mode="expand", borderaxespad=0)
ax.set_xlabel("Time")
ax.set_ylabel("2m temperature [°C]")
ax.set_xticks(range(len(era5_raw_df))[::14])
ax.set_xticklabels(era5_raw_df.index[::14].strftime("%Y-%m-%d"), rotation=15)
# %%
