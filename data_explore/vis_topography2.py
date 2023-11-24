"""
Plot topography - use new elevation file nz_elevation_25m.nc
"""

#%%

import os

import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

import rioxarray
import rasterio
import xarray
from rasterio.warp import transform
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling

from data_process.topography import ProcessTopography
from data_process.stations import ProcessStations

#%% process 25m elevation data to make it 100m 

def coarsen_and_save():
    # 'elevation'y: 56001x: 56001
    path = 'data/topography_25km'
    filename = 'nz_elevation_25m.nc'
    top = ProcessTopography(path=path)
    da = top.open_da(f'{path}/{filename}')  
    da_coarsened = top.coarsen_da(da, 4, boundary='pad')  # 1m20s
    da_coarsened = da_coarsened.rename({'x': 'longitude','y': 'latitude'})
    da_coarsened.plot()  # 2m11s
    da_coarsened.to_netcdf('data/topography_100km/nz_elevation_100m.nc')


def plot_hist_values(da):
    # Flatten the DataArray to a 1D array
    flat_data = da.values.flatten()

    # Plot the histogram
    plt.hist(flat_data, bins=30, color='blue', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    #da_subset = da.sel(x=slice(170, 173), y=slice(-43, -40))

#%% load 100m topography

file = 'data/topography_100km/nz_elevation_100m.nc'
da = xr.open_dataarray(file).squeeze()

#%% plot with coastline

minlon = np.array(da['longitude'].min())
maxlon = np.array(da['longitude'].max())
minlat = np.array(da['latitude'].min())
maxlat = np.array(da['latitude'].max())

# fig = plt.figure(figsize=(10, 12))
proj = ccrs.PlateCarree() #tried doesn't work 'EPSG:27200' # src.crs # 
# #fig = plt.figure(figsize=(10, 12))
# ax = fig.add_subplot(1, 1, 1, projection=proj)

fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=proj), figsize=(10, 12))
proj = ccrs.PlateCarree()
#ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.coastlines()
ax.set_xlim(minlon, maxlon)
ax.set_ylim(minlat, maxlat)

#ax.gridlines(draw_labels=True, crs=proj)
da.plot()
#top.plot_coastlines(fig)
plt.show()

#%% 

