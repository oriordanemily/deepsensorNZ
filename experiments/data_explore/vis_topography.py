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

#%% 

if False:
    file_to_open = 'data/topography/nz_elevation_25m.nc'
    save_as = 'data/topography/nz_elevation_200m.nc'
    coarsen_by = 8
    boundary = 'pad'
    coord_rename = {'lat': 'latitude','lon': 'longitude'}
    plot = False

    # coarsen and save
    top = ProcessTopography()
    da_original = top.open_da(f'{file_to_open}')
    da_coarsened = top.coarsen_da(da_original, coarsen_by=coarsen_by, boundary=boundary)  # 1m20s
    da_coarsened = top.rename_xarray_coords(da_coarsened, coord_rename)
    if plot:
        da_coarsened.plot()  # 2m11s
    top.save_nc(da_coarsened, save_as)
    print(f"Saved as {save_as}")

#%% load topography

file = 'data/topography/nz_elevation_200m.nc'  # this was created by running as main data_process/topography.py
top = ProcessTopography()
da = top.open_da(file)

#%% plot

# da.plot()  # ~2 min
top.plot_with_coastlines(da)
