"""
Plot ERA5
"""

#%%

import os

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import seaborn as sns

from nzdownscale.dataprocess.era5 import ProcessERA5

crs = ccrs.PlateCarree()


if False:

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

    #%% 

    era5 = ProcessERA5()
    ds = era5.get_ds('temperature')

    #%% 
    var = 'temperature'
    # var = 'precipitation'

    era5 = ProcessERA5()
    ds = era5.load_ds_specific_year(var, 2000)
    da = era5.ds_to_da(ds, var)
    da = da.rename({'longitude': 'lon','latitude': 'lat'})
    #era5_raw_ds = da.coarsen(latitude=5, longitude=5, boundary="trim").mean()
    #era5_raw_ds = era5_raw_ds.load()

    #%% 

    minlon = np.array(da['lon'].min())
    maxlon = np.array(da['lon'].max())
    minlat = np.array(da['lat'].min())
    maxlat = np.array(da['lat'].max())

    #print(da)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs), figsize=(10, 12))
    #fig = plt.figure(figsize=(10, 12))
    proj = ccrs.PlateCarree()
    #ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.coastlines()
    ax.set_xlim(minlon, maxlon)
    ax.set_ylim(minlat, maxlat)
    da.isel(time=0).plot()
    #ax.add_feature(cf.BORDERS)


    #%% 

    da = xr.open_dataarray('data/ftp.bodekerscientific.com/Greg/ForRisa2/2m_temperature/2000/ERA5_Land_2m_temperature_01.nc')

    da = xr.open_mfdataset('data/ftp.bodekerscientific.com/Greg/ForRisa2/2m_temperature/2000/*.nc')['t2m']


