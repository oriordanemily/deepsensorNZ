# Plot ERA5

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
from nzdownscale.dataprocess.utils import PlotData

#%% Load ERA5 data

var = 'temperature'
#var = 'precipitation'
era5 = ProcessERA5()

#%% Load all ERA5 data

#ds = era5.load_ds(var)  # ~2min 

#%% Load specific year only

ds = era5.load_ds(var, 2000)
da = era5.ds_to_da(ds, var)

#%% Plot timeslice

plotnz = PlotData()
ax = plotnz.nz_map_with_coastlines()
da_to_plot = da.isel(time=0)
da_to_plot.plot()
plt.title(f'ERA5-land: {var}\n{da_to_plot["time"].values}')
plt.savefig('./tmp/fig.png')
plt.show()
