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
from nzdownscale.dataprocess.utils import PlotData

#%% 

var = 'temperature'
era5 = ProcessERA5()
#ds = era5.load_ds(var)  # ~2min 
ds = era5.load_ds(var, 2000)
da = era5.ds_to_da(ds, var)

#%% 

plotnz = PlotData()
plotnz.plot_with_coastlines(da.isel(time=0))

