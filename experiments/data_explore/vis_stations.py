# Plot stations

#%%
import os

import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

from nzdownscale.dataprocess.stations import ProcessStations
from nzdownscale.dataprocess.utils import PlotData

#%% get station file paths

var = 'precipitation'
# var = 'temperature'
stations = ProcessStations()
all_stations = stations.get_path_all_stations(var)
print(len(all_stations))

#%% load example station

f = all_stations[200]
ds = stations.load_station(f)
print(ds)
da = stations.ds_to_da(ds, var)

#%% plot histogram of data for example station

plotdata = PlotData()
plotdata.plot_hist_values(da, n=500)

#%% get station metadata 

df = stations.get_metadata_df(var)

#%% plot metadata 

sns.histplot(df['start_year'])
plt.xlabel('Start date')
plt.show()

sns.histplot(df['end_year'])
plt.xlabel('End date')
plt.show()

sns.histplot(df['duration_years'])
plt.xlabel('Duration')
plt.show()

#%% plot stations on map 

ax = stations.plot_stations_on_map(df)
plt.title(f'Stations: {var}')
#plt.savefig('./tmp/fig.png')
plt.show()

#%% 