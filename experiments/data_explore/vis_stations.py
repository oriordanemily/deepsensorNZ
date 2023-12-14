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

#var = 'precipitation'
var = 'temperature'
process_stations = ProcessStations()
all_stations = process_stations.get_path_all_stations(var)
print(len(all_stations))

#%% load example station

f = all_stations[200]
ds = process_stations.load_station(f)
print(ds)
da = process_stations.ds_to_da(ds, var)

#%% plot histogram of data for example station

plotdata = PlotData()
plotdata.plot_hist_values(da, n=500)

#%% get station metadata 

df = process_stations.get_metadata_df(var)

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

ax = process_stations.plot_stations_on_map(df)
plt.title(f'Stations: {var}')
#plt.savefig('./tmp/fig.png')
plt.show()

#%%

# Select stations by specific year

year = 2000

process_stations = ProcessStations()
station_paths = process_stations.get_path_all_stations(var)
df = process_stations.get_metadata_df(var)

df_filtered = df[(df['start_year']<year) & (df['end_year']>=year)]
station_paths_filtered = list(df_filtered.index)
print(len(station_paths_filtered))

#%% Extract data from these stations

i = 0

ds = process_stations.load_station(station_paths_filtered[i])
da = process_stations.ds_to_da(ds, var)
df = da.to_dataframe()
#df.index = 


#%% 