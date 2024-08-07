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
from scipy.stats import gamma

from nzdownscale.dataprocess.stations import ProcessStations
from nzdownscale.dataprocess.utils import PlotData
from nzdownscale.dataprocess.config import VAR_STATIONS

#%% get station file paths

# var = 'precipitation'
# var = 'temperature'
# var = 'pressure'
# var = 'windspeed'
# var = 'winddirection'
var = 'humidity'
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

station_paths = station_paths_filtered
df_list = []
path = station_paths[0]
for path in tqdm(station_paths):
    df = process_stations.load_station_df(path, var, daily=True)
    df_list.append(df)
df = pd.concat(df_list)
station_raw_df = df.reset_index().set_index(['time', 'latitude', 'longitude']).sort_index()
print(station_raw_df)

#%% Look at distribution of station values

df_var = VAR_STATIONS[var]['var_name']

station_raw_df_values = station_raw_df[df_var]
station_raw_df_values = station_raw_df_values.dropna()
# Take a random sample of 5000 values
station_raw_df_values = station_raw_df_values.sample(5000)

# Plot histogram of the data points
fig, ax = plt.subplots()
ax.hist(station_raw_df_values, density=True, bins=100)

# Plot Gaussian distribution
mean, std = station_raw_df_values.mean(), station_raw_df_values.std()
x = np.linspace(mean - 3*std, mean + 3*std, 100)
y = (1/(std * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mean)/std)**2)
ax.plot(x, y, color='r', label='Gaussian Distribution')

# Plot Gamma distribution
def gamma_params(data):
        mean = np.mean(data)
        var = np.var(data)
        shape = mean**2 / var
        scale = var / mean
        return shape, scale

shape, scale = gamma_params(station_raw_df_values)
x_gamma = np.linspace(0, station_raw_df_values.max(), 100)
y_gamma = gamma.pdf(x_gamma, a=shape, scale=scale)
ax.plot(x_gamma, y_gamma, 'g-', label=f'Gamma PDF shape={shape:.2f}, scale={scale:.2f}')


ax.legend()
ax.set_title(f'{var.title()} Station Data Distribution')
# ax.set_xlim([0, 10])
plt.show()



#%% 
# Generate auxiliary dataset of x1/x2 coordinates to break translation equivariance in the model's CNN
# to enable learning non-stationarity
x1x2_ds = construct_x1x2_ds(aux_ds)
aux_ds['x1_arr'] = x1x2_ds['x1_arr']
aux_ds['x2_arr'] = x1x2_ds['x2_arr']
aux_ds
# %%
