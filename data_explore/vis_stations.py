"""
Plot stations

MaxMinTemp, mean_temp
min_years = 1984-2019
max_years = 1986-2019
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

#%% 

folder = 'MaxMinTemp'
var = 'mean_temp'

folder = 'Precipitation'
var = 'precipitation'

path = f'data/nz/{folder}'
# data = xr.open_mfdataset(f'{path}/*.nc')
all_stations = os.listdir(path)

#%% example

i = 200
f = all_stations[i]
ds = xr.open_dataset(f'{path}/{f}')

#%% example plots

n = 400

attribute_units = {
    'precipitation': 'mm per period',
    'frequency': 'reporting frequency, H=hourly, S=synoptic, D=daily',
    'period': 'hour',
    }

attr = 'mean_period'
da = ds[attr]
print(da)
sns.histplot(da[:n].values)
plt.title(attr)
plt.show()


#%% min max years and coords

dict_md = {}
for f in tqdm(all_stations):
    try:
        ds = xr.open_dataset(f'{path}/{f}')
        #lon = ds.longitude.values
        #lat = ds.latitude.values
        da = ds[var]
        years = np.unique([i.year for i in pd.DatetimeIndex(da.time.values)])
        start = years[0]
        end = years[-1]
        duration = int(end)-int(start)
        dict_md[f] = {
            'start': start, 
            'end':end, 
            'duration':duration,
            'lon': ds.longitude.values, 
            'lat':ds.latitude.values,
            }
    except:
        pass

df_md = pd.DataFrame(dict_md).T

#%% plot metadata 

sns.histplot(df_md['start'])
plt.xlabel('Start date')
plt.show()

sns.histplot(df_md['end'])
plt.xlabel('End date')
plt.show()

sns.histplot(df_md['duration'])
plt.xlabel('Duration')
plt.show()

#%% get station coords only

dict_coord = {}
for f in tqdm(all_stations):
    ds = xr.open_dataset(f'{path}/{f}')
    lon = ds.longitude.values
    lat = ds.latitude.values
    dict_coord[f] = {'lon': lon, 'lat':lat}

#%% plot stations - from lon lat dictionary

dict = dict_md

minlon = 165
maxlon = 179
minlat = -48
maxlat = -34
marker_size = 60

# proj = ccrs.PlateCarree(central_longitude=cm)
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.coastlines()
ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())
ax.gridlines(draw_labels=True, crs=proj)
for k, v in dict.items():
    ax.scatter(v['lon'], v['lat'], color='red', marker='o', s=marker_size)
plt.show()
