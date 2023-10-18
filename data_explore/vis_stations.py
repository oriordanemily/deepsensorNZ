"""
Plot stations

MaxMinTemp, mean_temp
min_years = 1984-2019
max_years = 1986-2019

# folder = 'MaxMinTemp'
# var = 'mean_temp'

# folder = 'Precipitation'
# var = 'precipitation'

# folder = 'ScreenObs'
# var = 'dry_bulb'

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

from visualisation.stations import ProcessStations

#%%

var = 'temperature'
ps = ProcessStations()
all_stations = ps.get_list_all_stations(var)
print(len(all_stations))

#%% example

i = 200
f = all_stations[i]
ds = xr.open_dataset(f'{ps.var_path(var)}/{f}')
da = ps.get_da_from_ds(ds, var)

print(da)
sns.histplot(da[:400].values)
#plt.title(attr)
plt.show()


#%% 

dict_md = ps.get_info_dict(var)  # min max years and coords

#%% plot metadata 
df_md = ps.dict_md_to_df(dict_md)

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

dict_coord = ps.get_coord_dict(var)

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
