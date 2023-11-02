"""
Plot topography
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

path = '/data/hpcdata/users/risno/code/deepweather-downscaling/data/LandcareResearchDEM/nzdem-north-island-25-metre'
top = ProcessTopography(path=path)

#%% 
### plot example
all_tif_files = top.get_all_tifs()
da = top.open_da(f'{path}/{all_tif_files[0]}')
da_masked = top.mask_da(da)
da_coarsened = top.coarsen_da(da_masked, 4, boundary='pad')
da_coarsened.plot()
###

#%% combine

da_combined = top.get_combined_da()
da_masked = top.mask_da(da_combined)
da = top.coarsen_da(da_masked, 4, boundary='pad')
#top.save_da_as_tif(da, 'data/topography_100km/north.tif')
#da.rio.to_raster(tif_original)

#%% change crs

tif_original = 'data/topography_100km/north.tif'

ds = rasterio.open(tif_original)
print(ds.crs) # CRS.from_epsg(27200)
print(ds.bounds)  # BoundingBox(left=2590000.0, bottom=6400000.0, right=2810000.0, top=6550000.0)
old_crs = ds.crs
new_crs = CRS.from_epsg(4326)
dst_crs = 'EPSG:4326'  # EPSG:4326, also known as the WGS84 projection
# "EPSG:4087 replaced EPSG:32663 which replaced EPSG:32662"

transform, width, height = calculate_default_transform(ds.crs, new_crs, ds.width, ds.height, *ds.bounds)

kwargs = ds.meta.copy()
kwargs.update({
    'crs': dst_crs,
    'transform': transform,
    'width': width,
    'height': height
})

save_tif = 'data/topography_100km/north_test2.tif'
with rasterio.open(save_tif, 'w', **kwargs) as dst:
    test = reproject(
        source=rasterio.band(ds, 1),
        destination=rasterio.band(dst, 1),
        src_transform=ds.transform,
        src_crs=ds.crs,
        dst_transform=transform,
        dst_crs=new_crs,
        resampling=Resampling.nearest,
        )

#%% test plot crs converted topography

f = 'data/topography_100km/north_test2.tif'
da = rioxarray.open_rasterio(f)
da = top.mask_da(da)
da.plot()
#top.save_da_as_tif(da, 'data/topography_100km/north_test.tif')

#%% plot coastline with topography

fig = plt.figure(figsize=(10, 12))
proj = ccrs.PlateCarree() #tried doesn't work 'EPSG:27200' # src.crs # 
#fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.coastlines()
ax.gridlines(draw_labels=True, crs=proj)
da.plot()
#top.plot_coastlines(fig)
plt.show()

#%% plot coasline + topography + stations

ps = ProcessStations()
dict_md = ps.get_info_dict('temperature')

#%%

# minlon = 165
# maxlon = 179
# minlat = -48
# maxlat = -34

# minlon = 171
# maxlon = 179
# minlat = -42
# maxlat = -34

area = 'north'
minlon = top.extent[area]['minlon']
maxlon = top.extent[area]['maxlon']
minlat = top.extent[area]['minlat']
maxlat = top.extent[area]['maxlat']

marker_size = 30

# proj = ccrs.PlateCarree(central_longitude=cm)
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.coastlines()
ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())
ax.gridlines(draw_labels=True, crs=proj)
da.plot()
for k, v in dict_md.items():
    ax.scatter(v['lon'], v['lat'], color='red', marker='o', s=marker_size)
plt.xlim(minlon, maxlon)
plt.ylim(minlat, maxlat)
plt.show()

plt.savefig('test.png')



#%% 

# src = top.open_with_rasterio(tif_original)

# src_crs = src.crs
# dst_crs = {'init': 'EPSG:4326'}

# transform, width, height = calculate_default_transform(src_crs, dst_crs, src.width, src.height, *src.bounds)

# # Reproject the raster data
# kwargs = src.meta.copy()
# kwargs.update({
#     'crs': dst_crs,
#     'transform': transform,
#     'width': width,
#     'height': height
# })

# dst = rasterio.open('data/topography_100km/north_test.tif', 'w', **kwargs)
# reproject(source=rasterio.band(src, 1), 
#           destination=rasterio.band(dst, 1), 
#           src_transform=src.transform,
#           src_crs=src.crs, 
#           dst_transform=transform, 
#           dst_crs=dst_crs, 
#           resampling=Resampling.nearest,
#           )

#%% 

da = rioxarray.open_rasterio('data/topography_100km/north.tif')

fig = plt.figure(figsize=(10, 12))
proj = ccrs.PlateCarree() #'EPSG:27200' # src.crs # 
#fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.coastlines()
ax.gridlines(draw_labels=True, crs=proj)
da.plot()
#top.plot_coastlines(fig)
plt.show()

#%% x and y values

# for file in all_tif_files:
#     print(f'\n{file}')
#     da = rioxarray.open_rasterio(f'{path}/{file}')
#     print('x')
#     print(np.unique(da.x.values))
#     print('y')
#     print(np.unique(da.y.values))

#%% change coordinate system 
"""
https://rasterio.readthedocs.io/en/stable/topics/reproject.html 
"""

file = 'aklnd_25r.tif'
ds = rasterio.open(f'{path}/{file}')
print(ds.crs) # CRS.from_epsg(27200)
print(ds.bounds)  # BoundingBox(left=2590000.0, bottom=6400000.0, right=2810000.0, top=6550000.0)
old_crs = ds.crs
new_crs = CRS.from_epsg(4326)
dst_crs = 'EPSG:4326'  # EPSG:4326, also known as the WGS84 projection 

transform, width, height = calculate_default_transform(ds.crs, new_crs, ds.width, ds.height, *ds.bounds)

kwargs = ds.meta.copy()
kwargs.update({
    'crs': dst_crs,
    'transform': transform,
    'width': width,
    'height': height
})

with rasterio.open('/tmp/test.tif', 'w', **kwargs) as dst:
    test = reproject(
        source=rasterio.band(ds, 1),
        destination=rasterio.band(dst, 1),
        src_transform=ds.transform,
        src_crs=ds.crs,
        dst_transform=transform,
        dst_crs=new_crs,
        resampling=Resampling.nearest,
        )
    

#%% join all then coarsen

dir = 'data/topography_100km'
print(os.listdir(dir))
save_tif_original = f'{dir}/north.tif'
save_tif = f'{dir}/north_wgs.tif'
save_nc = f'{dir}/north_wgs.nc'

###
dir = 'data/topography_25km'
print(os.listdir(dir))
save_tif_original = f'{dir}/north.tif'
save_tif = f'{dir}/north_wgs.tif'
save_nc = f'{dir}/north_wgs.nc'
###

#%% 

path = '/data/hpcdata/users/risno/code/deepweather-downscaling/data/LandcareResearchDEM/nzdem-north-island-25-metre'
all_tif_files = [f for f in os.listdir(path) if '.tif' in f[-4:] and '.' not in f[:1]]

#%% 1. combine, coarsen, mask

# save_path = 'data/topography_100km/north.nc'
# da_c.to_netcdf(save_path)

#%% 2. convert coords 

ds = rasterio.open(save_tif_original)
print(ds.crs) # CRS.from_epsg(27200)
print(ds.bounds)  # BoundingBox(left=2590000.0, bottom=6400000.0, right=2810000.0, top=6550000.0)
old_crs = ds.crs
new_crs = CRS.from_epsg(4326)
dst_crs = 'EPSG:4326'  # EPSG:4326, also known as the WGS84 projection
# "EPSG:4087 replaced EPSG:32663 which replaced EPSG:32662"

transform, width, height = calculate_default_transform(ds.crs, new_crs, ds.width, ds.height, *ds.bounds)

kwargs = ds.meta.copy()
kwargs.update({
    'crs': dst_crs,
    'transform': transform,
    'width': width,
    'height': height
})

with rasterio.open(save_tif, 'w', **kwargs) as dst:
    test = reproject(
        source=rasterio.band(ds, 1),
        destination=rasterio.band(dst, 1),
        src_transform=ds.transform,
        src_crs=ds.crs,
        dst_transform=transform,
        dst_crs=new_crs,
        resampling=Resampling.nearest,
        )


#%% 3. convert to nc

da = xr.open_dataarray(save_tif)
da = da.squeeze()
da.to_netcdf(save_nc)

#%% plot with coastline

da = rioxarray.open_rasterio(f'{dir}/north.tif')

proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(10, 12))
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.coastlines()
#ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())
ax.gridlines(draw_labels=True, crs=proj)
da.plot()
plt.show()

#%% gpt

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling

path = '/data/hpcdata/users/risno/code/deepweather-downscaling/data/LandcareResearchDEM/nzdem-north-island-25-metre'
all_tif_files = [f for f in os.listdir(path) if '.tif' in f[-4:] and '.' not in f[:1]]

dir = 'data/topography_100km'
save_tif_original = f'{dir}/north.tif'
src = rasterio.open(save_tif_original)
output = 'data/topography_100km/output.tif'

#%%

# Assuming src and dst are your source and destination datasets
src_crs = src.crs
dst_crs = {'init': 'EPSG:4326'}

transform, width, height = calculate_default_transform(src_crs, dst_crs, src.width, src.height, *src.bounds)

# Reproject the raster data
kwargs = src.meta.copy()
kwargs.update({
    'crs': dst_crs,
    'transform': transform,
    'width': width,
    'height': height
})

dst = rasterio.open(output, 'w', **kwargs)
reproject(source=rasterio.band(src, 1), 
          destination=rasterio.band(dst, 1), 
          src_transform=src.transform,
          src_crs=src.crs, 
          dst_transform=transform, 
          dst_crs=dst_crs, 
          resampling=Resampling.nearest,
          )

# reproject(source=rasterio.open(src), 
#           destination=rasterio.open(dst), src_transform=src.transform,
#           src_crs=src.crs, dst_transform=transform, dst_crs=dst_crs, resampling=Resampling.nearest)

# Plot with Cartopy

# # Read the reprojected data
# reprojected_data = dst.read(1)
with rasterio.open(output) as reprojected_ds:
    reprojected_data = reprojected_ds.read(1)


fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
img = ax.imshow(reprojected_data, extent=(transform[2], transform[2] + width * transform[0], transform[5] + height * transform[4], transform[5]), origin='upper', cmap='viridis')
ax.coastlines()
cbar = plt.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
plt.show()


# Plot with Cartopy
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
img = ax.imshow(reprojected_data, extent=(transform[2], transform[2] + width * transform[0], transform[5] + height * transform[4], transform[5]), origin='upper', cmap='viridis', transform=ccrs.PlateCarree())
ax.coastlines()

# Add colorbar for reference
cbar = plt.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
#cbar.set_label('Your Colorbar Label')
plt.show()