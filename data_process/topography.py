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


class ProcessTopography:
    def __init__(self, path:str) -> None:
        self.path = path
        self.extent = {
            'all': {
                'minlon': 165,
                'maxlon': 179,
                'minlat': -48,
                'maxlat': -34,
            },
            'north': {
                'minlon': 171,
                'maxlon': 179,
                'minlat': -42,
                'maxlat': -34,
            },
        }


    def set_source_path(self, path:str):
        self.path = path


    def get_all_tifs(self, 
                     path:str=None,
                     ) -> list:
        if path is None: path = self.path
        return [f for f in os.listdir(path) if '.tif' in f[-4:] and '.' not in f[:1]]


    def open_da(self, 
                da_file:str,
                ) -> xr.DataArray:
        return rioxarray.open_rasterio(da_file).squeeze()

    
    def mask_da(self, 
                      da: xr.DataArray, 
                      mask_value:float=-1e30,
                      ) -> xr.DataArray:
        return da.where(da > mask_value).squeeze()
    

    def coarsen_da(self, 
                         da:xr.DataArray, 
                         coarsen_by:int, 
                         boundary:str='exact',
                         ):
        """
        https://stackoverflow.com/questions/53886153/resample-xarray-object-to-lower-resolution-spatially
        """
        return da.coarsen(x=coarsen_by, boundary=boundary).mean().coarsen(y=coarsen_by, boundary=boundary).mean().squeeze()


    def get_combined_da(self, path:str=None):
        #names = [name.split('.tif')[0] for name in self.get_all_tifs(path)]
        #all_tifs = {n: None for n in names}
        # for i, file in enumerate(all_tif_files):
        #     #print(file)
        #     all_tifs[names[i]] = self.open_da(f'{path}/{file}')
        # return xr.combine_by_coords(list(all_tifs))

        if path is None: path = self.path
        all_tifs = self.get_all_tifs(path)
        da_list = []
        for file in all_tifs:
            da_list.append(self.open_da(f'{path}/{file}'))    
        return xr.combine_by_coords(da_list).squeeze()


    def open_with_rasterio(self, file:str):
        return rasterio.open(file)

    
    def plot_coastlines(self, fig):
        """
        cartopy projection https://scitools.org.uk/cartopy/docs/latest/reference/projections.html#platecarree
        """
        proj = ccrs.PlateCarree() #'EPSG:27200' # src.crs # 
        #fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.coastlines()
        ax.gridlines(draw_labels=True, crs=proj)
        #da.plot()
        #plt.show()
        return ax
    

    def save_da_as_tif(self, da, name):
        da.rio.to_raster(name)


    def change_crs(self, original_f, destination_f):
        """not working"""

        ds = rasterio.open(original_f)
        print(f'Original crs: {ds.crs}')  # CRS.from_epsg(27200)
        print(ds.bounds)  # BoundingBox(left=2590000.0, bottom=6400000.0, right=2810000.0, top=6550000.0)
        old_crs = ds.crs
        new_crs = CRS.from_epsg(4326)
        dst_crs = 'EPSG:4326'  # EPSG:4326, also known as the WGS84 projection

        transform, width, height = calculate_default_transform(old_crs, new_crs, ds.width, ds.height, *ds.bounds)

        kwargs = ds.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(destination_f, 'w', **kwargs) as dst:
            test = reproject(
                source=rasterio.band(ds, 1),
                destination=rasterio.band(dst, 1),
                src_transform=ds.transform,
                src_crs=ds.crs,
                dst_transform=transform,
                dst_crs=new_crs,
                resampling=Resampling.nearest,
                )
