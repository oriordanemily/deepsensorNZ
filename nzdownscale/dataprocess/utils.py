import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns


from nzdownscale.dataprocess.config import PLOT_EXTENT


class DataProcess:
    def __init__(self) -> None:
        pass


    def open_da(self, 
                da_file: str,
                ) -> xr.DataArray:
        """ Open as dataarray """
        #return rioxarray.open_rasterio(da_file)
        return xr.open_dataarray(da_file)

    
    def mask_da(self, 
                da: xr.DataArray, 
                mask_value: float=-1e30,
                ) -> xr.DataArray:
        """ 
        Set to None "no data" points below mask_value (e.g. -1e30) 
        """
        return da.where(da > mask_value).squeeze()


    def coarsen_da(self, 
                    da: xr.DataArray, 
                    coarsen_by: int, 
                    boundary: str='exact',
                    ):
        """
        Reduce resolution of data array by factor coarsen_by. e.g. coarsen_by=4 will reduce 25m resolution to 100m resolution.
        https://stackoverflow.com/questions/53886153/resample-xarray-object-to-lower-resolution-spatially
        """
        return da.coarsen(lon=coarsen_by, boundary=boundary).mean().coarsen(lat=coarsen_by, boundary=boundary).mean().squeeze()


    def rename_xarray_coords(self,
                             da,
                             rename_dict: dict,
                             ):
        """ Rename coordinates """
        return da.rename(rename_dict)


    def save_nc(self,
                da, 
                name: str,
                ) -> None:
        """ Save as .nc netcdf to name e.g. 'path/file.nc' """
        da.to_netcdf(name)



class PlotData:
    def __init__(self) -> None:
        pass


    def plot_with_coastlines(self,
                             da: xr.DataArray,
                             longitude_coord_name: str='longitude',
                             latitude_coord_name: str='latitude',
                             ):
        """
        Plot data with coastlines
        Can take ~3-4 min for ~100m resolution over NZ
        """
        minlon = np.array(da[longitude_coord_name].min())
        maxlon = np.array(da[longitude_coord_name].max())
        minlat = np.array(da[latitude_coord_name].min())
        maxlat = np.array(da[latitude_coord_name].max())

        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=proj), figsize=(10, 12))
        ax.coastlines()
        ax.set_xlim(minlon, maxlon)
        ax.set_ylim(minlat, maxlat)
        da.plot()
        plt.show()


    def nz_map_with_coastlines(self):
        minlon = PLOT_EXTENT['all']['minlon']
        maxlon = PLOT_EXTENT['all']['maxlon']
        minlat = PLOT_EXTENT['all']['minlat']
        maxlat = PLOT_EXTENT['all']['maxlat']

        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(10, 12))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.coastlines()
        ax.set_extent([minlon, maxlon, minlat, maxlat], proj)
        ax.gridlines(draw_labels=True, crs=proj)
        return ax


    def plot_hist_values(self,
                         da: xr.DataArray,
                         n: int=None,
                         ):
        """ 
        Plot values in data array as histogram 
        Args:
            da (xr.DataArray): data array
            n (int): number of random values to plot (for speed)
        """

        # Flatten the DataArray to a 1D array
        values = da.values.flatten()
        if n is not None:
            values = np.random.choice(values, size=n, replace=False)

        # Plot histogram
        sns.histplot(values)
        plt.xlabel('Value')
        plt.show()

        # plt.hist(values, bins=30, color='blue', edgecolor='black')
        # plt.xlabel('Value')
        # plt.ylabel('Frequency')
        # plt.show()

