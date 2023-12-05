"""
Process topography/elevation data 
"""

import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from nzdownscale.dataprocess.utils import DataProcess


class ProcessTopography(DataProcess):
    def __init__(self) -> None:
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


    def plot_hist_values(self,
                         da,
                         ):
        # Flatten the DataArray to a 1D array
        flat_data = da.values.flatten()

        # Plot the histogram
        plt.hist(flat_data, bins=30, color='blue', edgecolor='black')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

        #da_subset = da.sel(x=slice(170, 173), y=slice(-43, -40))


    def plot_with_coastlines(self,
                             da,
                             longitude_coord_name:str='longitude',
                             latitude_coord_name:str='latitude',
                             ):
        """
        Plot elevation data with coastlines
        Can take ~3-4 min for ~100m
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


if __name__ == '__main__':

    top = ProcessTopography()
    file_to_open = 'data/topography/nz_elevation_25m.nc'
    save_as = 'data/topography/nz_elevation_800m_test.nc'
    coarsen_by = 8
    boundary = 'pad'
    coord_rename = {'lat': 'latitude','lon': 'longitude'}
    plot = False

    # Coarsen and rename coords
    da = top.open_da(f'{file_to_open}').squeeze()
    da_coarsened = top.coarsen_da(da, coarsen_by=coarsen_by, boundary=boundary)  # 1m20s
    da_coarsened = top.rename_xarray_coords(da_coarsened, coord_rename)

    # Inspect
    if plot:
        da_coarsened.plot()  # 2m11s
    print(da_coarsened)

    # Save
    top.save_nc(da_coarsened, save_as)
    print(f"Saved: {save_as}")

