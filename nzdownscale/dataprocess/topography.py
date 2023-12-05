"""
Process topography/elevation data 
"""

import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from nzdownscale.dataprocess.utils import DataProcess
from nzdownscale.dataprocess.config import DATA_PATHS


class ProcessTopography(DataProcess):

    def __init__(self) -> None:
        super().__init__()


if __name__ == '__main__':

    file_to_open = f'{DATA_PATHS["topography"]}/nz_elevation_25m.nc'
    save_as = f'{DATA_PATHS["topography"]}/nz_elevation_800m_test.nc'
    coarsen_by = 8
    boundary = 'pad'
    coord_rename = {'lat': 'latitude','lon': 'longitude'}
    plot = False

    # Coarsen and rename coords
    top = ProcessTopography()
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

