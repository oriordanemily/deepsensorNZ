import os
from typing_extensions import Literal
import pickle

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
import pandas as pd

from nzdownscale.dataprocess.config import PLOT_EXTENT
from nzdownscale.dataprocess.config_local import DATA_PATHS
import argparse

def save_pickle(x, filename):
    with open(filename, 'wb') as pickle_file:
        pickle.dump(x, pickle_file)


def open_pickle(filename):
    with open(filename, 'rb') as pickle_file:
        x = pickle.load(pickle_file)
    return x

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


class Caching:
    def __init__(self) -> None:
        pass


class DataProcess:
    def __init__(self) -> None:
        pass


    def open_ds(self,
                file: str,
                ) -> xr.Dataset:
        """ Open file as xarray dataset """
        return xr.open_dataset(file)


    def open_da(self, 
                da_file: str,
                ) -> xr.DataArray:
        """ Open as dataarray """
        #return rioxarray.open_rasterio(da_file)
        return xr.open_dataarray(da_file)
    

    def ds_to_da(self,
                 ds: xr.Dataset,
                 var: str,
                 ) -> xr.DataArray:
        """ Get data array from dataset """
        return ds[var]

    
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
                    boundary: str = 'trim',
                    ):
        """
        Reduce resolution of data array by factor coarsen_by. e.g. coarsen_by=4 will reduce 25m resolution to 100m resolution.
        https://stackoverflow.com/questions/53886153/resample-xarray-object-to-lower-resolution-spatially
        """
        #return da.coarsen(longitude=coarsen_by, boundary=boundary).mean().coarsen(latitude=coarsen_by, boundary=boundary).mean().squeeze()
        if coarsen_by == 1:
            return da
        else:
            return da.coarsen(latitude=coarsen_by, longitude=coarsen_by, boundary=boundary).mean()


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


    def resolution(self,
                   ds: xr.Dataset,
                   coord: str,
                   ) -> float:
        """ Calculate resolution of coodinate in dataset """
        return np.round(np.abs(np.diff(ds.coords[coord].values)[0]), 5)


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


    def nz_map_with_coastlines(self, area=None):
        """ Get figure axis with coastlines for NZ """
        if area is None:
            area = 'all'

        minlon = PLOT_EXTENT[area]['minlon']
        maxlon = PLOT_EXTENT[area]['maxlon']
        minlat = PLOT_EXTENT[area]['minlat']
        maxlat = PLOT_EXTENT[area]['maxlat']

        ax = self.get_ax_nz_map((minlon, maxlon), (minlat, maxlat))
        return ax


    def get_ax_nz_map(self, lon_lim, lat_lim):
        fig = plt.figure(figsize=(10, 12))
        proj = ccrs.PlateCarree()
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.tick_params(axis='both', labelsize=15)
        ax.coastlines()
        ax.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]], proj)
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

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def int_or_none(value):
    if value is None or value == 'None':
        return None
    else:
        return int(value)

def str_or_none(value):
    if value is None or value == 'None':
        return None
    else:
        return str(value)
    
def bool_or_float_or_str(value):
    if value is True:
        return 1.0
    elif value is False:
        return 0.0
    elif value == 'random':
        return 'random'
    else:
        return float(value)

def validate_and_convert_args(args):

    type_functions = {
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
    'list': list,
    'int_or_none': int_or_none,
    'str_or_none': str_or_none,
    'bool_or_float_or_str': bool_or_float_or_str,
    }

    validated_args = {}
    for key, value in args.items():
        arg_value = value['arg']
        arg_type = value['type']
        if arg_type in type_functions:
            try:
                validated_args[key] = type_functions[arg_type](arg_value)
            except ValueError as e:
                raise ValueError(f"Invalid value for {key}: {arg_value}. Error: {e}")
    return validated_args

def debug_plot_da(da: xr.DataArray, save_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    da.plot(ax=ax)
    fig.savefig(save_path)

def random_hour_subset_xr(ds: xr.Dataset):
    """ Returns a dataset with a random hour selected for each day in the dataset"""
    ds['time'] = pd.to_datetime(ds['time'].values)
    daily_groups = ds.groupby('time.date')

    def select_random_hour(day_data):
        return day_data.isel(time=np.random.randint(0, len(day_data)))
    
    selected_hours = [select_random_hour(day) for date, day in daily_groups]
    ds_random_hour_per_day = xr.concat(selected_hours, dim='time')

    return ds_random_hour_per_day