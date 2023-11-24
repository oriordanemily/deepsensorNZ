

import os
from typing import Literal

import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


class ProcessStations:
    def __init__(self) -> None:
        
        self.path = 'data/nz'
        self.names = {
            'precipitation': {
                'folder': 'Precipitation',
                'var': 'precipitation'
            },
            'temperature': {
                'folder': 'ScreenObs',
                'var': 'dry_bulb'
            },
        }


    def get_var_path(self, 
                 variable: Literal['temperature', 'precipitation'],
                 ):
        self.var = variable
        self.var_path = f'{self.path}/{self.names[variable]["folder"]}'
        return self.var_path
    

    def get_da_from_ds(self, 
                       ds,
                       variable: Literal['temperature', 'precipitation'],
                       ):
        return ds[self.names[variable]['var']]
        

    def get_list_all_stations(self, 
                              variable: Literal['temperature', 'precipitation'],
                              ):
        self.all_stations = os.listdir(self.get_var_path(variable))
        return self.all_stations
    

    def get_station_ds(self,
                       variable: Literal['temperature', 'precipitation'],
                       station:str=None,
                       i_station:int=None,
                       ):
        if station is not None:
            return xr.open_dataset(f'{self.get_var_path(variable)}/{station}')
        elif i_station is not None:
            all_stations = self.get_list_all_stations(variable)
            return xr.open_dataset(f'{self.get_var_path(variable)}/{all_stations[i_station]}')
        else:
            raise ValueError('Provide station or i_station')


    def get_station_da(self,
                       variable: Literal['temperature', 'precipitation'],
                       station:str=None,
                       i_station:int=None,
                       ):
        ds = self.get_station_ds(variable, station, i_station)
        return self.get_da_from_ds(ds, variable)
    

    def get_info_dict(self, 
                       variable: Literal['temperature', 'precipitation'],
                       ):
        """ min max years and coords"""

        all_stations = self.get_list_all_stations(variable)
        
        dict_md = {}
        for f in tqdm(all_stations):
            try:
                ds = xr.open_dataset(f'{self.get_var_path(variable)}/{f}')
                #lon = ds.longitude.values
                #lat = ds.latitude.values
                da = ds[self.names[variable]["var"]]
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

        return dict_md
    

    def get_coord_dict(self, variable):
        all_stations = self.get_list_all_stations(variable)
        dict_coord = {}
        for f in tqdm(all_stations):
            ds = xr.open_dataset(f'{self.get_var_path(variable)}/{f}')
            lon = ds.longitude.values
            lat = ds.latitude.values
            dict_coord[f] = {'lon': lon, 'lat':lat}
        return dict_coord


    def dict_md_to_df(self, dict_md):
        return pd.DataFrame(dict_md).T
    

    def plot_stations_on_map(self, dict_md):
        minlon = 165
        maxlon = 179
        minlat = -48
        maxlat = -34
        marker_size = 60

        df = self.dict_md_to_df(dict_md)
        lon_lat_tuple = list(zip(df['lon'], df['lat']))
        self.plot_points(lon_lat_tuple, (minlon, maxlon), (minlat, maxlat), marker_size)

        # # proj = ccrs.PlateCarree(central_longitude=cm)
        # proj = ccrs.PlateCarree()
        # fig = plt.figure(figsize=(10, 12))
        # ax = fig.add_subplot(1, 1, 1, projection=proj)
        # ax.coastlines()
        # ax.set_extent([minlon, maxlon, minlat, maxlat], ccrs.PlateCarree())
        # ax.gridlines(draw_labels=True, crs=proj)
        # for k, v in dict_md.items():
        #     ax.scatter(v['lon'], v['lat'], color='red', marker='o', s=marker_size)
        # plt.show()


    def plot_points(self, 
                    lon_lat_tuples,
                    lon_lim=None,
                    lat_lim=None,
                    marker_size=30,
                    ):

        fig = plt.figure(figsize=(10, 12))
        proj = ccrs.PlateCarree()
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.coastlines()
        if lon_lim is not None and lat_lim is not None:
            # ax.set_extent([lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1]], ccrs.PlateCarree())
            ax.set_xlim(lon_lim)
            ax.set_ylim(lat_lim)
        ax.gridlines(draw_labels=True, crs=proj)
        for lon, lat in lon_lat_tuples:
            ax.scatter(lon, lat, color='red', marker='o', s=marker_size)
        plt.show()