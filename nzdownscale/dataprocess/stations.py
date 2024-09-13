import os
from typing import Literal, List

import xarray as xr
import pandas as pd
from tqdm import tqdm
import numpy as np

from nzdownscale.dataprocess.utils import DataProcess, PlotData
from nzdownscale.dataprocess.config import VARIABLE_OPTIONS, VAR_STATIONS, STATION_LATLON
from nzdownscale.dataprocess.config_local import DATA_PATHS



class ProcessStations(DataProcess):

    def __init__(self) -> None:
        super().__init__()


    def get_parent_path(self,
                        var: Literal[tuple(VARIABLE_OPTIONS)],
                        ):
        return f'{DATA_PATHS["stations"]["parent"]}/{VAR_STATIONS[var]["subdir"]}'


    def ds_to_da(self,
                 ds: xr.Dataset,
                 var: Literal[tuple(VARIABLE_OPTIONS)],
                 ) -> xr.DataArray:
        """
        Extracts dataarray from dataset (variable data only, loses some metadata)
        Args: 
            ds (xr.Dataset): dataset
            var (str): variable
        """
        return ds[VAR_STATIONS[var]['var_name']]


    def get_path_all_stations(self,
                              var: Literal[tuple(VARIABLE_OPTIONS)],
                              ) -> List[str]:
        """ Get list of filepaths for all stations for variable var """
        all_stations = os.listdir(self.get_parent_path(var))
        return [f"{self.get_parent_path(var)}/{s}" for s in all_stations]
    

    def load_station(self,
                     filepath: str,
                     ) -> xr.Dataset:
        return xr.open_dataset(filepath)


    def load_station_df(self,
                        filepath: str,
                        var: str,
                        daily: bool = False,
                        fill_missing: bool = True
                        ) -> pd.DataFrame:
        ds = self.load_station(filepath)
        da = self.ds_to_da(ds, var)
        df_station = da.to_dataframe()
        if daily: 
            df_station = df_station.reset_index().resample('D', on='time').mean()[[VAR_STATIONS[var]['var_name']]]
        lon, lat = self.get_lon_lat(ds)
        df_station['longitude'] = lon
        df_station['latitude'] = lat

        return df_station


    def get_lon_lat(self,
                    ds: xr.Dataset,
                    ) -> tuple:
        return float(ds.longitude), float(ds.latitude)
    

    def get_metadata_df(self,
                        var: Literal[tuple(VARIABLE_OPTIONS)],
                        ) -> pd.DataFrame: 
        """ get station metadata in dataframe format """
        dict_md = self.get_metadata_dict(var=var)
        df = self.dict_to_df(dict_md)
        df['station_id'] = df.index
        df['station_id'] = df['station_id'].apply(lambda row: row.split('.nc')[0].split('/')[-1])
        return df
        

    def get_metadata_dict(self, 
                          var: Literal[tuple(VARIABLE_OPTIONS)],
                          ) -> dict:
        """ get dictionary of min max years and coords"""

        station_paths = self.get_path_all_stations(var)        
        dict_md = {}
        for f in tqdm(station_paths, desc='Loading stations'):
            try:
                ds = self.load_station(f)
                lon, lat = self.get_lon_lat(ds)
                start_year, end_year = self.get_start_and_end_years(ds)
                duration = end_year - start_year

                dict_md[f] = {
                    'start_year': int(start_year), 
                    'end_year': int(end_year), 
                    'duration_years': int(duration),
                    'lon': lon, 
                    'lat':lat,
                    }
            except:
                pass

        return dict_md
    

    def get_start_and_end_years(self, ds: xr.Dataset):
        """ Get tuple of start year and end year of dataset e.g. (1991, 2019) """
        return tuple(int(t.year) for t in pd.DatetimeIndex([ds['time'][0].values, ds['time'][-1].values]))
    

    def get_coord_df(self, 
                     var: Literal[tuple(VARIABLE_OPTIONS)],
                     ) -> pd.DataFrame:
        station_paths = self.get_path_all_stations(var)
        dict_coord = {}
        for f in tqdm(station_paths):
            ds = self.load_station(f)
            lon, lat = self.get_lon_lat(ds)
            dict_coord[f] = {
                'lon': lon, 
                'lat':lat,
                }
        df = self.dict_to_df(dict_coord)
        return df


    def dict_to_df(self, 
                   d: dict,
                   ) -> pd.DataFrame:
        return pd.DataFrame.from_dict(d, orient='index')


    def plot_stations(self, 
                      df: pd.DataFrame, 
                      ax,
                      markersize=40,
                      color='red',
                      ):
        i = 0
        for lon, lat in zip(df['lon'].values, df['lat'].values):
            if i == 0:
                ax.scatter(lon, lat, color=color, marker='o', s=markersize, label='Stations')
                i += 1
            else:
                ax.scatter(lon, lat, color=color, marker='o', s=markersize)
        return ax


    def plot_stations_on_map(self,
                             df: pd.DataFrame,
                             # dict_md: dict,
                             ):
        nzplot = PlotData()
        ax = nzplot.nz_map_with_coastlines()
        ax = self.plot_stations(df, ax)

        # for lon, lat in zip(df['lon'].values, df['lat'].values):
        #     ax.scatter(lon, lat, color='red', marker='o', s=60)
        # for k, v in dict_md.items():
        #     ax.scatter(v['lon'], v['lat'], color='red', marker='o', s=60)

        return ax
    
    def get_station_info(self, var):
        paths = self.get_path_all_stations(var)

        stations_list = []
        for path in paths:
            stations_list.append(self.load_station(path))

        station_info = {}
        for station in stations_list:
            name = station.attrs['site name']
            station_info[name] = {'station_no': station.attrs['agent_number'],
                                'latitude': float(station['latitude'].values),
                                'longitude': float(station['longitude'].values),
                                'elevation': float(station['station_height'].values),}

        return station_info
    
    def load_stations_time(self, 
                           var: str, 
                           time, 
                           remove_stations: list = [], 
                           keep_stations: list = [],
                           daily: bool=False):
        """Load the stations at a given time (or list of given times)

        Args:
            var (str): Station variable
            time (_type_): Can be a list of times, a pd.Timestamp, or a np.datetime64
            remove_stations (list, optional): List of station names to be dropped. Defaults to [].
            keep_stations (list, optional): Only return these stations. Defaults to [].
            daily (bool, optional): Resample to daily. Defaults to False.

        Returns:
            df: DataFrame of stations
        """
        if isinstance(time, (list, np.ndarray)):
            time = np.array(time, dtype='datetime64[ns]')
            def condition(lst, ds_time): return len(set(lst).intersection(ds_time)) != 0 

        else:
            if isinstance(time, pd.Timestamp):
                time = np.datetime64(time.to_pydatetime())
            elif not isinstance(time, np.datetime64):
                time = np.datetime64(time)
            def condition(lst, ds_time): return lst in ds_time

        paths = self.get_path_all_stations(var)

        df_list = []
        for path in tqdm(paths, desc='Loading stations'):
            try:
                with xr.open_dataset(path) as ds:
                    if condition(time, ds.time.values):
                        da = self.ds_to_da(ds, var)
                        df_station = da.to_dataframe()
                        if daily: 
                            if var == 'precipitation':
                                df_station = df_station.reset_index().resample('D', on='time').sum()[[VAR_STATIONS[var]['var_name']]]
                            else:
                                df_station = df_station.reset_index().resample('D', on='time').mean()[[VAR_STATIONS[var]['var_name']]]
                        df_station = df_station.loc[time]
                        lon, lat = self.get_lon_lat(ds)
                        df_station['longitude'] = lon
                        df_station['latitude'] = lat
                        df_list.append(df_station)
            except:
                pass
        
        print(f'{len(df_list)} stations with data at prediction time(s)')
                    
        df = pd.concat(df_list)

        if len(remove_stations) > 0:
            for station in remove_stations:
                print(f'Removing {station}')
                latlon = (STATION_LATLON[station]['latitude'], STATION_LATLON[station]['longitude'])
                df = df[~((df['latitude'] == latlon[0]) & (df['longitude'] == latlon[1]))]
            print(f'Removed {len(remove_stations)} stations')
        elif len(keep_stations) > 0:
            new_df_list = []
            for station in keep_stations:
                print(f'Keeping {station}')
                latlon = (STATION_LATLON[station]['latitude'], STATION_LATLON[station]['longitude'])
                new_df_list.append(df[((df['latitude'] == latlon[0]) & (df['longitude'] == latlon[1]))])
            df = pd.concat(new_df_list)
            print(f'Kept {len(keep_stations)} stations')

        df = df.reset_index().rename(columns={'index': 'time'})
        df = df.set_index(['time', 'latitude', 'longitude']).sort_index()

        df_column_name = df.columns[0]
        df = df.rename(columns={df_column_name: f"{var}_station"})
        
        return df
    
        
        

       

if __name__ == '__main__':
    pass