import os
from typing import Literal, List

import xarray as xr
import pandas as pd
from tqdm import tqdm

from nzdownscale.dataprocess.utils import DataProcess, PlotData
from nzdownscale.dataprocess.config import VARIABLE_OPTIONS, DATA_PATHS, VAR_STATIONS


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
                     filepath:str=None,
                     ) -> xr.Dataset:
        return xr.open_dataset(filepath)
        

    def get_lon_lat(self,
                    ds: xr.Dataset,
                    ) -> tuple:
        return float(ds.longitude), float(ds.latitude)
    

    def get_metadata_df(self,
                        var: Literal[tuple(VARIABLE_OPTIONS)],
                        ) -> pd.DataFrame: 
        """ get station metadata in dataframe format """
        dict_md = self.get_metadata_dict(var=var)
        return self.dict_to_df(dict_md)
    

    def get_metadata_dict(self, 
                          var: Literal[tuple(VARIABLE_OPTIONS)],
                          ) -> dict:
        """ get dictionary of min max years and coords"""

        station_paths = self.get_path_all_stations(var)        
        dict_md = {}
        for f in tqdm(station_paths):
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
        for lon, lat in zip(df['lon'].values, df['lat'].values):
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


if __name__ == '__main__':
    pass