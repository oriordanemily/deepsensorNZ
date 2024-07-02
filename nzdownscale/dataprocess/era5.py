import os
from typing import Literal, List
import glob

import xarray as xr
from datetime import datetime

from nzdownscale.dataprocess.utils import DataProcess
from nzdownscale.dataprocess.config import VARIABLE_OPTIONS, VAR_ERA5
from nzdownscale.dataprocess.config_local import DATA_PATHS


class ProcessERA5(DataProcess):

    def __init__(self) -> None:
        super().__init__()


    def load_ds(self, 
                var: Literal[tuple(VARIABLE_OPTIONS)],
                years: List=None,
                ) -> xr.Dataset:
        """ 
        Loads dataset
        Args: 
            var (str): variable
            years (list): specific years, retrieves all if set to None
        """
        if type(years) == int:
            years = [years]
        elif type(years) == str:
            years = [int(years)]
        elif type(years) == list:
            years = [int(year) for year in years]
            if len(set(years)) == 1:
                years = [years[0]]
        else:
            ValueError (f'Years should be int, str or list, not {type(years)}')
        filenames = self.get_filenames(var, years)
        return xr.open_mfdataset(filenames)

    
    def ds_to_da(self,
                 ds: xr.Dataset,
                 var: Literal[tuple(VARIABLE_OPTIONS)],
                 ) -> xr.DataArray:
        """
        Extracts dataarray from dataset (variable data only, loses some metadata)
        If variable is temperature, converts from Kelvin to Celsius
        Args: 
            ds (xr.Dataset): dataset
            var (str): variable
        """
        da = ds[VAR_ERA5[var]['var_name']]
        if var == 'temperature':
            da = self.kelvin_to_celsius(da)
        return da


    def convert_hourly_to_daily(self, 
                                ds, 
                                function:Literal['mean', 'sum']='mean'):
        if function == 'mean':
            ds = ds.resample(time='D').mean()
        elif function == 'sum':
            ds = ds.resample(time='D').sum()
        else:
            raise ValueError(f'function={function} not recognised')
        # ! below line causes error in convNP run
        #ds['time'] = ds['time'].dt.strftime('%Y-%m-%d')
        return ds
    

    def coarsen_da(self, da: xr.DataArray, coarsen_by: int, boundary: str = 'trim'):
        return super().coarsen_da(da, coarsen_by, boundary)


    def get_parent_path(self,
                        var: Literal[tuple(VARIABLE_OPTIONS)],
                        ):
        if VAR_ERA5[var]['folder'] == 'NZ_land_processed':
            parent = "parent_processed"
        else:
            parent = "parent"
        return f'{DATA_PATHS["ERA5"][parent]}/{VAR_ERA5[var]["subdir"]}'
    

    def get_filenames(self,
                      var: Literal[tuple(VARIABLE_OPTIONS)],
                      years: List=None,
                      ) -> List[str]:
        """ Get list of ERA5 filenames for variable and list of years (if specified) """ 

        parent_path = self.get_parent_path(var)
        
        if VAR_ERA5[var]['folder'] == 'NZ_land': #'temperature': #surface pressure # 10m_u_component_of_wind # 10m_v_component_of_wind # 
            if years is None:
                filenames = glob.glob(f'{parent_path}/*/*.nc')
            else:
                filenames = []
                for year in years:
                    filenames_year = glob.glob(f'{parent_path}/{year}/*.nc')
                    if len(filenames_year) == 0:
                        filenames_year = glob.glob(f'{parent_path}/{year}/*/*.nc')
                    filenames = filenames + filenames_year

        elif VAR_ERA5[var]['folder'] == 'NZ_land_processed': #var == 'precipitation':  # relative_humidity
            if years is None:
                filenames = glob.glob(f'{parent_path}/*.nc')
            else:
                filenames = []
                for year in years:
                    filenames = filenames + [f'{parent_path}/{fname}' for fname in os.listdir(parent_path) if str(year) in fname]

        return filenames

    def load_ds_time(self, 
                     var: Literal[tuple(VARIABLE_OPTIONS)],
                     time,
                     ) -> xr.Dataset:
        """ 
        Loads dataset with time dimension
        Args: 
            var (str): variable
            time: datetime obj
                    """
        if isinstance(time, datetime):
            year = [time.year]
        elif isinstance(time, list):
            year = set(t.year for t in time)
        filenames = self.get_filenames(var, year)
        ds = xr.open_mfdataset(filenames)
        return ds.sel(time=time).load()


    def kelvin_to_celsius(self, da: xr.DataArray):
        return da - 273.15
    

if __name__ == '__main__':
    pass