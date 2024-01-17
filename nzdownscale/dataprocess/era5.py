import os
from typing import Literal, List
import glob

import xarray as xr

from nzdownscale.dataprocess.utils import DataProcess
from nzdownscale.dataprocess.config import VARIABLE_OPTIONS, DATA_PATHS, VAR_ERA5


class ProcessERA5(DataProcess):

    def __init__(self) -> None:
        super().__init__()


    def load_ds(self, 
                var: Literal[tuple(VARIABLE_OPTIONS)],
                years: List=None,
                daily: bool=False,
                ) -> xr.Dataset:
        """ 
        Loads dataset
        Args: 
            var (str): variable
            years (list): specific years, retrieves all if set to None
        """
        filenames = self.get_filenames(var, years)
        ds = xr.open_mfdataset(filenames)
        if daily:
            ds = ds.resample(time="D").mean()
            # ds['time'] = ds['time'].dt.strftime('%Y-%m-%d')
            return ds
        else:
            return ds

    
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
    

    def coarsen_da(self, da: xr.DataArray, coarsen_by: int, boundary: str = 'trim'):
        return super().coarsen_da(da, coarsen_by, boundary)


    def get_parent_path(self,
                        var: Literal[tuple(VARIABLE_OPTIONS)],
                        ):
        return f'{DATA_PATHS["ERA5"]["parent"]}/{VAR_ERA5[var]["subdir"]}'
    

    def get_filenames(self,
                      var: Literal[tuple(VARIABLE_OPTIONS)],
                      years: List=None,
                      ) -> List[str]:
        """ Get list of ERA5 filenames for variable and list of years (if specified) """ 

        parent_path = self.get_parent_path(var)
        
        if var == 'temperature':
            if years is None:
                filenames = glob.glob(f'{parent_path}/*/*.nc')
            else:
                filenames = []
                for year in years:
                    filenames = filenames + glob.glob(f'{parent_path}/{year}/*.nc')
        
        elif var == 'precipitation':  
            if years is None:
                filenames = glob.glob(f'{parent_path}/*.nc')
            else:
                filenames = []
                for year in years:
                    filenames = filenames + [f'{parent_path}/{fname}' for fname in os.listdir(parent_path) if str(year) in fname]
        
        return filenames

    def kelvin_to_celsius(self, da: xr.DataArray):
        return da - 273.15

if __name__ == '__main__':
    pass