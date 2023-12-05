import os
from typing import Literal
import glob

import xarray as xr

from nzdownscale.dataprocess.utils import DataProcess


class ProcessERA5(DataProcess):
    
    def __init__(self) -> None:
        self.path = 'data/ERA5-land'
        self.names = {
            'precipitation': {
                'folder': f'{self.path}/total_precipitation_hourly',
                'var_name': 'precipitation',
            },
            'temperature': {
                'folder': f'{self.path}/2m_temperature',
                'var_name': 't2m',
            },
        }


    def load_ds(self, 
               var:Literal['precipitation', 'temperature'],
               ):
        filenames = self.get_filenames(var)
        return xr.open_mfdataset(filenames)

    
    def ds_to_da(self,
                 ds,
                 var:Literal['precipitation', 'temperature'],
                 ):
        return ds[self.names[var]['var_name']]
    

    def load_ds_specific_year(self,
                    var:Literal['precipitation', 'temperature'],
                    year:int,
                    ):
        filenames = self.get_filenames_year(var, year)
        return xr.open_mfdataset(filenames)


    def get_filenames(self,
                      var:Literal['precipitation', 'temperature'],
                      ):
        if var == 'temperature':
            filenames = glob.glob(f'{self.names[var]["folder"]}/*/*.nc')
        elif var == 'precipitation':  
            filenames = glob.glob(f'{self.names[var]["folder"]}/*.nc')
        else:
            raise ValueError(f'var={var} not recognised')
        return filenames


    def get_filenames_year(self,
                           var:Literal['precipitation', 'temperature'],
                           year:int,
                           ):
        """ Get files for specific year """

        if var == 'temperature':
            filenames = glob.glob(f'{self.names[var]["folder"]}/{year}/*.nc')
        elif var == 'precipitation':
            filenames = [f'{self.names[var]["folder"]}/{fname}' for fname in os.listdir(f'{self.names["precipitation"]["folder"]}') if str(year) in fname]
        else:
            raise ValueError(f'var={var} not recognised')
        return filenames
        

if __name__ == '__main__':
    pass