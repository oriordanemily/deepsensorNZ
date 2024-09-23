import os
from types import NoneType
from typing import Literal, List
import glob
from scipy.interpolate import griddata

import xarray as xr
from datetime import datetime, timedelta
from scipy.interpolate import LinearNDInterpolator
import numpy as np
from functools import partial
from tqdm import tqdm
import dask.array as da
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
import xesmf as xe

from nzdownscale.dataprocess.utils import DataProcess
from nzdownscale.dataprocess.config import VARIABLE_OPTIONS, VAR_WRF
from nzdownscale.dataprocess.config_local import DATA_PATHS

from dask.diagnostics import ProgressBar

def generate_datetimes(start_str, end_str, interval_hours=12):
    # Convert strings to datetime objects
    start_date = datetime.strptime(str(start_str), '%Y%m%d%H')
    end_date = datetime.strptime(str(end_str), '%Y%m%d%H')

    # Create a list to hold all the datetimes
    datetimes = []

    # Generate datetimes at the specified interval
    current_date = start_date
    while current_date <= end_date:
        datetimes.append(current_date.strftime('%Y%m%d%H'))
        current_date += timedelta(hours=interval_hours)

    return datetimes


def get_filepaths(start_init, end_init,
                  model='nz4kmN-ECMWF-SIGMA'):
    """
    Returns list of filepaths for WRF data
    Args:
        start_init (int): start year
        end_init (int): end year
    """

    if model == 'nz4kmN-ECMWF-SIGMA':
        print('Currently using midnight runs only')
        interval_hours = 24 # change to 12
    else:
        ValueError(f'Model {model} not yet implemented')

    sub_dirs = generate_datetimes(start_init, end_init, interval_hours)
    wrf_base = DATA_PATHS["wrf"]["parent"]

    paths = []
    for subdir in sub_dirs:
        subdir_dt = datetime.strptime(subdir, '%Y%m%d%H')
        files = glob.glob(f'{wrf_base}/{subdir_dt.year}/{str(subdir_dt.month).zfill(2)}/{subdir}/{model}/*d02*00')
        files = sorted(files)[6:12] # just take the 6th-12th files for training
        paths.extend(files)
    print('Also only using the 6th-12th files from each directory')
    
    return paths


class ProcessWRF(DataProcess):

    def __init__(self) -> None:
        super().__init__()

    def _preprocess_load(self, ds, vars):
        """
        Preprocesses dataset before loading
        Args:
            ds (xr.Dataset): dataset
            vars (list): variables
        """
        return ds[vars]
    
    def load_ds(self, 
                years =  None,
                months = None,
                time = None,
                filenames = None,
                context_variables: List=None,
                subdirs: List[str]=None,
                ) -> xr.Dataset:
        """ 
        Loads dataset
        Args: 
            var (str): variable
            years (list): specific years, retrieves all if set to None
        """
        if filenames is None:
            if years is None:
                assert time is not None, 'One of years or time must be specified'
                years = np.unique([t.year for t in time])[0]
                months = np.unique([t.month for t in time])

            # if type(years) != int:
            #     ValueError (f'For WRF, years can only be int, not {type(years)}')
            if type(years) == list:
                years = years[0] 
                print(f'Only loading {years}')
            filenames = self.get_filenames(years, months, subdirs=subdirs)

        wrf_vars = [VAR_WRF[var]['var_name'] for var in context_variables]
        partial_preprocess = lambda ds: self._preprocess_load(ds, wrf_vars)


        with ProgressBar():
            ds = xr.open_mfdataset(filenames, 
                                preprocess = partial_preprocess,
                                parallel = True,
                                concat_dim='Time',
                                engine = 'netcdf4',
                                combine = 'nested'
                                )
        # if time is not None:
        #     ds.sel(Time=time)
        print('Loading data from dask')
        with ProgressBar():
            ds = ds.load()
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
        da = ds[VAR_WRF[var]['var_name']]
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
        return ds
    

    def coarsen_da(self, da: xr.DataArray, coarsen_by: int, boundary: str = 'trim'):
        return super().coarsen_da(da, coarsen_by, boundary)


    def get_filenames(self,
                        year: int,
                        months: Literal[tuple(VARIABLE_OPTIONS)],
                        model: str='nz4kmN-ECMWF-SIGMA',
                        subdirs: List[str] = None
                        ):
        paths = []
        for month in months:
            month_str = str(month).zfill(2)
            if subdirs is not None:
                for subdir in subdirs:
                    month_files = glob.glob(f'{DATA_PATHS["wrf"]["parent"]}/{year}/{month_str}/{subdir}/{model}/*d02*00')
                    month_files = sorted(month_files)
                    paths.extend(month_files[6:])
            else:
                month_files = glob.glob(f'{DATA_PATHS["wrf"]["parent"]}/{year}/{month_str}/*/{model}/*d02*00')
                # exclude the first 6 files as they are spinup files
                month_files = sorted(month_files)
                paths.extend(month_files[6:])
        return paths

    def load_ds_time(self, 
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
            year = np.unique([t.year for t in time])
        months = np.unique([t.month for t in time])
        filenames = self.get_filenames(year, months)
        ds = xr.open_mfdataset(filenames,
                               preprocess=self._preprocess_load, parallel=True, engine='netcdf4', combine='nested')
        return ds.sel(time=time)


    def kelvin_to_celsius(self, da: xr.DataArray):
        return da - 273.15

    def regrid_to_topo(self, ds: xr.Dataset, topo: xr.DataArray,) -> xr.Dataset:

        hold = ds.copy()
        hold = hold.where(hold['XLONG'] > 0, drop=True)
        # hold = hold.rename({'XLONG': 'lon', 'XLAT': 'lat'})
        hold.encoding.clear()  # Clear encoding to avoid integer scaling

        ds_out = xr.Dataset({
            'latitude': (['latitude'], topo.latitude.values),
            'longitude': (['longitude'], topo.longitude.values),
        })
        
        method = 'bilinear'
        Ny_in, Nx_in = len(hold.south_north), len(hold.west_east)
        Ny_out, Nx_out = topo['elevation'].shape
        filename = f'{method}_{Ny_in}x{Nx_in}_{Ny_out}x{Nx_out}.nc'
        filepath = os.path.join(DATA_PATHS['regridder_weights']['parent'], filename)
        if os.path.exists(filepath):
            regridder = xe.Regridder(hold.isel(Time=0), 
                                     ds_out, 
                                     'bilinear', 
                                     reuse_weights=True,
                                     filename=filepath
                                     )
        else:
            regridder = xe.Regridder(hold.isel(Time=0), 
                                     ds_out, 
                                     'bilinear', 
                                     reuse_weights=False,
                                     filename=filepath
                                     )
            regridder.to_netcdf(filepath)
        
        interp_hold = regridder(hold)
        return interp_hold.rename({'Time': 'time', 'XTIME': 'time'})


    def regrid_to_topo_old(self, ds: xr.DataArray, topo: xr.DataArray) -> xr.DataArray:
        """
        Generates a high resolution version of the field passed via data.
        Returns:
        """
        hold = ds.copy()
        hold = hold.where(hold['XLONG'] > 0, drop=True)
        hold.encoding.clear()  # Clear encoding to avoid integer scaling

        hires_lats = topo['latitude'].values
        hires_lons = topo['longitude'].values
        x_grid, y_grid = np.meshgrid(hires_lons, hires_lats)

        times = hold['XTIME'].values
        lats = hold['XLAT'].values[0].flatten()
        lons = hold['XLONG'].values[0].flatten()

        def _interp(values, lons, lats, new_lons, new_lats,):
            """
                values (_type_): values to be interpolated
                lons (_type_): lons to be interpolated
                lats (_type_): lats to be interpolated
                new_lons (_type_): interp to these lons
                new_lats (_type_): interp to these lats
            """
            interp = LinearNDInterpolator(list(zip(lons, lats)), values.flatten())
            return interp(new_lons, new_lats)

        def process_variable(values, var):
            num_times = len(times)
            x_flat, y_flat = x_grid.flatten(), y_grid.flatten()
            partial_interp = partial(_interp, lons=lons, lats=lats, 
                                     new_lons=x_flat, new_lats=y_flat)

            vals_interp = [partial_interp(values[i]) for i in range(num_times)]
            vals_interp = np.array(vals_interp).reshape(num_times, len(hires_lats), len(hires_lons))

            return xr.DataArray(vals_interp,
                                coords=[times, hires_lats, hires_lons],
                                dims=['time', 'latitude', 'longitude'],
                                name=var)
        list_of_arrays = []
        for var in tqdm(hold.data_vars, desc='Interpolating WRF data to topography grid'):
            values = hold[var].values
            list_of_arrays.append(process_variable(values, var))
        
        interp_hold = xr.merge(list_of_arrays)

        return interp_hold

    
        

if __name__ == '__main__':
    pass