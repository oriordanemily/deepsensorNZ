import os
from typing import Literal, List
import glob
from scipy.interpolate import griddata

import xarray as xr
from datetime import datetime
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
                years: int,
                months: Literal[tuple(VARIABLE_OPTIONS)],
                context_variables: List=None,
                ) -> xr.Dataset:
        """ 
        Loads dataset
        Args: 
            var (str): variable
            years (list): specific years, retrieves all if set to None
        """
        if type(years) != int:
            ValueError (f'For WRF, years can only be int, not {type(years)}')
        filenames = self.get_filenames(years, months)
        wrf_vars = [VAR_WRF[var]['var_name'] for var in context_variables]
        partial_preprocess = lambda ds: self._preprocess_load(ds, wrf_vars)
        # ds_list = []
        # for file in filenames[:10]:
        #     ds = xr.open_dataset(file)
        #     ds = self._preprocess_load(ds, wrf_vars)
        #     ds_list.append(ds)
        # ds = xr.concat(ds_list)

        print('#### REMEMBER TO CHANGE THIS BACK TO ALL FILENAMES ####')
        print('CURRENTLY INCORRECT: doesnt use all forecasts')
        # could create a 2d time variable: y axis = forecast initialization time, 
        # x axis = forecast hour
        # this will then be flattened in the interpolation step

        # ooooorrrr we just do entirely un-overlapping runs? 

        # can we have multiple time values that are the same?
        # in the pre-processing step, we can add an initialization time dimension
        # this then keeps forecast hours separate I think? 
        with ProgressBar():
            ds = xr.open_mfdataset(filenames[:10], 
                                preprocess = partial_preprocess,
                                parallel = True,
                                concat_dim='Time',
                                engine = 'netcdf4',
                                combine = 'nested'
                                ).load()
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
                        ):
        paths = []
        year = year[0]
        for month in months:
            month_str = str(month).zfill(2)
            month_files = glob.glob(f'{DATA_PATHS["base"]["parent"]}/{year}/{month_str}/*/{model}/*d02*00')
            # exclude the first 6 files as they are spinup files
            month_files = sorted(month_files)
            paths.extend(month_files[6:])
        return paths

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

    def regrid_to_topo(self, ds: xr.DataArray, topo: xr.DataArray) -> xr.DataArray:
        """
        Generates a high-resolution version of the field passed via data.
        Returns:
        """
        hold = ds.copy()
        hold.encoding.clear()  # Clear encoding to avoid integer scaling

        # High-resolution lat/lon from the topography
        hires_lats = topo['latitude'].values
        hires_lons = topo['longitude'].values
        x_grid, y_grid = np.meshgrid(hires_lons, hires_lats)  # Target high-res grid

        times = hold['XTIME'].values
        lats = hold['XLAT'].values[0].flatten()  # Flatten the original latitudes
        lons = hold['XLONG'].values[0].flatten()  # Flatten the original longitudes

        points = np.array([lons, lats]).T  # Original lat/lon points for interpolation
        grid_points = np.array([x_grid.flatten(), y_grid.flatten()]).T  # Target high-res grid points

        list_of_delayed_results = []

        # Interpolation function to be applied in parallel
        def interpolate_chunk(values_chunk):
            """Helper function to apply griddata for each time step chunk."""
            vals_interp = griddata(points, values_chunk.flatten(), grid_points, method='linear')
            return vals_interp.reshape(len(hires_lats), len(hires_lons))

        @delayed
        def process_variable(values):
            """Process a single variable in parallel using Dask."""

            timesteps = values.shape[0]
            values_flat = values.reshape(timesteps, -1)
            vals_interp = griddata(points, values_flat.T, grid_points, method='linear')

            return vals_interp.reshape(timesteps, len(hires_lats), len(hires_lons))

        # Create delayed tasks for each variable
        for var in tqdm(hold.data_vars, desc='Interpolating WRF data to topography grid'):
            values = hold[var].values
            # delayed_result = process_variable(values)
            # list_of_delayed_results.append((var, delayed_result))

        # # Setup LocalCluster
        # cluster = LocalCluster(n_workers=6, threads_per_worker=1, memory_limit='15GB')  # Adjust based on your machine specs
        # client = Client(cluster)  # Connect the client to the cluster

        # # Optional: View the dashboard
        # print(client.dashboard_link)

        # Compute all variables in parallel
        with ProgressBar():
            computed_results = compute(*[delayed_result for _, delayed_result in list_of_delayed_results])

        # After computing, construct DataArrays using the high-resolution grid
        list_of_arrays = []
        for (var, vals_interp) in zip(hold.data_vars, computed_results):
            list_of_arrays.append(xr.DataArray(vals_interp, 
                                            coords=[times, hires_lats, hires_lons], 
                                            dims=['time', 'latitude', 'longitude'], 
                                            name=var))

        # Merge the results
        interp_hold = xr.merge(list_of_arrays)

        return interp_hold

    def regrid_to_topo(self, ds: xr.DataArray, topo: xr.DataArray) -> xr.DataArray:
        """
        Generates a high resolution version of the field passed via data.
        Returns:
        """
        hold = ds.copy()
        hold.encoding.clear()  # Clear encoding to avoid integer scaling

        hires_lats = topo['latitude'].values
        hires_lons = topo['longitude'].values
        x_grid, y_grid = np.meshgrid(hires_lons, hires_lats)

        times = hold['XTIME'].values
        lats = hold['XLAT'].values[0].flatten()
        lons = hold['XLONG'].values[0].flatten()

        points = np.array([lons, lats]).T  # Original lat/lon points for interpolation
        grid_points = np.array([x_grid.flatten(), y_grid.flatten()]).T  # Grid for interpolation
        
        def process_variable_xesmf(values, var):
            regridder = xe.Regridder({'longitude': lons, 'latitude': lats}, 
                                     {'longitude': hires_lons, 'latitude': hires_lats}, 
                                     method='bilinear')
            
            vals_interp = regridder(values)
            return xr.DataArray(vals_interp,
                                coords=[times, hires_lats, hires_lons],
                                dims=['time', 'latitude', 'longitude'],
                                name=var)

        def process_variable(values, var):
            num_times = len(times)
            vals_interp = griddata(points, values.reshape(num_times, -1).T, grid_points, method='linear')
            vals_interp = vals_interp.reshape(num_times, len(hires_lats), len(hires_lons))

            return xr.DataArray(vals_interp,
                                coords=[times, hires_lats, hires_lons],
                                dims=['time', 'latitude', 'longitude'],
                                name=var)
        list_of_arrays = []
        for var in tqdm(hold.data_vars, desc='Interpolating WRF data to topography grid'):
            values = hold[var].values
            list_of_arrays.append(process_variable_xesmf(values, var))
        
        interp_hold = xr.merge(list_of_arrays)

        return interp_hold


    # def regrid_to_topo(self, ds: xr.DataArray, topo: xr.DataArray) -> xr.DataArray:
    #     """
    #     Generates a high resolution version of the field passed via data.
    #     Returns:
    #     """
    #     hold = ds.copy()  # So that we don't overwrite values in the incoming data array.
    #     hold.encoding.clear()  # Just in case there was some scaling to 2-byte integers.

    #     hires_lats = topo['latitude'].values
    #     hires_lons = topo['longitude'].values
    #     x_grid, y_grid = np.meshgrid(hires_lons, hires_lats)

    #     times = hold['XTIME'].values
    #     lats = hold['XLAT'].values.flatten()
    #     lons = hold['XLONG'].values.flatten()
        
    #     partial_interp = partial(LinearNDInterpolator, list(zip(lons, lats)))

    #     list_of_arrays = []
    #     for var in tqdm(hold.data_vars, desc='Interpolating WRF data to topography grid'):
    #         values = hold[var].values.flatten()
    #         interp = partial_interp(values)
    #         vals_interp = interp(x_grid, y_grid)

    #         # Build and return the DataArray.
    #         list_of_arrays.append(xr.DataArray(vals_interp, 
    #                                            coords=[times, hires_lats, hires_lons], 
    #                                            dims=['time', 'latitude', 'longitude'], 
    #                                            name=var))
            
    #     interp_hold = xr.merge(list_of_arrays)
    #     return interp_hold
        

if __name__ == '__main__':
    pass