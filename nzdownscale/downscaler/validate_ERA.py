from nzdownscale.dataprocess import era5, wrf, stations, topography, utils, config
from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train

from deepsensor.model.convnp import ConvNP
from deepsensor.data import construct_circ_time_ds

import xarray as xr
import torch
import pandas as pd
import pickle
from tqdm import tqdm
from datetime import datetime
from typing import Union
import numpy as np

class ValidateERA:
    def __init__(self,
                model_path, 
                data_processor_path,
                task_loader_path, 
                train_metadata_path):
        
        # Load necessary items
        self.model_path = model_path
        print('Unpickling data_processor')
        self.data_processor_dict = self.unpickle(data_processor_path)
        self.data_processor = self.data_processor_dict['data_processor']
        print('Unpickling task_loader')
        self.task_loader = self.unpickle(task_loader_path)
        print('Unpickling train_metadata')
        self.meta = self.unpickle(train_metadata_path)

        # Instantiate classes
        self.top = topography.ProcessTopography()
        self.station = stations.ProcessStations()
        self.base = 'era5' # self.meta['data_settings']['base']
        if self.base == 'era5':
            self.process_era = era5.ProcessERA5()
        else:
            self.process_wrf = wrf.ProcessWRF()

        # Load elevation data
        print('Loading elevation data')
        self.ds_elev = self.top.open_ds()
        high_res_coarsen_factor = self.meta['data_settings']['topography_highres_coarsen_factor']
        self.ds_elev = self.ds_elev.coarsen(latitude=high_res_coarsen_factor, 
                                            longitude=high_res_coarsen_factor,
                                   boundary='trim').mean()
        self.pred_mask = ~np.isnan(self.ds_elev['elevation'])
      
        
        pred_res = np.round(np.abs(np.diff(self.ds_elev.coords['latitude'].values)[0]), 5)
        print('Producing predictions at resolution:', pred_res)


        # Load metadata
        self.var = self.meta['data_settings']['var']
        self.use_daily_data = self.meta['date_info']['use_daily_data']

        # NoneType to start
        self.model = None

    def predict(self, 
                time: Union[datetime, str, list], 
                remove_stations: list = [],
                context_sampling: str = 'all',
                subdirs=None):
        self.load_data(time, remove_stations, subdirs=subdirs)
        self.task_loader = self.create_task_loader()
        if self.model is None:
            self.model = self.load_model()
        
        task = self.task_loader(time, context_sampling=context_sampling)
        pred = self.model.predict(task, 
                                  X_t=self.ds_elev, 
                                  progress_bar=True)

        for key in pred.keys():
            pred[key]['mean'] = pred[key]['mean'].where(self.pred_mask)
            pred[key]['std'] = pred[key]['std'].where(self.pred_mask)
        return pred

    def load_model(self):
        convnp_kwargs = self.meta['convnp_kwargs']

        model = ConvNP(self.data_processor,
                       self.task_loader,
                       **convnp_kwargs)
        
        model.model.load_state_dict(torch.load(self.model_path))

        return model
    
    def unpickle(self, file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    def create_task_loader(self):
        # Update context set to loaded data
        context = self.task_loader.context
        self.task_loader.context = (self.base_ds, 
                                    context[1], 
                                    context[2], 
                                    self.stations_df)
        
        # Update target set to loaded stations
        self.task_loader.target = tuple(self.stations_df)
        return self.task_loader

    def load_data(self, time, remove_stations, subdirs=None):
        self.aux_ds = self.data_processor_dict['aux_ds']
        self.highres_aux_ds = self.data_processor_dict['highres_aux_ds']
        self.landmask_ds = self.data_processor_dict['landmask_ds']

        self.base_ds_raw = self.load_ds(time, subdirs=subdirs)

        stations_df_raw = self.load_stations(time, remove_stations)
        self.stations_df_raw = stations_df_raw

        print(f'Pre-processing {self.base} data')
        base_ds = self.base_ds_raw.copy()
        for var in self.base_ds_raw:
            method = self.data_processor.config[var]['method']
            base_ds[var] = self.data_processor(self.base_ds_raw[var], method=method)
        self.base_ds = base_ds
        self.base_ds = self.add_time_of_year(self.base_ds)

        print('Pre-processing station data')
        method = self.data_processor.config[f"{self.var}_station"]['method']
        self.stations_df = self.data_processor(self.stations_df_raw, method=method)
        if self.var == 'humidity':
            self.stations_df = (self.stations_df + 1) / 2

    def load_stations(self, time, remove_stations=[], keep_stations=[]):
        stations_df = self.station.load_stations_time(self.var, 
                                                      time, 
                                                      remove_stations, 
                                                      keep_stations)
        return stations_df
        
    def load_ds(self, time, subdirs=None):
        ds_list = []

        context_variables = self.meta['data_settings']['context_variables']
        if context_variables[0] != self.var:
            idx = context_variables.index(self.var)
            context_variables[0], context_variables[idx] = context_variables[idx], context_variables[0]
            ds = xr.merge(ds_list)

        if self.base == 'era5':
            for var in tqdm(context_variables, desc=f'Loading {self.base}'):
                base_da = self.process_era.load_ds_time(var, time)
                ds_list.append(base_da)
            ds = xr.merge(ds_list)
            # precip_name = config.VAR_ERA5['precipitation']['var_name']
        
        elif self.base == 'wrf':
            ds = self.process_wrf.load_ds(time=time, 
                                               context_variables=context_variables,
                                               subdirs=subdirs)
            # aux_raw_ds = 
            ds = self.process_wrf.regrid_to_topo(ds, self.aux_ds)
            # precip_name = config.VAR_WRF['precipitation']['var_name']
            # ds = [vars].load()
                # probably need to put interp function here
        
        # ds[precip_name] = np.log10(1 + ds[precip_name])

        ds = self._trim_ds(ds, self.ds_elev)
        
        return ds
        
    def _trim_ds(self, ds, topo):
        # Slice base data to elevation data's spatial extent
        top_min_lat = topo['latitude'].min()
        top_max_lat = topo['latitude'].max()
        top_min_lon = topo['longitude'].min()
        top_max_lon = topo['longitude'].max()

        ds_trimmed = ds.sel(
            latitude=slice(top_max_lat, top_min_lat),
            longitude=slice(top_min_lon, top_max_lon))

        return ds_trimmed

    def add_time_of_year(self, ds):
        """ 
        Add cos_D and sin_D to output dataset to be used as context set, original ds must have the time coordinate. Info: https://alan-turing-institute.github.io/deepsensor/user-guide/convnp.html
        """
        if self.use_daily_data:
            freq='D'
        else:
            freq='H'
        dates = pd.date_range(ds.time.values.min(), ds.time.values.max(), freq=freq)
        doy_ds = construct_circ_time_ds(dates, freq=freq)
        
        ds[f"cos_{freq}"] = doy_ds[f"cos_{freq}"]
        ds[f"sin_{freq}"] = doy_ds[f"sin_{freq}"]
        return ds

    # def plot_timeseries()


    # def plot_ERA5_and_prediction(self, date: str = None, pred = None, era5 = None, location=None, closest_station=False, infer_extent=False, return_fig=False, remove_sea=True):
        # """Plot ERA5, and mean and std of model prediction at given date. 
        
        # Args:
        #     date (str, optional): date for prediction in format 'YYYY-MM-DD'. Default is None, in which case first validation date is used.
        #     location (str or tuple, optional): Location to zoom in on. If str, must be one of the keys in LOCATION_LATLON. If tuple, must be (lat, lon). Defaults to None.
        #     closest_station (bool, optional): If True, find closest station to location and plot that instead. Only used if location is not None. Defaults to False.
        #     infer_extent (bool, optional): Infer extent from data. If False, extent will be taken from config file. Defaults to True.
        #     return_fig (bool, optional): If True, return figure object. Defaults to False.
        # """
        # #setup
        # var = self.get_variable_name('era5')
        # if era5 is None:
        #     era5_raw_ds = self.processed_dict['era5_raw_ds'][var]
        # else:
        #     if isinstance(era5, xr.Dataset):
        #         era5_raw_ds = era5[var]
        #     elif isinstance(era5, xr.DataArray):
        #         era5_raw_ds = era5
        # station_raw_df = self.processed_dict['station_raw_df']

        # # get location if specified
        # if location is not None:
        #     if isinstance(location, str):
        #         X_t = self._get_location_coordinates(location)
        #     else:
        #         X_t = location
            
        #     if closest_station:
        #         # Find closest station to desired target location
        #         X_t = self._find_closest_station(X_t, station_raw_df)

        #     # zoom plot into location
        #     lat_slice = slice(X_t[0] + 2, X_t[0] - 2)
        #     lon_slice = slice(X_t[1] - 2, min(X_t[1] + 2, 180))
        #     era5_raw_ds = era5_raw_ds.sel(latitude=lat_slice, longitude=lon_slice)

        # # format date
        # date = self._format_date(date)

        # # get predictions and test_task
        # if pred is None:
        #     NotImplementedError('Need to implement this: swap era5_raw_ds in commented out line for highres_aux_raw_ds')
        #     # pred_db, _ = self._get_predictions_and_tasks(date, task_loader, model, era5_raw_ds)
        # else:
        #     pred_db = pred.sel(time=date)

        # if location is not None:
        #     lat_slice = slice(X_t[0] - 2, X_t[0] + 2)
        #     pred_db = pred_db.sel(latitude=lat_slice, longitude=lon_slice)

        # # plotting extent
        # if infer_extent:
        #     extent = utils._infer_extent()
        # else:
        #     extent = None

        # era5_var = self.get_variable_name('era5')
        # if era5_var == 't2m':
        #     label = '2m temperature [°C]'
        #     std_unit = '°C'
        # elif era5_var == 'precipitation':
        #     label = 'Precipitation [mm]'
        #     std_unit = 'mm'
        # # use test figure plot
        # fig, axes = self.gen_test_fig(
        #     era5_raw_ds.sel(time=date), 
        #     pred_db["mean"],
        #     pred_db["std"],
        #     add_colorbar=True,
        #     var_cbar_label=label,
        #     std_cbar_label=f"std dev [{std_unit}]",
        #     std_clim=(None, 5),
        #     figsize=(20, 20/3),
        #     fontsize=16,
        #     extent=extent,
        #     remove_sea=remove_sea
        # )

        # if location is not None:
        #     for ax in axes:
        #         ax.scatter(X_t[1], X_t[0], marker="s", color="black", transform=self.crs, s=10**2, facecolors='none', linewidth=2)

        # # fig.suptitle(date)
        # if return_fig:
        #     return fig, axes
