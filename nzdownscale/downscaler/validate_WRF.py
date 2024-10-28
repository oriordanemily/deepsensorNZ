from multiprocessing import process
from click import pass_context
from nzdownscale.dataprocess import era5, wrf, stations, topography, utils, config
from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train

from deepsensor.model.convnp import ConvNP
from deepsensor.data import construct_circ_time_ds

import xarray as xr
import glob
import torch
import pandas as pd
import pickle
from tqdm import tqdm
from datetime import datetime
from typing import Union
import numpy as np

from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.dataprocess import config, config_local

class ValidateWRF:
    def __init__(self,
                model_path, 
                data_processor_path,
                task_loader_path, 
                train_metadata_path):

        # Load necessary items
        self.model_path = model_path
        print('Loading data processor')
        self.data_processor_dict = self.unpickle(data_processor_path)
        print('Loading task loader')
        self.task_loader = self.unpickle(task_loader_path)
        print('Loading metadata')
        self.meta = self.unpickle(train_metadata_path)

        # Unpack data processor
        self.data_processor = self.data_processor_dict['data_processor']
        self.aux_ds = self.data_processor_dict['aux_ds']
        # self.aux_raw_ds = self.data_processor_dict['aux_raw_ds']
        self.highres_aux_ds = self.data_processor_dict['highres_aux_ds']
        self.landmask_ds = self.data_processor_dict['landmask_ds']

        # Unpack data settings
        self.base = self.meta['data_settings']['base']
        self.variable = self.meta['data_settings']['var']
        self.context_variables = self._order_context_variables(self.meta['data_settings']['context_variables'])

        self.data = PreprocessForDownscaling(base='wrf',
                                             variable = self.variable,
                                             context_variables=self.context_variables,
                                             validation=True)
        self.data.data_processor = self.data_processor

        # Unpack date info
        if self.base == 'era5':
            self.use_daily_data = self.meta['date_info']['use_daily_data']

        # Load model
        # print('Loading model')
        self.model = None#self.load_model()

        # Get topography data
        self.get_topo_data()

    
    def get_topo_data(self):
        self.top = topography.ProcessTopography()
        self.ds_elev = self.top.open_ds()
        self.ds_elev = self.ds_elev.coarsen(latitude=5, 
                                            longitude=5, 
                                            boundary='trim').mean()
        self.pred_mask = ~np.isnan(self.ds_elev['elevation'])
        pred_res = np.round(np.abs(np.diff(self.ds_elev.coords['latitude'].values)[0]), 5)
        print('Producing predictions at resolution:', pred_res)
        
    def get_filepaths(self, forecast_init, forecast_end=-1, forecast_start=0, model='nz4kmN-ECMWF-SIGMA'):
        base_dir = config_local.DATA_PATHS['wrf']['parent']
        year = str(forecast_init.year)
        month = str(forecast_init.month).zfill(2)
        day = str(forecast_init.day).zfill(2)
        hour = str(forecast_init.hour).zfill(2)

        subdir = f'{year}{month}{day}{hour}'

        path = f'{base_dir}/{year}/{month}/{subdir}/{model}/'

        all_files = glob.glob(f'{path}/*d02*00')
        all_files.sort()

        return all_files[forecast_start:forecast_end]
    
    def load_wrf(self, 
                  forecast_init: datetime=None, 
                  forecast_end: int=-1, 
                  forecast_start: int=0, 
                  filepaths: list=None,
                  model: str='nz4kmN-ECMWF-SIGMA'):
        """Load the WRF data and process it

        Args:
            forecast_init (datetime, optional): datetime.datetime of wrf initialisation. Can be None if list of paths is provided. Defaults to None.
            forecast_end (int, optional): How many hours of the forecast to load. Defaults to -1 (all hours).
            forecast_start (int, optional): Skip this many hours at the start of the forecast. Defaults to 0.
            filepaths (list, optional): Can provide a list filepaths instead of time details. Defaults to None.
            model (str, optional): Which model to use. Defaults to 'nz4kmN-ECMWF-SIGMA'.

        Returns:
            ds (xr.Dataset): WRF data processed and ready for prediction
        """
        # Instantiate wrf process class
        self.process_wrf = wrf.ProcessWRF()

        if filepaths is None:
            # Get filepaths
            assert forecast_init is not None, 'forecast_init or paths must be provided'
            filepaths = self.get_filepaths(forecast_init, forecast_end, forecast_start, model)
    
        # Load data
        ds = self.process_wrf.load_ds(filenames=filepaths, 
                                      context_variables=self.context_variables)
        ds = self.process_wrf.regrid_to_topo(ds, self.aux_raw_ds)
        self.original_data = ds.copy()

        for var in ds.data_vars:
            processor_method = self.data_processor.config[var]['method']
            ds[var] = self.data_processor(ds[var], method=processor_method)
        
        ds = self.add_time_of_year(ds)
        self.ds = ds.copy()
        
        return ds
    
    def load_stations(self, times, remove_stations=[], keep_stations=[]):
        self.station = stations.ProcessStations()
        stations_df = self.station.load_stations_time(self.variable, 
                                                times, 
                                                remove_stations, 
                                                keep_stations)
        self.stations_raw_df = stations_df.copy()

        processing_method = self.data_processor.config[f"{self.variable}_station"]['method']
        reset_index_stations = stations_df.reset_index()
        latitude = reset_index_stations['latitude'].values
        longitude = reset_index_stations['longitude'].values
        original_values = reset_index_stations[f"{self.variable}_station"].values
        stations_df = self.data_processor(stations_df, method=processing_method)
        stations_df[f"{self.variable}_station_original"] = original_values
        stations_df['latitude'] = latitude
        stations_df['longitude'] = longitude
        stations_df = stations_df.reset_index().set_index(['time', 'x1', 'x2', 'station_name', 'latitude', 'longitude'])
        self.stations_df = stations_df.copy()

        return stations_df

    def predict(self,
                filepaths,
                remove_stations=[],
                station_sampling=True,):
        
        self.data.all_paths = filepaths
        self.data.run_processing_sequence(topography_highres_coarsen_factor=5,
                                          topography_lowres_coarsen_factor=5,
                                        include_time_of_year=True,
                                        include_landmask=True,
                                        remove_stations=remove_stations,
                                        save_data_processor_dict=False,
                                        data_processor_dict=self.data_processor_dict,
                                        station_as_context=station_sampling)
        self.data.validation_fpaths = filepaths
        processed_output_dict = self.data.get_processed_output_dict(validation=True)

        self.ds = processed_output_dict['base_ds']
        stations_df = processed_output_dict['station_df']
        times = self.ds.time.values
        context_sampling = ['all', 'all', 'all', 'all']
        if station_sampling:
            context_sampling[-1] = 'all'
        else:
            context_sampling[-1] = 0

        task_loader = self.amend_task_loader(self.ds, stations_df)
        task = task_loader(list(times), context_sampling=context_sampling)        

        if self.model is None:
            self.model = self.load_model()

        pred = self.model.predict(task,
                                X_t = self.ds_elev,
                                progress_bar = True)

        for key in pred.keys():
            pred[key]['mean'] = pred[key]['mean'].where(self.pred_mask)
            pred[key]['std'] = pred[key]['std'].where(self.pred_mask)
        return pred


        # self.data.load_wrf()
        # base_raw_ds = self.data.preprocess_wrf()



    # def predict(self, 
    #             # forecast_init: datetime=None,
    #             # forecast_end: int = -1,
    #             # forecast_start: int = 0,
    #             filepaths=None,
    #             model = 'nz4kmN-ECMWF-SIGMA',
    #             remove_stations: list = [],
    #             station_sampling: bool = True,):

    #     # self.data.run_processing_sequence(topography_highres_coarsen_factor=5,
    #     #                                 topography_lowres_coarsen_factor=5,
    #     #                                 # era5_coarsen_factor=5,
    #     #                                 include_time_of_year=True,
    #     #                                 include_landmask=True,
    #     #                                 remove_stations=remove_stations,
    #     #                                 save_data_processor_dict=False,
    #     #                                 data_processor_dict=self.data_processor_dict,
    #     #                                 station_as_context=station_sampling)
    #     # processed_output_dict = self.data.get_processed_output_dict()

    #     # Load wrf data
    #     # ds = self.load_wrf(forecast_init=forecast_init, 
    #     #                     forecast_end=forecast_end, 
    #     #                     forecast_start=forecast_start, 
    #     #                     filepaths=filepaths, 
    #     #                     model=model)
    #     # print('REMBMBER TO CHANGE THIS BACK')
    #     # ds = xr.open_dataset('/home/emily/deepsensor/deepweather-downscaling/experiments/deepsensor/train/valid_example.nc')
    #     # ds = ds.drop_vars(['cos_H', 'sin_H'])
    #     # times = ds.time.values

    #     # # Load stations
    #     # stations_df = self.load_stations(times,
    #     #                                 remove_stations,
    #     #                                 )

    #     context_sampling = ['all', 'all', 'all', 'all']
    #     if station_sampling:
    #         context_sampling[-1] = 'all'
    #     else:
    #         context_sampling[-1] = 0
        
    #     ds = processed_output_dict['base_ds']
    #     times = ds.time.values
    #     stations_df = processed_output_dict['stations_df']
    #     task_loader = self.amend_task_loader(ds, stations_df)
    #     task = task_loader(list(times), context_sampling=context_sampling)        

    #     pred = self.model.predict(task,
    #                             X_t = self.ds_elev,
    #                             progress_bar = True)

    #     for key in pred.keys():
    #         pred[key]['mean'] = pred[key]['mean'].where(self.pred_mask)
    #         pred[key]['std'] = pred[key]['std'].where(self.pred_mask)
    #     return pred

    # def load_aux_data(self,):
    #     self.aux_ds = self.data_processor_dict['aux_ds']
    #     self.aux_raw_ds = self.data_processor_dict['aux_raw_ds']
    #     self.highres_aux_ds = self.data_processor_dict['highres_aux_ds']
    #     self.landmask_ds = self.data_processor_dict['landmask_ds']


    def load_model(self):
        convnp_kwargs = self.meta['convnp_kwargs']

        model = ConvNP(self.data_processor,
                       self.task_loader,
                       **convnp_kwargs)

        model.model.load_state_dict(torch.load(self.model_path))

        return model    
    
    def amend_task_loader(self, ds, stations_df):
        context = self.task_loader.context

        task_loader = self.task_loader
        task_loader.context = (ds,
                            context[1],
                            context[2],
                            stations_df)
        
        # if task_loader.target_var_IDs[0] != f'{self.variable}_station':
        station_id = stations_df.columns.values[0]
        if station_id != f'{self.variable}_station':
            Warning(f"First column in station_df is not '{self.variable}_station', it is '{station_id}'")
        context_id = list(task_loader.context_var_IDs)
        context_id[-1] = (station_id,)
        task_loader.context_var_IDs = tuple(context_id)
        task_loader.target_var_IDs = [(station_id,)]

        # should the target be stations?
        # task_loader.target = None

        return task_loader

    def unpickle(self, file):
        with open(file, 'rb') as f:
            return pickle.load(f)
        
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

    def _order_context_variables(self, context_variables):
        # Order context variables
        if context_variables[0] != self.variable:
            idx = context_variables.index(self.variable)
            context_variables[0], context_variables[idx] = context_variables[idx], context_variables[0]

        return context_variables

