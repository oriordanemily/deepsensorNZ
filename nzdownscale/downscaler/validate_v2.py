from nzdownscale.dataprocess import era5, stations, topography, utils, config
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

class ValidateV2:
    def __init__(self,
                model_path, 
                data_processor_path,
                task_loader_path, 
                train_metadata_path):
        
        # Load necessary items
        self.model_path = model_path
        self.data_processor_dict = self.unpickle(data_processor_path)
        self.data_processor = self.data_processor_dict['data_processor']
        self.task_loader = self.unpickle(task_loader_path)
        self.meta = self.unpickle(train_metadata_path)

        # Instantiate classes
        self.top = topography.ProcessTopography()
        self.station = stations.ProcessStations()
        self.e5 = era5.ProcessERA5()

        # Load elevation data
        self.ds_elev = self.top.open_ds()
        self.ds_elev = self.ds_elev.coarsen(latitude=5, longitude=5,
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
                remove_stations: list = []):
        self.load_data(time, remove_stations)
        self.task_loader = self.create_task_loader()
        if self.model is None:
            self.model = self.load_model()
        
        task = self.task_loader(time)
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
        self.task_loader.context = (self.era5_ds, 
                                    context[1], 
                                    context[2], 
                                    self.stations_df)
        
        # Update target set to loaded stations
        self.task_loader.target = tuple(self.stations_df)
        return self.task_loader

    def load_data(self, time, remove_stations):
        self.aux_ds = self.data_processor_dict['aux_ds']
        self.highres_aux_ds = self.data_processor_dict['highres_aux_ds']
        self.landmask_ds = self.data_processor_dict['landmask_ds']

        self.era5_ds_raw = self.load_era5(time)
        stations_df_raw = self.load_stations(time, remove_stations)

        print('Pre-processing ERA5 data')
        era5_ds = self.era5_ds_raw.copy()
        for var in self.era5_ds_raw:
            method = self.data_processor.config[var]['method']
            era5_ds[var] = self.data_processor(self.era5_ds_raw[var], method=method)
        self.era5_ds = era5_ds
        # self.era5_ds = self.data_processor(self.era5_ds_raw)
        self.ds_era = self.add_time_of_year(self.era5_ds)

        print('Pre-processing station data')
        method = self.data_processor.config[f"{self.var}_station"]['method']
        self.stations_df = self.data_processor(stations_df_raw, method=method)

    def load_stations(self, time, remove_stations=[], keep_stations=[]):
        stations_df = self.station.load_stations_time(self.var, 
                                                      time, 
                                                      remove_stations, 
                                                      keep_stations)
        return stations_df
        
    def load_era5(self, time):
        era5_list = []

        context_variables = self.meta['data_settings']['context_variables']
        if context_variables[0] != self.var:
            idx = context_variables.index(self.var)
            context_variables[0], context_variables[idx] = context_variables[idx], context_variables[0]

        for var in tqdm(context_variables, desc='Loading ERA5'):
            era5_da = self.e5.load_ds_time(var, time)

            if var == 'precipitation':
                era5_da['precipitation'] = np.log10(1 + era5_da['precipitation'])

            era5_list.append(era5_da)

        ds_era = xr.merge(era5_list)
        ds_era = self._trim_era5(ds_era, self.ds_elev)
        
        return ds_era
        
    def _trim_era5(self, da_era, topo):
        # Slice era5 data to elevation data's spatial extent
        top_min_lat = topo['latitude'].min()
        top_max_lat = topo['latitude'].max()
        top_min_lon = topo['longitude'].min()
        top_max_lon = topo['longitude'].max()

        era5_raw_ds = da_era.sel(
            latitude=slice(top_max_lat, top_min_lat),
            longitude=slice(top_min_lon, top_max_lon))

        return era5_raw_ds

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