
#%% 

import logging
logging.captureWarnings(True)
import os
import time

import xarray as xr
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import seaborn as sns
from scipy.ndimage import gaussian_filter
import torch
import cartopy.feature as cfeature
import pickle

import deepsensor.torch
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.model.convnp import ConvNP
from deepsensor.active_learning.algorithms import GreedyAlgorithm
from deepsensor.active_learning.acquisition_fns import Stddev
from deepsensor.train.train import train_epoch, set_gpu_default_device
from deepsensor.data.utils import construct_x1x2_ds
from tqdm import tqdm

from nzdownscale.downscaler.train import Train
from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.dataprocess import era5, stations, topography, utils, config
from nzdownscale.dataprocess.config import LOCATION_LATLON, STATION_LATLON


class ValidateV1:
    def __init__(self,
                 processed_output_dict: dict = None,
                 training_output_dict: dict = None,
                 training_metadata_path: str = None,
                 validation_date_range: list = None,
                 data_processor_dict: dict = None
                 ) -> None:
        """
        Args:
            processed_output_dict (dict, optional):
                Dict output from nzdownscale.downscaler.preprocess.PreprocessForDownscaling.get_processed_output_dict()
            training_output_dict (dict, optional): 
                Dict output from nzdownscale.downscaler.train.Train.get_training_output_dict()
            training_metadata_path (int, optional):
                (If loading pretrained model) Path to dictionary pickle file for training metadata, saved with model e.g. 'experiments/models/downscaling/metadata/test_model_1705594143.pkl'
            validation_date_range (list, optional):
                List of two years in format 'YYYY' for start and end of validation period (inclusive) e.g. ['2005', '2006']. Only include if different from model training period.
            data_processor_dict (dict, optional):
                Dict output from nzdownscale.downscaler.preprocess.PreprocessForDownscaling.process_all_for_training()
        """
        
        self.processed_output_dict = processed_output_dict
        self.training_output_dict = training_output_dict
        self.training_metadata_path = training_metadata_path
        self.validation_date_range = validation_date_range
        self.data_processor_dict = data_processor_dict
        self.crs = ccrs.PlateCarree()

        self.highres_aux_raw_ds = None

        self._check_args()


    def _check_args(self):
        if self.training_output_dict is None and self.training_metadata_path is None:
            raise ValueError('Either training_output_dict or training_metadata_path must be provided')
        
        if self.processed_output_dict is None and self.training_metadata_path is None:
            raise ValueError('Either processed_output_dict or training_metadata_path must be provided')
        

    def load_model(self, load_model_path=None, save_data_processing_dict=None):

        self.model_metadata = self.get_metadata()
        
        self._load_processed_dict_data(save_data_processing_dict=save_data_processing_dict)
        self._load_training_dict_data()

        if load_model_path is not None:
            self.model = self._load_pretrained_model(load_model_path)
        else:
            if self.training_output_dict is None: 
                raise ValueError('training_output_dict is required if not loading pretrained model. Please provide load_model_path or training_output_dict during class instantiation')
            self.model = self.training_output_dict['model']

    def get_metadata(self):
        if self.training_metadata_path is not None:
            metadata = utils.open_pickle(self.training_metadata_path)
        elif self.training_output_dict is not None:
            metadata = self.training_output_dict['metadata_dict']

        if self.validation_date_range is not None:
            # metadata['date_info']['val_start_year'] = self.validation_date_range[0]
            # metadata['date_info']['val_end_year'] = self.validation_date_range[1]
            metadata['date_info']['validation_years'] = self.validation_date_range

        return metadata

    def _load_processed_dict_data(self, save_data_processing_dict=None):
        
        if self.processed_output_dict is not None:
            processed_dict = self.processed_output_dict
        else:
            processed_dict = self._get_processed_output_dict_from_metadata(save_data_processing_dict=save_data_processing_dict)

        self.processed_dict = processed_dict 

    def _get_processed_output_dict_from_metadata(self, save_data_processing_dict=None):

        if not hasattr(self, 'model_metadata'):
            self.model_metadata = self.get_metadata()
        
        # can run preprocessed data once and try different models
        self.data = PreprocessForDownscaling(
            variable = self.model_metadata['data_settings']['var'],
            # start_year = self.model_metadata['date_info']['start_year'],
            # end_year = self.model_metadata['date_info']['end_year'],
            # val_start_year = self.model_metadata['date_info']['val_start_year'],
            # val_end_year = self.model_metadata['date_info']['val_end_year'],
            training_years = self.model_metadata['date_info']['training_years'],
            validation_years = self.model_metadata['date_info']['validation_years'],
            use_daily_data = self.model_metadata['date_info']['use_daily_data'],
            context_variables = self.model_metadata['data_settings']['context_variables'],
            validation = True
        )
        data = self.data

        data.run_processing_sequence(
            self.model_metadata['data_settings']['topography_highres_coarsen_factor'],
            self.model_metadata['data_settings']['topography_lowres_coarsen_factor'], 
            self.model_metadata['data_settings']['era5_coarsen_factor'],
            include_time_of_year=True,
            include_landmask=True,
            data_processor_dict=self.data_processor_dict,
            save_data_processor_dict=save_data_processing_dict,
            station_as_context='all',#self.model_metadata['station_as_context'],
            )
        processed_output_dict = data.get_processed_output_dict()

        return processed_output_dict


    def _load_training_dict_data(self):

        if self.training_output_dict is None:
            training_dict = self._get_training_output_dict_from_metadata()
        else:
            training_dict = self.training_output_dict
        
        self.data_processor = training_dict['data_processor']
        self.task_loader = training_dict['task_loader']
        self.val_tasks = training_dict['val_tasks']

        self.training_dict = training_dict


    def _get_training_output_dict_from_metadata(self):

        if not hasattr(self, 'model_metadata'):
            self.model_metadata = self.get_metadata()

        model_setup = Train(processed_output_dict=self.processed_dict)
        model_setup.setup_task_loader(validation=True)
        model_setup.initialise_model(**self.model_metadata['convnp_kwargs'])
        training_output_dict = model_setup.get_training_output_dict()
        training_output_dict['metadata_dict'] = self.model_metadata

        return training_output_dict


    def _load_pretrained_model(self, model_path):
        model = self._initialise_model()
        model.model.load_state_dict(torch.load(model_path))
        return model
    

    def _initialise_model(self, model_path=None):
        if self.training_output_dict is not None:
            model = self.training_output_dict['model']
            self.training_dict['model'] = model
        else:
            convnp_kwargs = self.training_dict['metadata_dict']['convnp_kwargs']
            if model_path is not None:
                model = ConvNP(self.data_processor,
                            self.task_loader, 
                            model_ID=model_path,
                            **convnp_kwargs,
                            ) 
            else:
                model = ConvNP(self.data_processor,
                            self.task_loader, 
                            **convnp_kwargs,
                            )
            _ = model(self.training_dict['val_tasks'][0])   # ? need ? 
        return model

    def calculate_loss(self, dates: list or str, 
                       locations: list or str, 
                       pred=None,
                       return_pred=False,
                       return_station=False,
                       verbose=True,
                       era=False):
        """Calculate loss

        Args:
            dates (listorstr): List of dates for loss to be calculated. 
            locations (listorstr): List of locations. 
            pred (_type_, optional): _description_. Defaults to None.
            return_pred (bool, optional): _description_. Defaults to False.
            return_station (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to True.
            era (bool, optional): If True, assumes you've given it ERA5 to calculate loss. Defaults to False.

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """        

        if isinstance(dates, pd._libs.tslibs.timestamps.Timestamp) or isinstance(dates, str):
            dates = [dates]
        
        if not isinstance(locations, list):
            locations = [locations]
        
        if verbose:
            print('Setting up...')
        # setup
        task_loader = self.task_loader
        model = self.model
        era5_raw_ds = self.processed_dict['era5_raw_ds']
        print('ERA5 data loaded')
        station_raw_df = self.processed_dict['station_raw_df']
        print('Station data loaded')

        # format date
        dates = [self._format_date(date) for date in dates]

        # get predictions and test_task
        if not isinstance(pred, xr.core.dataset.Dataset):
            pred, _ = self._get_predictions_and_tasks(dates, task_loader, model, era5_raw_ds, return_dataarray=True, verbose=verbose, save_preds=False)

        if era:
            key = self.get_variable_name('era5')
        else:
            key = 'mean'
        pred_db_mean = pred[key].sel(time=dates)

        # get location if specified
        norms = {}
        if return_pred:
            pred_values = {}
        if return_station:
            station_values = {}

        station_var = self.get_variable_name('station')
        for location in tqdm(locations, desc='Calculating losses'):
            # if verbose:
            #     print(f'Calculating loss for {location}')
            norms[location] = []
            if return_pred:
                pred_values[location] = []
            if return_station:
                station_values[location] = []
            if isinstance(location, str):
                if location in STATION_LATLON:
                    X_t = self._get_location_coordinates(location, station=True)
                else:
                    X_t = self._get_location_coordinates(location)
            else:
                X_t = location

            if location not in STATION_LATLON:
                # Find closest station to desired target location
                X_t = self._find_closest_station(X_t, station_raw_df)

            #get station values
            try:            
                station_val = station_raw_df.loc[dates, station_var].loc[pd.IndexSlice[:, X_t[0], X_t[1]]].groupby('time').mean().values.astype('float')
            except:
                raise ValueError(f'No station data for {location} on given date(s): {dates}')
            
            if era:
                pred_db_mean_location = pred_db_mean.sel(x1=X_t[0], x2=X_t[1], method='nearest').values.astype('float')
            else:
                pred_db_mean_location = pred_db_mean.sel(latitude=X_t[0], longitude=X_t[1], method='nearest').values.astype('float')
                
            if station_val.shape != pred_db_mean_location.shape:
                raise ValueError(f'Station and prediction shapes do not match for {location} on given date(s). Station shape: {station_val.shape}, prediction shape: {pred_db_mean_location.shape}')
            norms[location] = [utils.rmse(station, pred) for station, pred in zip(station_val, pred_db_mean_location)]
            if return_pred:
                pred_values[location] = pred_db_mean_location
            if return_station:
                station_values[location] = station_val

        items_to_return = [norms]
        if return_pred:
            items_to_return.append(pred_values)
        if return_station:
            items_to_return.append(station_values)

        if len(items_to_return) == 1:
            return items_to_return[0]
        else:
            return items_to_return
        
    def calculate_loss_era5(self, dates, locations, era5):
        station_raw_df = self.processed_dict['station_raw_df']

        era5_name = self.get_variable_name('era5')
        station_name = self.get_variable_name('station')

        if isinstance(era5, xr.Dataset):
            era5_values = era5[era5_name].sel(time=dates)
        else:
            era5_values = era5.sel(time=dates)

        norms_era5 = {}
        
        locations_kept = []
        for location in tqdm(locations, desc='Calculating losses'):
            # print(location)
            try:
                X_t = self._get_location_coordinates(location, station=True)
                station_val = station_raw_df.loc[dates, station_name].loc[pd.IndexSlice[:, X_t[0], X_t[1]]].groupby('time').mean().values.astype('float')
            
                era5_loc = era5_values.sel(latitude=X_t[0], longitude=X_t[1], method='nearest')
                if np.isnan(era5_loc).any():
                    # if era5 is nan, skip location
                    print(f'ERA5 has NANs at {location}: {sum(np.isnan(era5_loc).values)}/{len(era5_loc)}')
                else:
                    era5_rmse = [utils.rmse(station, era) for station, era in zip(station_val, era5_loc.values)]
                    norms_era5[location] = era5_rmse
                    locations_kept.append(location)
            except:
                print(f"Couldn't find station {location}")
    
        locations_kept = list(set(locations_kept))
        return norms_era5, locations_kept

    def calculate_loss_era5_pred(self, dates, locations, pred, era5=None, verbose=True):
        station_raw_df = self.processed_dict['station_raw_df']

        pred_db_mean = pred['mean'].sel(time=dates)

        if era5:
            era5_name = self.get_variable_name('era5')
            era5_values = era5[era5_name].sel(time=dates)

        norms = {}
        norms_era5 = {}
        station_var = self.get_variable_name('station')

        for location in tqdm(locations, desc='Calculating losses'):
            skip_location = False
            norms[location] = []
            if isinstance(location, str):
                if location in STATION_LATLON:
                    X_t = self._get_location_coordinates(location, station=True)
                else:
                    X_t = self._get_location_coordinates(location)
            else:
                X_t = location
            try:            
                station_val = station_raw_df.loc[dates, station_var].loc[pd.IndexSlice[:, X_t[0], X_t[1]]].groupby('time').mean().values.astype('float')
            except:
                raise ValueError(f'No station data for {location} on given date(s): {dates}')
            
            if era5:
                era5_loc = era5_values.sel(x1=X_t[0], x2=X_t[1], method='nearest')#.values.astype('float')
            
            pred_db_mean_loc = pred_db_mean.sel(latitude=X_t[0], longitude=X_t[1], method='nearest').values.astype('float')

            if era5:
                for station, era in zip(station_val, era5_loc):
                    skip_location = False
                    era5_rmse = utils.rmse(station, era)
                    print(location, era5_rmse)
                    if np.isnan(era5_rmse):
                        skip_location=True
                    else:
                        if location not in norms_era5:
                            norms_era5[location] = []
                        if location not in norms:
                            norms[location] = []
                        norms_era5[location].append(era5_rmse)
                        norms[location].append(utils.rmse(station, pred_db_mean_loc))


            # if not skip_location:
            #     norms_era5[location] = [utils.rmse(station, era5) for station, era5 in zip(station_val, era5_loc)]
        
            norms[location] = [utils.rmse(station, pred) for station, pred in zip(station_val, pred_db_mean_loc)]
            if np.isnan(norms[location]).any():
                print(f'Nans in norms for {location}: {sum(np.isnan(norms[location]))}/{len(norms[location])}')
                
        if era5:
            return norms, norms_era5
        else:
            return norms
                


    def plot_losses(self, dates, loss_dict,
                    pred_dict=None, station_dict=None, 
                    location=None, return_fig=False,
                    return_loss=False):
        """_summary_

        Args:
            dates (_type_): dates to plot
            loss_dict (_type_): dictionary of losses, output from calculate_loss
            pred_dict (_type_, optional): dictionary of predictions, output from calculate_loss. Defaults to None.
            station_dict (_type_, optional): dictionary of station values, output from calculate_loss. Defaults to None.
            location (_type_, optional): location to plot for. If None, calculates average over all stations
            return_fig (bool, optional): return fig. Defaults to False.

        """
        
        fig, ax = plt.subplots()

        if location is None:
            losses = np.nansum([i for i in loss_dict.values()], axis = 0)
        else:
            losses = loss_dict[location]
        ax.plot(dates, losses, color='b', label = 'losses')
        ax.set_ylabel('Losses (pred - station)', color='b')

        create_ax2 = False
        if pred_dict is not None or station_dict is not None:
            create_ax2 = True
            ax2 = ax.twinx()
            ax2.set_ylabel('Temperature (C)')
            if pred_dict is not None:
                if location is None:
                    preds = np.nanmean([i for i in pred_dict.values()], axis = 0)
                else:
                    preds = pred_dict[location]
                ax2.plot(dates, preds, color='r', label='preds', alpha=0.5)
            
            if station_dict is not None:
                if location is None:
                    stations = np.nanmean([i for i in station_dict.values()], axis = 0)
                else:
                    stations = station_dict[location]
                ax2.plot(dates, stations, color='g', label='stations', alpha=0.5)
            ax2.legend(loc='upper right')

        ax.legend(loc='upper left')
        
        if location is None:
            ax.set_title(f'Sum of losses for all stations from {dates[0]} to {dates[-1]}')
        else:
            ax.set_title(f'Losses for {location} from {dates[0]} to {dates[-1]}')
        
        if return_fig:
            if create_ax2:
                return fig, ax, ax2
            else:
                return fig, ax

    def stations_in_date_range(self, date_range):
        """Check if station is fully available for given date range

        Args:
            date_range (tuple): (start_date, end_date)
        """
        dates = pd.date_range(date_range[0], date_range[1])
        station_raw_df = self.processed_dict['station_raw_df']
        station_raw_df_dates = station_raw_df.loc[dates]

        dict_location = {}
        success = True
        keep_locations = []

        for location in STATION_LATLON.keys():
            if isinstance(location, str):
                X_t = self._get_location_coordinates(location, station=True)
            try:
                # is location in station_raw_df?
                dict_location[location] = station_raw_df_dates.loc[pd.IndexSlice[:, X_t[0], X_t[1]], :]
                # is location available for all dates?
                for d in dates:
                    if not success:
                        break
                    if d not in dict_location[location].index:
                        success = False
            except:
                success = False

            if success:
                if len(dict_location[location]) > 1:
                    keep_locations.append(location)
            success = True

        print(f'{len(keep_locations)}/{len(STATION_LATLON)} locations kept')
        return keep_locations
    
    ### Plotting functions

    def plot_nationwide_prediction(self, date: str = None, infer_extent=False, return_fig=False):
        """Plot mean and std of model prediction

        Args:
            date (str, optional): date for prediction in format 'YYYY-MM-DD'. Default is None, in which case first validation date is used.
            infer_extent (bool, optional): Infer extent from data. If False, extent will be taken from config file. Defaults to True.
            return (bool, optional): If True, return figure object. Defaults to False.
        """

        # ! to do: add option to plot different times once we're training hourly data

        # setup
        task_loader = self.task_loader
        model = self.model
        era5_raw_ds = self.processed_dict['era5_raw_ds']
        data_processor = self.data_processor
        crs = self.crs 

        # format date
        date = self._format_date(date)

        # get predictions and test_task
        pred, test_task = self._get_predictions_and_tasks(date, task_loader, model, era5_raw_ds, return_dataarray=False)
        
        # plotting extent
        if infer_extent:
            extent = self._infer_extent()
        else:
            extent = None
        
        # plot 
        fig = deepsensor.plot.prediction(pred, date, data_processor, task_loader, test_task, crs=crs, extent=extent)

        if return_fig:
            return fig
    
    def plot_errors_at_stations(self, date: str = None, pred = None, remove_sea=True, var_clim=None, diff_clim=None):
        #setup
        import cartopy.crs as ccrs
        from matplotlib.colors import TwoSlopeNorm
        task_loader = self.task_loader
        model = self.model
        era5_raw_ds = self.processed_dict['era5_raw_ds']
        station_raw_df = self.processed_dict['station_raw_df']
        fontsize=16

        date_str = date
        date = self._format_date(date)

                # get predictions and test_task
        if pred is None:
            pred_db, _ = self._get_predictions_and_tasks(date, task_loader, model, era5_raw_ds)
        else:
            pred_db = pred.sel(time=date)

        filtered_station_df = station_raw_df.loc[date]
        filtered_station_df = filtered_station_df.reset_index()

        filtered_station_df['prediction'] = np.nan
        filtered_station_df['differences'] = np.nan
        station_var = self.get_variable_name('station')
        # Iterate over the filtered DataFrame
        for index, row in filtered_station_df.iterrows():
            prediction = pred_db.sel(time=date, 
                                    latitude=row['latitude'], 
                                    longitude=row['longitude'],
                                    method='nearest')['mean'].values.item()
            
            # Add the prediction to the DataFrame
            filtered_station_df.at[index, 'prediction'] = prediction
            filtered_station_df.at[index, 'differences'] = row[station_var] - prediction

        ncols = 3

        fig, axes = plt.subplots(1, ncols, subplot_kw=dict(projection=self.crs), figsize=(20, 20/3))
        station_var = self.get_variable_name('station')
        ax_content = [station_var, 'prediction', 'differences']
        for i, ax in enumerate(axes):
            # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            ax.set_extent([166, 179, -47.5, -34], crs=ccrs.PlateCarree())  # Set the extent to cover New Zealand

            # Add features to the map
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            
            station_values = filtered_station_df[ax_content[0]]
            station_vmax = station_values.max()
            station_vmin = station_values.min()

            # Centre the cmap at 0
            if ax_content[i] == 'differences':
                data_values = filtered_station_df[ax_content[i]]
                vmax = max(data_values.max(), abs(data_values.min()))
                vmin = -vmax
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                scatter = ax.scatter(filtered_station_df['longitude'], filtered_station_df['latitude'], c=filtered_station_df[ax_content[i]],
                            cmap='coolwarm', marker='o', norm = norm, edgecolor='k', linewidth=0.5, s=100, transform=ccrs.PlateCarree())
            else:
                scatter = ax.scatter(filtered_station_df['longitude'], filtered_station_df['latitude'], c=filtered_station_df[ax_content[i]],
                            cmap='coolwarm', marker='o', edgecolor='k', vmin=station_vmin, vmax=station_vmax, linewidth=0.5, s=100, transform=ccrs.PlateCarree())

            # Add a colorbar
            if station_var == 'dry_bulb':
                label = 'Dry Bulb Temperature (°C)'
            elif station_var == 'precipitation':
                label = 'Precipitation (mm)'
            plt.colorbar(scatter, ax=ax, shrink=0.5, label=label)

            cbar = ax.collections[0].colorbar  # This assumes your plot is the first (or only) collection added to the axes
            # cbar.set_label(var_cbar_label, fontsize=fontsize)  # Set your desired fontsize here
            cbar.ax.tick_params(labelsize=fontsize) 
            ax.add_feature(cf.BORDERS)
            ax.coastlines()
            ax.tick_params(axis='both', labelsize=fontsize)
            if ax_content[i] == 'prediction':
                ax.set_title(f'Prediction at {date_str}', fontsize=fontsize)
            else:
                ax.set_title(ax_content[i], fontsize=fontsize)

        print('Average difference:' , filtered_station_df['differences'].mean())
        print('Std difference:', filtered_station_df['differences'].std())
        return fig, axes


    def plot_stations_and_prediction(self, date: str = None, pred = None,remove_sea=True):
        # task_loader = self.task_loader
        # model = self.model
        station_raw_df = self.processed_dict['station_raw_df']

        date = self._format_date(date)

        pred_db = pred.sel(time=date)

        filtered_station_df = station_raw_df.loc[date]
        filtered_station_df = filtered_station_df.reset_index()
        station_var = self.get_variable_name('station')
        station_values = filtered_station_df[station_var]
        station_vmax = station_values.max()
        station_vmin = station_values.min()

        fig, axes = plt.subplots(1, 3, subplot_kw=dict(projection=self.crs), figsize=(20, 20/3))
        
        for ax in axes:
            # Add features to the map
            ax.add_feature(cfeature.LAND)
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.BORDERS, linestyle=':')

        if station_var == 'precipitation':
            cmap = 'viridis'
        else:
            cmap='coolwarm'

        ### PLOT STATIONS
        axes[0].scatter(filtered_station_df['longitude'], filtered_station_df['latitude'], c=filtered_station_df[station_var],
                        cmap=cmap, marker='o', edgecolor='k', vmin=station_vmin, vmax=station_vmax, linewidth=0.5, s=100, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(axes[0].collections[0], ax=axes[0], shrink=1, label=station_var)
        axes[0].set_title('Stations', fontsize=14)

        ## PLOT PREDICTIONS
        mean_ds = pred_db['mean']
        if remove_sea:
            topo = self.processed_dict['highres_aux_ds']['elevation']
            topo_unnorm = self.data_processor.unnormalise(topo)
            interpolated_topo = topo_unnorm.interp_like(mean_ds)
            land_sea_mask = ~(interpolated_topo == 0)
            mean_ds = mean_ds.where(land_sea_mask)

        mean_ds.plot(ax=axes[1], cmap=cmap, vmin=station_vmin, vmax=station_vmax,
                        add_colorbar=True)
                    # cbar_kwargs=cbar_kwargs)
        axes[1].set_title("ConvNP mean", fontsize=14)

        ## PLOT STD
        # std_ds = pred_db['std']
        # if remove_sea:
        #     std_ds = std_ds.where(land_sea_mask)
        # std_ds.plot(ax=axes[2], cmap='Greys', vmin=0, vmax=5,
        #                 add_colorbar=True)
        #             # cbar_kwargs=cbar_kwargs)
        # axes[2].set_title("ConvNP std", fontsize=14)

        ## PLOT ERA
        # era5_var = self.get_variable_name('era5')
        era5 = self.processed_dict['era5_raw_ds']
        era5_values = era5.sel(time=date)
        # if remove_sea:
        #     era5_values = era5_values.where(land_sea_mask)
        era5_values.plot(ax=axes[2], cmap=cmap, vmin=station_vmin, vmax=station_vmax,
                        add_colorbar=True)
                    # cbar_kwargs=cbar_kwargs)
        axes[2].set_title("ERA5", fontsize=14)

        return fig, axes



    def plot_ERA5_and_prediction(self, date: str = None, pred = None, era5 = None, location=None, closest_station=False, infer_extent=False, return_fig=False, remove_sea=True):
        """Plot ERA5, and mean and std of model prediction at given date. 
        
        Args:
            date (str, optional): date for prediction in format 'YYYY-MM-DD'. Default is None, in which case first validation date is used.
            location (str or tuple, optional): Location to zoom in on. If str, must be one of the keys in LOCATION_LATLON. If tuple, must be (lat, lon). Defaults to None.
            closest_station (bool, optional): If True, find closest station to location and plot that instead. Only used if location is not None. Defaults to False.
            infer_extent (bool, optional): Infer extent from data. If False, extent will be taken from config file. Defaults to True.
            return_fig (bool, optional): If True, return figure object. Defaults to False.
        """
        #setup
        task_loader = self.task_loader
        model = self.model
        var = self.get_variable_name('era5')
        if era5 is None:
            era5_raw_ds = self.processed_dict['era5_raw_ds'][var]
        else:
            if isinstance(era5, xr.Dataset):
                era5_raw_ds = era5[var]
            elif isinstance(era5, xr.DataArray):
                era5_raw_ds = era5
        station_raw_df = self.processed_dict['station_raw_df']

        # get location if specified
        if location is not None:
            if isinstance(location, str):
                X_t = self._get_location_coordinates(location)
            else:
                X_t = location
            
            if closest_station:
                # Find closest station to desired target location
                X_t = self._find_closest_station(X_t, station_raw_df)

            # zoom plot into location
            lat_slice = slice(X_t[0] + 2, X_t[0] - 2)
            lon_slice = slice(X_t[1] - 2, min(X_t[1] + 2, 180))
            era5_raw_ds = era5_raw_ds.sel(latitude=lat_slice, longitude=lon_slice)

        # format date
        date = self._format_date(date)

        # get predictions and test_task
        if pred is None:
            NotImplementedError('Need to implement this: swap era5_raw_ds in commented out line for highres_aux_raw_ds')
            # pred_db, _ = self._get_predictions_and_tasks(date, task_loader, model, era5_raw_ds)
        else:
            pred_db = pred.sel(time=date)

        if location is not None:
            lat_slice = slice(X_t[0] - 2, X_t[0] + 2)
            pred_db = pred_db.sel(latitude=lat_slice, longitude=lon_slice)

        # plotting extent
        if infer_extent:
            extent = utils._infer_extent()
        else:
            extent = None

        era5_var = self.get_variable_name('era5')
        if era5_var == 't2m':
            label = '2m temperature [°C]'
            std_unit = '°C'
        elif era5_var == 'precipitation':
            label = 'Precipitation [mm]'
            std_unit = 'mm'
        # use test figure plot
        fig, axes = self.gen_test_fig(
            era5_raw_ds.sel(time=date), 
            pred_db["mean"],
            pred_db["std"],
            add_colorbar=True,
            var_cbar_label=label,
            std_cbar_label=f"std dev [{std_unit}]",
            std_clim=(None, 5),
            figsize=(20, 20/3),
            fontsize=16,
            extent=extent,
            remove_sea=remove_sea
        )

        if location is not None:
            for ax in axes:
                ax.scatter(X_t[1], X_t[0], marker="s", color="black", transform=self.crs, s=10**2, facecolors='none', linewidth=2)

        # fig.suptitle(date)
        if return_fig:
            return fig, axes

    def plot_prediction_with_stations(self, date: str = None, location=None, pred=None, closest_station=False, zoom_to_location=False, infer_extent=False, return_fig=False, labels=None):

        # setup
        task_loader = self.task_loader
        model = self.model
        era5_raw_ds = self.processed_dict['era5_raw_ds']
        station_raw_df = self.processed_dict['station_raw_df']

        if infer_extent:
            extent = self._infer_extent()

        date = self._format_date(date)

        if pred == None:
            pred_db, _ = self._get_predictions_and_tasks(date, task_loader, model, era5_raw_ds)
        else:
            pred_db = pred.sel(time=date)

        if location is not None:
            if isinstance(location, str):
                X_t = self._get_location_coordinates(location)
            else:
                X_t = location

            if closest_station:
                # Find closest station to desired target location
                X_t = self._find_closest_station(X_t, station_raw_df)

            if zoom_to_location:
                lat_slice = slice(X_t[0] + 2, X_t[0] - 2)
                lon_slice = slice(X_t[1] - 2, min(X_t[1] + 2, 180))
                pred_db = pred_db.sel(latitude=lat_slice, longitude=lon_slice)
            
        # Plot prediction mean
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=self.crs), figsize=(20, 20))
        pred_db['mean'].plot(ax=ax, cmap="jet")
        ax.coastlines()
        ax.add_feature(cf.BORDERS)
        
        if location is not None:
            ax.scatter(X_t[1], X_t[0], transform=self.crs, color="black", marker="*", s=200)
            size_of_stations = 60
        else:
            import matplotlib as mpl
            size_of_stations = mpl.rcParams['lines.markersize'] ** 2
        
        # Plot station locations
        locs = self._get_station_locations(station_raw_df)
        ax.scatter([loc[1] for loc in locs], [loc[0] for loc in locs], transform=self.crs, color="red", marker=".", s=size_of_stations)

        if zoom_to_location:
            ax.set_extent([lon_slice.start, lon_slice.stop, lat_slice.start, lat_slice.stop])
        elif infer_extent:
            ax.set_extent([extent['minlon'], extent['maxlon'], extent['minlat'], extent['maxlat']])

        if labels is not None:
            for location, losses in labels.items():
                ax.text(float(losses.longitude), float(losses.latitude), str(np.round(np.nanmean(losses.values),2)))

        if return_fig:
            return fig, ax

    def plot_timeseries_comparison(self, 
                                   location, 
                                   date_range: tuple, 
                                   pred=None,
                                   era5=None,
                                   closest_station=True,
                                   return_fig=False):

        # ! need to implement something to indicate how far ERA5/convnp location is from closest station. 
        # For now, this function plots ERA5/convnp at the closest station by default (when closest_station=True).

        # setup
        if not isinstance(pred, xr.core.dataset.Dataset):
            task_loader = self.task_loader
            model = self.model
        if type(era5) != xr.core.dataset.Dataset:
            era5_raw_ds = self.processed_dict['era5_raw_ds']
        else:
            era5_name = self.get_variable_name('era5')
            era5_raw_ds = era5[era5_name]
        station_raw_df = self.processed_dict['station_raw_df']
        station_var = self.get_variable_name('station')

        # get location
        if isinstance(location, str):
            X_t = self._get_location_coordinates(location)
        else:
            X_t = location

        dates = pd.date_range(date_range[0], date_range[1])

        if not isinstance(pred, xr.core.dataset.Dataset):
            pred_db, _ = self._get_predictions_and_tasks(dates, task_loader, model, era5_raw_ds)
        else:
            pred_db = pred.sel(time=dates)

        X_station_closest = self._find_closest_station(X_t, station_raw_df)

        # Get station data on dates in date_range
        station_closest_df = station_raw_df.reset_index().set_index(["latitude", "longitude"]).loc[tuple(X_station_closest)].set_index("time")
        # intersection = station_closest_df.index.intersection(dates)
        # if intersection.empty:
        #     raise ValueError(f"Station {X_station_closest} has no data for dates {dates}")
        # else:
        station_closest_df = station_closest_df.loc[dates]
        if closest_station:
            X_t = X_station_closest

        era5_raw_ds = era5_raw_ds.sel(time=dates, latitude=X_t[0], longitude=X_t[1], method="nearest")

        sns.set_style("white")

        convnp_mean = pred_db["mean"].sel(latitude=X_t[0], longitude=X_t[1], method='nearest').values.astype('float')
        stddev = pred_db["std"].sel(latitude=X_t[0], longitude=X_t[1], method='nearest').values.astype('float')
        era5_vals = era5_raw_ds.values.astype('float')

        # Plot mean
        fig, ax = plt.subplots(1, 1, figsize=(7*.9, 3*.9))
        ax.plot(convnp_mean, label="ConvNP", marker="o", markersize=3, zorder=2)
        # Make 95% confidence interval
        ax.fill_between(range(len(convnp_mean)), convnp_mean - 2 * stddev, convnp_mean + 2 * stddev, alpha=0.25, label="ConvNP 95% CI")
        # Plot true station data
        ax.plot(station_closest_df[station_var].values.astype('float'), label="Station", marker="o", markersize=3, zorder=1)
        ax.plot(era5_vals, label="ERA5", marker="o", markersize=3, zorder = 0)
        # Add legend
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=4, mode="expand", borderaxespad=0)
        ax.set_xlabel("Time")
        ax.set_ylabel("2m temperature [°C]",)

        dates_df = pd.DataFrame(index=pd.to_datetime(era5_raw_ds.time.values))
        first_of_each_month = dates_df.groupby([dates_df.index.year, dates_df.index.month]).apply(lambda x: x.index.min())
        positions = [dates.get_loc(pd.to_datetime(day)) for day in first_of_each_month]

        ax.set_xticks(positions, 
                      labels=[str(i)[:10] for i in first_of_each_month],
                    rotation=25)
        ax.tick_params(axis='x', which='major', length=10, width=1, 
                       direction='out')

        # ax.set_xticks(positions,)
        # ax.set_xticklabels([str(i)[:10] for i in era5_raw_ds.time.values[::30]], rotation=45)
        ax.set_title(f"ConvNP prediction for {location}", y=1.15)

        if return_fig:
            return fig, ax

    def gen_test_fig(self, era5_ds_plot=None, mean_ds=None, std_ds=None, 
                     samples_ds=None, add_colorbar=False, var_clim=None, 
                     std_clim=None, var_cbar_label=None, std_cbar_label=None,
                       fontsize=None, figsize=(15, 5), extent=None, 
                       remove_sea=True):
        # Plots ERA5, ConvNP mean, ConvNP std dev
        
        if extent is not None:
            NotImplementedError('extent not yet implemented')

        if remove_sea:
            topo = self.processed_dict['highres_aux_ds']['elevation']
            topo_unnorm = self.data_processor.unnormalise(topo)
            interpolated_topo = topo_unnorm.interp_like(mean_ds)
            land_sea_mask = ~(interpolated_topo == 0)

        if var_clim is None:
            vmin = np.array(mean_ds.min())
            vmax = np.array(mean_ds.max())
        else:
            vmin, vmax = var_clim

        if std_clim is None and std_ds is not None:
            std_vmin = np.array(std_ds.min())
            std_vmax = np.array(std_ds.max())
        elif std_clim is not None:
            std_vmin, std_vmax = std_clim
        else:
            std_vmin = None
            std_vmax = None

        ncols = 0
        if era5_ds_plot is not None:
            ncols += 1
        if mean_ds is not None:
            ncols += 1
        if std_ds is not None:
            ncols += 1
        if samples_ds is not None:
            ncols += samples_ds.shape[0]

        cmap = 'RdYlBu_r'

        fig, axes = plt.subplots(1, ncols, subplot_kw=dict(projection=self.crs), figsize=figsize)

        cbar_kwargs = {"location": 'right', 
                'pad': 0.06,
                'shrink': 0.8,
                'extend': 'both',
                'label': var_cbar_label}
        
        axis_i = 0
        if era5_ds_plot is not None:
            if era5_ds_plot.shape == (0, 0) or era5_ds_plot.shape == (1, 1):
                raise ValueError('era5_ds_plot resolution too coarse to plot at this scale. Please use a higher resolution or a larger region.')
            ax = axes[axis_i]
            era5_ds_plot.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                               add_colorbar=add_colorbar, 
                               cbar_kwargs=cbar_kwargs)
            ax.set_title("ERA5-Land", fontsize=fontsize)

        if mean_ds is not None:
            axis_i += 1
            ax = axes[axis_i]
            if remove_sea:
                mean_ds = mean_ds.where(land_sea_mask)
            mean_ds.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                          add_colorbar=add_colorbar,
                        cbar_kwargs=cbar_kwargs)
            ax.set_title("ConvNP mean", fontsize=fontsize)

        if samples_ds is not None:
            for i in range(samples_ds.shape[0]):
                axis_i += 1
                ax = axes[axis_i]
                samples_ds.isel(sample=i).plot(ax=ax, cmap="jet", vmin=vmin,
                                            vmax=vmax, 
                                            add_colorbar=add_colorbar, 
                                            cbar_kwargs=cbar_kwargs)
                ax.set_title(f"ConvNP sample {i+1}", fontsize=fontsize)

        if std_ds is not None:
            axis_i += 1
            ax = axes[axis_i]
            if remove_sea:
                std_ds = std_ds.where(land_sea_mask)
            std_ds.plot(ax=ax, cmap="Greys", add_colorbar=add_colorbar, 
                        vmin=std_vmin, vmax=std_vmax, 
                        cbar_kwargs=cbar_kwargs)
            ax.set_title("ConvNP std dev", fontsize=fontsize)

        for ax in axes:
            cbar = ax.collections[0].colorbar  # This assumes your plot is the first (or only) collection added to the axes
            cbar.set_label(var_cbar_label, fontsize=fontsize)  # Set your desired fontsize here
            cbar.ax.tick_params(labelsize=fontsize) 
            ax.add_feature(cf.BORDERS)
            ax.coastlines()
            ax.tick_params(axis='both', labelsize=fontsize)
        return fig, axes
    
    
    ### Plotting utils
    def _format_date(self, date: str = None):
        if date is None:
            # val_start_year = self.processed_dict['date_info']['val_start_year']
            validation_years = self.processed_dict['date_info']['validation_years']
            date = f"{validation_years[0]}-01-01T00:00:00.000000000"
        else:
            date = f'{date}T00:00:00.000000000'
        return date
    
    def get_predictions(self, dates, model, verbose=False, save_preds=False, remove_stations_from_tasks=[]):
        task_loader = self.task_loader
        # era5_raw_ds = self.processed_dict['era5_raw_ds']
        # highres_aux_raw_ds = self.processed_dict['highres_aux_raw_ds']
        if self.highres_aux_raw_ds is None:
            topo = self.processed_dict['highres_aux_ds']
            self.highres_aux_raw_ds = self.data_processor.unnormalise(topo)
        
        # era5_raw_ds = era5_raw_ds.sel({'time': dates})

        pred, _ = self._get_predictions_and_tasks(dates, task_loader, model, 
                                                  self.highres_aux_raw_ds,
                                                    return_dataarray=True, 
                                                    verbose=verbose, save_preds=save_preds,
                                                      remove_stations_from_tasks=remove_stations_from_tasks)

        return pred
    
    def _get_predictions_and_tasks(self, dates, task_loader, model, highres_aux_raw_ds, return_dataarray=True, verbose=False, save_preds=False, remove_stations_from_tasks=[]):
        if isinstance(dates, str):
            dates = [dates]
        if verbose:
            print('Loading test tasks...')
        
        # remove stations in remove_stations_from_tasks from the station data to be used in test tasks
        station_df = self.processed_dict['station_raw_df'].copy()
        self.processed_dict['station_raw_df'] = self._remove_stations_from_station_df(station_df, remove_stations_from_tasks)
        
        # if self.model_metadata['station_as_context']:
        #     context_sampling = ['all', 'all', 'all', 'all']
        # else:
        #     context_sampling = ['all', 'all', 'all']
        # feed context sampling into task_loader if needed

        test_task = task_loader(dates, target_sampling='all',  seed_override=42)
        if verbose:
            print('Test tasks loaded')
        if len(dates) == 1:
            test_task = test_task[0]
        if verbose:
            print('Calculating predictions...')
        pred = model.predict(test_task, X_t=highres_aux_raw_ds, progress_bar=1)
         
        # reset station data to include all for validation
        self.processed_dict['station_raw_df'] = station_df

        if save_preds:
            utils.save_pickle(pred, f'predictions_{self.training_metadata_path.split("/")[-1]}_{dates[0]}.pkl')

        if return_dataarray:
            var_ID = self.task_loader.target_var_IDs[0][0]
            pred = pred[var_ID]

        return pred, test_task

    def _remove_stations_from_station_df(self, station_df, stations):
        """Remove named stations from the station df

        Args:
            station_df (pd.DataFrame): df with time, latitude, longitude and data columns
            stations (list): list of station names (in STATION_LATLON) to be removed

        Returns:
            pd.DataFrame: df with stations removed
        """

        # create df of latitudes and longitudes of stations to be removed
        latlon_remove = {'latitude': [], 'longitude': []}
        for station in stations:
            latlon_remove['latitude'].append(STATION_LATLON[station]['latitude'])
            latlon_remove['longitude'].append(STATION_LATLON[station]['longitude'])
        removal_df = pd.DataFrame(latlon_remove)

        # merge station_df with removal_df to remove stations
        merged_df = station_df.reset_index().merge(removal_df, on=['latitude', 'longitude'], how='left', indicator=True).set_index(['time', 'latitude', 'longitude'])
        final_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        print(f"Removed the following stations from station_df: {stations}"        )
        return final_df
    
    def _infer_extent(self):
        """Get extent from config file. Return as tuple (minlon, maxlon, minlat, maxlat)
        """
        from nzdownscale.dataprocess.config import PLOT_EXTENT

        extent = PLOT_EXTENT['all']
        return (extent['minlon'], extent['maxlon'], extent['minlat'], extent['maxlat'])

    def _get_location_coordinates(self, location, station=False):
        if location in STATION_LATLON:
            station = True
            latlon_dict = STATION_LATLON
        else:
            latlon_dict = LOCATION_LATLON
        if location not in latlon_dict:
            raise ValueError(f"Location {location} not in {latlon_dict}, please set X_t manually")
        if station: 
            X_t = np.array([float(latlon_dict[location]['latitude']), float(latlon_dict[location]['longitude'])])
        else:
            X_t = latlon_dict[location]

        return X_t

    def _find_closest_station(self, coordinates, stations_raw_df):
        locs = self._get_station_locations(stations_raw_df)
        
        # Find closest station to desired target location
        X_station_closest = min(locs, key=lambda loc: np.linalg.norm(np.array(loc) - coordinates))
        X_t = np.array(X_station_closest)

        return X_t
    
    def _get_station_locations(self, stations_raw_df):
        locs = set(zip(stations_raw_df.reset_index()["latitude"], stations_raw_df.reset_index()["longitude"]))
        return locs

    def get_variable_name(self, dataset):
        if dataset == 'era5':
            from nzdownscale.dataprocess.config import VAR_ERA5 as VAR_DICT
        elif dataset == 'station':
            from nzdownscale.dataprocess.config import VAR_STATIONS as VAR_DICT

        variable = self.data.var
        return VAR_DICT[variable]['var_name']
# %%
