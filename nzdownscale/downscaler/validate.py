
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
                 validation_date_range: list = None
                 ) -> None:
        """
        Args:
            processed_output_dict (dict, optional):
                Dict output from nzdownscale.downscaler.preprocess.PreprocessForDownscaling.get_processed_output_dict()
            training_output_dict (dict, optional): 
                Dict output from nzdownscale.downscaler.train.Train.get_training_output_dict()
            training_metadata_path (int, optional):
                (If loading pretrained model) Path to dictionary pickle file for training metadata, saved with model e.g. 'models/downscaling/metadata/test_model_1705594143.pkl'
            validation_date_range (list, optional):
                List of two years in format 'YYYY' for start and end of validation period (inclusive) e.g. ['2005', '2006']. Only include if different from model training period.
        """
        
        self.processed_output_dict = processed_output_dict
        self.training_output_dict = training_output_dict
        self.training_metadata_path = training_metadata_path
        self.validation_date_range = validation_date_range
        self.crs = ccrs.PlateCarree()

        self._check_args()


    def _check_args(self):
        if self.training_output_dict is None and self.training_metadata_path is None:
            raise ValueError('Either training_output_dict or training_metadata_path must be provided')
        
        if self.processed_output_dict is None and self.training_metadata_path is None:
            raise ValueError('Either processed_output_dict or training_metadata_path must be provided')
        

    def load_model(self, load_model_path=None):

        self.model_metadata = self.get_metadata()
        
        self._load_processed_dict_data()
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
            metadata['date_info']['val_start_year'] = self.validation_date_range[0]
            metadata['date_info']['val_end_year'] = self.validation_date_range[1]

        return metadata

    def _load_processed_dict_data(self):
        
        if self.processed_output_dict is not None:
            processed_dict = self.processed_output_dict
        else:
            processed_dict = self._get_processed_output_dict_from_metadata()
        
        self.processed_dict = processed_dict 


    def _get_processed_output_dict_from_metadata(self):

        if not hasattr(self, 'model_metadata'):
            self.model_metadata = self.get_metadata()
        
        # ! to do : we don't need to load training data for validation
        data = PreprocessForDownscaling(
            variable = self.model_metadata['data_settings']['var'],
            start_year = self.model_metadata['date_info']['start_year'],
            end_year = self.model_metadata['date_info']['end_year'],
            val_start_year = self.model_metadata['date_info']['val_start_year'],
            val_end_year = self.model_metadata['date_info']['val_end_year'],
            use_daily_data = self.model_metadata['date_info']['use_daily_data'],
            validation = True
        )
        data.run_processing_sequence(
            self.model_metadata['data_settings']['topography_highres_coarsen_factor'],
            self.model_metadata['data_settings']['topography_lowres_coarsen_factor'], 
            self.model_metadata['data_settings']['era5_coarsen_factor'],
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
        model_setup.setup_task_loader()
        model_setup.initialise_model(**self.model_metadata['convnp_kwargs'])
        training_output_dict = model_setup.get_training_output_dict()
        training_output_dict['metadata_dict'] = self.model_metadata

        return training_output_dict


    def _load_pretrained_model(self, model_path):
        model = self._initialise_model()
        model.model.load_state_dict(torch.load(model_path))
        return model
    

    def _initialise_model(self):
        if self.training_output_dict is not None:
            model = self.training_output_dict['model']
            self.training_dict['model'] = model
        else:
            convnp_kwargs = self.training_dict['metadata_dict']['convnp_kwargs']
            model = ConvNP(self.data_processor,
                        self.task_loader, 
                        **convnp_kwargs,
                        ) 
            _ = model(self.training_dict['train_tasks'][0])   # ? need ? 
        return model

    def calculate_loss(self, dates: list or str, 
                       locations: list or str, 
                       function: str = 'l1',
                       predictions=None,
                       save_preds=False,
                       return_pred=False,
                       return_station=False,
                       verbose=True):
        """Calculate loss

        Args:
            dates (listorstr): List of dates for loss to be calculated. 
            locations (listorstr): List of locations. 
            function (str, optional): Function to calculate loss. Defaults to 'l1'.
            predictions (_type_, optional): _description_. Defaults to None.
            return_pred (bool, optional): _description_. Defaults to False.
            return_station (bool, optional): _description_. Defaults to False.
            verbose (bool, optional): _description_. Defaults to True.

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

        if function == 'l1':
            pass # was using np.linalg.norm but replaced for multidate calculations
        elif function == 'l2':
            NotImplementedError('l2 not yet implemented')
        else:
            raise ValueError('function must be one of l1 or l2')
        
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
        print('Dates formatted')

        # get predictions and test_task
        if predictions == None:
            pred, _ = self._get_predictions_and_tasks(dates, task_loader, model, era5_raw_ds, return_dataarray=False, verbose=verbose, save_preds=save_preds)
        else:
            with open(predictions, 'rb') as f:
                pred = pickle.load(f)
        pred_db = pred['dry_bulb']
        pred_db_mean = pred_db['mean'].sel(time=dates)

        # get location if specified
        norms = {}
        if return_pred:
            pred_values = {}
        if return_station:
            station_values = {}

        for location in locations:
            if verbose:
                print(f'Calculating loss for {location}')
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
                station_val = station_raw_df.loc[dates, 'dry_bulb'].loc[pd.IndexSlice[:, X_t[0], X_t[1]]].groupby('time').mean().values.astype('float')
            except:
                raise ValueError(f'No station data for {location} on given date(s): {dates}')
            
            pred_db_mean_location = pred_db_mean.sel(latitude=X_t[0], longitude=X_t[1], method='nearest').values.astype('float')
            if station_val.shape != pred_db_mean_location.shape:
                raise ValueError(f'Station and prediction shapes do not match for {location} on given date(s). Station shape: {station_val.shape}, prediction shape: {pred_db_mean_location.shape}')
            norms[location] = np.abs(pred_db_mean_location - station_val)
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

    def plot_ERA5_and_prediction(self, date: str = None, location=None, closest_station=False, infer_extent=False, return_fig=False):
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
        era5_raw_ds = self.processed_dict['era5_raw_ds']
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
        pred_db, _ = self._get_predictions_and_tasks(date, task_loader, model, era5_raw_ds)
        
        # plotting extent
        if infer_extent:
            extent = utils._infer_extent()
        else:
            extent = None

        # use test figure plot
        fig, axes = self.gen_test_fig(
            era5_raw_ds.sel(time=date), 
            pred_db["mean"],
            pred_db["std"],
            add_colorbar=True,
            var_cbar_label="2m temperature [°C]",
            std_cbar_label="std dev [°C]",
            std_clim=(None, 2),
            figsize=(20, 20/3),
            extent=extent
        )

        if location is not None:
            for ax in axes:
                ax.scatter(X_t[1], X_t[0], marker="s", color="black", transform=self.crs, s=10**2, facecolors='none', linewidth=2)

        fig.suptitle(date)
        if return_fig:
            return fig, axes

    def plot_prediction_with_stations(self, date: str = None, location=None, closest_station=False, zoom_to_location=False, infer_extent=False, return_fig=False, labels=None):

        # setup
        task_loader = self.task_loader
        model = self.model
        era5_raw_ds = self.processed_dict['era5_raw_ds']
        station_raw_df = self.processed_dict['station_raw_df']

        if infer_extent:
            extent = self._infer_extent()

        date = self._format_date(date)

        pred_db, _ = self._get_predictions_and_tasks(date, task_loader, model, era5_raw_ds)
        
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
                                   predictions=None,
                                   closest_station=True,
                                   return_fig=False):

        # ! need to implement something to indicate how far ERA5/convnp location is from closest station. 
        # For now, this function plots ERA5/convnp at the closest station by default (when closest_station=True).

        # setup
        if predictions == None:
            task_loader = self.task_loader
            model = self.model
        era5_raw_ds = self.processed_dict['era5_raw_ds']
        station_raw_df = self.processed_dict['station_raw_df']

        # get location
        if isinstance(location, str):
            X_t = self._get_location_coordinates(location)
        else:
            X_t = location

        dates = pd.date_range(date_range[0], date_range[1])

        if predictions == None:
            pred_db, _ = self._get_predictions_and_tasks(dates, task_loader, model, era5_raw_ds)
        else:
            print('Loading predictions from file')
            with open(predictions, 'rb') as f:
                pred = pickle.load(f)
            pred_db = pred['dry_bulb'].sel(time=dates)

        if closest_station:
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

        # era5_raw_df = era5_raw_ds.sel(latitude=X_t[0], longitude=X_t[1], method="nearest").to_dataframe()
        # era5_raw_df = era5_raw_df.loc[dates]
        # era5_raw_df
        era5_raw_ds = era5_raw_ds.sel(time=dates, latitude=X_t[0], longitude=X_t[1], method="nearest")

        sns.set_style("white")

        convnp_mean = pred_db["mean"].sel(latitude=X_t[0], longitude=X_t[1], method='nearest').values.astype('float')
        stddev = pred_db["std"].sel(latitude=X_t[0], longitude=X_t[1], method='nearest').values.astype('float')
        # era5_vals = era5_raw_df["t2m"].values.astype('float')
        era5_vals = era5_raw_ds.values.astype('float')

        # Plot mean
        fig, ax = plt.subplots(1, 1, figsize=(7*.9, 3*.9))
        ax.plot(convnp_mean, label="ConvNP", marker="o", markersize=3)
        # Make 95% confidence interval
        ax.fill_between(range(len(convnp_mean)), convnp_mean - 2 * stddev, convnp_mean + 2 * stddev, alpha=0.25, label="ConvNP 95% CI")
        ax.plot(era5_vals, label="ERA5", marker="o", markersize=3)
        # Plot true station data
        ax.plot(station_closest_df["dry_bulb"].values.astype('float'), label="Station", marker="o", markersize=3)
        # Add legend
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=4, mode="expand", borderaxespad=0)
        ax.set_xlabel("Time")
        ax.set_ylabel("2m temperature [°C]")
        ax.set_xticks(range(len(era5_raw_ds.time))[::14])
        ax.set_xticklabels([str(i)[:10] for i in era5_raw_ds.time.values[::14]], rotation=15)
        ax.set_title(f"ConvNP prediction for {location}", y=1.15)

        if return_fig:
            return fig, ax

    def gen_test_fig(self, era5_ds_plot=None, mean_ds=None, std_ds=None, samples_ds=None, add_colorbar=False, var_clim=None, std_clim=None, var_cbar_label=None, std_cbar_label=None, fontsize=None, figsize=(15, 5), extent=None):
        # Plots ERA5, ConvNP mean, ConvNP std dev

        if extent is not None:
            NotImplementedError('extent not yet implemented')

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

        fig, axes = plt.subplots(1, ncols, subplot_kw=dict(projection=self.crs), figsize=figsize)

        axis_i = 0
        if era5_ds_plot is not None:
            if era5_ds_plot.shape == (0, 0) or era5_ds_plot.shape == (1, 1):
                raise ValueError('era5_ds_plot resolution too coarse to plot at this scale. Please use a higher resolution or a larger region.')
            ax = axes[axis_i]
            era5_ds_plot.plot(ax=ax, cmap="jet", vmin=vmin, vmax=vmax, add_colorbar=add_colorbar, cbar_kwargs=dict(label=var_cbar_label))
            ax.set_title("ERA5", fontsize=fontsize)

        if mean_ds is not None:
            axis_i += 1
            ax = axes[axis_i]
            mean_ds.plot(ax=ax, cmap="jet", vmin=vmin, vmax=vmax, add_colorbar=add_colorbar, cbar_kwargs=dict(label=var_cbar_label))
            ax.set_title("ConvNP mean", fontsize=fontsize)

        if samples_ds is not None:
            for i in range(samples_ds.shape[0]):
                axis_i += 1
                ax = axes[axis_i]
                samples_ds.isel(sample=i).plot(ax=ax, cmap="jet", vmin=vmin, vmax=vmax, add_colorbar=add_colorbar, cbar_kwargs=dict(label=var_cbar_label))
                ax.set_title(f"ConvNP sample {i+1}", fontsize=fontsize)

        if std_ds is not None:
            axis_i += 1
            ax = axes[axis_i]
            std_ds.plot(ax=ax, cmap="Greys", add_colorbar=add_colorbar, vmin=std_vmin, vmax=std_vmax, cbar_kwargs=dict(label=std_cbar_label))
            ax.set_title("ConvNP std dev", fontsize=fontsize)

        for ax in axes:
            ax.add_feature(cf.BORDERS)
            ax.coastlines()
        return fig, axes
    
    ### Plotting utils
    def _format_date(self, date: str = None):
        if date is None:
            val_start_year = self.processed_dict['date_info']['val_start_year']
            date = f"{val_start_year}-01-01T00:00:00.000000000"
        else:
            date = f'{date}T00:00:00.000000000'
        return date
    
    def _get_predictions_and_tasks(self, dates, task_loader, model, era5_raw_ds, return_dataarray=True, verbose=False, save_preds=False):
        if isinstance(dates, str):
            dates = [dates]

        test_task = task_loader(dates, ["all", "all"], seed_override=42)
        if verbose:
            print('Test tasks loaded')
        if len(dates) == 1:
            test_task = test_task[0]
        if verbose:
            print('Calculating predictions...')
        pred = model.predict(test_task, X_t=era5_raw_ds.sel({'time': dates}), resolution_factor=2)

        if save_preds:
            with open(f'predictions_{self.training_metadata_path.split("/")[-1]}', 'wb') as f:
                pickle.dump(pred, f)
        # pred is of type deepsensor.model.pred.Prediction

        if return_dataarray:
            pred = pred['dry_bulb']

        return pred, test_task
    
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

# %%
