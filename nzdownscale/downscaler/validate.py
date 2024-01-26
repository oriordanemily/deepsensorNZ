
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
from nzdownscale.dataprocess.config import LOCATION_LATLON


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
        
        data = PreprocessForDownscaling(
            variable = self.model_metadata['data_settings']['var'],
            start_year = self.model_metadata['date_info']['start_year'],
            end_year = self.model_metadata['date_info']['end_year'],
            val_start_year = self.model_metadata['date_info']['val_start_year'],
            val_end_year = self.model_metadata['date_info']['val_end_year'],
            use_daily_data = self.model_metadata['date_info']['use_daily_data'],
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

    def infer_extent(self):
        """Get extent from config file. Return as tuple (minlon, maxlon, minlat, maxlat)
        """
        from nzdownscale.dataprocess.config import PLOT_EXTENT

        extent = PLOT_EXTENT['all']
        return (extent['minlon'], extent['maxlon'], extent['minlat'], extent['maxlat'])


    def plot_nationwide_prediction(self, date: str = None, infer_extent=False, return_fig=False):
        """Plot mean and std of model prediction

        Args:
            date (str, optional): date for prediction in format 'YYYY-MM-DD'. Default is None, in which case first validation date is used.
            infer_extent (bool, optional): Infer extent from data. If False, extent will be taken from config file. Defaults to True.
            return (bool, optional): If True, return figure object. Defaults to False.
        """

        # ! to do: add option to plot different times once we're training hourly data

        task_loader = self.task_loader
        model = self.model
        data_processor = self.data_processor
        if date is None:
            val_start_year = self.processed_dict['date_info']['val_start_year']
            date = f"{val_start_year}-01-01T00:00:00.000000000"
        else:
            date = f'{date}T00:00:00.000000000'
        test_task = task_loader(date, ["all", "all"], seed_override=42)
        pred = model.predict(test_task, X_t=self.processed_dict['era5_raw_ds'], resolution_factor=1)

        if infer_extent:
            extent = self.infer_extent()
        else:
            extent = None
        
        fig = deepsensor.plot.prediction(pred, date, data_processor, task_loader, test_task, crs=ccrs.PlateCarree(), extent=extent)

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
        task_loader = self.task_loader
        model = self.model
        era5_raw_ds = self.processed_dict['era5_raw_ds']

        if location is not None:
            if isinstance(location, str):
                if location not in LOCATION_LATLON:
                    raise ValueError(f"Location {location} not in LOCATION_LATLON, please set X_t manually")
                X_t = LOCATION_LATLON[location]
            else:
                X_t = location
            
            # Get ERA5 data at location
            station_raw_df = self.processed_dict['station_raw_df']
            locs = set(zip(station_raw_df.reset_index()["latitude"], station_raw_df.reset_index()["longitude"]))
            
            if closest_station:
                # Find closest station to desired target location
                X_station_closest = min(locs, key=lambda loc: np.linalg.norm(np.array(loc) - X_t))
                X_t = np.array(X_station_closest)
            lat_slice = slice(X_t[0] + 2, X_t[0] - 2)
            lon_slice = slice(X_t[1] - 2, min(X_t[1] + 2, 180))
            era5_raw_ds = era5_raw_ds.sel(latitude=lat_slice, longitude=lon_slice)

        if date is None:
            val_start_year = self.processed_dict['date_info']['val_start_year']
            date = f"{val_start_year}-01-01T00:00:00.000000000"
        else:
            date = f'{date}T00:00:00.000000000'
        test_task = task_loader(date, ["all", "all"], seed_override=42)

        pred = model.predict(test_task, X_t=era5_raw_ds, resolution_factor=1)
        pred_db = pred['dry_bulb']
        
        if infer_extent:
            extent = self.infer_extent()
        else:
            extent = None

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
                ax.scatter(X_t[1], X_t[0], marker="s", color="black", transform=ccrs.PlateCarree(), s=10**2, facecolors='none', linewidth=2)

        if return_fig:
            return fig, axes

    def gen_test_fig(self, era5_ds_plot=None, mean_ds=None, std_ds=None, samples_ds=None, add_colorbar=False, var_clim=None, std_clim=None, var_cbar_label=None, std_cbar_label=None, fontsize=None, figsize=(15, 5), extent=None):
        # Plots ERA5, ConvNP mean, ConvNP std dev
        crs = ccrs.PlateCarree()

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

        fig, axes = plt.subplots(1, ncols, subplot_kw=dict(projection=crs), figsize=figsize)

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

    def plot_prediction_with_stations(self, date: str = None, location=None, closest_station=False, 
    zoom_to_location=False, infer_extent=False, return_fig=False):

        task_loader = self.task_loader
        model = self.model
        era5_raw_ds = self.processed_dict['era5_raw_ds']
        
        if date is None:
            val_start_year = self.processed_dict['date_info']['val_start_year']
            date = f"{val_start_year}-01-01T00:00:00.000000000"
        else:
            date = f'{date}T00:00:00.000000000'

        test_task = task_loader(date, ["all", "all"], seed_override=42)
        pred = model.predict(test_task, X_t=era5_raw_ds, resolution_factor=2)
        pred_db = pred['dry_bulb']

        # Get ERA5 data at location
        station_raw_df = self.processed_dict['station_raw_df']
        locs = set(zip(station_raw_df.reset_index()["latitude"], station_raw_df.reset_index()["longitude"]))
        
        if location is not None:
            if isinstance(location, str):
                if location not in LOCATION_LATLON:
                    raise ValueError(f"Location {location} not in LOCATION_LATLON, please set X_t manually")
                X_t = LOCATION_LATLON[location]
            else:
                X_t = location

        if closest_station:
            # Find closest station to desired target location
            X_station_closest = min(locs, key=lambda loc: np.linalg.norm(np.array(loc) - X_t))
            X_t = np.array(X_station_closest)

        if zoom_to_location:
            lat_slice = slice(X_t[0] + 2, X_t[0] - 2)
            lon_slice = slice(X_t[1] - 2, min(X_t[1] + 2, 180))
            pred_db = pred_db.sel(latitude=lat_slice, longitude=lon_slice)
            
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()), figsize=(20, 20))
        pred_db['mean'].plot(ax=ax, cmap="jet")
        ax.coastlines()
        ax.add_feature(cf.BORDERS)
        
        if location is not None:
            ax.scatter(X_t[1], X_t[0], transform=ccrs.PlateCarree(), color="black", marker="*", s=200)
            size_of_stations = 60
        else:
            import matplotlib as mpl
            size_of_stations = mpl.rcParams['lines.markersize'] ** 2
        # Plot station locations
        ax.scatter([loc[1] for loc in locs], [loc[0] for loc in locs], transform=ccrs.PlateCarree(), color="red", marker=".", s=size_of_stations)
        if zoom_to_location:
            ax.set_extent([X_t[1] - 2, X_t[1] + 2, X_t[0] - 2, X_t[0] + 2])

        if return_fig:
            return fig, ax

    def plot_timeseries_comparison(self, 
                                   location, 
                                   date_range: tuple, 
                                   return_fig=False):

        task_loader = self.task_loader
        model = self.model
        era5_raw_ds = self.processed_dict['era5_raw_ds']
        station_raw_df = self.processed_dict['station_raw_df']

        if isinstance(location, str):
            if location not in LOCATION_LATLON:
                raise ValueError(f"Location {location} not in LOCATION_LATLON, please set X_t manually")
            X_t = LOCATION_LATLON[location]
        else:
            X_t = location

        dates = pd.date_range(date_range[0], date_range[1])

        test_task = task_loader(dates, ["all", "all"], seed_override=42)
        pred = model.predict(test_task, X_t=era5_raw_ds, resolution_factor=1)
        pred_db = pred['dry_bulb']

        locs = set(zip(station_raw_df.reset_index()["latitude"], station_raw_df.reset_index()["longitude"]))
        X_station_closest = min(locs, key=lambda loc: np.linalg.norm(np.array(loc) - X_t))
        X_t = np.array(X_station_closest)#.reshape(2, 1)
        station_closest_df = station_raw_df.reset_index().set_index(["latitude", "longitude"]).loc[X_station_closest].set_index("time")
        intersection = station_closest_df.index.intersection(dates)
        if intersection.empty:
            raise ValueError(f"Station {X_station_closest} has no data for dates {dates}")
        else:
            station_closest_df = station_closest_df.loc[dates]

        era5_raw_df = era5_raw_ds.sel(latitude=X_t[0], longitude=X_t[1], method="nearest").to_dataframe()
        era5_raw_df = era5_raw_df.loc[dates]
        era5_raw_df

        sns.set_style("white")

        convnp_mean = pred_db["mean"].sel(latitude=X_t[0], longitude=X_t[1], method='nearest').values.astype('float')
        stddev = pred_db["std"].sel(latitude=X_t[0], longitude=X_t[1], method='nearest').values.astype('float')
        era5_vals = era5_raw_df["t2m"].values.astype('float')

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
        ax.set_xticks(range(len(era5_raw_df))[::14])
        ax.set_xticklabels(era5_raw_df.index[::14].strftime("%Y-%m-%d"), rotation=15)
        ax.set_title(f"ConvNP prediction for {location}", y=1.15)

        if return_fig:
            return fig, ax
