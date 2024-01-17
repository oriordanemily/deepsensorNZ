
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

from nzdownscale.dataprocess import era5, stations, topography, utils, config
from nzdownscale.dataprocess.config import LOCATION_LATLON
from nzdownscale.downscaler.train import Train


class ValidateV1:
    def __init__(self,
                 processed_output_dict,
                 training_output_dict=None,                 
                 ) -> None:
        
        self.processed_output_dict = processed_output_dict
        self.training_output_dict = training_output_dict


    def initialise(self, load_model_path=None):
        self.load_data_processor()
        self.load_task_loader()
        self.load_val_tasks()
        if load_model_path is not None:
            self.model = self.load_pretrained_model(load_model_path)
        else:
            self.model = self.load_model_from_training_sequence()


    def load_pretrained_model(self, model_path):
        model = self.initialise_model()
        model.model.load_state_dict(torch.load(model_path))
        return model
    
    
    def load_model_from_training_sequence(self):
        assert self.training_output_dict is not None
        return self.training_output_dict['model']


    def initialise_model(self):
        if self.training_output_dict is not None:
            model = self.load_model_from_training_sequence()
        else:
            model = ConvNP(self.data_processor,
                        self.task_loader, 
                        unet_channels=self.convnp_settings['unet_channels'], 
                        likelihood=self.convnp_settings['likelihood'], 
                        internal_density=self.convnp_settings['internal_density'],
                        ) 
            
            _ = model(self.val_tasks[0])   # ? 
        return model
    
        
    def load_data_processor(self):
        self.data_processor = self.processed_output_dict['data_processor']


    def load_task_loader(self):
        if self.training_output_dict is not None:
            self.task_loader = self.training_output_dict['task_loader']
        else:
            raise NotImplementedError
            

    def load_val_tasks(self):
        if self.training_output_dict is not None:
            self.val_tasks = self.training_output_dict['val_tasks']
        else:
            raise NotImplementedError


    def initialise_plots(self):
        self.val_start_year = self.processed_output_dict['date_info']['val_start_year']
        self.era5_raw_ds = self.processed_output_dict['era5_raw_ds']
    

    # ! clean up below

    def plot_example_prediction(self, task_loader, data_processor, model):

        self.initialise_plots()

        crs = ccrs.PlateCarree()
        X_t = self.era5_raw_ds
        val_start_year = self.val_start_year
        date = f"{val_start_year}-06-25"

        test_task = task_loader(date, ["all", "all"], seed_override=42)
        pred = model.predict(test_task, X_t=X_t, resolution_factor=2)

        # Plot 1
        fig = deepsensor.plot.prediction(pred, date, data_processor, task_loader, test_task, crs=ccrs.PlateCarree())

        # Plot 2
        pred_db = pred['dry_bulb']
        fig, axes = self.gen_test_fig(
            X_t.isel(time=0), 
            pred_db["mean"],
            pred_db["std"],
            add_colorbar=True,
            var_cbar_label="2m temperature [°C]",
            std_cbar_label="std dev [°C]",
            std_clim=(None, 2),
            figsize=(20, 20/3)
        )
        

    def gen_test_fig(era5_raw_ds=None, mean_ds=None, std_ds=None, samples_ds=None, add_colorbar=False, var_clim=None, std_clim=None, var_cbar_label=None, std_cbar_label=None, fontsize=None, figsize=(15, 5)):
        
        crs = ccrs.PlateCarree()

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
        if era5_raw_ds is not None:
            ncols += 1
        if mean_ds is not None:
            ncols += 1
        if std_ds is not None:
            ncols += 1
        if samples_ds is not None:
            ncols += samples_ds.shape[0]

        fig, axes = plt.subplots(1, ncols, subplot_kw=dict(projection=crs), figsize=figsize)

        axis_i = 0
        if era5_raw_ds is not None:
            ax = axes[axis_i]
            # era5_raw_ds.sel(lat=slice(mean_ds["lat"].min(), mean_ds["lat"].max()), lon=slice(mean_ds["lon"].min(), mean_ds["lon"].max())).plot(ax=ax, cmap="jet", vmin=vmin, vmax=vmax, add_colorbar=False)
            era5_raw_ds.plot(ax=ax, cmap="jet", vmin=vmin, vmax=vmax, add_colorbar=add_colorbar, cbar_kwargs=dict(label=var_cbar_label))
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


    def emily_plots(self,
                    station_raw_df,
                    ):

        self.initialise_plots()
        val_start_year = self.val_start_year
        era5_raw_ds = self.era5_raw_ds
        task_loader = self.task_loader
        model = self.model
        crs = ccrs.PlateCarree()

        date = f"{val_start_year}-06-25"
        test_task = task_loader(date, ["all", "all"], seed_override=42)
        pred = model.predict(test_task, X_t=X_t, resolution_factor=2)
        pred_db = pred['dry_bulb']
        
        # %%

        location = "alexandra"
        if location not in LOCATION_LATLON:
            raise ValueError(f"Location {location} not in LOCATION_LATLON, please set X_t manually")
        X_t = LOCATION_LATLON[location]
        dates = pd.date_range(f"{val_start_year}-09-01", f"{val_start_year}-10-31")
        #station_raw_df

        # %%
        locs = set(zip(station_raw_df.reset_index()["latitude"], station_raw_df.reset_index()["longitude"]))
        locs
        # %%
        # Find closest station to desired target location
        X_station_closest = min(locs, key=lambda loc: np.linalg.norm(np.array(loc) - X_t))
        X_t = np.array(X_station_closest)#.reshape(2, 1)
        X_t

        # %%
        # As above but zooming in
        lat_slice = slice(X_t[0] + 2, X_t[0] - 2)
        lon_slice = slice(X_t[1] - 2, min(X_t[1] + 2, 180))
        fig, axes = self.gen_test_fig(
            era5_raw_ds.isel(time=0).sel(latitude=lat_slice, longitude=lon_slice),
            pred_db["mean"].sel(latitude=lat_slice, longitude=lon_slice),
            pred_db["std"].sel(latitude=lat_slice, longitude=lon_slice),
            add_colorbar=True,
            # var_clim=(10, -5),
            var_cbar_label="2m temperature [°C]",
            std_cbar_label="std dev [°C]",
            std_clim=(None, 2),
        )
        # Plot X_t
        for ax in axes:
            ax.scatter(X_t[1], X_t[0], marker="s", color="black", transform=crs, s=10**2, facecolors='none', linewidth=2)

        # %%
        # Get station target data
        station_closest_df = station_raw_df.reset_index().set_index(["latitude", "longitude"]).loc[X_station_closest].set_index("time").loc[dates]
        station_closest_df

        # %%
        # Plot location of X_t on map using cartopy
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs), figsize=(20, 20))
        pred_db['mean'].plot(ax=ax, cmap="jet")
        ax.coastlines()
        ax.add_feature(cf.BORDERS)
        ax.scatter(X_t[1], X_t[0], transform=crs, color="black", marker="*", s=200)
        # Plot station locations
        ax.scatter([loc[1] for loc in locs], [loc[0] for loc in locs], transform=crs, color="red", marker=".")
        # ax.set_extent([6, 15, 47.5, 55])
        # %%

        era5_raw_df = era5_raw_ds.sel(latitude=X_t[0], longitude=X_t[1], method="nearest").to_dataframe()
        era5_raw_df = era5_raw_df.loc[dates]
        era5_raw_df

        #%%

        test_tasks = task_loader(dates, "all")
        preds = model.predict(test_tasks, X_t=era5_raw_ds, resolution_factor=2)
        preds_db = preds['dry_bulb']

        #%%
        # Plot
        sns.set_style("white")
        fig, ax = plt.subplots(1, 1, figsize=(7*.9, 3*.9))
        convnp_mean = preds_db["mean"].sel(latitude=X_t[0], longitude=X_t[1], method='nearest').values.astype('float')
        ax.plot(convnp_mean, label="ConvNP", marker="o", markersize=3)
        stddev = preds_db["std"].sel(latitude=X_t[0], longitude=X_t[1], method='nearest').values.astype('float')
        # Make 95% confidence interval
        ax.fill_between(range(len(convnp_mean)), convnp_mean - 2 * stddev, convnp_mean + 2 * stddev, alpha=0.25, label="ConvNP 95% CI")
        era5_vals = era5_raw_df["t2m"].values.astype('float')
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

