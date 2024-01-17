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

# from nzdownscale.dataprocess.era5 import ProcessERA5
# from nzdownscale.dataprocess.stations import ProcessStations
# from nzdownscale.dataprocess.topography import ProcessTopography
# from nzdownscale.dataprocess.utils import DataProcess, PlotData
from nzdownscale.dataprocess.config import LOCATION_LATLON

class Train:
    def __init__(self,
                 processed_data,
                 convnp_settings='default',
                 save_model_path='models/downscaling'
                 ) -> None:

        if convnp_settings == 'default':
            convnp_settings = {
                'unet_channels': (64,)*4,
                'likelihood': 'gnp',
                'internal_density': 20,
            }

        self.convnp_settings = convnp_settings
        self.processed_data = processed_data
        self.save_model_path = save_model_path

        self.era5_ds = processed_data['era5_ds']
        self.highres_aux_ds = processed_data['highres_aux_ds']
        self.aux_ds = processed_data['aux_ds']
        self.station_df = processed_data['station_df']
        self.data_processor = processed_data['data_processor']

        self.train_start_year = processed_data['date_info']['train_start_year']
        self.val_start_year = processed_data['date_info']['val_start_year']
        self.years = processed_data['date_info']['years']


    def setup_task_loader(self):

        era5_ds = self.era5_ds
        highres_aux_ds = self.highres_aux_ds
        aux_ds = self.aux_ds
        station_df = self.station_df
        train_start_year = self.train_start_year
        val_start_year = self.val_start_year
        years = self.years
        
        task_loader = TaskLoader(context=[era5_ds, aux_ds],
                                target=station_df, 
                                aux_at_targets=highres_aux_ds)
        print(task_loader)

        train_start = f'{train_start_year}-01-01'
        train_end = f'{val_start_year-1}-12-31'
        val_start = f'{val_start_year}-01-01'
        val_end = f'{years[-1]}-12-31'

        train_dates = era5_ds.sel(time=slice(train_start, train_end)).time.values
        val_dates = era5_ds.sel(time=slice(val_start, val_end)).time.values

        train_tasks = []
        # only loaded every other date to speed up training for now
        for date in tqdm(train_dates[::2], desc="Loading train tasks..."):
            task = task_loader(date, context_sampling="all", target_sampling="all")
            train_tasks.append(task)

        val_tasks = []
        for date in tqdm(val_dates, desc="Loading val tasks..."):
            task = task_loader(date, context_sampling="all", target_sampling="all")
            val_tasks.append(task)

        print("Loading Dask arrays...")
        task_loader.load_dask()
        tic = time.time()
        print(f"Done in {time.time() - tic:.2f}s")                

        self.task_loader = task_loader     
        self.train_tasks = train_tasks
        self.val_tasks = val_tasks

        return task_loader     


    def initialise_model(self):

        # Set up model
        model = ConvNP(self.data_processor,
                    self.task_loader, 
                    unet_channels=self.convnp_settings['unet_channels'], 
                    likelihood=self.convnp_settings['likelihood'], 
                    internal_density=self.convnp_settings['internal_density'],
                    ) 

        #internal density edited to make model fit into memory -
        # may want to adjust down the line

        # Print number of parameters to check model is not too large for GPU memory
        _ = model(self.val_tasks[0])
        print(f"Model has {deepsensor.backend.nps.num_params(model.model):,} parameters")
        self.model = model


    def plot_context_encodings(self):
        
        model = self.model
        train_tasks = self.train_tasks
        task_loader = self.task_loader
        data_processor = self.data_processor

        fig = deepsensor.plot.context_encoding(model, train_tasks[0], task_loader)
        plt.show()

        #
        fig = deepsensor.plot.task(train_tasks[0], task_loader)
        plt.show()

        #

        crs = ccrs.PlateCarree()

        fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(projection=crs))
        ax.coastlines()
        ax.add_feature(cf.BORDERS)

        minlon = config.PLOT_EXTENT['all']['minlon']
        maxlon = config.PLOT_EXTENT['all']['maxlon']
        minlat = config.PLOT_EXTENT['all']['minlat']
        maxlat = config.PLOT_EXTENT['all']['maxlat']

        ax.set_extent([minlon, maxlon, minlat, maxlat], crs)
        # ax = nzplot.nz_map_with_coastlines()

        deepsensor.plot.offgrid_context(ax, val_tasks[0], data_processor, task_loader, plot_target=True, add_legend=True, linewidths=0.5)
        plt.show()

        # fig.savefig("tmp/train_stations.png", bbox_inches="tight")


    def train_model(self,
                    n_epochs=2,
                    plot_losses=True,
                    model_name='default',
                    ):

        model = self.model
        train_tasks = self.train_tasks
        val_tasks = self.val_tasks
        
        model_id = str(round(time.time()))
        if model_name == 'default':
            model_name = f'model_{model_id}'

        import lab as B
        def compute_val_loss(model, val_tasks):
            val_losses = []
            for task in val_tasks:
                val_losses.append(B.to_numpy(model.loss_fn(task, normalise=True)))
                val_losses_not_nan = [arr for arr in val_losses if~ np.isnan(arr)]
            return np.mean(val_losses_not_nan)

        #n_epochs = 2
        train_losses = []
        val_losses = []

        val_loss_best = np.inf

        for epoch in tqdm(range(n_epochs)):
            batch_losses = train_epoch(model, train_tasks)
            batch_losses_not_nan = [arr for arr in batch_losses if~ np.isnan(arr)]
            train_loss = np.mean(batch_losses_not_nan)
            train_losses.append(train_loss)

            val_loss = compute_val_loss(model, val_tasks)
            val_losses.append(val_loss)

            if val_loss < val_loss_best:
                import torch
                import os
                val_loss_best = val_loss
                if not os.path.exists(self.save_model_path): os.makedirs(self.save_model_path)
                torch.save(model.model.state_dict(), f"{self.save_model_path}/{model_name}.pt")

        #     print(f"Epoch {epoch} train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}")

        if plot_losses:
            fig = plt.figure()
            plt.plot(train_losses, label='Train loss')
            plt.plot(val_losses, label='Val loss')
            plt.show()
            folder = f"{self.save_model_path}/losses"
            if not os.path.exists(folder): os.makedirs(folder)
            fig.savefig(f"{folder}/model_{model_id}.png", bbox_inches="tight")

        self.model = model
        self.train_losses = train_losses
        self.val_losses = val_losses


# #%% Set up task loader

# task_loader = TaskLoader(context=[era5_ds, aux_ds],
#                         target=station_df, 
#                         aux_at_targets=highres_aux_ds)
# print(task_loader)

# train_start = f'{train_start_year}-01-01'
# train_end = f'{val_start_year-1}-12-31'
# val_start = f'{val_start_year}-01-01'
# val_end = f'{years[-1]}-12-31'

# train_dates = era5_ds.sel(time=slice(train_start, train_end)).time.values
# val_dates = era5_ds.sel(time=slice(val_start, val_end)).time.values

# train_tasks = []
# # only loaded every other date to speed up training for now
# for date in tqdm(train_dates[::2], desc="Loading train tasks..."):
#     task = task_loader(date, context_sampling="all", target_sampling="all")
#     train_tasks.append(task)

# val_tasks = []
# for date in tqdm(val_dates, desc="Loading val tasks..."):
#     task = task_loader(date, context_sampling="all", target_sampling="all")
#     val_tasks.append(task)

# print("Loading Dask arrays...")
# task_loader.load_dask()
# tic = time.time()
# print(f"Done in {time.time() - tic:.2f}s")

# #%% Inspect train task

# train_tasks[0]

# #%%

# # Set up model
# model = ConvNP(data_processor,
#                task_loader, 
#                unet_channels=(64,)*4, 
#                likelihood="gnp", 
#                #internal_density=50,
#                internal_density=20,
#                ) 

# #internal density edited to make model fit into memory -
# # may want to adjust down the line

# # Print number of parameters to check model is not too large for GPU memory
# _ = model(val_tasks[0])
# print(f"Model has {deepsensor.backend.nps.num_params(model.model):,} parameters")

# #%% Plot context encoding

# fig = deepsensor.plot.context_encoding(model, train_tasks[0], task_loader)
# plt.show()

# #
# fig = deepsensor.plot.task(train_tasks[0], task_loader)
# plt.show()

# #

# crs = ccrs.PlateCarree()

# fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(projection=crs))
# ax.coastlines()
# ax.add_feature(cf.BORDERS)

# minlon = config.PLOT_EXTENT['all']['minlon']
# maxlon = config.PLOT_EXTENT['all']['maxlon']
# minlat = config.PLOT_EXTENT['all']['minlat']
# maxlat = config.PLOT_EXTENT['all']['maxlat']

# ax.set_extent([minlon, maxlon, minlat, maxlat], crs)
# # ax = nzplot.nz_map_with_coastlines()

# deepsensor.plot.offgrid_context(ax, val_tasks[0], data_processor, task_loader, plot_target=True, add_legend=True, linewidths=0.5)
# plt.show()

# # fig.savefig("tmp/train_stations.png", bbox_inches="tight")

# #%% Train

# import lab as B
# def compute_val_loss(model, val_tasks):
#     val_losses = []
#     for task in val_tasks:
#         val_losses.append(B.to_numpy(model.loss_fn(task, normalise=True)))
#         val_losses_not_nan = [arr for arr in val_losses if~ np.isnan(arr)]
#     return np.mean(val_losses_not_nan)

# n_epochs = 2
# train_losses = []
# val_losses = []

# val_loss_best = np.inf

# for epoch in tqdm(range(n_epochs)):
#     batch_losses = train_epoch(model, train_tasks)
#     batch_losses_not_nan = [arr for arr in batch_losses if~ np.isnan(arr)]
#     train_loss = np.mean(batch_losses_not_nan)
#     train_losses.append(train_loss)

#     val_loss = compute_val_loss(model, val_tasks)
#     val_losses.append(val_loss)

#     if val_loss < val_loss_best:
#         import torch
#         import os
#         val_loss_best = val_loss
#         folder = "models/downscaling/"
#         if not os.path.exists(folder): os.makedirs(folder)
#         torch.save(model.model.state_dict(), folder + f"model_nosea_2.pt")

# #     print(f"Epoch {epoch} train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}")

# plt.figure()
# plt.plot(train_losses, label='Train loss')
# plt.plot(val_losses, label='Val loss')
# plt.show()
# fig.savefig("tmp/losses.png", bbox_inches="tight")
