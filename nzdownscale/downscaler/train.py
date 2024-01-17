#%% 

import logging
logging.captureWarnings(True)
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import lab as B
import torch

import deepsensor.torch
from deepsensor.data.loader import TaskLoader
from tqdm import tqdm

from deepsensor.model.convnp import ConvNP
from deepsensor.train.train import train_epoch, set_gpu_default_device
from nzdownscale.dataprocess import config, utils


class Train:
    def __init__(self,
                 processed_output_dict,
                 convnp_settings='default',
                 save_model_path='models/downscaling',
                 use_gpu=True,
                 ) -> None:
        """
        processed_output_dict: 
            output from nzdownscale.downscaler.getdata.GetData()
        """

        if convnp_settings == 'default':
            convnp_settings = {
                'unet_channels': (64,)*4,
                'likelihood': 'gnp',
                'internal_density': 20,
            }
        if use_gpu:
            set_gpu_default_device()

        self.convnp_settings = convnp_settings
        self.save_model_path = save_model_path
        self.processed_output_dict = processed_output_dict
        self.metadata_dict = {k: processed_output_dict[k] for k in ['data_settings', 'date_info']}

        self.era5_ds = processed_output_dict['era5_ds']
        self.highres_aux_ds = processed_output_dict['highres_aux_ds']
        self.aux_ds = processed_output_dict['aux_ds']
        self.station_df = processed_output_dict['station_df']
        self.data_processor = processed_output_dict['data_processor']

        self.train_start_year = processed_output_dict['date_info']['train_start_year']
        self.val_start_year = processed_output_dict['date_info']['val_start_year']
        self.years = processed_output_dict['date_info']['years']




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
        val_tasks = self.val_tasks
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
                    n_epochs=30,
                    plot_losses=True,
                    model_name='default',
                    model_name_prefix=None,
                    ):

        model = self.model
        train_tasks = self.train_tasks
        val_tasks = self.val_tasks
        
        model_id = str(round(time.time()))
        if model_name == 'default':
            model_name = f'model_{model_id}'
        if model_name_prefix is not None:
            model_name = f'{model_name_prefix}_{model_name}'

        def compute_val_loss(model, val_tasks):
            val_losses = []
            for task in val_tasks:
                val_losses.append(B.to_numpy(model.loss_fn(task, normalise=True)))
                val_losses_not_nan = [arr for arr in val_losses if~ np.isnan(arr)]
            return np.mean(val_losses_not_nan)

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
                val_loss_best = val_loss
                if not os.path.exists(self.save_model_path): os.makedirs(self.save_model_path)
                torch.save(model.model.state_dict(), f"{self.save_model_path}/{model_name}.pt")
                self.save_metadata(f"{self.save_model_path}/metadata", f'{model_name}')

                if plot_losses:
                    self.make_loss_plot(train_losses, 
                                    val_losses, 
                                    f"{self.save_model_path}/losses", 
                                    f"model_{model_id}.png")

        #     print(f"Epoch {epoch} train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}")

        if plot_losses:
            self.make_loss_plot(train_losses, 
                             val_losses, 
                             f"{self.save_model_path}/losses", 
                             f"model_{model_id}.png")

        self.model = model
        self.train_losses = train_losses
        self.val_losses = val_losses


    def get_training_output_dict(self):
        training_output_dict = {
            'model': self.model,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_tasks': self.train_tasks,
            'val_tasks': self.val_tasks,
            'task_loader': self.task_loader,
            'data_processor': self.data_processor,

            'model_settings': self.convnp_settings,
            'metadata': self.metadata_dict,
        }
        return training_output_dict
    

    def save_metadata(self, folder, name):
        if not os.path.exists(folder): os.makedirs(folder)
        utils.save_pickle(self.metadata_dict, f"{folder}/{name}.pkl")


    def make_loss_plot(self, train_losses, val_losses, folder='tmp', save_name="model_loss.png"):
        fig = plt.figure()
        plt.plot(train_losses, label='Train loss')
        plt.plot(val_losses, label='Val loss')
        plt.show()

        if not os.path.exists(folder): os.makedirs(folder)
        fig.savefig(f"{folder}/{save_name}", bbox_inches="tight")
        print(f"Saved: {folder}/{save_name}")

