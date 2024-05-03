import os
import time
import logging

logging.captureWarnings(True)

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
    def __init__(
        self,
        processed_output_dict,
        save_model_path: str = "models/downscaling",
        use_gpu: bool = True,
    ) -> None:
        """
        Args:
            processed_output_dict (dict):
                Output from nzdownscale.downscaler.getdata.GetData()
            save_model_path (int):
                Best models are saved in this directory
            use_gpu (bool):
                Uses GPU if True
        """

        if use_gpu:
            set_gpu_default_device()

        self.save_model_path = save_model_path
        self.processed_output_dict = processed_output_dict

        self.era5_ds = processed_output_dict["era5_ds"]
        self.highres_aux_ds = processed_output_dict["highres_aux_ds"]
        self.aux_ds = processed_output_dict["aux_ds"]
        self.station_df = processed_output_dict["station_df"]
        self.landmask_ds = processed_output_dict["landmask_ds"]

        self.data_processor = processed_output_dict["data_processor"]

        self.start_year = processed_output_dict["date_info"]["start_year"]
        self.end_year = processed_output_dict["date_info"]["end_year"]
        self.val_start_year = processed_output_dict["date_info"]["val_start_year"]
        self.val_end_year = processed_output_dict["date_info"]["val_end_year"]
        self.years = np.arange(self.start_year, self.end_year + 1)

        self.model = None
        self.train_tasks = None
        self.val_tasks = None
        self.task_loader = None
        self.train_losses = []
        self.val_losses = []
        self.metadata_dict = None
        self.convnp_kwargs = None

        self._check_inputs()

    def _check_inputs(self):
        # if self.landmask_ds is not None:
        #     raise NotImplementedError
        pass

    def run_training_sequence(
        self,
        n_epochs,
        model_name="default",
        batch=False,
        batch_size=1,
        **convnp_kwargs,
    ):
        self.setup_task_loader()
        self.initialise_model(**convnp_kwargs)
        self.train_model(
            n_epochs=n_epochs, model_name=model_name, batch=batch, batch_size=batch_size
        )

    def setup_task_loader(
        self,
        verbose=False,
        validation=False,
    ):
        era5_ds = self.era5_ds
        highres_aux_ds = self.highres_aux_ds
        aux_ds = self.aux_ds
        station_df = self.station_df
        landmask_ds = self.landmask_ds

        start_year = self.start_year
        end_year = self.end_year
        val_start_year = self.val_start_year
        val_end_year = self.val_end_year

        context = [era5_ds, aux_ds]
        if landmask_ds is not None:
            context += [landmask_ds]

        task_loader = TaskLoader(
            context=context, target=station_df, aux_at_targets=highres_aux_ds
        )
        if verbose:
            print(task_loader)

        train_start = f"{start_year}-01-01"
        train_end = f"{end_year}-12-31"
        val_start = f"{val_start_year}-01-01"
        val_end = f"{val_end_year}-12-31"

        if not validation:
            train_dates = era5_ds.sel(time=slice(train_start, train_end)).time.values
        val_dates = era5_ds.sel(time=slice(val_start, val_end)).time.values

        if not validation:
            train_tasks = []
            # only loaded every other date to speed up training for now
            for date in tqdm(train_dates[::2], desc="Loading train tasks..."):
                task = task_loader(date, context_sampling="all", target_sampling="all")
                train_tasks.append(task)

        val_tasks = []
        for date in tqdm(val_dates, desc="Loading val tasks..."):
            task = task_loader(date, context_sampling="all", target_sampling="all")
            val_tasks.append(task)

        if verbose:
            print("Loading Dask arrays...")
        task_loader.load_dask()
        tic = time.time()
        if verbose:
            print(f"Done in {time.time() - tic:.2f}s")

        self.task_loader = task_loader
        if not validation:
            self.train_tasks = train_tasks
        self.val_tasks = val_tasks
        self.context = context

        return task_loader

    def initialise_model(self, **convnp_kwargs):
        """
        Args:
            convnp_kwargs (dict):
                Inputs to deepsensor.model.convnp.ConvNP(). Uses default
                CONVNP_KWARGS_DEFAULT if not provided.
        """

        if convnp_kwargs is None:
            convnp_kwargs = config.CONVNP_KWARGS_DEFAULT

        # Set up model
        model = ConvNP(self.data_processor, self.task_loader, **convnp_kwargs)

        # Print number of parameters to check model is not too large for GPU memory
        _ = model(self.val_tasks[0])
        print(
            f"Model has {deepsensor.backend.nps.num_params(model.model):,} parameters"
        )

        self.convnp_kwargs = dict(convnp_kwargs)
        self.model = model

    def plot_context_encodings(self):
        model = self.model
        train_tasks = self.train_tasks
        val_tasks = self.val_tasks
        task_loader = self.task_loader
        data_processor = self.data_processor

        fig = deepsensor.plot.context_encoding(model, train_tasks[0], task_loader)
        plt.show()

        fig = deepsensor.plot.task(train_tasks[0], task_loader)
        plt.show()

        crs = ccrs.PlateCarree()
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(projection=crs))
        ax.coastlines()
        ax.add_feature(cf.BORDERS)

        minlon = config.PLOT_EXTENT["all"]["minlon"]
        maxlon = config.PLOT_EXTENT["all"]["maxlon"]
        minlat = config.PLOT_EXTENT["all"]["minlat"]
        maxlat = config.PLOT_EXTENT["all"]["maxlat"]

        ax.set_extent([minlon, maxlon, minlat, maxlat], crs)

        deepsensor.plot.offgrid_context(
            ax,
            val_tasks[0],
            data_processor,
            task_loader,
            plot_target=True,
            add_legend=True,
            linewidths=0.5,
        )
        plt.show()

    def train_model(
        self,
        n_epochs=30,
        plot_losses=True,
        model_name="default",
        batch=False,
        batch_size=1,
    ):
        model = self.model
        train_tasks = self.train_tasks
        val_tasks = self.val_tasks

        if model_name == "default":
            model_id = str(round(time.time()))
            model_name = f"model_{model_id}"
        else:
            model_name = f"model_{model_name}"
        self.set_save_dir(model_name)

        def compute_val_loss(model, val_tasks):
            val_losses = []
            for task in val_tasks:
                val_losses.append(B.to_numpy(model.loss_fn(task, normalise=True)))
                val_losses_not_nan = [arr for arr in val_losses if ~np.isnan(arr)]
            return np.mean(val_losses_not_nan)

        train_losses = []
        val_losses = []

        val_loss_best = np.inf

        if batch:
            print(f"Using batched data with batch size {batch_size}")
            # if batch is True and batch_size is None, then batches are created by number of stations
            batched_train_tasks = self.batch_data_by_num_stations(
                train_tasks, batch_size=batch_size
            )
            batched_val_tasks = self.batch_data_by_num_stations(
                val_tasks, batch_size=batch_size
            )

        for epoch in tqdm(range(n_epochs)):
            if batch:
                batch_losses = [
                    train_epoch(model, batched_train_tasks[f"{num_stations}"])
                    for num_stations in batched_train_tasks.keys()
                ]
                batch_losses = [item for sublist in batch_losses for item in sublist]
            else:
                breakpoint()
                batch_losses = train_epoch(model, train_tasks)
            batch_losses_not_nan = [arr for arr in batch_losses if ~np.isnan(arr)]
            train_loss = np.mean(batch_losses_not_nan)
            train_losses.append(train_loss)

            if batch:
                batch_val_losses = [
                    compute_val_loss(model, batched_val_tasks[f"{num_stations}"])
                    for num_stations in batched_val_tasks.keys()
                ]
                val_loss = np.mean(batch_val_losses)
            else:
                val_loss = compute_val_loss(model, val_tasks)
            val_losses.append(val_loss)

            if val_loss < val_loss_best:
                val_loss_best = val_loss

                torch.save(model.model.state_dict(), f"{self.save_dir}/{model_name}.pt")
                self.save_metadata(f"{self.save_dir}", f"metadata_{model_name}")

                self.train_losses = train_losses
                self.val_losses = val_losses

            if plot_losses:
                self.make_loss_plot(
                    train_losses,
                    val_losses,
                    f"{self.save_dir}",
                    f"losses_{model_name}.png",
                )

        self.model = model
        self.train_losses = train_losses
        self.val_losses = val_losses

    # def train_epoch_and_print(self, model, train_tasks):
    #     # used for debugging
    #     te = train_epoch(model, train_tasks)
    #     return te

    def set_save_dir(self, model_name):
        self.save_dir = f"{self.save_model_path}/{model_name}"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def batch_data_by_num_stations(self, tasks, batch_size=None):
        batched_tasks = {}
        for task in tasks:
            num_stations = task["X_t"][0].shape[1]
            if f"{num_stations}" not in batched_tasks.keys():
                batched_tasks[f"{num_stations}"] = [task]
            else:
                batched_tasks[f"{num_stations}"].append(task)

        if batch_size is not None:
            batched_tasks_copy = batched_tasks.copy()
            batched_tasks = {}
            for num_stations in batched_tasks_copy.keys():
                number_tasks_in_batch = len(batched_tasks_copy[f"{num_stations}"])
                for idx, i in enumerate(range(0, number_tasks_in_batch, batch_size)):
                    batched_tasks[f"{num_stations}_{idx}"] = batched_tasks_copy[
                        f"{num_stations}"
                    ][i : i + batch_size]

        return batched_tasks

    def get_training_output_dict(self):
        if self.metadata_dict is None:
            self._construct_metadata_dict()

        training_output_dict = {
            "model": self.model,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_tasks": self.train_tasks,
            "val_tasks": self.val_tasks,
            "task_loader": self.task_loader,
            "data_processor": self.data_processor,
            "metadata_dict": self.metadata_dict,
        }
        return training_output_dict

    def save_metadata(self, folder, name):
        self._construct_metadata_dict()
        if not os.path.exists(folder):
            os.makedirs(folder)
        utils.save_pickle(self.metadata_dict, f"{folder}/{name}.pkl")

    def _construct_metadata_dict(self):
        metadata_dict = {
            k: self.processed_output_dict[k] for k in ["data_settings", "date_info"]
        }
        metadata_dict["convnp_kwargs"] = self.convnp_kwargs
        metadata_dict["train_losses"] = self.train_losses
        metadata_dict["val_losses"] = self.val_losses
        self.metadata_dict = metadata_dict

    def make_loss_plot(
        self, train_losses, val_losses, folder="tmp", save_name="model_loss.png"
    ):
        fig = plt.figure()
        plt.plot(train_losses, label="Train loss")
        plt.plot(val_losses, label="Val loss")
        plt.show()

        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig(f"{folder}/{save_name}", bbox_inches="tight")
        print(f"Saved: {folder}/{save_name}")
