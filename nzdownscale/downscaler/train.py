import time
import logging
from pathlib import Path

logging.captureWarnings(True)

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import lab as B
import torch
import deepsensor.torch  # noqa
from tqdm import tqdm
from deepsensor.data.loader import TaskLoader
from deepsensor.data.task import Task
from deepsensor.model.convnp import ConvNP
from deepsensor.train import Trainer, set_gpu_default_device
# from neuralprocesses.model.loglik import loglik
# from neuralprocesses.model import Model
# from neuralprocesses.mask import Masked

from nzdownscale.dataprocess import config, utils


# @loglik.dispatch
# def loglik(model: Model, contexts: list, xt, yt: Masked, **kwargs):
#     yt = torch.where(yt.mask > 0, yt.y, B.nan)
#     state = B.global_random_state(B.dtype(xt))
#     state, logpdfs = loglik(state, model, contexts, xt, yt, **kwargs)
#     B.set_global_random_state(state)
#     return logpdfs


def train_epoch(
    model: ConvNP,
    tasks: list[Task],
    lr: float = 5e-5,
    opt=None,
    progress_bar=False,
    tqdm_notebook=False,
) -> list[float]:
    """
    Train model for one epoch.

    Args:
        model (:class:`~.model.convnp.ConvNP`):
            Model to train.
        tasks (list[:class:`~.data.task.Task`]):
            List of tasks to train on.
        lr (float, optional):
            Learning rate, by default 5e-5.
        opt (Optimizer, optional):
            TF or Torch optimizer. Defaults to None. If None,
            :class:`tensorflow:tensorflow.keras.optimizer.Adam` is used.
        progress_bar (bool, optional):
            Whether to display a progress bar. Defaults to False.
        tqdm_notebook (bool, optional):
            Whether to use a notebook progress bar. Defaults to False.

    Returns:
        list[float]: List of losses for each task/batch.
    """
    import torch.optim as optim

    if opt is None:
        opt = optim.Adam(model.model.parameters(), lr=lr)

    tasks = np.random.permutation(tasks)

    if tqdm_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    batch_losses = []
    for batch_i in tqdm(range(len(tasks)), disable=not progress_bar):
        opt.zero_grad()
        task_loss = model.loss_fn(tasks[batch_i], normalise=True)
        task_loss.backward()
        opt.step()
        batch_losses.append(task_loss.detach().cpu().numpy())

    return batch_losses


class SimpleTrainer(Trainer):
    def __call__(
        self,
        tasks: list[Task],
        progress_bar: bool = False,
        tqdm_notebook: bool = False,
    ) -> list[float]:
        return train_epoch(
            model=self.model,
            tasks=tasks,
            opt=self.opt,
            progress_bar=progress_bar,
            tqdm_notebook=tqdm_notebook,
        )


def make_loss_plot(train_losses, val_losses, filename="model_loss.png"):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Train loss")
    ax.plot(val_losses, label="Val loss")
    ax.legend()
    fig.savefig(filename, bbox_inches="tight")


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
            save_model_path (str):
                Best models are saved in this directory
            use_gpu (bool):
                Uses GPU if True
        """

        if use_gpu:
            set_gpu_default_device()

        self.save_model_path = Path(save_model_path)
        self.processed_output_dict = processed_output_dict

        self.model = None
        self.train_tasks = None
        self.val_tasks = None
        self.task_loader = None
        self.train_losses = []
        self.val_losses = []
        self.convnp_kwargs = None

        self._check_inputs()

    @property
    def data_processor(self):
        return self.processed_output_dict["data_processor"]

    @property
    def metadata_dict(self):
        metadata = {
            k: self.processed_output_dict[k] for k in ["data_settings", "date_info"]
        }
        metadata["convnp_kwargs"] = self.convnp_kwargs
        metadata["train_losses"] = self.train_losses
        metadata["val_losses"] = self.val_losses
        return metadata

    def _check_inputs(self):
        # if self.landmask_ds is not None:
        #     raise NotImplementedError
        pass

    def run_training_sequence(
        self,
        n_epochs,
        model_name="default",
        lr=5e-5,
        **convnp_kwargs,
    ):
        self.setup_task_loader()
        self.initialise_model(**convnp_kwargs)
        self.train_model(n_epochs=n_epochs, model_name=model_name, lr=lr)

    def setup_task_loader(
        self,
        verbose=False,
        validation=False,
    ):
        # extract datasets from the data preprocessing output dictionary
        era5_ds = self.processed_output_dict["era5_ds"]
        highres_aux_ds = self.processed_output_dict["highres_aux_ds"]
        aux_ds = self.processed_output_dict["aux_ds"]
        station_df = self.processed_output_dict["station_df"]
        landmask_ds = self.processed_output_dict["landmask_ds"]

        context = [era5_ds, aux_ds]
        if landmask_ds is not None:
            context += [landmask_ds]

        self.task_loader = TaskLoader(
            context=context, target=station_df, aux_at_targets=highres_aux_ds
        )
        if verbose:
            print(self.task_loader)

        # create training tasks
        if not validation:
            start_year = self.processed_output_dict["date_info"]["start_year"]
            end_year = self.processed_output_dict["date_info"]["end_year"]
            train_start = f"{start_year}-01-01"
            train_end = f"{end_year}-12-31"

            train_slice = slice(train_start, train_end)
            train_dates = era5_ds.sel(time=train_slice).time.values
            self.train_tasks = []
            # only loaded every other date to speed up training for now
            for date in tqdm(train_dates[::2], desc="Loading train tasks..."):
                task = self.task_loader(
                    date, context_sampling="all", target_sampling="all"
                )
                task = task.add_batch_dim().cast_to_float32().mask_nans_numpy()
                self.train_tasks.append(task)

        # create validation tasks
        val_start_year = self.processed_output_dict["date_info"]["val_start_year"]
        val_end_year = self.processed_output_dict["date_info"]["val_end_year"]
        val_start = f"{val_start_year}-01-01"
        val_end = f"{val_end_year}-12-31"

        val_dates = era5_ds.sel(time=slice(val_start, val_end)).time.values
        self.val_tasks = []
        for date in tqdm(val_dates, desc="Loading val tasks..."):
            task = self.task_loader(date, context_sampling="all", target_sampling="all")
            task = task.add_batch_dim().cast_to_float32().mask_nans_numpy()
            self.val_tasks.append(task)

        if verbose:
            print("Loading Dask arrays...")
        self.task_loader.load_dask()
        tic = time.time()
        if verbose:
            print(f"Done in {time.time() - tic:.2f}s")

        return self.task_loader

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
        fig = deepsensor.plot.context_encoding(
            self.model, self.train_tasks[0], self.task_loader
        )
        plt.show()

        fig = deepsensor.plot.task(self.train_tasks[0], self.task_loader)
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
            self.val_tasks[0],
            self.data_processor,
            self.task_loader,
            plot_target=True,
            add_legend=True,
            linewidths=0.5,
        )
        plt.show()

    def train_model(self, n_epochs=30, plot_losses=True, model_name="default", lr=5e-5):
        if model_name == "default":
            model_id = str(round(time.time()))
            model_name = f"model_{model_id}"
        else:
            model_name = f"model_{model_name}"

        save_dir = self.save_model_path / model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        val_loss_best = min(self.val_losses) if self.val_losses else np.inf
        trainer = SimpleTrainer(self.model, lr=lr)

        for epoch in tqdm(range(n_epochs)):
            train_losses = trainer(self.train_tasks)
            assert not np.isnan(train_losses).any(), "NaN train loss"
            train_loss = np.mean(train_losses)
            self.train_losses.append(train_loss)

            val_losses = [
                B.to_numpy(self.model.loss_fn(task, normalise=True))
                for task in self.val_tasks
            ]
            assert not np.isnan(val_losses).any(), "NaN validation loss"
            val_loss = np.mean(val_losses)
            self.val_losses.append(val_loss)

            if val_loss < val_loss_best:
                val_loss_best = val_loss
                model_state = self.model.model.state_dict()
                torch.save(model_state, save_dir / f"{model_name}.pt")

        if plot_losses:
            make_loss_plot(
                self.train_losses,
                self.val_losses,
                save_dir / f"losses_{model_name}.png",
            )

        utils.save_pickle(self.metadata_dict, save_dir / f"metadata_{model_name}.pkl")

    # def train_epoch_and_print(self, model, train_tasks):
    #     # used for debugging
    #     te = train_epoch(model, train_tasks)
    #     return te

    def get_training_output_dict(self):
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
