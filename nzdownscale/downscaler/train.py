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
import random
import xarray as xr
import pandas as pd
import pickle

import deepsensor.torch
from deepsensor.data.loader import TaskLoader
from tqdm import tqdm

from deepsensor.model.convnp import ConvNP
from deepsensor.train.train import train_epoch, set_gpu_default_device
from nzdownscale.dataprocess import config, config_local, utils
from deepsensor.data.task import Task
from sklearn.model_selection import train_test_split


class Train:
    def __init__(self,
                 processed_output_dict,
                 save_model_path: str = 'default', 
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

        if save_model_path == 'default':
            save_model_path=config_local.DATA_PATHS['save_model']['fpath']
        self.save_model_path = save_model_path
        self.processed_output_dict = processed_output_dict

        self.variable = processed_output_dict['data_settings']['var']
        self.base_ds = processed_output_dict['base_ds']
        self.highres_aux_ds = processed_output_dict['highres_aux_ds']
        self.aux_ds = processed_output_dict['aux_ds']
        self.station_df = processed_output_dict['station_df']
        self.landmask_ds = processed_output_dict['landmask_ds']
        self.station_as_context = processed_output_dict['station_as_context']
        
        self.data_processor = processed_output_dict['data_processor']

        # self.start_year = processed_output_dict['date_info']['start_year']
        # self.end_year = processed_output_dict['date_info']['end_year']
        # self.val_start_year = processed_output_dict['date_info']['val_start_year']
        # self.val_end_year = processed_output_dict['date_info']['val_end_year']
        self.training_years = processed_output_dict['date_info']['training_years']
        self.validation_years = processed_output_dict['date_info']['validation_years']
        # self.years = np.arange(self.start_year, self.end_year+1)

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


    def run_training_sequence(self, n_epochs, model_name='default', 
                              batch=False, batch_size=1, lr=5e-5, 
                              weight_decay=0, time_intervals=1, 
                              **convnp_kwargs,):
        
        self.setup_task_loader(model_name=model_name, time_intervals=time_intervals)
        self.initialise_model(**convnp_kwargs)
        self.train_model(n_epochs=n_epochs, model_name=model_name, batch=batch, batch_size=batch_size, lr=lr, weight_decay=weight_decay)


    def setup_task_loader(self, 
                          model_name,
                          verbose=False, 
                          validation=False,
                          val_tasks=None, 
                          time_intervals=1,
                          ):

        base_ds = self.base_ds
        highres_aux_ds = self.highres_aux_ds
        aux_ds = self.aux_ds
        station_df = self.station_df
        landmask_ds = self.landmask_ds
        station_as_context = self.station_as_context
        self.val_tasks = val_tasks
        
        # start_year = self.start_year
        # end_year = self.end_year
        # val_start_year = self.val_start_year
        # val_end_year = self.val_end_year
        
        training_years = self.training_years
        validation_years = self.validation_years

        context = [base_ds, aux_ds]
        context_sampling = ["all", "all"]

        if landmask_ds is not None:
            context += [landmask_ds]
            context_sampling += ["all"]

        if station_as_context != 0:
            context += [station_df]        
            if validation:
                context_sampling += ['all']
            elif type(station_as_context) == float:
                context_sampling += [station_as_context]
            elif station_as_context == 'all':
                context_sampling += ['all']
            elif station_as_context == 'random':
                context_sampling += ['random']

        self.task_loader = TaskLoader_SampleStations(context=context,
                                                target=station_df, 
                                                aux_at_targets=highres_aux_ds,)
        task_loader = self.task_loader

        if verbose:
            print(task_loader)

        task_loader_path = f"{self.save_model_path}/{self.variable}/{model_name}/task_loader.pkl"
        with open(task_loader_path, 'wb+') as f:
                pickle.dump(task_loader, f)

        # if not validation:
        #     train_start = f'{start_year}-01-01'
        #     train_end = f'{end_year}-12-31'
        # val_start = f'{val_start_year}-01-01'
        # val_end = f'{val_end_year}-12-31'

        # From here downwards, could this be a separate function? 
        # E.g. load_tasks() ?
        
        if not validation:
            train_dates = [base_ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31')).time.values for year in training_years]
            train_dates = [date for sublist in train_dates for date in sublist]
        val_dates = [base_ds.sel(time=slice(f'{year}-01-01', f'{year}-12-31')).time.values for year in validation_years]
        val_dates = [date for sublist in val_dates for date in sublist]

        if not validation:
            train_tasks = []
            for date in tqdm(train_dates[::time_intervals], desc="Loading train tasks..."):
                if context_sampling[-1] == 'random': #currently only implemented for stations
                    context_sampling_ = context_sampling[:-1] + [np.random.rand()]
                else:
                    context_sampling_ = context_sampling
                task = task_loader(date, context_sampling=context_sampling_, target_sampling="all")
                train_tasks.append(task)

            # if self.val_tasks is None:
        val_tasks = []
        for date in tqdm(val_dates[::time_intervals], desc="Loading val tasks..."):
        # for date in tqdm(val_dates[::hours_interval], desc="Loading val tasks..."):
            if context_sampling[-1] == 'random': #currently only implemented for stations
                context_sampling_ = context_sampling[:-1] + [np.random.rand()]
            else:
                context_sampling_ = context_sampling
            task = task_loader(date, context_sampling=context_sampling_, target_sampling="all")
            # task["ops"] = ["numpy_mask", "nps_mask"]
            val_tasks.append(task)
        # else:
        #     val_tasks = self.val_tasks

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
    
        # think saving task_loader here saves all the tasks as well
        # task_loader_path = f"{self.save_model_path}/{self.variable}/{model_name}/task_loader.pkl"
        # with open(task_loader_path, 'wb+') as f:
        #         pickle.dump(task_loader, f)

        return task_loader     


    def initialise_model(self, **convnp_kwargs):
        """
        Args:
            convnp_kwargs (dict):
                Inputs to deepsensor.model.convnp.ConvNP(). Uses default CONVNP_KWARGS_DEFAULT if not provided.
        """

        if convnp_kwargs is None:
            convnp_kwargs = config.CONVNP_KWARGS_DEFAULT
    
        # Set up model
        model = ConvNP(self.data_processor,
                    self.task_loader, 
                    **convnp_kwargs,
                    )
    
        # Print number of parameters to check model is not too large for GPU memory
        # _ = model(self.val_tasks[0])
        print(f"Model has {deepsensor.backend.nps.num_params(model.model):,} parameters")
        
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

        deepsensor.plot.offgrid_context(ax, val_tasks[0], data_processor, task_loader, plot_target=True, add_legend=True, linewidths=0.5)
        plt.show()


    def train_model(self,
                    n_epochs=30,
                    plot_losses=True,
                    model_name='default',
                    # model_name_prefix=None,
                    batch=False,
                    batch_size=1,
                    shuffle_tasks=True,
                    lr=5e-5,
                    weight_decay=0,
                    scheduler_patience=5,
                    early_stopping_patience=10,
                    ):

        model = self.model
        train_tasks = self.train_tasks
        val_tasks = self.val_tasks
        weight_decay = weight_decay
        opt = torch.optim.AdamW(model.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=scheduler_patience, verbose=True)

        if shuffle_tasks:
            random.shuffle(train_tasks)
            random.shuffle(val_tasks)
        
        if model_name == 'default':
            model_id = str(round(time.time()))
            model_name = f'model_{model_id}'

        self.set_save_dir(model_name)

        def compute_val_loss(model, val_tasks):
            val_losses = []
            for task in val_tasks:
                val_losses.append(B.to_numpy(model.loss_fn(task, normalise=True)))
                val_losses_not_nan = [arr for arr in val_losses if~ np.isnan(arr)]
            return np.mean(val_losses_not_nan)

        train_losses = []
        val_losses = []

        val_loss_best = np.inf
        epochs_no_improve = 0

        if batch:
            print(f'Using batched data with batch size {batch_size}')
            # if batch is True and batch_size is None, then batches are created by number of stations
            batched_train_tasks = self.batch_data_by_num_stations(train_tasks, batch_size=batch_size)
            batched_val_tasks = self.batch_data_by_num_stations(val_tasks, batch_size=batch_size)

        for epoch in tqdm(range(n_epochs), desc="Training"):
            if batch:
                batch_losses = [train_epoch(model, batched_train_tasks[f'{num_stations}'], 
                                            batch_size=len(batched_train_tasks[f'{num_stations}']), 
                                            lr=lr, opt=opt) for num_stations in batched_train_tasks.keys()]
                # batch_losses = [train_epoch(model, batched_train_tasks[f'{num_stations}']) for num_stations in batched_train_tasks.keys()]
                batch_losses = [item for sublist in batch_losses for item in sublist]
            else:
                batch_losses = train_epoch(model, train_tasks, opt=opt)
            batch_losses_not_nan = [arr for arr in batch_losses if~ np.isnan(arr)]
            
            train_loss = np.mean(batch_losses_not_nan)
            train_losses.append(train_loss)

            if batch:
                batch_val_losses = [compute_val_loss(model, batched_val_tasks[f'{num_stations}']) for num_stations in batched_val_tasks.keys()]
                val_loss = np.mean(batch_val_losses)
            else:
                val_loss = compute_val_loss(model, val_tasks)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            if val_loss < val_loss_best:
                val_loss_best = val_loss
                epochs_no_improve = 0
                print(f'Saving model at epoch {epoch}')
                torch.save(model.model.state_dict(), f"{self.save_dir}/{model_name}.pt")
                self.save_metadata(f"{self.save_dir}", f'metadata_{model_name}')
                
                self.train_losses = train_losses
                self.val_losses = val_losses
            else:
                epochs_no_improve += 1

            if plot_losses:
                self.make_loss_plot(train_losses, 
                                val_losses, 
                                f"{self.save_dir}", 
                                f"losses_{model_name}.png")

            if epochs_no_improve >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch}')
                break


        self.model = model
        self.train_losses = train_losses
        self.val_losses = val_losses


    # def train_epoch_and_print(self, model, train_tasks):
    #     # used for debugging
    #     te = train_epoch(model, train_tasks)
    #     return te 

    def set_save_dir(self, model_name):
        self.save_dir = f'{self.save_model_path}/{self.variable}/{model_name}'
        if not os.path.exists(self.save_dir): 
            os.makedirs(self.save_dir)

    
    def batch_data_by_num_stations(self, tasks, batch_size=None):
        # if batch_size == None, return a dict in which each key value pair is a 
        # list of *all of the tasks* with the same number of stations
        batched_tasks = {}
        for task in tasks:
            num_stations = task['X_t'][0].shape[1]
            if f'{num_stations}' not in batched_tasks.keys():
                batched_tasks[f'{num_stations}'] = [task]
            else:
                batched_tasks[f'{num_stations}'].append(task)

        # if batch_size is not None, return a dict in which each key value pair is a
        # list of tasks with the same number of stations, but with batch_size number of tasks
        # e.g. if batch_size = 4 but there are 10 tasks with 100 stations, then there will be 100 keys:
        # '100_0' with 4 tasks, '100_1' with 4 tasks, '100_2' with 2 tasks
                
        # reason for doing it like this: if we set a large batch_size, e.g. 16, and there are 
        # only 10 tasks with 100 stations, then the deepsensor train_epoch function will 
        # ignore these tasks 
        if batch_size is not None:
            batched_tasks_copy = batched_tasks.copy()
            batched_tasks = {}
            for num_stations in batched_tasks_copy.keys():
                number_tasks_in_batch = len(batched_tasks_copy[f'{num_stations}'])
                for idx, i in enumerate(range(0, number_tasks_in_batch, batch_size)):
                    batched_tasks[f'{num_stations}_{idx}'] = batched_tasks_copy[f'{num_stations}'][i:i+batch_size]

        return batched_tasks
    

    def get_training_output_dict(self):

        if self.metadata_dict is None:
            self._construct_metadata_dict()

        training_output_dict = {
            'model': self.model,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,

            'train_tasks': self.train_tasks,
            'val_tasks': self.val_tasks,
            'task_loader': self.task_loader,
            'data_processor': self.data_processor,

            'metadata_dict': self.metadata_dict,
        }
        return training_output_dict
    

    def save_metadata(self, folder, name):
        self._construct_metadata_dict()
        if not os.path.exists(folder): os.makedirs(folder)
        utils.save_pickle(self.metadata_dict, f"{folder}/{name}.pkl")


    def _construct_metadata_dict(self):
        metadata_dict = {k: self.processed_output_dict[k] for k in ['data_settings', 'date_info']}
        metadata_dict['convnp_kwargs'] = self.convnp_kwargs
        metadata_dict['train_losses'] = self.train_losses
        metadata_dict['val_losses'] = self.val_losses
        metadata_dict['station_as_context'] = self.station_as_context
        self.metadata_dict = metadata_dict


    def make_loss_plot(self, train_losses, val_losses, folder='tmp', save_name="model_loss.png"):
        fig = plt.figure()
        plt.plot(train_losses, label='Train loss')
        plt.plot(val_losses, label='Val loss')
        plt.legend()
        plt.show()

        if not os.path.exists(folder): os.makedirs(folder)
        fig.savefig(f"{folder}/{save_name}", bbox_inches="tight")
        print(f"Saved: {folder}/{save_name}")


class TaskLoader_SampleStations(TaskLoader):
    def __init__(self, context, target, aux_at_targets, **kwargs):
        super().__init__(context=context, target=target, aux_at_targets=aux_at_targets, **kwargs)
    
    def sample_df(self, df, sampling_strat, seed=None):
        df = df.dropna(how="any")  # If any obs are NaN, drop them

        if isinstance(sampling_strat, float):
            sampling_strat = int(sampling_strat * df.shape[0])

        if isinstance(sampling_strat, (int, np.integer)):
            N = sampling_strat
            rng = np.random.default_rng(seed)
            idx = rng.choice(df.index, N, replace=False)
            X_c = df.loc[idx].reset_index()[["x1", "x2"]].values.T.astype(self.dtype)
            Y_c = df.loc[idx].values.T

            # Get the target data as the complementary set
            X_t = df.drop(idx).reset_index()[["x1", "x2"]].values.T.astype(self.dtype)
            Y_t = df.drop(idx).values.T
            
        elif isinstance(sampling_strat, str) and sampling_strat in [
            "all",
            "split",
        ]:
            # NOTE if "split", we assume that the context-target split has already been applied to the df
            # in an earlier scope with access to both the context and target data. This is maybe risky!
            X_c = df.reset_index()[["x1", "x2"]].values.T.astype(self.dtype)
            Y_c = df.values.T
            X_t = X_c
            Y_t = Y_c
        #     raise InvalidSamplingStrategyError(f"Unknown sampling strategy {sampling_strat}")

        return X_c, Y_c, X_t, Y_t

    def task_generation(self, date, context_sampling="all", target_sampling=None, split_frac=0.5, datewise_deterministic=False, seed_override=None):
        
        def check_sampling_strat(sampling_strat, set):
            if sampling_strat is None:
                return None
            if not isinstance(sampling_strat, (list, tuple)):
                sampling_strat = tuple([sampling_strat] * len(set))
            return sampling_strat

        context_sampling = check_sampling_strat(context_sampling, self.context)
        target_sampling = check_sampling_strat(target_sampling, self.target)

        if split_frac < 0 or split_frac > 1:
            raise ValueError(f"split_frac must be between 0 and 1, got {split_frac}")

        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)

        if seed_override is not None:
            seed = seed_override
        elif datewise_deterministic:
            seed = int(date.strftime("%Y%m%d"))
        else:
            seed = None

        task = {}
        task["time"] = date
        task["ops"] = []
        task["X_c"] = []
        task["Y_c"] = []
        task["X_t"] = []
        task["Y_t"] = []

        context_slices = [
            self.time_slice_variable(var, date, delta_t)
            for var, delta_t in zip(self.context, self.context_delta_t)
        ]
        # target_slices = [
        #     self.time_slice_variable(var, date, delta_t)
        #     for var, delta_t in zip(self.target, self.target_delta_t)
        # ]

        for i, (var, sampling_strat) in enumerate(zip(context_slices, context_sampling)):
            context_seed = seed + i if seed is not None else None
            if isinstance(var, (pd.DataFrame, pd.Series)):
                X_c, Y_c, X_t, Y_t = self.sample_df(var, sampling_strat, context_seed)
                task["X_t"].append(X_t)
                task["Y_t"].append(Y_t)
            elif isinstance(var, (xr.Dataset, xr.DataArray)):
                X_c, Y_c = self.sample_da(var, sampling_strat, context_seed)

            task["X_c"].append(X_c)
            task["Y_c"].append(Y_c)
        
        if self.aux_at_contexts is not None:
            X_c_offgrid = [X_c for X_c in task["X_c"] if not isinstance(X_c, tuple)]
            if len(X_c_offgrid) == 0:
                X_c_offrid_all = np.empty((2, 0), dtype=self.dtype)
            else:
                X_c_offrid_all = np.concatenate(X_c_offgrid, axis=1)
            Y_c_aux = self.sample_offgrid_aux(
                X_c_offrid_all,
                self.time_slice_variable(self.aux_at_contexts, date),
            )
            task["X_c"].append(X_c_offrid_all)
            task["Y_c"].append(Y_c_aux)

        if self.aux_at_targets is not None:
            if len(task["X_t"]) > 1:
                raise ValueError(
                    "Cannot add auxiliary variable to target set when there are multiple target variables (not supported by default `ConvNP` model)."
                )
            task["Y_t_aux"] = self.sample_offgrid_aux(
                task["X_t"][0],
                self.time_slice_variable(self.aux_at_targets, date),
            )

        return Task(task)
