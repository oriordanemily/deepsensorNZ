#%% 

import deepsensor.torch
from deepsensor.data import DataProcessor, TaskLoader
from deepsensor.model import ConvNP
from deepsensor.train import Trainer

import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch;print(torch.cuda.is_available())

#%% 

# Load raw data
ds_raw = xr.tutorial.open_dataset("air_temperature")

# Normalise data
data_processor = DataProcessor(x1_name="lat", x2_name="lon")
ds = data_processor(ds_raw)

# Set up task loader
task_loader = TaskLoader(context=ds, target=ds)

# Set up model
model = ConvNP(data_processor, task_loader)

# Generate training tasks with up to 10% of grid cells passed as context and all grid cells
# passed as targets
train_tasks = []
for date in pd.date_range("2013-01-01", "2014-11-30")[::7]:
    task = task_loader(date, context_sampling=np.random.uniform(0.0, 0.1), target_sampling="all")
    train_tasks.append(task)

# Train model
trainer = Trainer(model, lr=5e-5)
for epoch in tqdm(range(10)):
    batch_losses = trainer(train_tasks)

# Predict on new task with 10% of context data and a dense grid of target points
test_task = task_loader("2014-12-31", 0.1)
pred = model.predict(test_task, X_t=ds_raw)

#%% 
import logging
logging.captureWarnings(True)

import deepsensor.torch
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.model.convnp import ConvNP
from deepsensor.active_learning.algorithms import GreedyAlgorithm
from deepsensor.active_learning.acquisition_fns import Stddev

from deepsensor.train.train import Trainer, set_gpu_default_device
from deepsensor.data.utils import construct_x1x2_ds

import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns

#%% 
crs = ccrs.PlateCarree()

use_gpu = True
if use_gpu:
    set_gpu_default_device()

#%% 

# Load raw data
ds_raw = xr.tutorial.open_dataset("air_temperature")

print(ds_raw)
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=crs))
ds_raw.isel(time=0).air.plot()
ax.coastlines()


data_processor = DataProcessor(x1_name="lat", x1_map=(ds_raw["lat"].min(), ds_raw["lat"].max()), x2_name="lon", x2_map=(ds_raw["lon"].min(), ds_raw["lon"].max()))
ds = data_processor(ds_raw)

x1x2_ds = construct_x1x2_ds(ds)
ds['x1_arr'] = x1x2_ds['x1_arr']
ds['x2_arr'] = x1x2_ds['x2_arr']
aux_ds = ds[['x1_arr', 'x2_arr']]
ds = ds[['air']]

task_loader = TaskLoader(context=ds, target=ds)

train_tasks = []
for date in pd.date_range("2013-01-01", "2014-11-30")[::7]:
    # Pass up to 10% of grid cells as context and use all grid cells as targets
    task = task_loader(date, context_sampling=np.random.uniform(0.0, 0.1), target_sampling="all")
    train_tasks.append(task)

#%% ttain 

from tqdm import tqdm

n_epochs = 20
losses = []
trainer = Trainer(model, lr=5e-5)
for epoch in tqdm(range(n_epochs)):
    batch_losses = trainer(train_tasks)
    losses.append(np.mean(batch_losses))

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")

