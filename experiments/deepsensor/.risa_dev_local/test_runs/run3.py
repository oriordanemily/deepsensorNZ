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
#from nzdownscale import downscaler
from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train

var = 'temperature'
start_year = 2000
end_year = 2005
train_start_year = 2000
val_start_year = 2005

topography_highres_coarsen_factor = 30
topography_lowres_coarsen_factor = 10
era5_coarsen_factor = 5

model_name_prefix = 'run3'
epochs = 30

#%%

data = PreprocessForDownscaling(
    variable = 'temperature',
    start_year = start_year,
    end_year = end_year,
    train_start_year = train_start_year,
    val_start_year = val_start_year,
)

data.load_topography()
data.load_era5()
data.load_stations()

highres_aux_raw_ds, aux_raw_ds = data.preprocess_topography(highres_coarsen_factor=topography_highres_coarsen_factor, lowres_coarsen_factor=topography_lowres_coarsen_factor)
era5_raw_ds = data.preprocess_era5(coarsen_factor=era5_coarsen_factor)
station_raw_df = data.preprocess_stations()

processed_data = data.process_all(era5_raw_ds, highres_aux_raw_ds, aux_raw_ds, station_raw_df)

print(processed_data.keys())

#%% 

training = Train(
    processed_data=processed_data,
)

training.setup_task_loader()
training.initialise_model()
training.train_model(n_epochs=epochs, model_name_prefix=model_name_prefix)

