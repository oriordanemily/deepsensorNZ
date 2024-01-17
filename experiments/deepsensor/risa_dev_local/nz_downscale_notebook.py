import logging
logging.captureWarnings(True)
import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train
from nzdownscale.downscaler.validate import ValidateV1

#%% 
# ------------------------------------------
# Settings
# ------------------------------------------

var = 'temperature'
start_year = 2000
end_year = 2001
train_start_year = 2000
val_start_year = 2001
use_daily_data = True

topography_highres_coarsen_factor = 30
topography_lowres_coarsen_factor = 10
era5_coarsen_factor = 10

model_name_prefix = 'run_test'
n_epochs = 2

#%%
# ------------------------------------------
# Preprocess data
# ------------------------------------------

data = PreprocessForDownscaling(
    variable = var,
    start_year = start_year,
    end_year = end_year,
    val_start_year = val_start_year,
    use_daily_data = use_daily_data,
)

data.load_topography()
data.load_era5()
data.load_stations()

highres_aux_raw_ds, aux_raw_ds = data.preprocess_topography(topography_highres_coarsen_factor, topography_lowres_coarsen_factor)
era5_raw_ds = data.preprocess_era5(coarsen_factor=era5_coarsen_factor)
station_raw_df = data.preprocess_stations()

data.process_all(era5_raw_ds, highres_aux_raw_ds, aux_raw_ds, station_raw_df)
processed_output_dict = data.get_processed_output_dict()

# ------------------------------------------
# Plot info
# ------------------------------------------

data.print_resolutions()

data.plot_dataset('era5')
data.plot_dataset('top_highres')
data.plot_dataset('top_lowres')

# ------------------------------------------
# Train model
# ------------------------------------------

training = Train(processed_output_dict=processed_output_dict,
                convnp_settings='default',
                )

training.setup_task_loader()
training.initialise_model()
training.train_model(n_epochs=n_epochs, model_name_prefix=model_name_prefix)

training_output_dict = training.get_training_output_dict()

# ------------------------------------------
# Inspect trained model
# ------------------------------------------

validate = ValidateV1(
    processed_output_dict=processed_output_dict,
    #training_output_dict=training_output_dict,
    training_output_dict=None,
    )

validate.initialise(load_model_path='models/downscaling/run1_model_1705453547.pt')

#%% ! bug

validate.plot_example_prediction()
validate.emily_plots()
