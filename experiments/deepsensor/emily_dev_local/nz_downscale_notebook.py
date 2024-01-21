#%%
import logging
logging.captureWarnings(True)
import os
import time
import importlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train
from nzdownscale.downscaler.validate import ValidateV1
from nzdownscale.dataprocess import utils

#%% 
# Settings
# ------------------------------------------

var = 'temperature'
start_year = 2000
end_year = 2001
val_start_year = 2002
val_end_year = 2002
use_daily_data = True

topography_highres_coarsen_factor = 30
topography_lowres_coarsen_factor = 30
era5_coarsen_factor = 30

model_name_prefix = 'new_test'
n_epochs = 2

convnp_kwargs = {
    'unet_channels': (64,)*4,
    'likelihood': 'gnp',
    'internal_density': 5,
}

#%% 
# Preprocess data
# ------------------------------------------

data = PreprocessForDownscaling(
    variable = var,
    start_year = start_year,
    end_year = end_year,
    val_start_year = val_start_year,
    val_end_year = val_start_year,
    use_daily_data = use_daily_data,
)
data.run_processing_sequence(
    topography_highres_coarsen_factor,
    topography_lowres_coarsen_factor, 
    era5_coarsen_factor,
    )
processed_output_dict = data.get_processed_output_dict()

#%% 
# Plot info
# ------------------------------------------

data.print_resolutions()

data.plot_dataset('era5')
data.plot_dataset('top_highres')
data.plot_dataset('top_lowres')

#%% 
# Train model
# ------------------------------------------

training = Train(processed_output_dict=processed_output_dict)

training.run_training_sequence(n_epochs, model_name_prefix, **convnp_kwargs)

training_output_dict = training.get_training_output_dict()

#%%
# Load trained model
# ------------------------------------------

# Option 1: load from processed_output_dict and training_output_dict just created

validate = ValidateV1(
    processed_output_dict=processed_output_dict,
    training_output_dict=training_output_dict,
    )
validate.load_model()

metadata = validate.get_metadata()
print(metadata)
model = validate.model

#%% 
# Option 2: load from saved model

train_metadata_path = 'models/downscaling/metadata/new_test_model_1705607353.pkl'
model_path = 'models/downscaling/new_test_model_1705607353.pt'

validate = ValidateV1(
training_metadata_path=train_metadata_path)
validate.load_model(load_model_path=model_path)

metadata = validate.get_metadata()
print(metadata)
model = validate.model

#%%
# Inspect trained model
# ------------------------------------------
# ! bug

validate.plot_example_prediction()
validate.emily_plots()
