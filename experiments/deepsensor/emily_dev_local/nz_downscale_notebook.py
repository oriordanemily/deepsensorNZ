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
from datetime import datetime
from dateutil.relativedelta import relativedelta

from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train
from nzdownscale.downscaler.validate import ValidateV1
from nzdownscale.dataprocess import utils

#%% 
training = False

#%%

# Settings
# ------------------------------------------

if training:

    var = 'temperature'
    start_year = 2000
    end_year = 2004
    val_start_year = 2005
    val_end_year = 2005
    use_daily_data = True

    topography_highres_coarsen_factor = 5
    topography_lowres_coarsen_factor = 30
    era5_coarsen_factor = 3

    model_name_prefix = 'test'
    n_epochs = 2

    convnp_kwargs = {
        'unet_channels': (64,)*4,
        'likelihood': 'gnp',
        'internal_density': 5,
    }

#%% 
# Preprocess data
# ------------------------------------------
if training: 
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
if training: 
    data.print_resolutions()

    data.plot_dataset('era5')
    data.plot_dataset('top_highres')
    data.plot_dataset('top_lowres')

#%% 
# Train model
# ------------------------------------------
if training:
    training = Train(processed_output_dict=processed_output_dict)

    training.run_training_sequence(n_epochs, model_name_prefix, **convnp_kwargs)

    training_output_dict = training.get_training_output_dict()



#%%
# Load trained model
# ------------------------------------------

# Option 1: load from processed_output_dict and training_output_dict just created
if training: 
    validate = ValidateV1(
        processed_output_dict=processed_output_dict,
        training_output_dict=training_output_dict,
        validation_date_range=[val_start_year, val_end_year],
        )
    validate.load_model()

    metadata = validate.get_metadata()
    print(metadata)
    model = validate.model

#%% 
# Option 2: load from saved model
if not training:
    model_name = 'hr_1_model_1705857990'
    train_metadata_path = f'models/downscaling/metadata/{model_name}.pkl'
    model_path = f'models/downscaling/{model_name}.pt'

    validation_date_range = [2005, 2006] #[start_date, end_date]
    validate = ValidateV1(
    training_metadata_path=train_metadata_path, 
    validation_date_range=validation_date_range)
    validate.load_model(load_model_path=model_path)

    metadata = validate.get_metadata()
    print(metadata)
    model = validate.model

#%%
# Inspect trained model
# ------------------------------------------
date = '2005-02-02'
location = 'christchurch'

validate.plot_nationwide_prediction(date=date)

#%%
#nationwide
validate.plot_ERA5_and_prediction(date=date)

#at location
validate.plot_ERA5_and_prediction(date=date, location=location, closest_station=False)

#%%
validate.plot_prediction_with_stations(date=date, location=location)

validate.plot_prediction_with_stations(date=date, location=location, zoom_to_location=True)

validate.plot_prediction_with_stations(date=date)

#%%
# Add two months to the date
date_obj = datetime.strptime(date, '%Y-%m-%d')
new_date_obj = date_obj + relativedelta(months=2)
new_date_str = new_date_obj.strftime('%Y-%m-%d')

date_range = (date, new_date_str)
validate.plot_timeseries_comparison(location=location, date_range=date_range)

# %%
