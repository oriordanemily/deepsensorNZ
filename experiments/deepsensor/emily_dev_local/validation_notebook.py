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
# Load from saved model
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
