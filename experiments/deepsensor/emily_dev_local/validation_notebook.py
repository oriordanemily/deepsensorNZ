#%%
# use below when running interactively to reload modules
# %reload_ext autoreload
# %autoreload 2

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
import pickle
from tqdm import tqdm

from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train
from nzdownscale.downscaler.validate import ValidateV1
from nzdownscale.dataprocess import utils
from nzdownscale.dataprocess.config import LOCATION_LATLON, STATION_LATLON

#%% 
# Load from saved model
model_name = 'hr_5_model_1706121800'
train_metadata_path = f'models/downscaling/metadata/{model_name}.pkl'
model_path = f'models/downscaling/{model_name}.pt'

with open(train_metadata_path, 'rb') as f:
    meta = pickle.load(f)

print(f'Model training date range: {meta["date_info"]["start_year"]} - {meta["date_info"]["end_year"]}')
print(f'Model validation date range: {meta["date_info"]["val_start_year"]} - {meta["date_info"]["val_end_year"]}')
#%%
# change the validation date range to a testing range to see how the model performs on unseen data
validation_date_range = [meta["date_info"]["val_end_year"] + 1, meta["date_info"]["val_end_year"] + 1] #[start_date, end_date]
validate = ValidateV1(
training_metadata_path=train_metadata_path, 
validation_date_range=validation_date_range)
validate.load_model(load_model_path=model_path)

metadata = validate.get_metadata()
print(metadata)
model = validate.model

#%%
# Inspect trained model over given dates
# ------------------------------------------
date = f'{meta["date_info"]["val_end_year"] + 1}-10-01'
number_of_months=3
location = 'taupo'

date_obj = datetime.strptime(date, '%Y-%m-%d')
new_date_obj = date_obj + relativedelta(months=number_of_months) - relativedelta(days=1)
new_date_str = new_date_obj.strftime('%Y-%m-%d')

date_range = (date, new_date_str)
dates = pd.date_range(date_range[0], date_range[1])

#%%
# check to see if station data exists there
validation_stations = validate.stations_in_date_range(date_range)

#%% 
# Note that the full set predictions can be opened with the following:
# with open('/home/emily/deepsensor/deepweather-downscaling/experiments/deepsensor/emily_dev_local/predictions_hr_1_model_1705857990.pkl', 'rb') as f:
#     nz_pred = pickle.load(f)
# %%
prediction_fpath = f'/home/emily/deepsensor/deepweather-downscaling/experiments/deepsensor/emily_dev_local/predictions_{model_name}.pkl'
if os.path.exists(prediction_fpath):
    print('Loading predictions from file')
    predictions = prediction_fpath
    save_preds = False
else:
    predictions = None
    save_preds = True

loss = {}
# if running for the first time, set save_preds=True and don't use the predictions argument.
loss_dict, pred_dict, station_dict = validate.calculate_loss(dates, 
                                              validation_stations, 
                                              predictions=predictions,
                                              save_preds=save_preds,
                                              return_pred=True, 
                                              return_station=True,
                                              verbose=True)
#%%
loss_mean_std = {}
for location, loss in loss_dict.items():
    loss_mean_std[location] = (np.nanmean(loss), np.nanstd(loss))

overall_mean = np.nanmean([loss_mean_std[location][0] for location in loss_mean_std.keys()])
overall_std = np.nanmean([loss_mean_std[location][1] for location in loss_mean_std.keys()])

print(f'Mean loss: {overall_mean}')
print(f'Mean of std loss: {overall_std}')

total_loss = 0 
for location, loss in loss_mean_std.items():
    if np.isnan(loss[0]):
        continue
    total_loss += loss[0]
print(f'Total loss: {total_loss}')

#%%


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
sample_locations = ['TAUPO AWS', 'CHRISTCHURCH AERO', 'KAITAIA AERO', 'MT COOK EWS']
if location not in validation_stations:
    sample_locations.remove(location)

# takes about 3 mins to run the below per station
for location in tqdm(sample_locations):
    validate.plot_timeseries_comparison(location=location, date_range=date_range, predictions=predictions)


#%%
# validate.calculate_loss(dates, validation_stations[2], 'l1', save_preds=True)



#%%
# plot average losses for all stations
validate.plot_losses(dates, loss_dict, pred_dict, station_dict)

# plot for a single station
validate.plot_losses(dates, loss_dict, pred_dict, station_dict, location='MT COOK EWS')

# validate.plot_losses(dates, loss)

# %%
# this has plot with loss at stations over a given time period
# but obvs only plots the prediction values on the map on one date
# misleading plot? 
validate.plot_prediction_with_stations(labels=loss)
# %%
fig, ax = plt.subplots()

for location, losses in loss_dict.items():
    elev = STATION_LATLON[location]['elevation']
    loss_mean = np.nanmean(losses)
    ax.plot(elev, loss_mean, 'o', label=location, c='green') 

ax.set_xlabel('Elevation (m)')
ax.set_ylabel('Loss (C)')  
# %%
