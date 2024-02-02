#%%
# use below when running interactively to reload modules
%reload_ext autoreload
%autoreload 2

import logging
logging.captureWarnings(True)
import os
import time
import importlib
import glob

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

#%%  Load from saved model
model_name_pattern = 'hr_5_model_*'
model_name = glob.glob(f'models/downscaling/{model_name_pattern}.pt')[0].split('/')[-1].split('.')[0]
train_metadata_path = f'models/downscaling/metadata/{model_name}.pkl'
model_path = f'models/downscaling/{model_name}.pt'

with open(train_metadata_path, 'rb') as f:
    meta = pickle.load(f)

print(f'Model training date range: {meta["date_info"]["start_year"]} - {meta["date_info"]["end_year"]}')
print(f'Model validation date range: {meta["date_info"]["val_start_year"]} - {meta["date_info"]["val_end_year"]}')

#%% Load validation class
# Takes 10 mins or so to run 
# change the validation date range to a testing range to see how the model performs on unseen data
# validation_date_range = [meta["date_info"]["val_end_year"] + 1, meta["date_info"]["val_end_year"] + 1] #[start_date, end_date]
validation_date_range = [2018, 2018] #inclusive
print(f'Validating model on date range: {validation_date_range[0]} - {validation_date_range[1]}')

# takes ~10 mins to load 1 year of station data
# ! to do: might be able to speed this up by saving DataProcessor
# tried saving processed_output_dict and training_output_dict to file but it's still slow
validate = ValidateV1(
    training_metadata_path=train_metadata_path, 
    validation_date_range=validation_date_range
    )
validate.load_model(load_model_path=model_path)

metadata = validate.get_metadata()
print(metadata)
model = validate.model

#%% ____________________________________________________________________________
# Inspect trained model over given dates
# ------------------------------------------
date = f'{validation_date_range[0]}-01-01'
number_of_months=12

date_obj = datetime.strptime(date, '%Y-%m-%d')
new_date_obj = date_obj + relativedelta(months=number_of_months) - relativedelta(days=1)
new_date_str = new_date_obj.strftime('%Y-%m-%d')

date_range = (date, new_date_str)
dates = pd.date_range(date_range[0], date_range[1])
print(f'Validating for dates between {date_range}')

#%%
# check to see if station data exists there
validation_stations = validate.stations_in_date_range(date_range)

#%%
prediction_fpath = f'/home/emily/deepsensor/deepweather-downscaling/experiments/deepsensor/emily_dev_local/predictions_{model_name}_{validation_date_range[0]}.pkl'
if os.path.exists(prediction_fpath):
    print('Loading predictions from file')
    pred = utils.open_pickle(prediction_fpath)
    save_preds = False
else:
    # Takes 5 mins to run for a year of data
    print(f'Calculating predictions from {date_range[0]} to {date_range[-1]}')
    pred = validate.get_predictions(dates, model, verbose=True, save_preds=False)
    if number_of_months == 12:
        print(f'Saving to {prediction_fpath}')
        utils.save_pickle(pred, prediction_fpath)
    else:
        print('Not saving predictions as number_of_months is not 12')


# %%
loss = {}
# could speed this up with multiprocessing ?
loss_dict, pred_dict, station_dict = validate.calculate_loss(dates, 
                                              validation_stations, 
                                              pred=pred,
                                              return_pred=True, 
                                              return_station=True,
                                              verbose=True)
#%%
loss_mean_std = {}
for location, loss in loss_dict.items():
    loss_mean_std[location] = (np.nanmean(loss), np.nanstd(loss))

overall_mean = np.round(np.nanmean([loss_mean_std[location][0] for location in loss_mean_std.keys()]), 6)
overall_std = np.round(np.nanmean([loss_mean_std[location][1] for location in loss_mean_std.keys()]), 6)

print(f'Mean loss across all stations: {overall_mean}')
print(f'Mean of std loss across all stations: {overall_std}')

total_loss = []
for location, loss in loss_dict.items():
    total_loss.extend(loss)
total_loss = np.round(np.nansum(total_loss), 6)
print(f'Total loss: {total_loss}')

#%%
validate.plot_nationwide_prediction(date=date)
#%%
#nationwide
validate.plot_ERA5_and_prediction(date=date, pred=pred)

#at location
validate.plot_ERA5_and_prediction(date=date,
                                location=location, 
                                pred=pred, 
                                closest_station=False)

#%%
sample_locations = ['TAUPO AWS', 'CHRISTCHURCH AERO', 'KAITAIA AERO', 'MT COOK EWS']
if location not in validation_stations:
    sample_locations.remove(location)

for location in tqdm(sample_locations):
    validate.plot_timeseries_comparison(location=location, 
                                        date_range=date_range, 
                                        pred=pred,
                                        return_fig=True)
#%%

validate.plot_prediction_with_stations(date=date, 
                                       location=location, 
                                       pred=pred)

validate.plot_prediction_with_stations(date=date, 
                                       location=location, 
                                       pred=pred, 
                                       zoom_to_location=True)

validate.plot_prediction_with_stations(date=date, 
                                       pred=pred)

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
