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
model_name = 'hr_1_model_1705857990'
train_metadata_path = f'models/downscaling/metadata/{model_name}.pkl'
model_path = f'models/downscaling/{model_name}.pt'

with open(train_metadata_path, 'rb') as f:
    meta = pickle.load(f)

print(f'Model training date range: {meta["date_info"]["start_year"]} - {meta["date_info"]["end_year"]}')
print(f'Model validation date range: {meta["date_info"]["val_start_year"]} - {meta["date_info"]["val_end_year"]}')

# change the validation date range to a testing range to see how the model performs on unseen data
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
date = '2005-07-02'
location = 'taupo'
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
# Add two months to the date
date_obj = datetime.strptime(date, '%Y-%m-%d')
new_date_obj = date_obj + relativedelta(months=2)
new_date_str = new_date_obj.strftime('%Y-%m-%d')

date_range = (date, new_date_str)
# takes about 3 mins to run the below
# validate.plot_timeseries_comparison(location=location, date_range=date_range)

# %%

dates = pd.date_range(date_range[0], date_range[1])

losses, preds, stats = [], [], []
for date in dates:
    loss, pred, stat = validate.calculate_loss(date, location, 'l1', return_pred=True, return_station=True)
    losses.append(loss)
    preds.append(pred)
    stats.append(stat)

#%%

fig, ax = plt.subplots()

ax.plot(dates, losses, color='b', label = 'losses')
ax.set_ylabel('Losses (pred - station)', color='b')

ax2 = ax.twinx()
ax2.plot(dates, preds, color='r', label='preds', alpha=0.5)
ax2.plot(dates, stats, color='g', label='stations', alpha=0.5)
ax2.set_ylabel('Temperature (C)')

ax.legend(loc='upper left')
ax2.legend(loc='upper right')

ax.set_title(f'Losses for {location} from {date_range[0]} to {date_range[1]}')
 #%%
date = '2006-01-01'
date_obj = datetime.strptime(date, '%Y-%m-%d')
new_date_obj = date_obj + relativedelta(months=2)
new_date_str = new_date_obj.strftime('%Y-%m-%d')

# check to see if station data exists there
date_range = (date, new_date_str)
validation_stations = validate.stations_in_date_range(date_range)
#%%
validate.calculate_loss(date, validation_stations[0], 'l1')

# %%
loss = {}
dates = pd.date_range(date_range[0], date_range[1])

loss, pred, station = validate.calculate_loss(dates, validation_stations, return_pred=True, return_station=True)
# %%
loss_dict = {}
for location, losses in loss.items():
    loss_dict[location] = (float(np.mean(losses).values), float(np.std(losses).values))
# %%
location = 'HAMILTON AERO'
mean_loss = loss_dict[location][0]
mean_loss
# %%
def mean_loss_by_date(date, loss, std=False):
    date_losses = []
    for loc, losses in loss.items():
        date_losses.append(float(losses.sel({'time': date}).values))
    if std:
        return np.nanmean(date_losses), np.nanstd(date_losses)
    else:
        return np.nanmean(date_losses)
#%%
fig, ax = plt.subplots()
mean_losses, std_losses = [], []
for date in dates:
    mean_loss, std_loss = mean_loss_by_date(date, loss, std=True)
    mean_losses.append(mean_loss)
    std_losses.append(std_loss)
ax.plot(dates, mean_losses, label = 'mean')
ax.plot(dates, [mean_losses[i] + std_losses[i] for i in range(len(mean_losses))], alpha = 0.5, label = 'mean + std')
ax.legend();
# %%
# this has plot with loss at stations over a given time period
# but obvs only plots the prediction values on the map on one date
# misleading plot? 
validate.plot_prediction_with_stations(labels=loss)
# %%
