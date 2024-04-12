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

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from scipy.interpolate import griddata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle
from tqdm import tqdm
import xarray as xr
from scipy.interpolate import griddata


from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train
from nzdownscale.downscaler.validate import ValidateV1
from nzdownscale.dataprocess import utils
from nzdownscale.dataprocess.config import LOCATION_LATLON, STATION_LATLON
#%%
era5_interp = None
era5_interp_filled = None
#%%  Load from saved model
# model_name = 'test_model_1712875091' # 100
# model_name = 'test_model_1712813969' # 250
# model_name = 'test_model_1712551308' # 500
# model_name = 'test_model_1712551280' # 750
model_name = 'test_model_1712551296' # 1000
# model_name = 'test_model_1712647507' # 1500
# model_name = 'test_model_1712649253' # 2000
# model_name = 'test_model_1712792079' # 2000
base = '/home/emily/deepsensor/deepweather-downscaling/experiments/deepsensor/emily_dev_local/'
train_metadata_path = base + f'models/downscaling/{model_name}/metadata_{model_name}.pkl'
model_path = base + f'models/downscaling/{model_name}/{model_name}.pt'

with open(train_metadata_path, 'rb') as f:
    meta = pickle.load(f)

print(f'Model training date range: {meta["date_info"]["start_year"]} - {meta["date_info"]["end_year"]}')
print(f'Model validation date range: {meta["date_info"]["val_start_year"]} - {meta["date_info"]["val_end_year"]}')
keys_to_exclude = ['train_losses', 'val_losses']
filtered_dict = {key: meta[key] for key in meta if key not in keys_to_exclude}
filtered_dict
print(f'Metadata: {filtered_dict}')
#%% Load validation class
# Takes 10 mins or so to run 
# change the validation date range to a testing range to see how the model performs on unseen data
# validation_date_range = [meta["date_info"]["val_end_year"] + 1, meta["date_info"]["val_end_year"] + 1] #[start_date, end_date]
validation_date_range = [2016, 2020] #inclusive
print(f'Validating model on date range: {validation_date_range[0]} - {validation_date_range[1]}')

save_validate = False
load_validate = not save_validate

if load_validate:
    with open(base + 'data_processor_dict_era1_topohr5_topolr5_2000_2011.pkl', 'rb') as handle:
        data_processor_dict = pickle.load(handle)
        print('Creating validate object using loaded processor dict')
else:
    data_processor_dict = None
    print('Creating validate object without loaded processor dict,\
           may be slow')

# if not load_validate:
# if 'validate' not in locals():
validate = ValidateV1(
    training_metadata_path=train_metadata_path, 
    validation_date_range=validation_date_range,
    data_processor_dict=data_processor_dict
    )
validate.load_model(load_model_path=model_path, 
                    save_data_processing_dict=save_validate)

metadata = validate.get_metadata()
print(metadata)
model = validate.model

#%% ____________________________________________________________________________
# Inspect trained model over given dates
# ------------------------------------------
# date = f'{validation_date_range[0]}-01-01'
date = '2016-01-01'
number_of_months=24
# note that the stations only have data up to 2018-08-01

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
    print(f'Calculating predictions from {date_range[0]} to {date_range[-1]}')
    pred = validate.get_predictions(dates, model, verbose=True, save_preds=False)
    if number_of_months == 12:
        print(f'Saving to {prediction_fpath}')
        utils.save_pickle(pred, prediction_fpath)
    else:
        print('Not saving predictions as number_of_months is not 12')


 #%%
if era5_interp is None:
    era5_unnorm = validate.data_processor.unnormalise(validate.processed_dict['era5_ds'].drop_vars(('cos_D', 'sin_D')))
    era5_interp = era5_unnorm.interp(coords={
        'latitude': pred['latitude'], 
        'longitude': pred['longitude']
        }, 
        method='linear')
    print('ERA5 interpolated')

if era5_interp_filled is None:
    topo = validate.processed_dict['highres_aux_ds']['elevation']
    topo_unnorm = validate.data_processor.unnormalise(topo)
    interpolated_topo = topo_unnorm.interp_like(pred['mean'])
    land_sea_mask = ~(interpolated_topo == 0)
    missing_land_values = xr.where(land_sea_mask & np.isnan(era5_interp),
                                    True, False)
    print('Missing land values calculated')

    # Get the coordinates of valid (non-NaN) and invalid (NaN) data points
    era5_da = era5_interp['t2m'].isel(time=0)
    valid_points = np.array(np.nonzero(~np.isnan(era5_da))).T
    missing_land_values_da = missing_land_values['t2m'].isel(time=0)
    invalid_points = np.array(np.nonzero(missing_land_values_da)).T
    print('Valid and invalid points calculated')

    era5_interp_filled = era5_interp.copy()
    for t in tqdm(era5_interp['t2m'].time):
        era5_da = era5_interp['t2m'].sel(time=t)
        valid_values = era5_da.values[~np.isnan(era5_da)]
        # Perform nearest neighbor interpolation
        interpolated_values = griddata(valid_points,
                                    valid_values, 
                                    invalid_points, 
                                    method='nearest')
        # Fill the era5_interp DataArray with the interpolated values
        era5_interp_filled['t2m'].sel(time=t).values[tuple(invalid_points.T)] = interpolated_values

#%%
loss_dict_era5, locations = validate.calculate_loss_era5(dates, validation_stations, era5_interp_filled) 
 #%% 
# could speed this up with multiprocessing ?
loss_dict, pred_dict, station_dict = validate.calculate_loss(dates, 
                                              locations, 
                                              pred=pred,
                                              return_pred=True, 
                                              return_station=True,
                                              verbose=True)
 
with open(base + f'models/downscaling/{model_name}/loss_dict_{date}_{number_of_months}months.pkl', 'wb') as handle:
    pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
 #%%
print('------ ConvNP predictions -------')
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
# Mean losses per month
monthly_losses = {}
monthly_losses_list = {}
monthly_stds = {}
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
print('ConvNP mean loss for month')
for month in range(1, 13):
    monthly_losses_list[month] = []
    monthly_losses[month] = []
    for location, loss in loss_dict.items():
        monthly_losses_list[month].extend([loss[i] for i in range(len(loss)) if dates[i].month == month])

    monthly_losses[month] = np.nanmean(monthly_losses_list[month])
    monthly_stds[month] = np.nanstd(monthly_losses_list[month])
    print(f'{months[month -1]}: {monthly_losses[month], monthly_stds[month]}')

#%%
print('------ ERA5 -------')
loss_mean_std_era5 = {}
for location, loss in loss_dict_era5.items():
    loss_mean_std_era5[location] = (np.nanmean(loss), np.nanstd(loss))

overall_mean_era5 = np.round(np.nanmean([loss_mean_std_era5[location][0] for location in loss_mean_std_era5.keys()]), 6)
overall_std_era5 = np.round(np.nanmean([loss_mean_std_era5[location][1] for location in loss_mean_std_era5.keys()]), 6)

print(f'Mean loss across all stations: {overall_mean_era5}')
print(f'Mean of std loss across all stations: {overall_std_era5}')

total_loss_era5 = []
for location, loss in loss_dict_era5.items():
    total_loss_era5.extend(loss)
total_loss_era5 = np.round(np.nansum(total_loss_era5), 6)
print(f'Total loss: {total_loss_era5}')
#%%
# Mean losses per month
monthly_losses_era5 = {}
monthly_losses_era5_list = {}
monthly_stds_era5 = {}
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
print('ERA5 mean loss for month')
for month in range(1, 13):
    monthly_losses_era5_list[month] = []
    for location, loss in loss_dict_era5.items():
        monthly_losses_era5_list[month].extend([loss[i] for i in range(len(loss)) if dates[i].month == month])
    
    monthly_losses_era5[month] = np.nanmean(monthly_losses_era5_list[month])
    monthly_stds_era5[month] = np.nanstd(monthly_losses_era5_list[month])
    print(f'{months[month -1]}: {monthly_losses_era5[month], monthly_stds_era5[month]}')

#%%
fig, ax = plt.subplots()
ax.plot(monthly_losses.keys(), monthly_losses.values(), label = 'ConvNP')
# ax.fill_between(monthly_losses.keys(), [monthly_losses[i] - monthly_stds[i] for i in monthly_losses.keys()], [monthly_losses[i] + monthly_stds[i] for i in monthly_losses.keys()], alpha=0.2)

ax.plot(monthly_losses.keys(), monthly_losses_era5.values(), label='ERA5')
# ax.fill_between(monthly_losses.keys(), [monthly_losses_era5[i] - monthly_stds_era5[i] for i in monthly_losses_era5.keys()], [monthly_losses_era5[i] + monthly_stds_era5[i] for i in monthly_losses_era5.keys()], alpha=0.2)
ax.legend()
ax.set_xlabel('Month')
ax.set_ylabel('Average loss at stations')
# %%
# validate.plot_nationwide_prediction(date=date)

#%%
#nationwide
date = '2016-01-01'
validate.plot_ERA5_and_prediction(date=date, 
                                  pred=pred, 
                                  era5=era5_interp_filled, 
                                  remove_sea=True)

# #at location
# for location in ['THE BROTHERS LIGHT', 'WHITE ISLAND AWS', 'BROTHERS ISLAND AWS', 'HAWERA AWS', 'FAREWELL SPIT AWS', 'CASTLEPOINT AWS', 'CAPE KIDNAPPERS WXT AWS']:
#     validate.plot_ERA5_and_prediction(date=date,
#                                     location=location, 
#                                     pred=pred, 
#                                     era5=era5_interp_filled,
#                                     closest_station=False,
#                                     remove_sea=True)
#%% 
# plot errors at stations
validate.plot_errors_at_stations('2016-01-01')
validate.plot_errors_at_stations('2016-07-01')
#%%
sample_locations = ['TAUPO AWS', 'CHRISTCHURCH AERO', 'KAITAIA AERO', 'MT COOK EWS']
if location not in validation_stations:
    sample_locations.remove(location)

for location in tqdm(sample_locations):
    date_range_location = (date_range, '2017-01-01')
    validate.plot_timeseries_comparison(location=location, 
                                        date_range=date_range, 
                                        pred=pred,
                                        return_fig=True)
    
#%%

losses_list = []
for k, v in loss_dict.items():
    losses_list.extend(v)

era5_losses_list = []
for k, v in loss_dict_era5.items():
    era5_losses_list.extend(v)

fig, ax = plt.subplots()
positions = [1, 1.6]
ax.violinplot([losses_list, era5_losses_list], positions=positions)
ax.tick_params(axis='y', labelsize=16)
ax.set_xticks(positions,
                  labels=['ConvNP', 'ERA5'], fontsize=12)
ax.set_ylabel('RMSE', fontsize=16)
# ax.set_title('Violin plot of error for ConvNP and ERA5', fontsize=14)

#%%
import matplotlib.dates as mdates

num_days = 365
fontsize=16
daily_averages = [np.mean([loss_dict[location][day] for location in loss_dict]) for day in range(num_days)]
daily_averages_era5 = [np.mean([loss_dict_era5[location][day] for location in loss_dict_era5]) for day in range(num_days)]
# Plotting
days = range(1, num_days + 1)
days = [datetime(2016, 1, 1) + relativedelta(days=day) for day in days]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(days, daily_averages, label='ConvNP')
ax.plot(days, daily_averages_era5, label='ERA5')

ax.tick_params(labelsize=fontsize)
# ax.set_xlabel('Day of the Year', fontsize=12)
ax.set_ylabel('Average station RMSE', fontsize=fontsize)
# plt.title('Average RMSE Over All Available Stations in 2016')



# Format the x-axis to only show the year and month
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator())  # Optional: to have a tick for each month

# Ensure the x-axis ticks only show the year and month
plt.xticks(rotation=45)  # Optional: Rotate the tick labels for better readability

current_ticks = ax.get_xticks()
new_ticks = current_ticks[:-1]  # Remove the last tick
ax.set_xticks(new_ticks)

ax.legend(fontsize=fontsize)
# ax.grid(True)
#%%

import cartopy.crs as ccrs
fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()), figsize=(15, 15))
ax.spines['geo'].set_visible(False)
ax.coastlines()

lats, lons, elevations = [], [], []
for station, values in STATION_LATLON.items():
    lats.append(values['latitude'])
    lons.append(values['longitude'])
    elevations.append(values['elevation'])

# Convert lists to numpy arrays
lats = np.array(lats)
lons = np.array(lons)
elevations = np.array(elevations)

# Normalize elevations for colormap
norm = plt.Normalize(elevations.min(), elevations.max())

# Plot using scatter, with colormap to indicate elevation
scatter = ax.scatter(lons, lats, s=100, c=elevations, cmap='OrRd',  norm=norm, edgecolor='k', transform=ccrs.Geodetic())

# Adding a colorbar
cbar = plt.colorbar(scatter, shrink=0.5, aspect=10, pad=0.02)
cbar.ax.tick_params(labelsize=16)  # Make ticks larger
cbar.set_label('Elevation (m)', fontsize=16)

plt.show()
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
# validate.plot_prediction_with_stations(labels=loss_dict)
# %%
fig, ax = plt.subplots()

for location, losses in loss_dict.items():
    elev = STATION_LATLON[location]['elevation']
    loss_mean = np.nanmean(losses)
    ax.plot(elev, loss_mean, 'o', label=location, c='green') 

ax.set_xlabel('Elevation (m)')
ax.set_ylabel('Loss (C)')  
# %%
