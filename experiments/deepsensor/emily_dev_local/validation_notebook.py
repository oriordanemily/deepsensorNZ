# %%
# use below when running interactively to reload modules
%reload_ext autoreload
%autoreload 2

import logging

logging.captureWarnings(True)
import os
import time
import importlib
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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



# %%
era5_interp = None
era5_interp_filled = None
# %%  Load from saved model
#### TEMP 
# model_name = "test_model_1713164678"  # 100 - left out stations
# model_name = "model_test_var_batching"
# model_name = 'model_temp_incstations'
# model_name = 'model_all_stations_context'
# model_name = 'model_1714684457'
# model_name = 'model_1716532171'
# model_name = 'model_model_radiation'
model_name = 'model_full_allcontextvar'

# --- from mahuika ----
# model_name = 'test_model_1712898775' # 50
# model_name = 'test_model_1712875091' # 100
# model_name = 'test_model_1712813969' # 250
# model_name = 'test_model_1712551308' # 500
# model_name = 'test_model_1712551280' # 750
# model_name = 'test_model_1712551296' # 1000
# model_name = 'test_model_1712647507' # 1500
# model_name = 'test_model_1712649253' # 2000
# model_name = 'test_model_1712792079' # 2000
# model_name = '_model_1713135569'
# model_name = '_model_1713136369'

# PRECIP
# model_name = 'model_precip_100epochs'


base = '/home/emily/deepsensor/deepweather-downscaling/experiments/'
base2 = "/home/emily/deepsensor/deepweather-downscaling/experiments/deepsensor/emily_dev_local/"

model_dir = base + f"models/downscaling/{model_name}/"

train_metadata_path = (
    model_dir + f"metadata_{model_name}.pkl"
)
model_path = model_dir + f"{model_name}.pt"

with open(train_metadata_path, "rb") as f:
    meta = pickle.load(f)

st_yr = meta["date_info"]["start_year"]
end_yr = meta["date_info"]["end_year"]
var = meta['data_settings']['var']

print(
    f'Model training date range: {st_yr} - {end_yr}'
)
print(
    f'Model validation date range: {meta["date_info"]["val_start_year"]} - {meta["date_info"]["val_end_year"]}'
)
keys_to_exclude = ["train_losses", "val_losses"]
filtered_dict = {key: meta[key] for key in meta if key not in keys_to_exclude}
filtered_dict
print(f"Metadata: {filtered_dict}")
# %% Load validation class

# change the validation date range to a testing range to see how the model performs on unseen data
validation_date_range = [2016, 2016]  # inclusive
print(
    f"Validating model on date range: {validation_date_range[0]} - {validation_date_range[1]}"
)
# dp_path = base + f'models/{model_name[6:]}/not'

dp_path = base + f'models/downscaling/{model_name}/data_processor_dict_{var}_{model_name}.pkl'
# dp_path = 'not'
# dp_path = '/home/emily/deepsensor/deepweather-downscaling/data_processor_dict_temp_model_radiation.pkl'
# if os.path.exists(dp_path):
#     with open(
#         dp_path, "rb"
#     ) as handle:
#         data_processor_dict = pickle.load(handle)
#         print("Creating validate object using loaded processor dict")
#     save_dp = None # don't save as it's already saved
# else:
data_processor_dict = None
print(
    "Creating validate object without loaded processor dict,\
        may be slow"
)
save_dp = dp_path # save the processor dict to this fpath

remove_stations_list = [
    "TAUPO AWS",
    "CHRISTCHURCH AERO",
    "KAITAIA AERO",
    "MT COOK EWS",
    "AUCKLAND AERO",
    "ALEXANDRA AWS",
    "TOLAGA BAY WXT AWS",
    "WELLINGTON AERO",
    "BLENHEIM AERO",
    "DUNEDIN AERO AWS",
]
# if not load_validate:
# if 'validate' not in locals():
validate = ValidateV1(
    training_metadata_path=train_metadata_path,
    validation_date_range=validation_date_range,
    data_processor_dict=data_processor_dict,
)
# validate.station_as_context = True
#%%
validate.load_model(
    load_model_path=model_path,
    save_data_processing_dict=save_dp,
)

# validate.model = validate._load_pretrained_model(model_path)
metadata = validate.get_metadata()
print(metadata)
model = validate.model

# %% ____________________________________________________________________________
# Inspect trained model over given dates
# ------------------------------------------
# date = f'{validation_date_range[0]}-01-01'
# date = "2016-01-01"
date = '2016-01-01'
number_of_months = 12
# note that the stations only have data up to 2019-08-01

date_obj = datetime.strptime(date, "%Y-%m-%d")
new_date_obj = date_obj + relativedelta(months=number_of_months) - relativedelta(days=1)
new_date_str = new_date_obj.strftime("%Y-%m-%d")

date_range = (date, new_date_str)
dates = pd.date_range(date_range[0], date_range[1])
print(f"Validating for dates between {date_range}")

# %%
# check to see if station data exists there
validation_stations = validate.stations_in_date_range(date_range)

# %%
prediction_fpath = f"/home/emily/deepsensor/deepweather-downscaling/experiments/models/downscaling/{model_name}/predictions_{model_name}_{validation_date_range[0]}.pkl"
if os.path.exists(prediction_fpath):
    print("Loading predictions from file")
    pred = utils.open_pickle(prediction_fpath)
    save_preds = False
else:
    print(f"Calculating predictions from {date_range[0]} to {date_range[-1]}")
    pred = validate.get_predictions(
        dates,
        model,
        verbose=True,
        save_preds=False,
        remove_stations_from_tasks=remove_stations_list,
    )
    if number_of_months == 12:
        print(f"Saving to {prediction_fpath}")
        utils.save_pickle(pred, prediction_fpath)
    else:
        print("Not saving predictions as number_of_months is not 12")

# %%
dataprocess = utils.DataProcess()
resolution = dataprocess.resolution(pred["mean"].isel(time=0), "latitude")
print("Prediction resolution:", resolution)
# %%
# if 2016 - 2018 then just use a pre-done one,
# model_with_era5_done = 'model_1714684457' #precip
model_with_era5_done = model_name
# otherwise use model_name instead of model_with_era5_done
era5_interp = None
if era5_interp is None:
    era5_interp_path = (
        base
        + f"models/downscaling/{model_with_era5_done}/era5_interp_{date}_{number_of_months}months.pkl"
    )
    if os.path.exists(era5_interp_path):
        print("Loading interpolated ERA5 from file")
        with open(era5_interp_path, "rb") as handle:
            era5_interp = pickle.load(handle)
    else:
        era5_unnorm = validate.data_processor.unnormalise(
            validate.processed_dict["era5_ds"].drop_vars(("cos_D", "sin_D"), errors='ignore')
        )
        print("ERA5 unnormalised")
        pred_coarse = pred.coarsen(latitude=5, longitude=5, boundary="trim").mean()
        # pred_coarse = pred
        era5_interp = era5_unnorm.interp(
            coords={
                "latitude": pred_coarse["latitude"],
                "longitude": pred_coarse["longitude"],
            },
            method="linear",
        )
        with open(era5_interp_path, "wb") as handle:
            pickle.dump(era5_interp, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("ERA5 interpolated")

era5_var = validate.get_variable_name('era5')
era5_interp = era5_interp[era5_var]
#%%
era5_interp_filled = None
if era5_interp_filled is None:
    era5_interp_filled_path = (
        base
        + f"models/downscaling/{model_with_era5_done}/era5_interp_filled_{date}_{number_of_months}months.pkl"
    )
    if os.path.exists(era5_interp_filled_path):
        print("Loading filled ERA5 from file")
        with open(era5_interp_filled_path, "rb") as handle:
            era5_interp_filled = pickle.load(handle)
    else:
        topo = validate.processed_dict["highres_aux_ds"]["elevation"]
        topo_unnorm = validate.data_processor.unnormalise(topo)
        interpolated_topo = topo_unnorm.interp_like(pred_coarse["mean"])
        land_sea_mask = ~(interpolated_topo == 0)
        missing_land_values = xr.where(
            land_sea_mask & np.isnan(era5_interp), True, False
        )
        print("Missing land values calculated")

        era5_interp_filled = era5_interp.copy()
        era5_var = validate.get_variable_name('era5')
        for t in tqdm(era5_interp.time, desc="Filling missing values"):
            era5_da = era5_interp.sel(time=t)
            valid_points = np.array(np.nonzero(~np.isnan(era5_da))).T
            valid_values = era5_da.values[~np.isnan(era5_da)]
            missing_land_values_da = missing_land_values.sel(time=t)
            invalid_points = np.array(np.nonzero(missing_land_values_da)).T
            # Perform nearest neighbor interpolation
            interpolated_values = griddata(
                valid_points, valid_values, invalid_points, method="nearest"
            )
            # Fill the era5_interp DataArray with the interpolated values
            era5_interp_filled.sel(time=t).values[tuple(invalid_points.T)] = (
                interpolated_values
            )

        with open(era5_interp_filled_path, "wb") as handle:
            pickle.dump(era5_interp_filled, handle, protocol=pickle.HIGHEST_PROTOCOL)

era5_interp_filled = era5_interp_filled
if era5_var == 't2m':
    era5_interp_filled -= 273.15
# %%
validation_stations.remove("CAPE CAMPBELL AWS")
validation_stations.remove("CAPE KIDNAPPERS WXT AWS")
#%%
loss_dict_era5, locations = validate.calculate_loss_era5(
    dates, validation_stations, era5_interp_filled
)
# %%
# could speed this up with multiprocessing ?
loss_dict, pred_dict, station_dict = validate.calculate_loss(
    dates, locations, pred=pred, return_pred=True, return_station=True, verbose=True
)

with open(
    base
    + f"models/downscaling/{model_name}/loss_dict_{date}_{number_of_months}months.pkl",
    "wb",
) as handle:
    pickle.dump(loss_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
print("------ ConvNP predictions -------")
loss_mean_std = {}
for location, loss in loss_dict.items():
    loss_mean_std[location] = (np.nanmean(loss), np.nanstd(loss))

overall_mean = np.round(
    np.nanmean([loss_mean_std[location][0] for location in loss_mean_std.keys()]), 6
)
overall_std = np.round(
    np.nanmean([loss_mean_std[location][1] for location in loss_mean_std.keys()]), 6
)

print(f"Mean loss across all stations: {overall_mean}")
print(f"Mean of std loss across all stations: {overall_std}")

total_loss = []
for location, loss in loss_dict.items():
    total_loss.extend(loss)
total_loss = np.round(np.nansum(total_loss), 6)
print(f"Total loss: {total_loss}")

# %%
# Mean losses per month
monthly_losses = {}
monthly_losses_list = {}
monthly_stds = {}
months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
print("ConvNP mean loss for month")
for month in range(1, 13):
    monthly_losses_list[month] = []
    monthly_losses[month] = []
    for location, loss in loss_dict.items():
        monthly_losses_list[month].extend(
            [loss[i] for i in range(len(loss)) if dates[i].month == month]
        )

    monthly_losses[month] = np.nanmean(monthly_losses_list[month])
    monthly_stds[month] = np.nanstd(monthly_losses_list[month])
    print(f"{months[month -1]}: {monthly_losses[month], monthly_stds[month]}")

# %%
print("------ ERA5 -------")
loss_mean_std_era5 = {}
for location, loss in loss_dict_era5.items():
    loss_mean_std_era5[location] = (np.nanmean(loss), np.nanstd(loss))

overall_mean_era5 = np.round(
    np.nanmean(
        [loss_mean_std_era5[location][0] for location in loss_mean_std_era5.keys()]
    ),
    6,
)
overall_std_era5 = np.round(
    np.nanmean(
        [loss_mean_std_era5[location][1] for location in loss_mean_std_era5.keys()]
    ),
    6,
)

print(f"Mean loss across all stations: {overall_mean_era5}")
print(f"Mean of std loss across all stations: {overall_std_era5}")

total_loss_era5 = []
for location, loss in loss_dict_era5.items():
    total_loss_era5.extend(loss)
total_loss_era5 = np.round(np.nansum(total_loss_era5), 6)
print(f"Total loss: {total_loss_era5}")
# %%
# Mean losses per month
monthly_losses_era5 = {}
monthly_losses_era5_list = {}
monthly_stds_era5 = {}
months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
print("ERA5 mean loss for month")
for month in range(1, 13):
    monthly_losses_era5_list[month] = []
    for location, loss in loss_dict_era5.items():
        monthly_losses_era5_list[month].extend(
            [loss[i] for i in range(len(loss)) if dates[i].month == month]
        )

    monthly_losses_era5[month] = np.nanmean(monthly_losses_era5_list[month])
    monthly_stds_era5[month] = np.nanstd(monthly_losses_era5_list[month])
    print(f"{months[month -1]}: {monthly_losses_era5[month], monthly_stds_era5[month]}")
# %%
df_losses = pd.DataFrame(
    [
        (location, loss_mean_std[location][0], loss_mean_std_era5[location][0])
        for location in loss_dict.keys()
    ],
    columns=["Location", "ConvNP", "ERA5"],
)
df_losses.sort_values(by="ERA5", ascending=False, inplace=True)
pd.set_option("display.max_rows", None)  # None means show all rows

df_losses
# %%
print("------ ConvNP predictions at unseen stations -------")
loss_mean_std_unseen = {}
unseen_locations = [
    "TAUPO AWS",
    "CHRISTCHURCH AERO",
    "KAITAIA AERO",
    "MT COOK EWS",
    "AUCKLAND AERO",
    "ALEXANDRA AWS",
    "TOLAGA BAY WXT AWS",
    "WELLINGTON AERO",
    "BLENHEIM AERO",
    "DUNEDIN AERO AWS",
]
for location, loss in loss_dict.items():
    if location in unseen_locations:
        loss_mean_std_unseen[location] = (np.nanmean(loss), np.nanstd(loss))

print(f"Found {len(loss_mean_std_unseen)}/{len(unseen_locations)} unseen stations")
overall_mean = np.round(
    np.nanmean(
        [loss_mean_std_unseen[location][0] for location in loss_mean_std_unseen.keys()]
    ),
    6,
)
overall_std = np.round(
    np.nanmean(
        [loss_mean_std_unseen[location][1] for location in loss_mean_std_unseen.keys()]
    ),
    6,
)

print(f"Mean loss across all unseen stations: {overall_mean}")
print(f"Mean of std loss across all unseen stations: {overall_std}")

# total_loss = []
# for location, loss in loss_dict.items():
#     if location in unseen_locations:
#         total_loss.extend(loss)
# total_loss = np.round(np.nansum(total_loss), 6)
# print(f'Total loss: {total_loss}')

# %%
print("------ ERA5-Land at unseen stations -------")
loss_mean_std_era5_unseen = {}
for location, loss in loss_dict_era5.items():
    if location in unseen_locations:
        loss_mean_std_era5_unseen[location] = (np.nanmean(loss), np.nanstd(loss))

print(f"Found {len(loss_mean_std_era5_unseen)}/{len(unseen_locations)} unseen stations")
overall_mean = np.round(
    np.nanmean(
        [
            loss_mean_std_era5_unseen[location][0]
            for location in loss_mean_std_era5_unseen.keys()
        ]
    ),
    6,
)
overall_std = np.round(
    np.nanmean(
        [
            loss_mean_std_era5_unseen[location][1]
            for location in loss_mean_std_era5_unseen.keys()
        ]
    ),
    6,
)

print(f"Mean loss across all unseen stations: {overall_mean}")
print(f"Mean of std loss across all unseen stations: {overall_std}")

# %%
fig, ax = plt.subplots()
ax.plot(monthly_losses.keys(), monthly_losses.values(), label="ConvNP")
# ax.fill_between(monthly_losses.keys(), [monthly_losses[i] - monthly_stds[i] for i in monthly_losses.keys()], [monthly_losses[i] + monthly_stds[i] for i in monthly_losses.keys()], alpha=0.2)

ax.plot(monthly_losses.keys(), monthly_losses_era5.values(), label="ERA5")
# ax.fill_between(monthly_losses.keys(), [monthly_losses_era5[i] - monthly_stds_era5[i] for i in monthly_losses_era5.keys()], [monthly_losses_era5[i] + monthly_stds_era5[i] for i in monthly_losses_era5.keys()], alpha=0.2)
ax.legend()
ax.set_xlabel("Month")
ax.set_ylabel("Average loss at stations")
# %%
# validate.plot_nationwide_prediction(date=date)

# %%
# nationwide
date = "2016-01-01"
validate.plot_ERA5_and_prediction(
    date=date,
    pred=pred,
    #   era5=era5_interp_filled,
    remove_sea=True,
)
# 'THE BROTHERS LIGHT', 'WHITE ISLAND AWS',
# # #at location
# for location in validation_stations:
#     if location not in locations:
#         print(location)
#         validate.plot_ERA5_and_prediction(date=date,
#                                         location=location,
#                                         pred=pred,
#                                         era5=era5_interp_filled,
#                                         closest_station=False,
#                                         remove_sea=True)
# %%
# plot errors at stations
validate.plot_errors_at_stations("2016-01-01")
validate.plot_errors_at_stations("2016-07-01")

#%% 
validate.plot_stations_and_prediction("2017-5-2", pred=pred, )

# %%
# sample_locations = ['TAUPO AWS', 'CHRISTCHURCH AERO', 'KAITAIA AERO', 'MT COOK EWS']
# sample_locations = ['TAUPO AWS', 'MT COOK EWS', 'ALEXANDRA AWS', 'TOLAGA BAY WXT AWS', 'WELLINGTON AERO']
# sample_locations = ['MILFORD SOUND', 'MT COOK EWS', 'FRANZ JOSEF EWS', 'WELLINGTON AERO AWS', 'TAUPO AERO', 'BLENHEIM AERO', 'DUNEDIN AERO', 'GISBORNE AERO']
# sample_locations =  ['WELLINGTON AERO']
sample_locations = ["MT COOK EWS", "WELLINGTON AERO"]
# sample_locations = ['CAPE REINGA AWS']
# sample_locations = ['KAIKOURA AWS']
# sample_locations = ['NEW PLYMOUTH AERO']
# sample_locations = ['TAUPO AERO']
# sample_locations = ['SECRETARY ISLAND AWS']
# sample_locations = ['MT POTTS EWS']
for location in sample_locations:
    if location not in validation_stations:
        sample_locations.remove(location)

for location in tqdm(sample_locations):
    date_range_location = (date_range[0], "2016-07-01")
    # date_range_location = ('2016-07-01', '2016-12-31')
    # date_range_locations = ('2016-01-01', '2016-12-31')
    validate.plot_timeseries_comparison(
        location=location,
        date_range=date_range_location,
        pred=pred,
        era5=era5_interp_filled,
        return_fig=True,
    )

# %%

# %%

sample_locations = [
    "TAUPO AWS",
    "CHRISTCHURCH AERO",
    "KAITAIA AERO",
    "MT COOK EWS",
    "ALEXANDRA AWS",
    "WELLINGTON AERO",
    "DUNEDIN AERO AWS",
    "AUCKLAND AERO",
    "BLENHEIM AERO",
    "TOLAGA BAY WXT AWS",
]

losses_list = []
for k, v in loss_dict.items():
    # if k in sample_locations:
    if k != "WHITE ISLAND AWS":
        v = [x for x in v if str(x) != "nan"]
        losses_list.extend(v)

era5_losses_list = []
for k, v in loss_dict_era5.items():
    # if k in sample_locations:
    if k != "WHITE ISLAND AWS":
        v = [x for x in v if str(x) != "nan"]
        era5_losses_list.extend(v)

fig, ax = plt.subplots()
positions = [1, 1.6]
ax.violinplot([losses_list, era5_losses_list], positions=positions)
ax.tick_params(axis="y", labelsize=16)
ax.set_xticks(positions, labels=["ConvNP", "ERA5"], fontsize=12)
ax.set_ylabel("RMSE", fontsize=16)
ax.set_title("Violin plot of error for ConvNP and ERA5", fontsize=14)

# %%
import matplotlib.dates as mdates

num_days = 365
fontsize = 16
daily_averages = [
    np.mean([loss_dict[location][day] for location in loss_dict])
    for day in range(num_days)
]
daily_averages_era5 = [
    np.mean([loss_dict_era5[location][day] for location in loss_dict_era5])
    for day in range(num_days)
]
# Plotting
days = range(1, num_days + 1)
days = [datetime(2016, 1, 1) + relativedelta(days=day) for day in days]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(days, daily_averages, label="ConvNP")
ax.plot(days, daily_averages_era5, label="ERA5")

ax.tick_params(labelsize=fontsize)
# ax.set_xlabel('Day of the Year', fontsize=12)
ax.set_ylabel("Average station RMSE", fontsize=fontsize)
# plt.title('Average RMSE Over All Available Stations in 2016')


# Format the x-axis to only show the year and month
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(
    mdates.MonthLocator()
)  # Optional: to have a tick for each month

# Ensure the x-axis ticks only show the year and month
plt.xticks(rotation=45)  # Optional: Rotate the tick labels for better readability

current_ticks = ax.get_xticks()
new_ticks = current_ticks[:-1]  # Remove the last tick
ax.set_xticks(new_ticks)

ax.legend(fontsize=fontsize)
# ax.grid(True)
# %%

import cartopy.crs as ccrs

fig, ax = plt.subplots(
    1, 1, subplot_kw=dict(projection=ccrs.PlateCarree()), figsize=(15, 15)
)
ax.spines["geo"].set_visible(False)
ax.coastlines()

lats, lons, elevations = [], [], []
for station, values in STATION_LATLON.items():
    lats.append(values["latitude"])
    lons.append(values["longitude"])
    elevations.append(values["elevation"])

# Convert lists to numpy arrays
lats = np.array(lats)
lons = np.array(lons)
elevations = np.array(elevations)

# Normalize elevations for colormap
norm = plt.Normalize(elevations.min(), elevations.max())

# Plot using scatter, with colormap to indicate elevation
scatter = ax.scatter(
    lons,
    lats,
    s=100,
    c=elevations,
    cmap="OrRd",
    norm=norm,
    edgecolor="k",
    transform=ccrs.Geodetic(),
)

# Adding a colorbar
cbar = plt.colorbar(scatter, shrink=0.5, aspect=10, pad=0.02)
cbar.ax.tick_params(labelsize=16)  # Make ticks larger
cbar.set_label("Elevation (m)", fontsize=16)

plt.show()
# %%

validate.plot_prediction_with_stations(date=date, location=location, pred=pred)

validate.plot_prediction_with_stations(
    date=date, location=location, pred=pred, zoom_to_location=True
)

validate.plot_prediction_with_stations(date=date, pred=pred)

# %%
# plot average losses for all stations
validate.plot_losses(dates, loss_dict, pred_dict, station_dict)

# plot for a single station
validate.plot_losses(dates, loss_dict, pred_dict, station_dict, location="MT COOK EWS")

# validate.plot_losses(dates, loss)

# %%
# this has plot with loss at stations over a given time period
# but obvs only plots the prediction values on the map on one date
# misleading plot?
# validate.plot_prediction_with_stations(labels=loss_dict)
# %%
fig, ax = plt.subplots()

for location, losses in loss_dict.items():
    elev = STATION_LATLON[location]["elevation"]
    loss_mean = np.nanmean(losses)
    ax.plot(elev, loss_mean, "o", label=location, c="green")

ax.set_xlabel("Elevation (m)")
ax.set_ylabel("Loss (C)")
# %%

fig, ax = plt.subplots()

loss_mean = {"low": [], "mid": [], "high": []}
loss_mean_era5 = {"low": [], "mid": [], "high": []}

for location, losses in loss_dict.items():
    elev = STATION_LATLON[location]["elevation"]
    mean_loss = np.nanmean(losses)
    mean_loss_era5 = np.nanmean(loss_dict_era5[location])

    if not np.isnan(mean_loss):
        if elev < 500:
            loss_mean["low"].append(mean_loss)
            if not np.isnan(mean_loss_era5):
                loss_mean_era5["low"].append(mean_loss_era5)
        elif elev > 1000:
            loss_mean["high"].append(mean_loss)
            if not np.isnan(mean_loss_era5):
                loss_mean_era5["high"].append(mean_loss_era5)
        else:
            loss_mean["mid"].append(mean_loss)
            if not np.isnan(mean_loss_era5):
                loss_mean_era5["mid"].append(mean_loss_era5)

positions = [1, 2, 3]
positions_era5 = [x + 0.5 for x in positions]

# Using patch_artist to apply fill color
box1 = ax.boxplot(
    [loss_mean["low"], loss_mean["mid"], loss_mean["high"]],
    positions=positions,
    widths=0.4,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue"),
)
box2 = ax.boxplot(
    [loss_mean_era5["low"], loss_mean_era5["mid"], loss_mean_era5["high"]],
    positions=positions_era5,
    widths=0.4,
    patch_artist=True,
    boxprops=dict(facecolor="darkblue"),
)

ax.set_xticks([np.mean(pair) for pair in zip(positions, positions_era5)])
ax.set_xticklabels(["Low (<500m)", "Mid (500m - 1000m)", "High (>1000m)"], fontsize=10)

# Adding a legend
ax.legend(
    [box1["boxes"][0], box2["boxes"][0]],
    ["ConvNP", "ERA5"],
    loc="upper left",
    fontsize=12,
)

ax.set_xlabel("Elevation (m)", fontsize=14)
ax.set_ylabel("RMSE", fontsize=14)

plt.show()

# %%
# bias correction for ERA5
era5_interp_filled
# %%
time_slice = slice("2000-01-01", "2011-12-31")
df = validate.processed_dict["station_raw_df"]
df = df.reset_index()
df = df[df["time"].between(time_slice.start, time_slice.stop)]
# station_points = df[['latitude', 'longitude']].drop_duplicates().to_xarray()
# era5_interpolated = era5_unnorm.interp(latitude=station_points.latitude, longitude=station_points.longitude).sel(time=time_slice)
if era5_unnorm == None:
    era5_unnorm = validate.data_processor.unnormalise(
        validate.processed_dict["era5_ds"].drop_vars(("cos_D", "sin_D"))
    )
    print("ERA5 unnormalised")

for row in tqdm(df.iterrows(), total=len(df)):
    era5_val = era5_unnorm.sel(
        time=row[1]["time"],
        latitude=row[1]["latitude"],
        longitude=row[1]["longitude"],
        method="nearest",
    )
    df.at[row[0], "t2m"] = era5_val["t2m"].values

# %%
print(f"df has {len(df)} rows")
print(f'There are {len(df[df["t2m"].isna()])} NaN ERA5 values')
print(f'There are {len(df[df["dry_bulb"].isna()])} NaN station values')
print("Dropping rows with missing temperature values")
df_dropped_na = df.dropna(subset=["t2m"])
df_dropped_na = df_dropped_na.dropna(subset=["dry_bulb"])
print(f"df_dropped_na has {len(df_dropped_na)} rows")
# %%
from sklearn.linear_model import LinearRegression
import numpy as np

X = df_dropped_na["t2m"].values.reshape(-1, 1)
y = df_dropped_na["dry_bulb"].values.reshape(-1, 1)

model = LinearRegression().fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope: {slope}, Intercept: {intercept}")
# %%
df = validate.processed_dict["station_raw_df"]
df_test = df.reset_index()
df_test = df_test[df_test["time"].between("2016-01-01", "2017-12-31")]

for row in tqdm(df_test.iterrows(), total=len(df_test)):
    era5_val = era5_unnorm.sel(
        time=row[1]["time"],
        latitude=row[1]["latitude"],
        longitude=row[1]["longitude"],
        method="nearest",
    )
    df_test.at[row[0], "t2m"] = era5_val["t2m"].values
    df_test.at[row[0], "regression"] = (era5_val["t2m"].values * slope) + intercept
df_test
# %%
loss_regression = np.nanmean(utils.rmse(df_test["dry_bulb"], df_test["regression"]))
print(f"Loss for regression: {loss_regression}")
# %%
import matplotlib.pyplot as plt

# Plotting the regression line
plt.scatter(df_test["t2m"], df_test["dry_bulb"], color="blue", label="Data Points")
plt.plot(df_test["t2m"], df_test["regression"], color="red", label="Regression Line")

# Plotting 1000 random points
# random_indices = np.random.choice(len(df_test), size=1000, replace=False)
# random_points_t2m = df_test.iloc[random_indices]['t2m']
# random_points_dry_bulb = df_test.iloc[random_indices]['dry_bulb']
# plt.scatter(random_points_t2m, random_points_dry_bulb, color='green', label='Random Points')

plt.xlabel("t2m")
plt.ylabel("dry_bulb")
plt.legend()
plt.show()

# %%
era5_hr_path = "/mnt/datasets/ERA5/NZ_land_processed/HiRes_ERA5/"
dates = pd.date_range("2016-01-01", "2017-09-30")
era5_hr_files = [
    f"{era5_hr_path}hires_era5_t2m_{date.year}_{str(date.month).zfill(2)}_{str(date.day).zfill(2)}.nc"
    for date in dates
]
era5_hr_files
# %%
era5_hr = xr.open_mfdataset(era5_hr_files)
era5_hr
# %%
era5_hr_daily = era5_hr.resample(time="1D").mean()
era5_hr_daily.load()
era5_hr_daily = era5_hr_daily - 273.15

# %%
loss_dict_era5_hr, locations = validate.calculate_loss_era5(
    dates, locations, era5_hr_daily
)
# %%
print("------ HiRes ERA5 -------")
loss_mean_std_era5_hr = {}
for location, loss in loss_dict_era5_hr.items():
    loss_mean_std_era5_hr[location] = (np.nanmean(loss), np.nanstd(loss))

overall_mean_era5_hr = np.round(
    np.nanmean(
        [
            loss_mean_std_era5_hr[location][0]
            for location in loss_mean_std_era5_hr.keys()
        ]
    ),
    6,
)
overall_std_era5_hr = np.round(
    np.nanmean(
        [
            loss_mean_std_era5_hr[location][1]
            for location in loss_mean_std_era5_hr.keys()
        ]
    ),
    6,
)

print(f"Mean loss across all stations: {overall_mean_era5_hr}")
print(f"Mean of std loss across all stations: {overall_std_era5_hr}")

# %%
