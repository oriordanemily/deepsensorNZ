#%% 

import logging
logging.captureWarnings(True)
import os

import pandas as pd
import numpy as np
import xarray as xr

from deepsensor.data import DataProcessor, TaskLoader
from deepsensor.data import construct_circ_time_ds
from deepsensor.data.utils import construct_x1x2_ds
from deepsensor.data.sources import get_ghcnd_station_data, get_era5_reanalysis_data, get_earthenv_auxiliary_data, get_gldas_land_mask

#%% 

# Using the same settings allows use to use pre-downloaded cached data
data_range = ("2015-06-25", "2015-06-30")
extent = "europe"
station_var_IDs = ["TAVG", "PRCP"]
era5_var_IDs = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
auxiliary_var_IDs = ["elevation", "tpi"]
cache_dir = "data/.datacache"

station_raw_df = get_ghcnd_station_data(station_var_IDs, extent, date_range=data_range, cache=True, cache_dir=cache_dir)
era5_raw_ds = get_era5_reanalysis_data(era5_var_IDs, extent, date_range=data_range, cache=True, cache_dir=cache_dir)
auxiliary_raw_ds = get_earthenv_auxiliary_data(auxiliary_var_IDs, extent, "1KM", cache=True, cache_dir=cache_dir)
land_mask_raw_ds = get_gldas_land_mask(extent, cache=True, cache_dir=cache_dir)

#%% 

# works

data_processor = DataProcessor(x1_name="lat", x2_name="lon")
#data_processor = DataProcessor("../deepsensor_config/")
era5_ds = data_processor(era5_raw_ds)
aux_ds, land_mask_ds = data_processor([auxiliary_raw_ds, land_mask_raw_ds], method="min_max")

station_df = data_processor(station_raw_df)

# Add 2D circular day of year variable to land mask context set

dates = pd.date_range(era5_ds.time.values.min(), era5_ds.time.values.max(), freq="D")
doy_ds = construct_circ_time_ds(dates, freq="D")

len_days = len(dates)
land_mask_ds = xr.DataArray(
    np.tile(land_mask_ds.values, (len_days, 1, 1)),
    dims=('time', 'x1', 'x2'),
    coords={
        'time': dates,
        'x1': land_mask_ds['x1'],
        'x2': land_mask_ds['x2'],
    },
    name='land_mask',
)
land_mask_ds = xr.Dataset({
    'land_mask': land_mask_ds,
    'cos_D': doy_ds["cos_D"],
    'sin_D': doy_ds['sin_D']
})

print(land_mask_ds)

#%% 
#%% 
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
import xarray as xr

from deepsensor.data import DataProcessor, TaskLoader
from deepsensor.data import construct_circ_time_ds
from deepsensor.data.sources import get_ghcnd_station_data, get_era5_reanalysis_data, get_earthenv_auxiliary_data, get_gldas_land_mask
from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train
from nzdownscale.downscaler.validate import ValidateV1
from nzdownscale.dataprocess import utils

#
# Settings
# ------------------------------------------


###
include_time_of_year=True
include_landmask=True
###

var = 'temperature'
start_year = 2000
end_year = 2001
val_start_year = 2002
val_end_year = 2002
use_daily_data = True

topography_highres_coarsen_factor = 5
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
    include_time_of_year=include_time_of_year,
    include_landmask=include_landmask,
    )

processed_output_dict = data.get_processed_output_dict()
# ['data_processor', 'era5_ds', 'highres_aux_ds', 'aux_ds', 'landmask_ds', 'station_df', 'station_raw_df', 'era5_raw_ds', 'data_settings', 'date_info']

print(processed_output_dict['landmask_ds']) #[x]
print(processed_output_dict['era5_ds'])  #[x]

#%% train 

"""
TaskLoader(3 context sets, 1 target sets)
Context variable IDs: (('t2m', 'cos_D', 'sin_D'), ('elevation', 'TPI_0.1', 'TPI_0.05', 'TPI_0.025', 'x1_arr', 'x2_arr'), ('landmask',))
Target variable IDs: (('dry_bulb',),)
Auxiliary-at-target variable IDs: ('elevation', 'TPI_0.1', 'TPI_0.05', 'TPI_0.025')

Context data dimensions: (3, 6, 1)
Target data dimensions: (1,)
Auxiliary-at-target data dimensions: 4
"""

training = Train(processed_output_dict=processed_output_dict)

#training.run_training_sequence(n_epochs, model_name_prefix, **convnp_kwargs)

batch=False
batch_size=1

training.setup_task_loader()

print(len(training.context))
training.context[0]
training.context[1]
training.context[2]

#%% 
training.initialise_model(**convnp_kwargs)
training.train_model(n_epochs=n_epochs, model_name_prefix=model_name_prefix, batch=batch, batch_size=batch_size)

training_output_dict = training.get_training_output_dict()

#%% dev - expanded preprocess

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

data.load_topography()
data.load_era5()
data.load_stations()

highres_aux_raw_ds, aux_raw_ds = data.preprocess_topography(topography_highres_coarsen_factor, topography_lowres_coarsen_factor)
era5_raw_ds = data.preprocess_era5(coarsen_factor=era5_coarsen_factor)
station_raw_df = data.preprocess_stations()

#data.process_all_for_training(era5_raw_ds, highres_aux_raw_ds, aux_raw_ds, station_raw_df)

print('Creating DataProcessor...')

min(highres_aux_raw_ds['latitude'].min(), era5_raw_ds['latitude'].min())
max(highres_aux_raw_ds['latitude'].max(), era5_raw_ds['latitude'].max())

# map=ERA:
# {'coords': {'time': {'name': 'time'},
#             'x1': {'map': (-47.45000076293945, -35.45000076293945),
#                    'name': 'latitude'},
#             'x2': {'map': (168.4499969482422, 177.4499969482422),
#                    'name': 'longitude'}}}

# map=highrestop:
# {'coords': {'time': {'name': 'time'},
#             'x1': {'map': (-47.99512481689453, -34.005126953125),
#                    'name': 'latitude'},
#             'x2': {'map': (166.00486755371094, 179.994873046875),
#                    'name': 'longitude'}}}


data_processor = DataProcessor(
    x1_name="latitude", 
    # x1_map=(era5_raw_ds["latitude"].min(), 
    #         era5_raw_ds["latitude"].max()),
    #    x1_map=(highres_aux_raw_ds["latitude"].min(), 
    #            highres_aux_raw_ds["latitude"].max()),
    x1_map=(
        min(highres_aux_raw_ds['latitude'].min(), era5_raw_ds['latitude'].min()),
        max(highres_aux_raw_ds['latitude'].max(), era5_raw_ds['latitude'].max())
    ),
    x2_name="longitude", 
    # x2_map = (era5_raw_ds["longitude"].min(), 
    #             era5_raw_ds["longitude"].max()),
    # )
    #    x2_map=(highres_aux_raw_ds["longitude"].min(), 
    #            highres_aux_raw_ds["longitude"].max()))
    x2_map=(
        min(highres_aux_raw_ds['longitude'].min(), era5_raw_ds['longitude'].min()),
        max(highres_aux_raw_ds['longitude'].max(), era5_raw_ds['longitude'].max())
    ),
)

print(data_processor)

#%%

# compute normalisation parameters
era5_ds, station_df = data_processor([era5_raw_ds, station_raw_df]) #meanstd
aux_ds, highres_aux_ds = data_processor([aux_raw_ds, highres_aux_raw_ds], method="min_max") #minmax
print(data_processor)

#%% v3 
# ! [x] Add to: 
# data.process_all_for_training(era5_raw_ds, highres_aux_raw_ds, aux_raw_ds, station_raw_df)

# land mask (new)
landmask_raw_ds = data.load_landmask()
landmask_ds = data_processor(land_mask_raw_ds, method="min_max")
print(data_processor)
data.landmask_ds = landmask_ds  ##

# add time of year to era5_ds 
dates = pd.date_range(era5_ds.time.values.min(), era5_ds.time.values.max(), freq="D")
doy_ds = construct_circ_time_ds(dates, freq="D")
era5_ds = xr.Dataset({
    't2m': era5_ds, 
    'cos_D': doy_ds["cos_D"], 
    'sin_D': doy_ds["sin_D"],
    })

# ! [] add to context set:
# land_mask_ds
# era5_ds

#%% 

x1x2_ds = construct_x1x2_ds(aux_ds)
aux_ds['x1_arr'] = x1x2_ds['x1_arr']
aux_ds['x2_arr'] = x1x2_ds['x2_arr']

#%% v2.1 (old)

# land mask
land_mask_raw_ds = highres_aux_raw_ds['elevation'] > 0.000
land_mask_ds = data_processor(land_mask_raw_ds, method="min_max")

# time of year
dates = pd.date_range(era5_raw_ds.time.values.min(), era5_raw_ds.time.values.max(), freq="D")
doy_ds = construct_circ_time_ds(dates, freq="D")
era5_raw_ds["cos_D"] = doy_ds["cos_D"]
era5_raw_ds["sin_D"] = doy_ds["sin_D"]

era5_ds_ = data_processor(era5_raw_ds)

#%% v1 (bug)

# Add 2D circular day of year variable to land mask context set
land_mask_raw_ds = highres_aux_raw_ds['elevation'] > 0.000
#land_mask_raw_ds.name = 'landmask'

land_mask_ds = data_processor(land_mask_raw_ds, method="min_max")
#land_mask_ds = land_mask_raw_ds
dates = pd.date_range(era5_raw_ds.time.values.min(), era5_raw_ds.time.values.max(), freq="D")
doy_ds = construct_circ_time_ds(dates, freq="D")

len_days = len(dates)
land_mask_ds = xr.DataArray(
    np.tile(land_mask_ds.values, (len_days, 1, 1)),
    dims=('time', 'x1', 'x2'),
    coords={
        'time': dates,
        'x1': land_mask_ds['x1'],
        'x2': land_mask_ds['x2'],
    },
    name='landmask',
)
land_mask_ds = xr.Dataset({
    'landmask': land_mask_ds,
    'cos_D': doy_ds["cos_D"],
    'sin_D': doy_ds['sin_D']
})
print(land_mask_ds)

land_mask_ds = data_processor(land_mask_ds, method="min_max")
print(data_processor)

#%% v2 (bug)

# Add 2D circular day of year variable to land mask context set
land_mask_raw_ds = highres_aux_raw_ds['elevation'] > 0.000

dates = pd.date_range(era5_raw_ds.time.values.min(), era5_raw_ds.time.values.max(), freq="D")
doy_ds = construct_circ_time_ds(dates, freq="D")

len_days = len(dates)
land_mask_ds = xr.DataArray(
    np.tile(land_mask_raw_ds.values, (len_days, 1, 1)),
    dims=('time', 'latitude', 'longitude'),
    coords={
        'time': dates,
        'latitude': land_mask_raw_ds['latitude'],
        'longitude': land_mask_raw_ds['longitude'],
    },
    name='landmask',
)
land_mask_ds = xr.Dataset({
    'landmask': land_mask_ds,
    'cos_D': doy_ds["cos_D"],
    'sin_D': doy_ds['sin_D']
})
print(land_mask_ds)

land_mask_ds = data_processor(land_mask_ds, method="min_max")
print(data_processor)
print(land_mask_ds)

 
#%% 

data = PreprocessForDownscaling(
    variable = var,
    start_year = start_year,
    end_year = end_year,
    val_start_year = val_start_year,
    val_end_year = val_start_year,
    use_daily_data = use_daily_data,
)

# data.run_processing_sequence(
#     topography_highres_coarsen_factor,
#     topography_lowres_coarsen_factor, 
#     era5_coarsen_factor,
#     )

data.load_topography()
data.load_era5()
data.load_stations()

highres_aux_raw_ds, aux_raw_ds = data.preprocess_topography(topography_highres_coarsen_factor, topography_lowres_coarsen_factor)
era5_raw_ds = data.preprocess_era5(coarsen_factor=era5_coarsen_factor)
station_raw_df = data.preprocess_stations()

data.process_all_for_training(era5_raw_ds, highres_aux_raw_ds, aux_raw_ds, station_raw_df)

processed_output_dict = data.get_processed_output_dict()

#%% 
# Train model
# ------------------------------------------

training = Train(processed_output_dict=processed_output_dict)

#training.run_training_sequence(n_epochs, model_name_prefix, **convnp_kwargs)

batch=False
batch_size=1

training.setup_task_loader()
training.initialise_model(**convnp_kwargs)
training.train_model(n_epochs=n_epochs, model_name_prefix=model_name_prefix, batch=batch, batch_size=batch_size)

training_output_dict = training.get_training_output_dict()

#%% 

# data.ds_elev['elevation']
from nzdownscale.dataprocess import era5, stations, topography
import matplotlib.colors as mcolors

process_top = topography.ProcessTopography()
da_elev_copy = process_top.coarsen_da(data.ds_elev['elevation'], topography_highres_coarsen_factor)
da_landmask = xr.where(np.isnan(da_elev_copy), 0, 1)
da_landmask.plot()


#%% 
# coarsened_da = da.coarsen(x=coarsen_window[0], y=coarsen_window[1], boundary='trim').reduce(custom_aggregation)

# coarsen_factor = topography_highres_coarsen_factor

# da_landmask = xr.where(np.isnan(data.ds_elev['elevation']), 0, 1)
# da_landmask.coarsen(latitude=coarsen_factor, longitude=coarsen_factor, boundary='trim').reduce(lambda x: 1 if x.any() else 0)

# custom_aggregation = lambda arr: 1 if np.any(arr) else 0
# def custom_agg(arr): return 1 if np.any(arr) else 0
# da_landmask.coarsen(latitude=coarsen_factor, longitude=coarsen_factor, boundary='trim').reduce(custom_agg)

# #%% 