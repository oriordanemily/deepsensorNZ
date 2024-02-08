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

#
# Settings
# ------------------------------------------

var = 'temperature'
start_year = 2000
end_year = 2001
val_start_year = 2002
val_end_year = 2002
use_daily_data = True

topography_highres_coarsen_factor = 5
topography_lowres_coarsen_factor = 5
era5_coarsen_factor = 1

model_name_prefix = 'new_test'
n_epochs = 2

convnp_kwargs = {
    'unet_channels': (64,)*4,
    'likelihood': 'gnp',
    'internal_density': None,
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

    area = 'christchurch'
)

###
# data.run_processing_sequence(
#     topography_highres_coarsen_factor,
#     topography_lowres_coarsen_factor, 
#     era5_coarsen_factor,
#     )

data.load_topography()
data.load_stations()

#%%

station_raw_df = data.preprocess_stations()

#%% 

df = data.station_metadata_all 
years = data.years

dff = df[(df['start_year']<years[-1]) & (df['end_year']>=years[0])]

from nzdownscale.dataprocess.config import LOCATION_LATLON, PLOT_EXTENT

area = data.area
dff2 = dff[(dff['lon'] > PLOT_EXTENT[area]['minlon']) & (dff['lon'] < PLOT_EXTENT[area]['maxlon']) & (dff['lat'] > PLOT_EXTENT[area]['minlat']) & (dff['lat'] < PLOT_EXTENT[area]['maxlat'])]


minlon = PLOT_EXTENT[area]['minlon']
maxlon = PLOT_EXTENT[area]['maxlon']
minlat = PLOT_EXTENT[area]['minlat']
maxlat = PLOT_EXTENT[area]['maxlat']


#%% 
from nzdownscale.dataprocess.config_local import DATA_PATHS

savedir = f'{DATA_PATHS["cache"]}/station_data'
filepath = f'{savedir}/station_metadata_all.pkl'
print(f'Loading station metadata from cache: {filepath}, set use_cache=False if you want to manually load them.')

#%% cache

savedir = 'data/.datacache/test2'
filepath = f'{savedir}/station_metadata_all.pkl'
if os.path.exists(filepath):
    pass
else:
    os.makedirs(savedir, exist_ok=True)
    utils.save_pickle(data.station_metadata_all, filepath)


#%% dev 
area = 'christchurch'
from nzdownscale.dataprocess.config import PLOT_EXTENT
minlon = PLOT_EXTENT[area]['minlon']
maxlon = PLOT_EXTENT[area]['maxlon']
minlat = PLOT_EXTENT[area]['minlat']
maxlat = PLOT_EXTENT[area]['maxlat']

self = data
self.ds_elev

ds2 = self.ds_elev.sel(latitude=slice(minlat, maxlat), longitude=slice(minlon, maxlon))
#ds2.elevation.plot()

ax = self.nzplot.nz_map_with_coastlines(area='christchurch')
ds2.elevation.plot()
ax = self.process_stations.plot_stations(self.station_metadata_all, ax)
plt.show()

ax = self.nzplot.nz_map_with_coastlines(area='christchurch')
ds2.elevation.plot()
ax = self.process_stations.plot_stations(self.station_metadata_filtered, ax)
plt.show()

#%% cont.

data.load_era5()
#data.load_stations()

highres_aux_raw_ds, aux_raw_ds = data.preprocess_topography(topography_highres_coarsen_factor, topography_lowres_coarsen_factor)
era5_raw_ds = data.preprocess_era5(coarsen_factor=era5_coarsen_factor)
station_raw_df = data.preprocess_stations()

data.process_all_for_training(era5_raw_ds, highres_aux_raw_ds, aux_raw_ds, station_raw_df)

###

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

#training.run_training_sequence(n_epochs, model_name_prefix, **convnp_kwargs)

batch=False
batch_size=1

training.setup_task_loader()
training.initialise_model(**convnp_kwargs)

#%% 

training.train_model(n_epochs=n_epochs, model_name_prefix=model_name_prefix, batch=batch, batch_size=batch_size)

training_output_dict = training.get_training_output_dict()
