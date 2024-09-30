from nzdownscale.downscaler.validate_ERA import ValidateERA
from nzdownscale.dataprocess.utils import save_netcdf
from datetime import datetime, timedelta
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from time import strftime, gmtime
import sys
import os
import calendar
import argparse

top_dir = '/mnt/temp/projects/DeepWeather/data_delete/DeepSensor/models'

def get_paths(var, model_name):
    
    model_dir = f'{top_dir}/{var}/{model_name}'
    model_path = f'{model_dir}/{model_name}.pt'
    train_metadata_path = f'{model_dir}/metadata_{model_name}.pkl'

    data_processor_path = f'{model_dir}/data_processor.pkl'
    task_loader_path = f'{model_dir}/task_loader.pkl'
    return model_path, data_processor_path, task_loader_path, train_metadata_path

def setup_validation_class(var, model_name):
    model_path, data_processor_path, task_loader_path, train_metadata_path = get_paths(var, model_name)
    validate = ValidateERA(model_path, data_processor_path, task_loader_path, train_metadata_path)
    return validate

def get_dates(year, month):
    last_day = calendar.monthrange(year, month)[1]
    date_range = pd.date_range(start=datetime(year, month, 1, 0), 
                               end=datetime(year, month, last_day, 23), 
                               freq='H')
    time = date_range
    time = [date.to_pydatetime() for date in time]
    return time

def write_standard_metadata(ds: xr.Dataset):
    ds.attrs['institution'] = 'Bodeker Scientific'
    ds.attrs['author'] = "Emily O'Riordan"
    ds.attrs['email'] = 'emily@bodekerscientific.com'
    ds.attrs['created'] = strftime("%Y-%m-%d %H:%M:%S UTC", gmtime())
    ds.attrs['script'] = sys.argv[0]

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--year', type=int)
    argparser.add_argument('--gpu', type=int, default=0)

    args = argparser.parse_args()
    year = args.year
    gpu = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Model setup
    var = 'temperature'
    model_name = 'high_res' #'hourly_1e-5_v2'
    print('Predictions for:', var, model_name)
    save_dir = f'{top_dir}/{var}/{model_name}/outputs/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Remove stations
    remove_stations_list = [
    "TAUPO AERO AWS",
    "CHRISTCHURCH AERO",
    "MT COOK EWS",
    "AUCKLAND AERO",
    "ALEXANDRA AWS",
    "TOLAGA BAY WXT AWS",
    "WELLINGTON AERO",
    "BLENHEIM AERO",
    "DUNEDIN AERO AWS",
    ]

    validate = setup_validation_class(var, model_name)

    save_netcdf_partial = partial(save_netcdf, 
                                 compress=5, 
                                 dtype='float32', 
                                 engine='netcdf4')
    # Dates
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    print('Predicting year:', year)

    for month in months:
        print('Predicting month:', month)
        # Predict
        time = get_dates(year, month)
        save_netcdf_chunk_dict = {'time': len(time), 
                                  'lat': 250, 
                                  'lon': 250}
        preds = validate.predict(time, remove_stations=remove_stations_list)
        preds = preds[f'{var}_station']
        preds = preds.rename({'mean': var})

        # Save
        write_standard_metadata(preds)
        save_path = f'{save_dir}predictions_{year}{str(month).zfill(2)}.nc'
        print('Saving to:', save_path)
        save_netcdf_partial(preds, save_path, chunk_dict=save_netcdf_chunk_dict)
        # preds.to_netcdf(save_path, )