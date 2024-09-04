from nzdownscale.downscaler.validate_v2 import ValidateV2
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

# python validation_script.py --model_name hourly_1e-5_v2 --variable temperature --year 2016

def get_paths(model_name, var):

    top_dir = '/mnt/temp/projects/DeepWeather/data_delete/DeepSensor/models'

    model_dir = f'{top_dir}/{var}/{model_name}'
    model_path = f'{model_dir}/{model_name}.pt'
    train_metadata_path = f'{model_dir}/metadata_{model_name}.pkl'
    data_processor_path = f'{model_dir}/data_processor.pkl'
    task_loader_path = f'{model_dir}/task_loader.pkl'

    return model_path,  data_processor_path, task_loader_path, train_metadata_path,


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name of the model to validate')
    parser.add_argument('--variable', type=str, help='Variable to validate')
    parser.add_argument('--year', type=int, help='Year to validate')

    args = parser.parse_args()
    model_name = args.model_name
    variable = args.variable
    year = args.year

    ########## Load model ##########
    model_path, data_processor_path, task_loader_path, train_metadata_path  = get_paths(model_name, variable)
    print('Loading validation object')
    validate = ValidateV2(model_path,  data_processor_path, task_loader_path, train_metadata_path,)
    print('Validation object loaded')

    ########## Date information ##########
    date_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='H')
    time = [date.to_pydatetime() for date in date_range]

    ########## Remove stations ##########
    remove_stations_list = [
    "TAUPO AWS",
    "CHRISTCHURCH AERO",
    "MT COOK EWS",
    "AUCKLAND AERO",
    "ALEXANDRA AWS",
    "TOLAGA BAY WXT AWS",
    "WELLINGTON AERO",
    "BLENHEIM AERO",
    "DUNEDIN AERO AWS",
    ]

    ########## Make predictions ##########
    print('Making predictions')
    months = set(date.month for date in time)
    for month in months:
        print('Predicting month:', month)
        time_month = [date for date in time if date.month == month]
        predictions = validate.predict(time_month, 
                                        remove_stations=remove_stations_list)
        
        ########## Save predictions ##########
        predictions = predictions[f'{variable}_station']
        prediction_dir = f'/mnt/no_backup/DeepSensor_outputs/{variable}/{model_name}/'
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
        prediction_fpath = f'{prediction_dir}{year}{str(month).zfill(2)}.nc'

        print(f'Saving predictions to {prediction_fpath}')
        predictions.to_netcdf(prediction_fpath)