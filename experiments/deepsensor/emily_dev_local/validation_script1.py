from nzdownscale.downscaler.validate_v2 import ValidateV2
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-year", 
                    help="The year to produce predictions for",
                    dest="year")
args = parser.parse_args()
year = int(args.year)

months = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
           'August', 'September', 'October', 'November', 'December']

if __name__ == '__main__':
    model_name = 'hourly_1e-5'
    var = 'temperature'

    model_base = f'/mnt/temp/projects/DeepWeather/data_keep/DeepSensor/models/{var}'
    model_dir = f'{model_base}/{model_name}'
    model_path = f'{model_dir}/{model_name}.pt'
    train_metadata_path = f'{model_dir}/metadata_{model_name}.pkl'

    model_dir2 = f'{model_base}/hourly_v2'
    data_processor_path = f'{model_dir2}/data_processor.pkl'
    task_loader_path = f'{model_dir2}/task_loader.pkl'

    validate = ValidateV2(model_path,
                        data_processor_path,
                        task_loader_path,
                        train_metadata_path)

    # time = [datetime(2010, 1, 1, i) for i in range(24)]
    # once the other notebook has finished loading val_tasks, check what dates the val_tasks are. 
    # every 10 hours? 
    remove_stations_list = [
        "TAUPO AWS",
        "CHRISTCHURCH AERO",
        # "KAITAIA AERO",
        "MT COOK EWS",
        "AUCKLAND AERO",
        "ALEXANDRA AWS",
        "TOLAGA BAY WXT AWS",
        "WELLINGTON AERO",
        "BLENHEIM AERO",
        "DUNEDIN AERO AWS",
    ]

    date_range = pd.date_range(start=datetime(year, 1, 1, 0), end=datetime(year, 12, 31, 0), freq='H')
    # for month in range(1, 13):
    for month in range(8, 13):
        time = date_range[date_range.month == month]
        print(time)
        time = [date.to_pydatetime() for date in time]
        print(f'Predicting {len(time)} hours for {months[month - 1]} {year}')

        pred = validate.predict(time, remove_stations=remove_stations_list)

        netcdf_path = f'{model_dir}/predictions/{year}/{month}.nc'
        if not os.path.exists(f'{model_dir}/predictions'):
            os.makedirs(f'{model_dir}/predictions/{year}')
        if not os.path.exists(f'{model_dir}/predictions/{year}'):
            os.makedirs(f'{model_dir}/predictions/{year}', )
        pred['dry_bulb'].to_netcdf(netcdf_path)
