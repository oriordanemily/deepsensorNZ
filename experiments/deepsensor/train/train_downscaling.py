import logging

logging.captureWarnings(True)
import yaml

from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train
from nzdownscale.dataprocess import config, config_local, era5, wrf
from nzdownscale.dataprocess.utils import validate_and_convert_args
from nzdownscale.dataprocess.config_local import DATA_PATHS

import os 
import shutil
import argparse


def main():
    """
    Example:

    python experiments/deepsensor/train/train_downscaling.py 
    """
    
    print('Starting training script')
    # ------------------------------------------
    # Settings
    # ------------------------------------------
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-arg_path", 
                        help="The filepath of the arguments file",
                        default=None,
                        dest="arg_path")
    args = parser.parse_args()
    
    # extract args from arg parser
    if args.arg_path is None:
        arg_path = DATA_PATHS['arguments']
    else:
        arg_path = args.arg_path
            
    with open(arg_path) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    args = validate_and_convert_args(args)

    print('ARGUMENTS:', args)    
    variable = args["variable"]
    base = args["base"]

    training_years = None
    validation_years = None
    training_fpaths = None
    validation_fpaths = None
    use_daily_data = None

    if base == "era5":
        start_year = args["start_year"]
        end_year = args["end_year"]
        training_years_step = args["training_years_step"]
        val_start_year = args["val_start_year"]
        val_end_year = args["val_end_year"]
        validation_years_step = args["val_years_step"]
        use_daily_data = args["use_daily_data"]

        training_years = list(range(start_year, end_year+1, training_years_step))
        validation_years = list(range(val_start_year, val_end_year+1, validation_years_step))
        print('Training years:', training_years)
        print('Validation years:', validation_years)

    elif base == "wrf":
        train_start = args["start_init"]
        train_end = args["end_init"]
        val_start = args["val_start_init"]
        val_end = args["val_end_init"]
        print(f'Training period: {train_start} - {train_end}')
        print(f'Validation period: {val_start} - {val_end}')

        print('ONLY USING FIRST 30 TRAINING FILES AND 10 VALIDATION FILES')
        training_fpaths = wrf.get_filepaths(train_start, train_end)[:30]
        validation_fpaths = wrf.get_filepaths(val_start, val_end)[:10]
        
    time_intervals = args["time_intervals"]
    include_time_of_year = args["include_time_of_year"]
    include_landmask = args["include_landmask"]
    context_variables = args["context_variables"]
    area = args["area"]
    batch = args["batch"]
    batch_size = args["batch_size"]
    model_name = f'{args["model_name"]}'
    lr = args["lr"]
    weight_decay = args["weight_decay"]
    topography_highres_coarsen_factor = args["topography_highres_coarsen_factor"]
    topography_lowres_coarsen_factor = args["topography_lowres_coarsen_factor"]
    era5_coarsen_factor = args["era5_coarsen_factor"]
    n_epochs = args["n_epochs"]
    station_as_context = args["station_as_context"]

    convnp_kwargs = config.CONVNP_KWARGS_DEFAULT
    if args["unet_channels"] is not None:
        convnp_kwargs["unet_channels"] = args["unet_channels"]
    if args["likelihood"] is not None:
        convnp_kwargs["likelihood"] = args["likelihood"]
    else:
        # if precipitation, use bernoulli likelihood, otherwise default to cnp
        if variable == "precipitation": 
            convnp_kwargs["likelihood"] = "bernoulli-gamma"
        else:
            convnp_kwargs["likelihood"] = "cnp"
    if args["internal_density"] is not None:
        convnp_kwargs["internal_density"] = args["internal_density"]

    # If not setting internal_density, remove from convnp kwargs
    if args["auto_set_internal_density"]:
        convnp_kwargs = {
            k: v for k, v in convnp_kwargs.items() if k != "internal_density"
        }

    remove_stations = args['remove_stations']

    # ------------------------------------------
    # Preprocess data
    # ------------------------------------------
    data = PreprocessForDownscaling(
        variable=variable,
        base=base,
        training_years=training_years, #era
        validation_years=validation_years, #era
        training_fpaths=training_fpaths, #wrf
        validation_fpaths=validation_fpaths, #wrf
        use_daily_data=use_daily_data, #era
        area=area,
        context_variables=context_variables,
    )
    
    model_dir = os.path.join(DATA_PATHS['save_model']['fpath'], variable)#, model_name)
    if use_daily_data:
        suffix = ''
    else:
        suffix = '_hourly'

    model_name_dir = f'{model_dir}/{model_name}'
    data_processor_fpath = f'{model_name_dir}/data_processor.pkl'

    print('Looking for dataprocessor at:', data_processor_fpath)
    if os.path.exists(data_processor_fpath):
        print('Using loaded dataprocessor')
        data_processor_dict = data.load_data_processor_dict(data_processor_fpath)
        save_data_processor_dict=False
    else:
        print('No dataprocessor found, will be created')
        data_processor_dict = None
        save_data_processor_dict=data_processor_fpath
        if not os.path.exists(model_name_dir):
            os.makedirs(model_name_dir)
    
    shutil.copy(DATA_PATHS['arguments'], model_name_dir)
    
    print('Starting data processing')
    data.run_processing_sequence(
        topography_highres_coarsen_factor,
        topography_lowres_coarsen_factor,
        era5_coarsen_factor,
        include_time_of_year=include_time_of_year,
        include_landmask=include_landmask,
        remove_stations=remove_stations,
        save_data_processor_dict=save_data_processor_dict,
        data_processor_dict=data_processor_dict,
        station_as_context=station_as_context,
    )
    processed_output_dict = data.get_processed_output_dict()
    data.print_resolutions()

    # ------------------------------------------
    # Train model
    # ------------------------------------------
    print('Starting training')
    training = Train(processed_output_dict=processed_output_dict,
                     base=base)
    training.run_training_sequence(n_epochs, model_name, batch=batch, 
                                   batch_size=batch_size, lr=lr,
                                   weight_decay=weight_decay, 
                                   time_intervals=time_intervals,
                                   **convnp_kwargs)
    training.model.save(model_dir)

if __name__ == "__main__":

    main()
