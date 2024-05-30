import logging

logging.captureWarnings(True)
import yaml

from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train
from nzdownscale.dataprocess import config, config_local
from nzdownscale.dataprocess.utils import validate_and_convert_args
from nzdownscale.dataprocess.config_local import DATA_PATHS

import os 


def main():
    """
    Example:

    python experiments/deepsensor/emily_dev_local/train_downscaling.py 
    """

    # ------------------------------------------
    # Settings
    # ------------------------------------------
    with open(DATA_PATHS['arguments']) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

    args = validate_and_convert_args(args)
    
    var = args["variable"]
    start_year = args["start_year"]
    end_year = args["end_year"]
    val_start_year = args["val_start_year"]
    val_end_year = args["val_end_year"]
    use_daily_data = args["use_daily_data"]
    include_time_of_year = args["include_time_of_year"]
    include_landmask = args["include_landmask"]
    context_variables = args["context_variables"]
    area = args["area"]
    batch = args["batch"]
    batch_size = args["batch_size"]
    model_name = f'{args["model_name"]}'
    lr = args["lr"]
    topography_highres_coarsen_factor = args["topography_highres_coarsen_factor"]
    topography_lowres_coarsen_factor = args["topography_lowres_coarsen_factor"]
    era5_coarsen_factor = args["era5_coarsen_factor"]
    n_epochs = args["n_epochs"]

    convnp_kwargs = config.CONVNP_KWARGS_DEFAULT
    if args["unet_channels"] is not None:
        convnp_kwargs["unet_channels"] = args["unet_channels"]
    if args["likelihood"] is not None:
        convnp_kwargs["likelihood"] = args["likelihood"]
    # else:
        # if precipitation, use bernoulli likelihood, otherwise default to cnp
        # if var == "precipitation": 
        #     convnp_kwargs["likelihood"] = "bernoulli-gamma"
    if args["internal_density"] is not None:
        convnp_kwargs["internal_density"] = args["internal_density"]

    # If not setting internal_density, remove from convnp kwargs
    if args["auto_set_internal_density"]:
        convnp_kwargs = {
            k: v for k, v in convnp_kwargs.items() if k != "internal_density"
        }

    remove_stations = [
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

    # ------------------------------------------
    # Preprocess data
    # ------------------------------------------
    data = PreprocessForDownscaling(
        variable=var,
        start_year=start_year,
        end_year=end_year,
        val_start_year=val_start_year,
        val_end_year=val_end_year,
        use_daily_data=use_daily_data,
        area=area,
        context_variables=context_variables,
    )
    data_processor_dict_fpath = f'data_processor_dict_{var}_{model_name}.pkl'
    if os.path.exists(data_processor_dict_fpath):
        # data_processor_dict_fpath = 'data_processor_dict_era1_topohr5_topolr5_2000_2001.pkl'
        data_processor_dict = data.load_data_processor_dict(data_processor_dict_fpath)
        save_data_processor_dict=False
    else:
        data_processor_dict = None
        save_data_processor_dict=data_processor_dict_fpath

    data.run_processing_sequence(
        topography_highres_coarsen_factor,
        topography_lowres_coarsen_factor,
        era5_coarsen_factor,
        include_time_of_year=include_time_of_year,
        include_landmask=include_landmask,
        remove_stations=remove_stations,
        save_data_processor_dict=save_data_processor_dict,
        data_processor_dict=data_processor_dict,
        station_as_context=False
    )
    processed_output_dict = data.get_processed_output_dict()
    data.print_resolutions()

    # ------------------------------------------
    # Train model
    # ------------------------------------------

    training = Train(processed_output_dict=processed_output_dict)
    # training.run_training_sequence(n_epochs, model_name, batch=False, **convnp_kwargs)
    training.run_training_sequence(n_epochs, model_name, batch=batch, batch_size=batch_size, lr=lr, **convnp_kwargs)


if __name__ == "__main__":
    main()
