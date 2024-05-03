import logging

logging.captureWarnings(True)
import argparse

from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train
from nzdownscale.dataprocess import config, config_local
from nzdownscale.dataprocess.utils import str2bool

import os 


def main():
    """
    Example:

    python experiments/deepsensor/emily_dev_local/train_downscaling.py --var='temperature' --start_year=2000 --end_year=2001 --val_start_year=2002 --val_end_year=2002 --topography_highres_coarsen_factor=5 --topography_lowres_coarsen_factor=5 --era5_coarsen_factor=1 --include_time_of_year=True --include_landmask=True  --n_epochs=10  --internal_density=250 --auto_set_internal_density=False
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--var",
        type=str,
        default="temperature",
        # default="precipitation"
    )
    parser.add_argument(
        "--start_year", 
        type=int, 
        default=2000, 
        help="Training start year"
    ),
    parser.add_argument(
        "--end_year",
        type=int, 
        # default=2005, 
        default=2001,
        help="Training end year is inclusive"
    ),
    parser.add_argument(
        "--val_start_year", 
        type=int, 
        # default=2006,
        default=2002, 
        help="Validation start year"
    ),
    parser.add_argument(
        "--val_end_year",
        type=int,
        # default=2007,
        default=2002,
        help="Validation end year is inclusive",
    ),
    parser.add_argument(
        "--use_daily_data",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--topography_highres_coarsen_factor",
        type=int,
        default=5,
    ),
    parser.add_argument(
        "--topography_lowres_coarsen_factor",
        type=int,
        default=5,
        help="Note that the lowres topo is coarsened from the highres topo, so the lowres topo resolution is actually topography_highres_coarsen_factor * topography_lowres_coarsen_factor.",
    ),
    parser.add_argument(
        "--era5_coarsen_factor",
        type=int,
        default=1,
    ),
    parser.add_argument(
        "--include_time_of_year",
        type=str2bool,
        default=True,
    ),
    parser.add_argument(
        "--include_landmask",
        type=str2bool,
        default=True,
    ),
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=30,
    ),
    parser.add_argument(
        "--unet_channels",
        default=None,
        type=tuple,
        help="ConvNP model argument, uses default CONVNP_KWARGS_DEFAULT if not set",
    ),
    parser.add_argument(
        "--likelihood",
        default=None,
        type=str,
        help="ConvNP model argument, uses default CONVNP_KWARGS_DEFAULT if not set",
    ),
    parser.add_argument(
        "--batch",
        type=str2bool,
        default=False, 
        help="Use batch training (True) or not (False)",
    ),
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training, only used if batch=True",
    ),
    parser.add_argument(
        "--internal_density",
        type=int,
        default=100,
        help="ConvNP model argument, uses default CONVNP_KWARGS_DEFAULT if not set",
    ),
    parser.add_argument(
        "--auto_set_internal_density",
        type=str2bool,
        default=False,
        help="Allow automatic setting of internal density by ConvNP",
    )
    parser.add_argument(
        "--area",
        type=str,
        default=None,
        help="Select area of map, options specified in: PLOT_EXTENT in nzdownscale.dataprocess.config.py. PLOT_EXTENT['all'] (all of NZ) is used as default",
    )
    parser.add_argument(
        "--remove_stations",
        type=list,
        default=[None],
        help=" ! CURRENTLY NOT IMPLEMENTED ! List of station names to remove from the dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="default",
        help="Name of the model to be saved, if default it will be the time",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for training",
    )

    args = parser.parse_args()

    # ------------------------------------------
    # Settings
    # ------------------------------------------
    var = args.var
    start_year = args.start_year
    end_year = args.end_year
    val_start_year = args.val_start_year
    val_end_year = args.val_end_year
    use_daily_data = args.use_daily_data
    include_time_of_year = args.include_time_of_year
    include_landmask = args.include_landmask
    area = args.area
    batch = args.batch
    batch_size = args.batch_size
    model_name = f'{args.model_name}'
    lr = args.lr
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

    topography_highres_coarsen_factor = args.topography_highres_coarsen_factor
    topography_lowres_coarsen_factor = args.topography_lowres_coarsen_factor
    era5_coarsen_factor = args.era5_coarsen_factor

    n_epochs = args.n_epochs

    convnp_kwargs = config.CONVNP_KWARGS_DEFAULT
    if args.unet_channels is not None:
        convnp_kwargs["unet_channels"] = args.unet_channels
    if args.likelihood is not None:
        convnp_kwargs["likelihood"] = args.likelihood
    # else:
        # if precipitation, use bernoulli likelihood, otherwise default to cnp
        # if var == "precipitation": 
        #     convnp_kwargs["likelihood"] = "bernoulli-gamma"
    if args.internal_density is not None:
        convnp_kwargs["internal_density"] = args.internal_density
    

    # If not setting internal_density, remove from convnp kwargs
    if args.auto_set_internal_density:
        convnp_kwargs = {
            k: v for k, v in convnp_kwargs.items() if k != "internal_density"
        }

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
    )
    if False:
        data_processor_dict_fpath = 'data_processor_dict_era1_topohr5_topolr5_2000_2001.pkl'
        data_processor_dict = data.load_data_processor_dict(data_processor_dict_fpath)
        save_data_processor_dict=False
    else:
        data_processor_dict = None
        save_data_processor_dict = True
        
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
