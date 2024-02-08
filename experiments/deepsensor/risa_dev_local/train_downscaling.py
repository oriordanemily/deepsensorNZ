
import logging
logging.captureWarnings(True)
import argparse

from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train
from nzdownscale.dataprocess import config


def main():
    """
    Example:

    python experiments/deepsensor/risa_dev_local/train_downscaling.py --var='temperature' --start_year=2000 --end_year=2001 --val_start_year=2002 --val_end_year=2002 --topography_highres_coarsen_factor=30 --topography_lowres_coarsen_factor=30 --era5_coarsen_factor=30 --include_time_of_year=True --include_landmask=True --model_name_prefix='test' --n_epochs=3  --internal_density=5 --area='christchurch' --auto_set_internal_density=False
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--var",
            type=str,
            default='temperature',
        )
    parser.add_argument(
        "--start_year",
        type=int,
        default=2000,
        help='Training start year'
    ),
    parser.add_argument(
        "--end_year",
        type=int,
        default=2001,
        help='Training end year is inclusive'
    ),
    parser.add_argument(
        "--val_start_year",
        type=int,
        default=2002,
        help='Validation start year'
    ),
    parser.add_argument(
        "--val_end_year",
        type=int,
        default=2002,
        help='Validation end year is inclusive'
    ),
    
    parser.add_argument(
        "--use_daily_data",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--topography_highres_coarsen_factor",
        type=int,
        default=10,
    ),
    parser.add_argument(
        "--topography_lowres_coarsen_factor",
        type=int,
        default=10,
        help='Note that the lowres topo is coarsened from the highres topo, so the lowres topo resolution is actually topography_highres_coarsen_factor * topography_lowres_coarsen_factor.'
    ),
    parser.add_argument(
        "--era5_coarsen_factor",
        type=int,
        default=5,
    ),

    parser.add_argument(
        "--include_time_of_year",
        type=bool,
        default=False,
    ),
    parser.add_argument(
        "--include_landmask",
        type=bool,
        default=False,
    ),

    parser.add_argument(
        "--model_name_prefix",
        type=str,
        default='',
        help='Prefix string for saved model'
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
        "--internal_density",
        type=int,
        default=None,
        help="ConvNP model argument, uses default CONVNP_KWARGS_DEFAULT if not set",
    ),    
    parser.add_argument(
        "--auto_set_internal_density",
        type=bool,
        default=False,
        help="Allow automatic setting of internal density by ConvNP"
    )
    parser.add_argument(
        "--area",
        type=str,
        default=None,
        help="Select area of map",
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

    topography_highres_coarsen_factor = args.topography_highres_coarsen_factor
    topography_lowres_coarsen_factor = args.topography_lowres_coarsen_factor
    era5_coarsen_factor = args.era5_coarsen_factor

    model_name_prefix = args.model_name_prefix
    n_epochs = args.n_epochs

    convnp_kwargs = config.CONVNP_KWARGS_DEFAULT
    if args.unet_channels is not None:
        convnp_kwargs['unet_channels'] = args.unet_channels
    if args.likelihood is not None:
        convnp_kwargs['likelihood'] = args.likelihood
    if args.internal_density is not None:
        convnp_kwargs['internal_density'] = args.internal_density
    
    # If not setting internal_density, remove from convnp kwargs
    if args.auto_set_internal_density:
        convnp_kwargs = {k: v for k, v in convnp_kwargs.items() if k != 'internal_density'}

    # ------------------------------------------
    # Preprocess data
    # ------------------------------------------
    data = PreprocessForDownscaling(
        variable = var,
        start_year = start_year,
        end_year = end_year,
        val_start_year = val_start_year,
        val_end_year = val_end_year,
        use_daily_data = use_daily_data,
        area=area,
    )
    data.run_processing_sequence(
        topography_highres_coarsen_factor,
        topography_lowres_coarsen_factor, 
        era5_coarsen_factor,
        include_time_of_year=include_time_of_year,
        include_landmask=include_landmask,
        )
    processed_output_dict = data.get_processed_output_dict()
    data.print_resolutions()

    # ------------------------------------------
    # Train model
    # ------------------------------------------

    training = Train(processed_output_dict=processed_output_dict)

    training.run_training_sequence(n_epochs, model_name_prefix, **convnp_kwargs)


if __name__ == '__main__':
    main()