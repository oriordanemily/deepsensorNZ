
import logging
logging.captureWarnings(True)
import argparse

from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train
from nzdownscale.downscaler.validate import ValidateV1


CONVNP_KWARGS_DEFAULT = {
    'unet_channels': (64,)*4,
    'likelihood': 'gnp',
    'internal_density': 50,
}


def main():
    """
    Example:

    python experiments/deepsensor/risa_dev_local/train_downscaling.py --var='temperature' --start_year=2000 --end_year=2001 --val_start_year=2001 --topography_highres_coarsen_factor=30 --topography_lowres_coarsen_factor=10 --era5_coarsen_factor=5 --model_name_prefix='test' --n_epochs=3

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
    ),
    parser.add_argument(
        "--end_year",
        type=int,
        default=2001,
        help='End year is inclusive'
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
        help="ConvNP model argument",
    ),
    parser.add_argument(
        "--likelihood",
        default=None,
        help="ConvNP model argument",
    ),
    parser.add_argument(
        "--internal_density",
        default=None,
        help="ConvNP model argument",
    ),    

    args = parser.parse_args()




if __name__ == '__main__':
    main()