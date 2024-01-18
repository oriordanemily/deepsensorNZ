import logging
logging.captureWarnings(True)
import argparse

from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train
from nzdownscale.downscaler.validate import ValidateV1


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
        default=2001,
        help='Validation start year'
    ),
    parser.add_argument(
        "--topography_highres_coarsen_factor",
        type=int,
        default=10,
    ),
    parser.add_argument(
        "--topography_lowres_coarsen_factor",
        type=int,
        default=10,
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

    args = parser.parse_args()

    # ------------------------------------------
    # Settings
    # ------------------------------------------
    use_daily_data = True

    var = args.var
    start_year = args.start_year
    end_year = args.end_year
    val_start_year = args.val_start_year

    topography_highres_coarsen_factor = args.topography_highres_coarsen_factor
    topography_lowres_coarsen_factor = args.topography_lowres_coarsen_factor
    era5_coarsen_factor = args.era5_coarsen_factor

    model_name_prefix = args.model_name_prefix
    n_epochs = args.n_epochs

    # ------------------------------------------
    # Preprocess data
    # ------------------------------------------

    data = PreprocessForDownscaling(
        variable = var,
        start_year = start_year,
        end_year = end_year,
        val_start_year = val_start_year,
        use_daily_data = use_daily_data,
    )

    data.load_topography()
    data.load_era5()
    data.load_stations()

    highres_aux_raw_ds, aux_raw_ds = data.preprocess_topography(topography_highres_coarsen_factor, topography_lowres_coarsen_factor)
    era5_raw_ds = data.preprocess_era5(coarsen_factor=era5_coarsen_factor)
    station_raw_df = data.preprocess_stations()

    data.process_all_for_training(era5_raw_ds, highres_aux_raw_ds, aux_raw_ds, station_raw_df)
    processed_output_dict = data.get_processed_output_dict()

    # ------------------------------------------
    # Plot info
    # ------------------------------------------

    data.print_resolutions()
    if False:
        data.plot_dataset('era5')
        data.plot_dataset('top_highres')
        data.plot_dataset('top_lowres')

    # ------------------------------------------
    # Train model
    # ------------------------------------------

    training = Train(processed_output_dict=processed_output_dict)

    training.setup_task_loader()
    training.initialise_model()
    training.train_model(n_epochs=n_epochs, model_name_prefix=model_name_prefix)

    # training_output_dict = training.get_training_output_dict()

    # # ------------------------------------------
    # # Inspect trained model
    # # ------------------------------------------

    # validate = ValidateV1(
    #     processed_output_dict=processed_output_dict,
    #     #training_output_dict=training_output_dict,
    #     training_output_dict=None,
    #     )

    # validate.initialise(load_model_path='models/downscaling/run1_model_1705453547.pt')


if __name__ == '__main__':
    main()