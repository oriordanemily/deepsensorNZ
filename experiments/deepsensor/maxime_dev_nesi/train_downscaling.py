import os
import logging
import itertools as it
from typing import Iterable

logging.captureWarnings(True)

import defopt
import joblib
import numpy as np
import pandas as pd

from nzdownscale.downscaler.preprocess import PreprocessForDownscaling
from nzdownscale.downscaler.train import Train
from nzdownscale.dataprocess import config, config_local


def preprocess(
    var,
    start_year,
    end_year,
    val_start_year,
    val_end_year,
    use_daily_data,
    area,
    topography_highres_coarsen_factor,
    topography_lowres_coarsen_factor,
    era5_coarsen_factor,
    include_time_of_year,
    include_landmask,
    remove_stations,
):
    data = PreprocessForDownscaling(
        variable=var,
        start_year=start_year,
        end_year=end_year,
        val_start_year=val_start_year,
        val_end_year=val_end_year,
        use_daily_data=use_daily_data,
        area=area,
    )
    data.run_processing_sequence(
        topography_highres_coarsen_factor,
        topography_lowres_coarsen_factor,
        era5_coarsen_factor,
        include_time_of_year=include_time_of_year,
        include_landmask=include_landmask,
        remove_stations=remove_stations,
    )
    processed_output_dict = data.get_processed_output_dict()
    data.print_resolutions()
    return processed_output_dict


if "JOBLIB_CACHEDIR" in os.environ:
    memory = joblib.Memory(os.environ["JOBLIB_CACHEDIR"], verbose=10, compress=3)
    preprocess = memory.cache(preprocess)


DEFAULT_REMOVED_STATIONS = (
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
)


def main(
    *,
    var: str = "temperature",
    start_year: int = 2000,
    end_year: int = 2001,
    val_start_year: int = 2002,
    val_end_year: int = 2002,
    use_daily_data: bool = True,
    topography_highres_coarsen_factor: int = 5,
    topography_lowres_coarsen_factor: int = 5,
    era5_coarsen_factor: int = 1,
    include_time_of_year: bool = True,
    include_landmask: bool = True,
    n_epochs: int = 30,
    unet_channels: tuple[int] | None = None,
    likelihood: str | None = None,
    internal_density: int = 100,
    auto_set_internal_density: bool = False,
    area: str | None = None,
    remove_stations: Iterable[str] = DEFAULT_REMOVED_STATIONS,
    model_name: str = "default",
    use_gpu: bool = False,
    lr: float = 5e-5,
    n_workers: int = 1,
    batch_size: int = 1,
):
    """
    Note: the lowres topography is coarsened from the highres topography, so the
    lowres topography resolution is actually::

        topography_highres_coarsen_factor * topography_lowres_coarsen_factor

    :param start_year: training start year
    :param end_year: training end year is inclusive
    :param val_start_year: validation start year
    :param val_end_year: validation end year, inclusive
    :param unet_channels: ConvNP model argument, uses default CONVNP_KWARGS_DEFAULT if not set
    :param likelihood: ConvNP model argument, uses default CONVNP_KWARGS_DEFAULT if not set
    :param internal_density: ConvNP model argument, uses default CONVNP_KWARGS_DEFAULT if not set
    :param auto_set_internal_density: allow automatic setting of internal density by ConvNP
    :param area: select area of map, options specified in PLOT_EXTENT in nzdownscale.dataprocess.config.py,
                 PLOT_EXTENT['all'] (all of NZ) is used as default
    :param remove_stations: list of station names to remove from the dataset
    :param model_name: name of the model to be saved, if default it will be the time
    :param use_gpu: use a GPU for model training
    :param lr: learning rate
    :param n_workers: number of parallel workers used for training in a minibatch
    :param batch_size: number of tasks per minibatch
    """

    convnp_kwargs = config.CONVNP_KWARGS_DEFAULT
    if unet_channels is not None:
        convnp_kwargs["unet_channels"] = unet_channels
    if likelihood is not None:
        convnp_kwargs["likelihood"] = likelihood
    if internal_density is not None:
        convnp_kwargs["internal_density"] = internal_density

    # If not setting internal_density, remove from convnp kwargs
    if auto_set_internal_density:
        convnp_kwargs = {
            k: v for k, v in convnp_kwargs.items() if k != "internal_density"
        }

    # ------------------------------------------
    # Preprocess data
    # ------------------------------------------
    processed_output_dict = preprocess(
        var,
        start_year,
        end_year,
        val_start_year,
        val_end_year,
        use_daily_data,
        area,
        topography_highres_coarsen_factor,
        topography_lowres_coarsen_factor,
        era5_coarsen_factor,
        include_time_of_year,
        include_landmask,
        remove_stations,
    )

    # TODO move this into nzdownscale.downscaler.preprocess

    # replace missing stations with mean of variable
    dset = processed_output_dict["station_df"]

    time = dset.reset_index()["time"].unique()
    latlon = list(dset.reset_index().groupby(["x1", "x2"]).groups.keys())
    index = pd.MultiIndex.from_tuples(
        [(time, lat, lon) for time, (lat, lon) in it.product(time, latlon)],
        names=["time", "x1", "x2"],
    )

    dset_full = pd.DataFrame(data={col: np.NaN for col in dset.columns}, index=index)
    dset_full.loc[dset.index] = dset

    processed_output_dict["station_df"] = dset_full

    # ------------------------------------------
    # Train model
    # ------------------------------------------
    training = Train(
        processed_output_dict=processed_output_dict,
        use_gpu=use_gpu,
        n_workers=n_workers,
        batch_size=batch_size,
    )
    training.run_training_sequence(n_epochs, model_name, lr=lr, **convnp_kwargs)


if __name__ == "__main__":
    """
    Example:

    python experiments/deepsensor/risa_dev_local/train_downscaling.py \
        --var='temperature' \
        --start-year=2000 \
        --end-year=2001 \
        --val-start-year=2002 \
        --val-end-year=2002 \
        --topography-highres-coarsen-factor=30 \
        --topography-lowres-coarsen-factor=30 \
        --era5-coarsen-factor=30 \
        --include-time-of-year=True \
        --include-landmask=True \
        --model-name-prefix='test' \
        --n-epochs=3 \
        --internal-density=5 \
        --area='christchurch' \
        --auto-set-internal-density=False
    """
    defopt.run(main)
