# a script to create the commands to move data from bodeker scientific data stores to nesi
from nzdownscale.dataprocess import config, config_local
import os

NZ_land_variables = [
                     'surface_pressure',
                     '10m_u_component_of_wind',
                     '10m_v_component_of_wind',
                     'surface_net_solar_radiation',
                     'surface_solar_radiation_downwards',
                     '2m_temperature',
                     ]

NZ_land_proc_variables = ['total_precipitation_hourly',
                          'relative_humidity',
                          ]

# ------------------------------------------

if __name__ == "__main__":
#     print("Moving data from Bodeker Scientific to NeSI via Emily's computer")

# for var in NZ_land_variables:
#     # print(f"To move {var} data")
#     print(f"scp -3vr scp://emily@hpc:47471/{config_local.DATA_PATHS['ERA5']['parent']}/{var} scp://mahuika//nesi/nobackup/nesi03947/deepsensor_data/era5/NZ_land/")

# for var in NZ_land_proc_variables:
#     # print(f"To move {var} data")
#     print(f"scp -3vr scp://emily@hpc:47471/{config_local.DATA_PATHS['ERA5']['parent_processed']}/{var} scp://mahuika//nesi/nobackup/nesi03947/deepsensor_data/era5/NZ_land_processed/")

    # print("Moving data from Bodeker Scientific to NeSI directly")

    # for var in NZ_land_variables:
    #     print(f"Moving {var} data")
    #     os.system(f"scp -vr {config_local.DATA_PATHS['ERA5']['parent']}/{var} scp://mahuika//nesi/nobackup/nesi03947/deepsensor_data/era5/NZ_land/")

    # for var in NZ_land_proc_variables:
    #     print(f"Moving {var} data")
    #     os.system(f"scp -vr {config_local.DATA_PATHS['ERA5']['parent_processed']}/{var} scp://mahuika//nesi/nobackup/nesi03947/deepsensor_data/era5/NZ_land_processed/")

    print('Making script for moving data from Bodeker Scientific to NeSI')
    script_content = """#!/bin/bash
    echo 'Making script for moving data from Bodeker Scientific to NeSI'
    """
    for var in NZ_land_variables:
        script_content += f"scp -vr {config_local.DATA_PATHS['ERA5']['parent']}/{var} scp://mahuika//nesi/nobackup/nesi03947/deepsensor_data/era5/NZ_land/ &\n"

    for var in NZ_land_proc_variables:
        script_content += f"scp -vr {config_local.DATA_PATHS['ERA5']['parent_processed']}/{var} scp://mahuika//nesi/nobackup/nesi03947/deepsensor_data/era5/NZ_land_processed/ &\n"

    script_content += "wait\n"

    # Write the script to a file
    with open("experiments/move_data_bs_nesi.sh", "w") as script_file:
        script_file.write(script_content)

    print("Script 'move_data.sh' has been created.")