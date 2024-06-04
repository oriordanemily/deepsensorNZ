#!/bin/bash
echo 'Making script for moving data from Bodeker Scientific to NeSI'
#scp -vr /mnt/datasets/ERA5/NZ_land/surface_pressure scp://mahuika//nesi/nobackup/nesi03947/deepsensor_data/era5/NZ_land/ 
scp -vr /mnt/datasets/ERA5/NZ_land/10m_u_component_of_wind scp://mahuika//nesi/nobackup/nesi03947/deepsensor_data/era5/NZ_land/ 
scp -vr /mnt/datasets/ERA5/NZ_land/10m_v_component_of_wind scp://mahuika//nesi/nobackup/nesi03947/deepsensor_data/era5/NZ_land/ 
scp -vr /mnt/datasets/ERA5/NZ_land/surface_net_solar_radiation scp://mahuika//nesi/nobackup/nesi03947/deepsensor_data/era5/NZ_land/ 
scp -vr /mnt/datasets/ERA5/NZ_land/surface_solar_radiation_downwards scp://mahuika//nesi/nobackup/nesi03947/deepsensor_data/era5/NZ_land/ 
scp -vr /mnt/datasets/ERA5/NZ_land/2m_temperature scp://mahuika//nesi/nobackup/nesi03947/deepsensor_data/era5/NZ_land/ 
scp -vr /mnt/datasets/ERA5/NZ_land_processed/total_precipitation_hourly scp://mahuika//nesi/nobackup/nesi03947/deepsensor_data/era5/NZ_land_processed/
scp -vr /mnt/datasets/ERA5/NZ_land_processed/relative_humidity scp://mahuika//nesi/nobackup/nesi03947/deepsensor_data/era5/NZ_land_processed/
wait
