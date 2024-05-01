#!/bin/bash -e
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=15G
#SBATCH --gpus-per-node=A100:1
#SBATCH --partition=hgx,gpu
#SBATCH --output logs/%j-%x.out
#SBATCH --error logs/%j-%x.out

module purge
module load Python/3.11.6-foss-2023a

scontrol show job $SLURM_JOB_ID
scontrol write batch_script $SLURM_JOB_ID -

# monitor GPU usage
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv,nounits -l 5 > "logs/gpustats-${SLURM_JOB_ID}.csv" &

venv/bin/python train_downscaling.py \
    --var='temperature' \
    --start_year=2000 \
    --end_year=2011 \
    --val_start_year=2012 \
    --val_end_year=2015 \
    --topography_highres_coarsen_factor=5 \
    --topography_lowres_coarsen_factor=5 \
    --era5_coarsen_factor=1 \
    --include_time_of_year=True \
    --include_landmask=True \
    --model_name=$SLURM_JOB_ID \
    --n_epochs=10 \
    --internal_density=250 
