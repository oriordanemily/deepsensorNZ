#!/bin/bash -e
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH --gpus-per-node=A100:1
#SBATCH --partition=hgx
#SBATCH --output logs/%j-%x.out
#SBATCH --error logs/%j-%x.out

module purge
module load Python/3.11.6-foss-2023a
. venv/bin/activate

scontrol show job $SLURM_JOB_ID
scontrol write batch_script $SLURM_JOB_ID -

# monitor GPU usage
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv,nounits -l 5 > "logs/gpustats-${SLURM_JOB_ID}.csv" &

export JOBLIB_CACHEDIR=cache

pyinstrument -o logs/profile-${SLURM_JOB_ID}.html -r html \
    train_downscaling.py \
    --var='temperature' \
    --start-year=2000 \
    --end-year=2011 \
    --val-start-year=2012 \
    --val-end-year=2015 \
    --topography-highres-coarsen-factor=5 \
    --topography-lowres-coarsen-factor=5 \
    --era5-coarsen-factor=1 \
    --include-time-of-year \
    --include-landmask \
    --model-name=$SLURM_JOB_ID \
    --n-epochs=1 \
    --internal-density=250 \
    --use-gpu \
    --lr 5e-5