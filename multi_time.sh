#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks=8
#SBATCH --array=0
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --error=/pbs/home/a/astropart22/private/notebooks/job.%A_%a.err
#SBATCH --output=/pbs/home/a/astropart22/private/notebooks/job.%A_%a.out
#SBATCH --time=10:00:00
#SBATCH --partition=htc

/pbs/software/redhat-9-x86_64/jnp/3.11/bin/python3 -u multi_time_sr.py