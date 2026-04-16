#!/bin/bash

#SBATCH --job-name=nolook_ewma_pass2_overlapping
#SBATCH --output=/home/bpinoli/sub-industry-correlation/src/signal/naics_strats/EWMA_2_pass/nolookahead/slurm_%j.out
#SBATCH --error=/home/bpinoli/sub-industry-correlation/src/signal/naics_strats/EWMA_2_pass/nolookahead/slurm_%j.err
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=65536M
#SBATCH --mail-user=bpinoli@byu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

cd ~/sub-industry-correlation
source .venv/bin/activate

echo "=== Starting nolook EWMA pass2 monthly overlapping ==="
python src/signal/naics_strats/EWMA_2_pass/nolookahead/nolook_EWMA_pass2_monthly_overlapping.py
echo "=== Complete. Output: data/signal_monthly_overlapping_nolook.parquet ==="
