#!/bin/bash

#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=65536M
#SBATCH -J "monthly-signals"
#SBATCH --mail-user=bpinoli@byu.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

cd ~/sub-industry-correlation
source .venv/bin/activate

# echo "=== Starting overlapping ==="
# python src/signal/naics_strats/EWMA_2_pass/monthly_cum_idio_ret/EWMA_pass2_monthly_overlapping.py

echo "=== Starting non-overlapping ==="
python src/signal/naics_strats/EWMA_2_pass/monthly_cum_idio_ret/EWMA_pass2_monthly_nonoverlapping.py

echo "=== nonoverlapping complete ==="