#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=128gb
#SBATCH --job-name="gp_$1-$2_$3"
#SBATCH --time=02:00:00
#SBATCH -p short
#SBATCH --output=gp_$1-$2_$3.out
#SBATCH --mail-user=mho026@ucr.edu
#SBATCH --mail-type=ALL

date

# cd $SLURM_SUBMIT_DIR

echo "----"

# loading matlab module
# module load matlab

# run matlab script
matlab -nodesktop -nosplash -r "qso_start_ind = $1; qso_end_ind = $2; num_quasars = $3; offset = $4; parpool('local', 24); run_process_tau0; exit;"

echo "----"

hostname
