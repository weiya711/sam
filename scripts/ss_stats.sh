#!/bin/sh
#SBATCH -N 1 
#SBATCH --exclusive
/home/owhsu/anaconda3/condabin/conda init bash 
/home/owhsu/anaconda3/condabin/conda activate aha

rm -rf /home/owhsu/aha/scadi_graph/scripts/logs 

python suitesparse_stats.py --overall -nstop 250 
