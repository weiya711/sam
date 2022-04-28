#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360
outdir=/nobackup/owhsu/sparse-datasets/suitesparse-formatted

DATASET_NAMES=(
  bcsstm04
  bcsstm02
  bcsstm03
  lpi_bgprtr
  cage4
  klein-b1
  GD02_a
  GD95_b
  Hamrle1
  LF10
)

export SUITESPARSE_PATH=/nobackup/owhsu/sparse-datasets/suitesparse
export SUITESPARSE_FORMATTED_PATH=$outdir

mkdir -p $outdir
cd $outdir

for i in ${!DATASET_NAMES[@]}; do
    name=${DATASET_NAMES[$i]} 
    echo "Generating input format files for $name..."
    python /home/owhsu/aha-sparsity/sam/scripts/datastructure_suitesparse.py -n $name
done
