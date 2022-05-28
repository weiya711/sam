#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

# THIS FILE MUST BE RUN FROM sam/ location
outdir=/nobackup/owhsu/sparse-datasets/suitesparse-formatted
basedir=$(pwd)

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

errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color

export SUITESPARSE_PATH=/nobackup/owhsu/sparse-datasets/suitesparse
export SUITESPARSE_FORMATTED_PATH=$outdir

mkdir -p $outdir

for i in ${!DATASET_NAMES[@]}; do
    name=${DATASET_NAMES[$i]} 

    cd $outdir
    echo "Generating input format files for $name..."
    python $basedir/scripts/datastructure_suitesparse.py -n $name 
    chgrp -R sparsity $outdir
    chmod -R 777 $outdir

    cd $basedir/sam/sim/test/apps
    echo "Testing $name..."
    pytest --ssname $name
    status=$?

    if [ $status -gt 0 ]
    then 
      errors+=($name)
    fi

    cd $outdir
    echo "Removing format files for $name..."
    rm ./$name*.txt
done

echo -e "${RED}Failed tests:"
for i in ${!errors[@]}; do
    error=${errors[$i]} 
    echo -e "${RED}$error,"
done
echo -e "${NC}"

