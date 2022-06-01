#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

DATASET_NAMES=(
  bcsstm04
  bcsstm02
  bcsstm03
  lpi_bgprtr
  cage4
#  klein-b1
#  GD02_a
#  GD95_b
#  Hamrle1
#  LF10
)

errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color


export SUITESPARSE_PATH=/nobackup/owhsu/sparse-datasets/suitesparse

cwd=$(pwd)
resultdir=results

mkdir -p $cwd/$resultdir

cd ./sam/sim

for i in ${!DATASET_NAMES[@]}; do
    name=${DATASET_NAMES[$i]} 

    echo "Testing $name..."

    pytest test/apps/  --ssname $name -s --benchmark-json=$cwd/$resultdir/$name.json #--debug-sim 
    python $cwd/scripts/converter.py --json_name $cwd/$resultdir/$name.json	
	    
    status=$?
    if [ $status -gt 0 ]
    then 
      errors+=("${name}")
    fi


done

echo -e "${RED}Failed tests:"
for i in ${!errors[@]}; do
    error=${errors[$i]} 
    echo -e "${RED}$error,"
done
echo -e "${NC}"
