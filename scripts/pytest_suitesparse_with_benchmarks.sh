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
export SUITESPARSE_FORMATTED_PATH=$outdir

cwd=$(pwd)

mkdir -p $outdir
cd ./sam/sim

for i in ${!DATASET_NAMES[@]}; do
    name=${DATASET_NAMES[$i]} 

    echo "Testing $name..."

    pytest test/apps/  --ssname $name -s --benchmark-json=$name.json #--debug-sim 
    $cwd/scripts/converter.py --json_name $name.json	
	    
    status=$?
    if [ $status -gt 0 ]
    then 
      errors+=("${name} matmul_ijk")
    fi


done

echo -e "${RED}Failed tests:"
for i in ${!errors[@]}; do
    error=${errors[$i]} 
    echo -e "${RED}$error,"
done
echo -e "${NC}"
