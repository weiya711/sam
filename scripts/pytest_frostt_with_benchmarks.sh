#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360


BENCHMARKS=(
  tensor3_elemmul
  tensor3_identity
  tensor3_elemadd
  tensor3_ttm
  tensor3_ttv
  tensor3_innerprod
  tensor_mttkrp
)

DATASET_NAMES=(
   facebook
   fb10k
   fb1k
   nell-1
   nell-2
   taco-tensor
)

outdir=/nobackup/owhsu/sparse-datasets/frostt-formatted

export FROSTT_PATH=/nobackup/owhsu/sparse-datasets/frostt
export FROSTT_FORMATTED_PATH=$outdir


errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color


cwd=$(pwd)
resultdir=results


cd ./sam/sim

for b in ${!BENCHMARKS[@]}; do
    bench=${BENCHMARKS[$b]}
    path=$cwd/$resultdir/$bench

    mkdir -p $cwd/$resultdir/$bench
    echo "Testing $bench..."

    for i in ${!DATASET_NAMES[@]}; do
        name=${DATASET_NAMES[$i]} 

        echo "Testing $name..."

        pytest test/apps/test_$bench.py --ssname $name -s --benchmark-json=$path/$name.json 
        python $cwd/scripts/converter.py --json_name $path/$name.json	
            
        status=$?
        if [ $status -gt 0 ]
        then 
          errors+=("${name}, ${bench}")
        fi
    done
    
    python $cwd/scripts/bench_csv_aggregator.py $path $cwd/suitesparse_$bench.csv

done

echo -e "${RED}Failed tests:"
for i in ${!errors[@]}; do
    error=${errors[$i]} 
    echo -e "${RED}$error,"
done
echo -e "${NC}"
