#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

BENCHMARKS=(
#  matmul_kij
#  matmul_kji
#  matmul_ikj
#  matmul_jki
  matmul_ijk
#  matmul_jik
#  mat_elemmul
  mat_identity
#  vecmul_ij
#  vecmul_ji
#  vec_elemmul
#  vec_identity
#  vec_elemadd
#  vec_scalar_mul
#  tensor3_elemmul
#  tensor3_identity
)


DATASET_NAMES=(
  bcsstm04
#  bcsstm02
#  bcsstm03
#  lpi_bgprtr
#  cage4
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
export FROSTT_PATH=/nobackup/owhsu/sparse-datasets/frostt-formatted
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
