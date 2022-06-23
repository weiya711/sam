#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

# Vars
if [ -z "$2" ]
then
    export SYNTHETIC_PATH="$(pwd)/synthetic/"
else
    export SYNTHETIC_PATH="$2"
fi

BENCHMARKS=(
#  matmul_kij
#  matmul_kji
#  matmul_ikj
#  matmul_jki
  test_vec_elemmul_bittree
  test_vec_elemmul_bitvector
  test_vec_elemmul_compress_skip
  test_vec_elemmul_compress_split
  test_vec_elemmul_compressed
  test_vec_elemmul_uncompressed
#  matmul_jik
#  mat_elemmul
#   mat_identity
#  vecmul_ij
#  vecmul_ji
#  vec_elemmul
#  vec_identity
#  vec_elemadd
#  vec_scalar_mul
#  tensor3_elemmul
#  tensor3_identity
)



# errors=()
# RED='\033[0;31m'
# NC='\033[0m' # No Color


# export SUITESPARSE_PATH=/nobackup/owhsu/sparse-datasets/suitesparse
# export FROSTT_PATH=/nobackup/owhsu/sparse-datasets/frostt-formatted
cwd=$(pwd)
resultdir=results


# cd ./sam/sim

for b in ${!BENCHMARKS[@]}; do
    bench=${BENCHMARKS[$b]}
    path=$resultdir/

    mkdir -p $resultdir/
    echo "Testing $bench..."

    pytest sam/sim/test/study-apps/$bench.py --synth -k "random or 0.2-blocks or 0.2-runs" --benchmark-json="$path/$bench.json"
    python $cwd/scripts/converter.py --json_name $path/$bench.json
    python $cwd/scripts/bench_csv_aggregator.py $path $cwd/SYNTH_OUT.csv
#     for i in ${!DATASET_NAMES[@]}; do
#         name=${DATASET_NAMES[$i]} 

#         echo "Testing $name..."

#         pytest test/apps/test_$bench.py --ssname $name -s --benchmark-json=$path/$name.json 
#         python $cwd/scripts/converter.py --json_name $path/$name.json
            
#         status=$?
#         if [ $status -gt 0 ]
#         then 
#           errors+=("${name}, ${bench}")
#         fi
#     done
    
done

# echo -e "${RED}Failed tests:"
# for i in ${!errors[@]}; do
#     error=${errors[$i]} 
#     echo -e "${RED}$error,"
# done
# echo -e "${NC}"
