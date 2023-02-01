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
  test_vec_elemmul_bittree
  test_vec_elemmul_bitvector
  test_vec_elemmul_compress_skip
  test_vec_elemmul_compress_split
  test_vec_elemmul_compressed
  test_vec_elemmul_uncompressed
)

cwd=$(pwd)
resultdir=results

for b in ${!BENCHMARKS[@]}; do
    bench=${BENCHMARKS[$b]}
    path=$resultdir/

    mkdir -p $resultdir/
    echo "Testing $bench..."

    pytest sam/sim/test/study-apps/$bench.py --synth --check-gold -k "random-40 or 0.2-blocks or 0.2-runs" --benchmark-json="$path/$bench.json"
    python $cwd/scripts/converter.py --json_name $path/$bench.json
    python $cwd/scripts/bench_csv_aggregator.py $path $cwd/SYNTH_OUT_ACCEL.csv

done

BENCHMARKS=(
    test_reorder_matmul_ijk
    test_reorder_matmul_ikj
    test_reorder_matmul_jik
    test_reorder_matmul_jki
    test_reorder_matmul_kij
    test_reorder_matmul_kji
)

cwd=$(pwd)
resultdir=results_reorder

for b in ${!BENCHMARKS[@]}; do
    bench=${BENCHMARKS[$b]}
    path=$resultdir/

    mkdir -p $resultdir/
    echo "Testing $bench..."

    pytest sam/sim/test/reorder-study/$bench.py --synth --check-gold --benchmark-json="$path/$bench.json"
    python $cwd/scripts/converter.py --json_name $path/$bench.json
    python $cwd/scripts/bench_csv_aggregator.py $path $cwd/SYNTH_OUT_REORDER.csv

done

BENCHMARKS=(
    test_mat_sddmm_coiter_fused
    test_mat_sddmm_locate_fused
    test_mat_sddmm_unfused
)

cwd=$(pwd)
resultdir=results_fusion

for b in ${!BENCHMARKS[@]}; do
    bench=${BENCHMARKS[$b]}
    path=$resultdir/

    mkdir -p $resultdir/
    echo "Testing $bench..."

    pytest sam/sim/test/fusion-study/$bench.py --synth --check-gold --benchmark-json="$path/$bench.json"
    python $cwd/scripts/converter.py --json_name $path/$bench.json
    python $cwd/scripts/bench_csv_aggregator.py $path $cwd/SYNTH_OUT_FUSION.csv

done

# echo -e "${RED}Failed tests:"
# for i in ${!errors[@]}; do
#     error=${errors[$i]} 
#     echo -e "${RED}$error,"
# done
# echo -e "${NC}"
