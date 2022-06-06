#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH --exclusive

set -u

sspath=/nobackup/owhsu/sparse-datasets/suitesparse
# out=suitesparse-bench/sam
out=suitesparse-bench/taco

mkdir -p "$out"

while read line; do
	matrix="$sspath/$line.mtx"
	csvout="$out/result-$line.csv"
    GEN=ON SUITESPARSE_TENSOR_PATH="$matrix" make -j8 validate-bench BENCHES="bench_suitesparse" VALIDATION_OUTPUT_PATH="/nobackup/owhsu/validate/taco"
done <$1

