#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

# ./frostt_runner.sh <tensor_names.txt> <sched | unsched> 


cwd=$(pwd)

out=frostt-bench/taco

mkdir -p "$out"

while read line; do
	name=$line
	tensor_path="$FROSTT_PATH/$name.tns"

	csvout="$out/result-$name.csv"
	FROSTT_TENSOR_PATH=$tensor_path make -j8 taco-bench BENCHES="bench_frostt_$2" TACO_OUT="$csvout" GEN=OFF NEVA=ON
done <$1

