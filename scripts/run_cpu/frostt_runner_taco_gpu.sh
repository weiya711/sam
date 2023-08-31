#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

# ./frostt_runner_taco_gpu.sh <tensor_names.txt> 


cwd=$(pwd)

out=frostt-bench-gpu/taco

# mkdir -p "$out"
# 
# 
export CC=/usr/bin/gcc-7
export CXX=/usr/bin/g++-7

pushd . 
mkdir -p compiler/build/ && cd compiler/build/ && cmake -DOPENMP=ON -DNEVA= OFF -DCUDA=ON -DCMAKE_C_COMPILER=/usr/bin/gcc-7 -DCMAKE_CXX_COMPILER=/usr/bin/g++-7 ../ && make taco-bench -j8
popd

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:compiler/build/lib/:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=2

while read line; do
	name=$line
	tensor_path="$FROSTT_PATH/$name.tns"

	csvout="$out/result-$name.csv"
	CUDA_VISIBLE_DEVICES=2 FROSTT_TENSOR_PATH=$tensor_path GEN=OFF OMP_PROC_BIND=true  compiler/build/taco-bench --benchmark_filter="bench_frostt_gpu" --benchmark_out_format="csv" --benchmark_out="$csvout" --benchmark_repetitions=10 --benchmark_counters_tabular=true

done <$1

