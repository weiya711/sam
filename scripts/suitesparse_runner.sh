#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

set -u

cwd=$(pwd)
sspath=$SUITESPARSE_PATH
# LANKA
if [ $2 -eq 1 ]; then
	lanka=ON
	neva=OFF
elif [ $2 -eq 2 ]; then
	lanka=OFF
	neva=ON
else
	lanka=OFF
	neva=OFF
fi

out=suitesparse-bench/taco

mkdir -p "$out"

while read line; do
	if [ $2 -eq 1 ]; then
		matrix="$sspath/$line/$line.mtx"
	elif [ $2 -eq 2 ]; then
		matrix="$sspath/$line.mtx"
	else
		matrix="$sspath/$line.mtx"
	fi
	csvout="$out/result-$line.csv"
	SUITESPARSE_TENSOR_PATH="$matrix" TACO_OUT="$csvout" make -j8 taco-bench BENCHES="bench_suitesparse" NEVA=$neva LANKA=$lanka GEN=ON
done <$1

# for path in $sspath/*; do
#     if [ ! -d $path ]; then
#         continue
#     fi
#     name="$(cut -d'/' -f3 <<< "$path")"
#     matrix="$path/$name.mtx"
# 
#     csvout="$out/result-$name.csv"
# 
#     LANKA=ON SUITESPARSE_TENSOR_PATH="$matrix" TACO_OUT="$csvout" make -j8 taco-bench BENCHES="bench_suitesparse_ufunc"
# done
