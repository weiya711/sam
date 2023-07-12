#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

cwd=$(pwd)
frostt_path=$FROSTT_PATH

out=frostt-bench/taco

mkdir -p "$out"

while read line; do
	name=${line} 
	csvout="$out/result-$name.csv"
	make -j8 taco-bench BENCHES="$name" GEN=ON
done <$1

