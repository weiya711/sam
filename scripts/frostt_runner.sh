#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive


DATASET_NAMES=(
  fb1k
  fb10k
  facebook
  nell-2
  nell-1
)

cwd=$(pwd)
# LANKA
if [ $1 -eq 1 ]; then
	sspath=/data/scratch/changwan/florida_all/.
	lanka=ON
	neva=OFF
elif [ $1 -eq 2 ]; then
	sspath=/nobackup/owhsu/sparse-datasets/suitesparse
	lanka=OFF
	neva=ON
else
	sspath=cwd/.
	lanka=OFF
	neva=OFF
fi

out=frostt-bench/taco

mkdir -p "$out"

for i in ${!DATASET_NAMES[@]}; do
	name=${DATASET_NAMES[$i]} 
	csvout="$out/result-$name.csv"
	make -j8 taco-bench BENCHES="$name" NEVA=$neva LANKA=$lanka GEN=ON
done

