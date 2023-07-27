#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive


DATASET_NAMES=(
  facebook
  rand_tensor19
  facebook_copy
  tensorsample
)

cwd=$(pwd)
# LANKA
if [ $1 -eq 1 ]; then
	#sspath=/data/scratch/changwan/florida_all/.,  this code not used
	sspath=/home/jadivara/sam/sam/onyx
	lanka=ON
	neva=OFF
elif [ $1 -eq 2 ]; then
	#sspath=/nobackup/owhsu/sparse-datasets/suitesparse
	sspath=/home/jadivara/sam/sam/onyx
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
	#print(name)
	csvout="$out/result-$name.csv"
	make -j8 taco-bench BENCHES="$name" NEVA=$neva LANKA=$lanka GEN=ON
done

