#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive


DATASET_NAMES=(
#   facebook
#   rand_tensor19
#   facebook_copy
#   tensorsample

	# rand_tensor19
	# rand_tensor18
	# rand_tensor17
	# rand_tensor16
	# rand_tensor15
	# rand_tensor14
	# rand_tensor13
	# rand_tensor12
	# rand_tensor11
	# rand_tensor10
	# rand_tensor1
	# rand_tensor29
	# rand_tensor28
	# rand_tensor29
	# rand_tensor27
	# rand_tensor26
	# rand_tensor25
	# rand_tensor24
	# rand_tensor23
	# rand_tensor22
	# rand_tensor21
	# rand_tensor20
	# rand_tensor2
	# rand_tensor39
	# rand_tensor38
	# rand_tensor37
	# rand_tensor36
	# rand_tensor35
	# rand_tensor34
	# rand_tensor33
	# rand_tensor32
	# rand_tensor31
	# rand_tensor30
	# rand_tensor3
	# rand_tensor49
	# rand_tensor48
	# rand_tensor47
	# rand_tensor46
	# rand_tensor45
	# rand_tensor44
	# rand_tensor43
	# rand_tensor42
	# rand_tensor41
	# rand_tensor40
	# rand_tensor4
	# rand_tensor50
	# 'rand_tensor5'
	# rand_tensor6
	# rand_tensor7
	# rand_tensor8
	# rand_tensor9
	# rand_tensor31
	# rand_large_tensor1
	# rand_large_tensor2
	# rand_large_tensor3
	# rand_large_tensor4
	# rand_large_tensor5
	# rand_large_tensor6
	# rand_large_tensor7
	# rand_large_tensor8
	# rand_large_tensor9
	rand_large_tensor10
	# rand_large_tensor11
	# rand_large_tensor12
	# rand_large_tensor13
	# rand_large_tensor14
	# rand_large_tensor15
	# rand_large_tensor16
	# rand_large_tensor17
	# rand_large_tensor18
	# rand_large_tensor19
	# rand_large_tensor20
	# rand_large_tensor21
	# rand_large_tensor22
	# rand_large_tensor23
	# rand_large_tensor24
	# rand_large_tensor25
	# rand_large_tensor26
	# rand_large_tensor27
	# rand_large_tensor28
	# rand_large_tensor29
	# rand_large_tensor30
	# rand_large_tensor31
	# rand_large_tensor32
	# rand_large_tensor33
	# rand_large_tensor34
	# rand_large_tensor35
	# rand_large_tensor36
	# rand_large_tensor37
	# rand_large_tensor38
	# rand_large_tensor39
	# rand_large_tensor40
	# rand_large_tensor41
	# rand_large_tensor42
	# rand_large_tensor43
	# rand_large_tensor44
	# rand_large_tensor45
	# rand_large_tensor46
	# rand_large_tensor47
	# rand_large_tensor48
	# rand_large_tensor49
	# rand_large_tensor50
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

