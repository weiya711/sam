#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive


DATASET_NAMES=(
  #fb1k
  #fb10k
  #used bef for testing: facebook
  #newtensor
  #newtensor_correctformat

  #tensor1
  #tensor2
  #tensorspec
  #tensor4
  #tensor5
  tensor6
  tensor7
  tensor8
  tensor9
  tensor10
  tensor11
  tensor12
  tensor13
  tensor14
  tensor15
  tensor16
  tensor17
  tensor18
  tensor19
  tensor20
  tensor21
  tensor22
  tensor23
  tensor24
  tensor25
  tensor26
  tensor27
  tensor28
  tensor29
  tensor30
  tensor31
  tensor32
  tensor33
  tensor34
  tensor35
  tensor36
  tensor37
  tensor38
  tensor39
  tensor40
  tensor41
  tensor42
  tensor43
  tensor44
  tensor45
  tensor46
  tensor47
  tensor48
  tensor49
  tensor50
  #nell-2
  #nell-1
)

cwd=$(pwd)
# LANKA
if [ $1 -eq 1 ]; then
	#sspath=/data/scratch/changwan/florida_all/.
	sspath=/nobackup/jadivara/sam/sam/onyx
	lanka=ON
	neva=OFF
elif [ $1 -eq 2 ]; then
	#sspath=/nobackup/owhsu/sparse-datasets/suitesparse
	sspath=/nobackup/jadivara/sam/sam/onyx
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

