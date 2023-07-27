#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

# ./frostt_runner.sh <machine> 
# Arg1 <machine> - Which machine is being used (0:local, 1:Lanka, 2:Kiwi/Neva) 

DATASET_NAMES=(
  fb1k
  fb10k
  facebook
  nell-2
  nell-1
)

cwd=$(pwd)
frostt_path=$FROSTT_PATH

out=frostt-bench/taco

mkdir -p "$out"

while read line; do
	name=${line} 
	csvout="$out/result-$name.csv"
	make -j8 taco-bench BENCHES="$name" GEN=ON
done <$1

