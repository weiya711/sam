#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

# ./suitesparse_runner.sh  <tensor_names.txt> <machine> <sched | unshed | mkl>
# Arg1 <tensor_names.txt> - Textfile with names of suitesparse tensors to run
# Arg2 <machine> - Which machine is being used (0:local, 1:Lanka, 2:Kiwi/Neva) 

set -u

cwd=$(pwd)
sspath=$TACO_TENSOR_PATH
# sspath=/home/max/Documents/SPARSE/GPU/mats

out=suitesparse-bench/taco
mkdir -p "$out"

while read line; do
	# if [ $2 -eq 1 ]; then
		# matrix="$sspath/$line/$line.mtx"
    # else
    matrix="$sspath/$line/$line.mtx"
	# fi
	csvout="$out/result-$line.csv"
	echo "RUNNING $line"
    SUITESPARSE_TENSOR_PATH="$matrix" TACO_OUT="$csvout" BENCHES="" make -j cusparse-bench  GEN=OFF
done <$1