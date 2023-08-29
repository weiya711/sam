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
sspath=/home/max/Documents/SPARSE/GPU/mats

out=suitesparse-bench/taco

mkdir -p "$out"
make -j cusparsething

while read line; do
	# if [ $2 -eq 1 ]; then
		# matrix="$sspath/$line/$line.mtx"
    # else
    matrix="$sspath/$line/$line.mtx"
	# fi
	csvout="$out/result-$line.csv"
	export SUITESPARSE_TENSOR_PATH="$matrix"
    export TACO_OUT="$csvout"
    export BENCHES="cusparse_benchmark"
    export GEN=OFF
    ./cusparsething
done <$1