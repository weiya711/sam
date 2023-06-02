#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

BENCHMARKS=(
  matmul_ikj
  matmul_ijk
  matmul_kij
  mat_elemmul
  mat_elemadd
  mat_elemadd3
  mat_residual
  mat_mattransmul
  mat_identity
)

# This is a list of benchmarks that have "other" tensors that are generated
OTHERBENCHES='["mat_residual", "mat_mattransmul", "mat_vecmul"]'

# THIS FILE MUST BE RUN FROM sam/ location
outdir=${SUITESPARSE_FORMATTED_PATH} 
basedir=$(pwd)
textfile=$basedir/$1

mkdir -p $outdir
cd $outdir

for b in ${!BENCHMARKS[@]}; do
	bench=${BENCHMARKS[$b]}
	while read line; do
		name=$line 
		sspath=${SUITESPARSE_PATH}/$name
		echo "Generating input format files for $name..."

		SUITESPARSE_TENSOR_PATH=$sspath python $basedir/scripts/datastructure_suitesparse.py -n $name -hw -b $bench 
		if [[ $OTHERBENCHES =~ "$bench" ]]; then
			echo "Generating format of 'other' tensor"
			python $basedir/scripts/datastructure_tns.py -n $line -f ss01 --other -ss -b $bench -hw
		fi
	
	done <$textfile
done
