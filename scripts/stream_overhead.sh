#!/bin/bash

basedir=$(pwd)
path=$basedir/jsons

mkdir -p $path

while read line; do
	echo "Generating input format files for $line..."
	SUITESPARSE_TENSOR_PATH=$matrix python $basedir/scripts/datastructure_suitesparse.py -n $line 
	
		
	cd $basedir/sam/sim/test/final-apps/
	pytest test_mat_identity_FINAL.py --ssname $line --check-gold -s --benchmark-json=$path/$line.json

	cd $basedir
done <$1

python $basedir/scripts/bench_csv_aggregator.py $path $basedir/suitesparse_stream_overhead.csv
